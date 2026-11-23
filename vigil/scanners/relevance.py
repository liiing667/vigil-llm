"""LLM-backed relevance scanner.

The model is asked to return a small JSON object::

    {"detected": bool, "abbreviated_chunks": [], "irregular": []}

Real models do not always honour that contract: they wrap JSON in markdown or
unknown tags, sometimes return a Python-style dictionary, and occasionally
return a response that is cut off mid-object.  The scanner must treat all of
those responses as evidence about the *same* scan request.  In particular:

* the template is rendered safely (input text never becomes a format field);
* truncated output is retried before the scanner reports a failure;
* a parsed conclusion is the only thing handed to the dispatcher; and
* a response that cannot be parsed is an error, never an empty successful
  scan.
"""

import ast
import json
import logging
import re
import uuid

import yaml

from typing import Any, Dict, List, Optional, Tuple

from vigil.schema import BaseScanner
from vigil.schema import RelevanceMatch
from vigil.schema import ScanModel
from vigil.core.llm import LLM
from vigil.registry import Registration


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RelevanceResponseError(ValueError):
    """The relevance model returned something that is not a conclusion."""


class EmptyResponseError(RelevanceResponseError):
    """The model returned no content at all."""


class TruncatedResponseError(RelevanceResponseError):
    """The model returned JSON that was cut off before it was complete."""


# The template used ``{input_data}`` historically.  Accept the common
# alternatives as well, but only ever substitute the template itself: a
# replacement is never rescanned, so braces/placeholders supplied in the user
# input cannot rewrite another template variable.
_INPUT_VARIABLE_RE = re.compile(
    r"\{\{\s*input_data\s*\}\}"
    r"|\{\s*input_data\s*\}"
    r"|\$\{\s*input_data\s*\}"
    r"|\$\s*input_data\b"
)

_TAG_NAMES = ('detected', 'abbreviated_chunks', 'chunks', 'irregular', 'irregular_chunks')
_LIST_TAG_NAMES = ('abbreviated_chunks', 'chunks', 'irregular', 'irregular_chunks')


def render_prompt(template: str, input_data: str) -> str:
    """Render ``input_data`` into ``template`` without running ``str.format``.

    ``str.format`` interprets every pair of braces in the prompt (including
    the JSON example) as a format field and lets an input that contains braces
    interfere with the template.  A single regular-expression substitution is
    both safer and keeps the literal JSON in the prompt intact.
    """

    template = '' if template is None else str(template)
    input_text = '' if input_data is None else str(input_data)

    rendered, substitutions = _INPUT_VARIABLE_RE.subn(lambda _match: input_text, template)
    if substitutions:
        return rendered

    # Be forgiving for custom prompt files that predate the placeholder.  Put
    # the untrusted text in the same delimiters the built-in prompt uses so
    # the model still knows exactly what it is analysing.
    suffix = '\n\n</!~~TEXT~~>\n{}\n</!~~TEXT~~>\n'.format(input_text)
    return template.rstrip() + suffix


# Friendly/public aliases used by callers and tests.
render_template = render_prompt
format_prompt = render_prompt


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [item if isinstance(item, str) else str(item) for item in value]
        return [item for item in items if item.strip()]
    if isinstance(value, tuple):
        items = [item if isinstance(item, str) else str(item) for item in value]
        return [item for item in items if item.strip()]
    if isinstance(value, dict):
        return [str(item) for item in value.values() if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    text = str(value)
    return [text] if text.strip() else []


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {
            'true', 'yes', 'y', '1', 'detected', 'injection', 'prompt_injection',
            'jailbreak', 'malicious', 'unsafe',
        }
    return bool(value)


def _normalize_conclusion(payload: Any) -> Dict[str, Any]:
    """Return the one conclusion shape used by every consumer."""

    if not isinstance(payload, dict):
        raise RelevanceResponseError(
            'relevance model response must be a JSON object, '
            f'got {type(payload).__name__}'
        )

    # Unknown/extra keys are deliberately ignored.  They are not evidence and
    # must not make an otherwise valid model response fail.
    irregular = payload.get('irregular')
    if irregular is None:
        irregular = payload.get('irregular_chunks')
    irregular = _coerce_list(irregular)
    detected = _coerce_bool(payload.get('detected', False))
    if payload.get('detected') is None and isinstance(payload.get('label'), str):
        detected = _coerce_bool(payload['label'])
    # A non-empty irregular list is itself detection evidence.  Keeping the
    # flag in sync with the evidence prevents the scanner status, the message
    # and the stored match from disagreeing about the same model response.
    detected = detected or bool(irregular)
    chunks = payload.get('abbreviated_chunks')
    if chunks is None:
        chunks = payload.get('chunks')
    abbreviated_chunks = _coerce_list(chunks)

    return {
        'detected': detected,
        'abbreviated_chunks': abbreviated_chunks,
        'irregular': irregular,
    }


_CONCLUSION_KEYS = frozenset(
    ('detected', 'irregular', 'irregular_chunks', 'abbreviated_chunks', 'chunks')
)


def _has_conclusion_keys(payload: Dict[str, Any]) -> bool:
    return any(key in payload for key in _CONCLUSION_KEYS)


def _regex_candidates(text: str) -> List[str]:
    candidates = []

    # Markdown code fences, including ```json and unlabelled fences.
    for match in re.finditer(r'```(?:json|javascript|text)?\s*(.*?)```', text, re.I | re.S):
        candidates.append(match.group(1).strip())

    # Some providers wrap JSON in a single pair of quotes.
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] in ('"', "'") and stripped[-1] == stripped[0]:
        candidates.append(stripped[1:-1].strip())

    return candidates


def _json_slices(text: str) -> Tuple[List[str], bool]:
    """Return complete JSON-looking slices and whether an incomplete one exists."""

    complete: List[str] = []
    incomplete = False

    for start, char in enumerate(text):
        if char not in '{[':
            continue

        depth = 0
        in_string = False
        escaped = False
        closed_at = None

        for index in range(start, len(text)):
            current = text[index]

            if in_string:
                if escaped:
                    escaped = False
                elif current == '\\':
                    escaped = True
                elif current == '"':
                    in_string = False
                continue

            if current == '"':
                in_string = True
            elif current in '{[':
                depth += 1
            elif current in '}]':
                depth -= 1
                if depth == 0:
                    closed_at = index
                    break
                if depth < 0:
                    break

        if closed_at is not None:
            complete.append(text[start:closed_at + 1])
        elif depth > 0 or in_string:
            incomplete = True

    return complete, incomplete


_UNPARSED = object()


def _pythonize_literals(candidate: str) -> str:
    """Turn JSON/JavaScript literals into Python literals outside strings."""

    result: List[str] = []
    index = 0
    quote: Optional[str] = None
    escaped = False

    while index < len(candidate):
        char = candidate[index]

        if quote is not None:
            result.append(char)
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue

        if char in ('"', "'"):
            quote = char
            result.append(char)
            index += 1
            continue

        replaced = False
        for word, replacement in (('true', 'True'), ('false', 'False'), ('null', 'None')):
            end = index + len(word)
            if candidate[index:end].lower() != word:
                continue
            before_ok = index == 0 or not (candidate[index - 1].isalnum() or candidate[index - 1] == '_')
            after_ok = end == len(candidate) or not (candidate[end].isalnum() or candidate[end] == '_')
            if before_ok and after_ok:
                result.append(replacement)
                index = end
                replaced = True
                break

        if not replaced:
            result.append(char)
            index += 1

    return ''.join(result)


def _parse_literal(candidate: str) -> Any:
    """Parse JSON, falling back to a safe Python-literal parser.

    The prompt example itself uses Python-style single quotes, and some models
    copy that style.  ``ast.literal_eval`` handles those responses without
    executing anything.
    """

    try:
        return json.loads(candidate)
    except (TypeError, ValueError):
        pass

    try:
        return json.loads(candidate, strict=False)
    except (TypeError, ValueError):
        pass

    try:
        return ast.literal_eval(candidate)
    except (SyntaxError, ValueError):
        pass

    try:
        return ast.literal_eval(_pythonize_literals(candidate))
    except (SyntaxError, ValueError):
        return _UNPARSED


def _tagged_payload(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort support for models that answer with simple XML-like tags."""

    values: Dict[str, Any] = {}

    for name in _TAG_NAMES:
        match = re.search(
            r'<{0}\b[^>]*>(.*?)</{0}\s*>'.format(re.escape(name)),
            text,
            re.I | re.S,
        )
        if not match:
            continue

        raw_value = match.group(1).strip()
        if name in _LIST_TAG_NAMES:
            parsed = _parse_literal(raw_value)
            values[name] = raw_value if parsed is _UNPARSED else parsed
        elif name == 'detected':
            values[name] = raw_value

    if 'detected' in values or any(name in values for name in _LIST_TAG_NAMES):
        return values
    return None


def _looks_truncated(text: str, had_incomplete_slice: bool) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if had_incomplete_slice:
        return True
    if stripped[0] in '{[' and stripped[-1] not in '}]':
        return True
    # A response that ends inside an unterminated JSON string is also
    # unambiguously truncated.
    if stripped.count('"') % 2 == 1:
        return True
    return False


def parse_model_response(raw_response: Any) -> Dict[str, Any]:
    """Parse a relevance-model response into the shared conclusion shape.

    Raises :class:`TruncatedResponseError` when the response looks cut off so
    the scanner can retry it, and :class:`RelevanceResponseError` for other
    invalid responses.
    """

    if isinstance(raw_response, dict):
        if _has_conclusion_keys(raw_response):
            return _normalize_conclusion(raw_response)

        # Be tolerant of an SDK-style response object if a caller passes the
        # full completion instead of ``content_only`` text.
        choices = raw_response.get('choices')
        if isinstance(choices, list) and choices:
            message = choices[0].get('message') if isinstance(choices[0], dict) else None
            if isinstance(message, dict) and isinstance(message.get('content'), str):
                raw_response = message['content']
            elif isinstance(choices[0], dict) and isinstance(choices[0].get('text'), str):
                raw_response = choices[0]['text']
        elif isinstance(raw_response.get('content'), str):
            raw_response = raw_response['content']
        elif isinstance(raw_response.get('response'), str):
            raw_response = raw_response['response']

        if isinstance(raw_response, dict):
            return _normalize_conclusion(raw_response)

    if raw_response is None:
        raise EmptyResponseError('relevance model returned an empty response')

    if isinstance(raw_response, (bytes, bytearray)):
        raw_response = raw_response.decode('utf-8', errors='replace')

    text = str(raw_response).lstrip('\ufeff').strip()
    if not text:
        raise EmptyResponseError('relevance model returned an empty response')

    candidates: List[str] = [text]
    wrapped_candidates = _regex_candidates(text)
    candidates.extend(wrapped_candidates)
    slices, had_incomplete_slice = _json_slices(text)
    # Prefer the largest complete object.  Wrappers such as unknown tags or
    # prose around the JSON only add noise; the largest slice is the model's
    # actual answer.
    candidates.extend(sorted(slices, key=len, reverse=True))

    whole_candidates = {text.strip()}
    whole_candidates.update(candidate.strip() for candidate in wrapped_candidates)

    parsed_candidates: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not candidate:
            continue
        parsed = _parse_literal(candidate)
        if parsed is _UNPARSED:
            continue
        # A nested object from a truncated outer response is not the model's
        # conclusion.  Accept an unknown-key object only when it is the whole
        # response (or a complete fenced block); otherwise keep looking and
        # let truncation detection report the cut-off response.
        if isinstance(parsed, dict) \
                and not _has_conclusion_keys(parsed) \
                and candidate.strip() not in whole_candidates:
            continue
        try:
            parsed_candidates.append(_normalize_conclusion(parsed))
        except RelevanceResponseError:
            continue

    if parsed_candidates:
        # Prefer a conclusion that actually carries a detection signal.  This
        # avoids selecting a nested incidental object when the response also
        # contains the real top-level answer.
        for conclusion in parsed_candidates:
            if conclusion.get('detected') or conclusion.get('irregular'):
                return conclusion
        return parsed_candidates[0]

    tagged = _tagged_payload(text)
    if tagged is not None:
        return _normalize_conclusion(tagged)

    if _looks_truncated(text, had_incomplete_slice):
        raise TruncatedResponseError(
            'relevance model returned truncated JSON: {!r}'.format(text[:200])
        )

    raise RelevanceResponseError(
        'relevance model returned invalid JSON: {!r}'.format(text[:200])
    )


# Friendly/public aliases.  The scanner itself only calls one of these.
parse_response = parse_model_response
parse_llm_response = parse_model_response
parse_json_response = parse_model_response


@Registration.scanner(name='relevance', requires_config=True)
class RelevanceScanner(BaseScanner):
    def __init__(self, config_dict: dict):
        self.name = 'scanner:relevance'
        self.prompt_path = config_dict.get('prompt') or config_dict.get('prompt_path')

        if self.prompt_path is None:
            logger.error(f'[{self.name}] prompt path is not defined; check config')
            raise ValueError('[scanner:relevance] prompt path is not defined')

        try:
            retries = config_dict.get('max_retries', config_dict.get('retries', 1))
            self.max_retries = max(0, int(retries))
        except (TypeError, ValueError):
            self.max_retries = 1

        # Optional dependency injection is useful for tests and lets a caller
        # share a pre-configured client.  Production configuration continues
        # to construct the normal LiteLLM client.
        provided_llm = config_dict.get('llm')
        if provided_llm is not None:
            self.llm = provided_llm
        else:
            model_name = config_dict.get('model_name') or config_dict.get('model')
            if not model_name:
                logger.error(f'[{self.name}] model name is not defined; check config')
                raise ValueError('[scanner:relevance] model name is not defined')

            self.llm = LLM(
                model_name=model_name,
                api_key=config_dict.get('api_key'),
                api_base=config_dict.get('api_base')
            )

    def load_prompt(self) -> dict:
        logger.info(f'[{self.name}] Loading prompt from {self.prompt_path}')

        with open(self.prompt_path, 'r') as fp:
            data = yaml.safe_load(fp)
        return data or {}

    def build_prompt(self, input_data: str) -> str:
        prompt = self.load_prompt().get('prompt', '')
        return render_prompt(prompt, input_data)

    def analyze(self, scan_obj: Any, scan_id: uuid.UUID = None) -> ScanModel:
        logger.info(f'[{self.name}] performing scan; id="{scan_id}"')

        # The dispatcher passes a ScanModel, but keep accepting a raw prompt
        # for callers of the older scanner API.
        if isinstance(scan_obj, ScanModel):
            model = scan_obj
            input_data = scan_obj.prompt
        elif hasattr(scan_obj, 'prompt'):
            model = scan_obj
            input_data = getattr(scan_obj, 'prompt', '') or ''
        else:
            input_data = '' if scan_obj is None else str(scan_obj)
            model = ScanModel(prompt=input_data)

        prompt = self.build_prompt(input_data)

        attempts = self.max_retries + 1
        last_error: Optional[RelevanceResponseError] = None

        for attempt in range(1, attempts + 1):
            try:
                output = self.llm.generate(prompt, content_only=True)
                logger.info(f'[{self.name}] LLM output: {output}')
            except Exception as err:
                logger.error(f'[{self.name}] Failed to perform relevance scan (call to LLM): {err}')
                if last_error is not None:
                    # The retry itself failed after a truncated/empty response.
                    # Preserve the parse failure (and therefore the manager's
                    # failed/partial status) rather than turning it into a
                    # generic transport error.
                    raise last_error.__class__(
                        '{}; retry failed: {}'.format(last_error, err)
                    ) from err
                raise

            try:
                conclusion = parse_model_response(output)
            except EmptyResponseError as err:
                last_error = err
                logger.warning(
                    f'[{self.name}] empty model response; attempt={attempt}/{attempts} id="{scan_id}"'
                )
                if attempt < attempts:
                    continue
                raise
            except TruncatedResponseError as err:
                last_error = err
                logger.warning(
                    f'[{self.name}] truncated model response; retrying; '
                    f'attempt={attempt}/{attempts} id="{scan_id}"'
                )
                if attempt < attempts:
                    continue
                raise
            except RelevanceResponseError:
                # Unknown wrappers/tags and extra JSON keys are handled by the
                # parser.  A genuinely malformed answer is not a successful
                # empty scan and must reach the manager as an error.
                raise

            if conclusion['detected'] or conclusion['irregular']:
                model.results.append(
                    RelevanceMatch(
                        detected=conclusion['detected'],
                        abbreviated_chunks=conclusion['abbreviated_chunks'],
                        irregular=conclusion['irregular'],
                    )
                )
            else:
                logger.info(
                    f'[{self.name}] no relevance evidence detected; id="{scan_id}"'
                )

            return model

        # Only reachable if a retry path above was changed to fall through.
        if last_error is not None:
            raise last_error
        raise RelevanceResponseError('relevance scan failed without a model response')
