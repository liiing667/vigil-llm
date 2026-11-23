import uuid
import math

from loguru import logger

from typing import List, Dict

from vigil.schema import BaseScanner
from vigil.schema import ScanModel
from vigil.schema import ResponseModel

from vigil.common import timestamp_str


messages = {
    'scanner:yara': 'Potential prompt injection detected: YARA signature(s)',
    'scanner:transformer': 'Potential prompt injection detected: transformer model',
    'scanner:vectordb': 'Potential prompt injection detected: vector similarity',
    'scanner:response-similarity': 'Potential prompt injection detected: prompt-response similarity',
    'scanner:relevance': 'Potential prompt injection detected: relevance analysis',
}


def calculate_entropy(text) -> float:
    if not text:
        return 0.0
    prob = [text.count(c) / len(text) for c in set(text)]
    entropy = -sum(p * math.log2(p) for p in prob)
    return entropy


def _serialize_match(result) -> dict:
    """Serialize any evidence object produced by a scanner."""
    if isinstance(result, dict):
        return result
    if hasattr(result, 'dict'):
        return result.dict()
    return result


def _result_is_error(results) -> bool:
    return isinstance(results, dict) and 'error' in results


def _result_is_empty_failure(results) -> bool:
    """An absent/blank scanner result is a failure, not a clean scan."""
    if results is None:
        return True
    if isinstance(results, str) and not results.strip():
        return True
    if isinstance(results, (bytes, bytearray)) and not results.strip():
        return True
    return False


def _matches_from_results(results):
    """Normalize the different shapes scanners/tests can return into a list."""
    if isinstance(results, dict):
        if 'matches' in results:
            return results['matches']
        # A single evidence dictionary (for example a parsed relevance
        # conclusion) is still one match, not a successful empty scan.
        if 'detected' in results or 'irregular' in results:
            return [results]
        return []
    if results is None:
        return []
    if isinstance(results, (list, tuple)):
        return list(results)
    return [results]


def _has_detection(scanner_name: str, matches) -> bool:
    matches = _matches_from_results(matches)
    if scanner_name == 'scanner:relevance':
        for match in matches:
            if not isinstance(match, dict) and hasattr(match, 'dict'):
                match = match.dict()
            if not isinstance(match, dict):
                continue
            if match.get('detected'):
                return True
            if match.get('irregular'):
                return True
        return False
    return len(matches) > 0


class Manager:
    def __init__(self,
                 scanners: List[BaseScanner],
                 auto_update: bool = False,
                 update_threshold: int = 3,
                 db_client=None,
                 name: str = 'input'):
        self.name = f'dispatch:{name}'
        self.dispatcher = Scanner(scanners)
        self.auto_update = auto_update
        self.update_threshold = update_threshold
        self.db_client = db_client

        if self.auto_update:
            if self.db_client is None:
                logger.warn(f'{self.name} Auto-update disabled: db client is None')
            else:
                logger.info(f'{self.name} Auto-update vectordb enabled: threshold={self.update_threshold}')

    def perform_scan(self, prompt: str, prompt_response: str = None) -> dict:
        resp = ResponseModel(
            status='success',
            prompt=prompt,
            prompt_response=prompt_response,
            prompt_entropy=calculate_entropy(prompt),
        )

        resp.uuid = str(resp.uuid)

        if not prompt:
            resp.errors.append('Input prompt value is empty')
            resp.status = 'failed'
            logger.error(f'{self.name} Input prompt value is empty')
            return resp.dict()

        logger.info(f'{self.name} Dispatching scan request id={resp.uuid}')

        try:
            scan_results = self.dispatcher.run(
                prompt=prompt,
                prompt_response=prompt_response,
                scan_id=resp.uuid
            )
        except Exception as err:
            resp.status = 'failed'
            resp.errors.append(f'Error dispatching scan: {err}')
            logger.error(f'{self.name} Error dispatching scan id={resp.uuid}: {err}')
            return resp.dict()

        total_matches = 0
        error_count = 0
        success_count = 0

        for scanner_name, results in scan_results.items():
            if _result_is_error(results):
                error_count += 1
                resp.errors.append(f'Error in {scanner_name}: {results["error"]}')
            elif _result_is_empty_failure(results):
                error_count += 1
                resp.errors.append(
                    f'Error in {scanner_name}: scanner returned an empty result'
                )
            else:
                success_count += 1
                matches = [
                    _serialize_match(match)
                    for match in _matches_from_results(results)
                ]
                resp.results[scanner_name] = {'matches': matches}
                if _has_detection(scanner_name, matches) and scanner_name != 'scanner:sentiment':
                    total_matches += 1

        # A scanner error must never be hidden by the empty results of the
        # other scanners.  If every scanner failed the request failed; if only
        # some failed it is a partial success.
        if error_count:
            resp.status = 'failed' if success_count == 0 else 'partial_success'
        else:
            resp.status = 'success'

        for scanner_name, message in messages.items():
            scanner_result = resp.results.get(scanner_name)
            if scanner_result is not None \
                    and _has_detection(scanner_name, scanner_result.get('matches', [])) \
                    and message not in resp.messages:
                resp.messages.append(message)

        logger.info(f'{self.name} Total scanner matches: {total_matches}')
        if self.auto_update and self.db_client is not None \
                and (total_matches >= self.update_threshold):
            logger.info(f'{self.name} (auto-update) Adding detected prompt to db id={resp.uuid}')
            doc_id = self.db_client.add_texts(
                [prompt],
                [
                    {
                        'uuid': resp.uuid,
                        'source': 'auto-update',
                        'timestamp': timestamp_str(),
                        'threshold': self.update_threshold
                    }
                ]
            )
            logger.success(f'{self.name} (auto-update) Successful doc_id={doc_id} id={resp.uuid}')

        logger.info(f'{self.name} Returning response object id={resp.uuid}')

        return resp.dict()


class Scanner:
    def __init__(self, scanners: List[BaseScanner]):
        self.name = 'dispatch:scan'
        self.scanners = scanners

    def run(self, prompt: str, scan_id: uuid.uuid4, prompt_response: str = None) -> Dict:
        response = {}

        for scanner in self.scanners:
            scan_obj = ScanModel(
                prompt=prompt,
                prompt_response=prompt_response
            )

            try:
                logger.info(f'Running scanner: {scanner.name}; id={scan_id}')

                updated = scanner.analyze(scan_obj, scan_id)
                if hasattr(updated, 'results'):
                    matches = updated.results
                elif isinstance(updated, dict) and 'results' in updated:
                    matches = updated['results']
                elif isinstance(updated, (list, tuple)):
                    matches = updated
                else:
                    raise TypeError(
                        f'{scanner.name} returned {type(updated).__name__}; '
                        'expected a ScanModel'
                    )
                response[scanner.name] = [_serialize_match(res) for res in matches]
                logger.success(f'Successfully ran scanner: {scanner.name} id={scan_id}')

            except Exception as err:
                logger.error(f'Failed to run scanner: {scanner.name}, Error: {str(err)} id={scan_id}')
                response[scanner.name] = {'error': str(err)}

        return response
