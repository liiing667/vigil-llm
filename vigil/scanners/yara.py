import os
import yara
import threading
import uuid

from loguru import logger

from vigil.schema import YaraMatch
from vigil.schema import ScanModel
from vigil.schema import BaseScanner

from vigil.registry import Registration


@Registration.scanner(name='yara', requires_config=True)
class YaraScanner(BaseScanner):
    def __init__(self, rules_dir: str):
        self.name = 'scanner:yara'
        self.rules_dir = rules_dir
        self.compiled_rules = None
        self.rule_sources = {}
        self._load_lock = threading.Lock()

        if not os.path.exists(self.rules_dir):
            logger.error(f'Directory not found: {self.rules_dir}')
            raise Exception

        if not os.path.isdir(self.rules_dir):
            logger.error(f'Path is not a valid directory: {self.rules_dir}')
            raise Exception
        
        self.load_rules()
        logger.success('Loaded scanner')

    def discover_rules(self) -> dict:
        """Discover loadable YARA rule files in the rules directory.

        Returns a dict mapping namespace (bare file name) to the full file
        path. Only regular files with a .yar/.yara extension are included;
        directories, dotfiles, and any other non-rule files are excluded so
        they can never leak into compilation.
        """
        discovered = {}
        for entry in sorted(os.listdir(self.rules_dir)):
            full_path = os.path.join(self.rules_dir, entry)

            if entry.startswith('.'):
                continue

            if not os.path.isfile(full_path):
                logger.warning(f'Skipping non-file entry in rules directory: {entry}')
                continue

            if not self.is_yara_file(entry):
                logger.warning(f'Skipping non-rule file in rules directory: {entry}')
                continue

            discovered[entry] = full_path

        return discovered

    def load_rules(self) -> bool:
        """Compile all YARA rules in a directory and atomically swap them in.

        The new rule set is fully discovered and compiled before it replaces
        the current generation, so scanning consumers never observe a
        half-updated rule set. If discovery or compilation fails, the
        previous generation is kept and False is returned. On the initial
        load (no usable generation yet) failures are raised instead.
        """
        with self._load_lock:
            logger.info(f'Loading rules from directory: {self.rules_dir}')
            yara_paths = self.discover_rules()

            if len(yara_paths) == 0:
                logger.error(f'No YARA rules found in directory: {self.rules_dir}')
                if self.compiled_rules is None:
                    raise Exception(f'No YARA rules found in directory: {self.rules_dir}')
                logger.warning('Keeping previous rule set')
                return False

            try:
                compiled_rules = yara.compile(filepaths=yara_paths)
            except Exception as err:
                logger.error(f'YARA compilation error: {err}')
                if self.compiled_rules is None:
                    raise err
                logger.warning('Compilation failed; keeping previous rule set')
                return False

            # Atomic generation swap: compiled rules and their source
            # mapping are only published together, as a complete set.
            self.compiled_rules = compiled_rules
            self.rule_sources = yara_paths

            logger.success(f'Loaded {len(yara_paths)} YARA rule file(s): {sorted(yara_paths.keys())}')
            return True

    def loaded_rule_names(self) -> list:
        """Names (namespaces) of the currently active rule generation"""
        return sorted(self.rule_sources.keys())

    def is_yara_file(self, file_path: str) -> bool:
        """Check if file is rule by extension"""
        if file_path.lower().endswith('.yara') or file_path.lower().endswith('.yar'):
            return True
        return False

    def analyze(self, scan_obj: ScanModel, scan_id: uuid.uuid4) -> ScanModel:
        """Run scan against input data and return list of YaraMatchs"""
        logger.info(f'Performing scan; id="{scan_id}"')

        if scan_obj.prompt.strip() == '':
            logger.error(f'No input data; id="{scan_id}"')
            return scan_obj

        # Snapshot the current generation so a concurrent reload can never
        # swap rules mid-scan.
        compiled_rules = self.compiled_rules
        if compiled_rules is None:
            logger.error(f'No compiled YARA rules loaded; id="{scan_id}"')
            return scan_obj

        try:
            matches = compiled_rules.match(data=scan_obj.prompt)
        except Exception as err:
            logger.error(f'Failed to perform yara scan; id="{scan_id}" error="{err}"')
            return scan_obj

        for match in matches:
            m = YaraMatch(rule_name=match.rule, tags=match.tags, category=match.meta.get('category', None))
            logger.warning(f'Matched rule rule="{m.rule_name} tags="{m.tags}" category="{m.category}"')
            scan_obj.results.append(m)

        if len(scan_obj.results) == 0:
            logger.info(f'No matches found; id="{scan_id}"')

        return scan_obj
