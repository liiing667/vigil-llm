import os
import yara
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
        # Explicit compiler health state shared by load/scan/reload paths:
        #   'healthy'   -> compiled_rules is usable
        #   'unhealthy' -> last compile/scan failed; scanner must not report
        #                  an empty result list as a legitimate "no match"
        #   'empty'     -> rules dir contains no YARA rule files
        self.status = 'unhealthy'
        self.last_error = None

        if not os.path.exists(self.rules_dir):
            logger.error(f'Directory not found: {self.rules_dir}')
            raise FileNotFoundError(f'Directory not found: {self.rules_dir}')

        if not os.path.isdir(self.rules_dir):
            logger.error(f'Path is not a valid directory: {self.rules_dir}')
            raise NotADirectoryError(f'Path is not a valid directory: {self.rules_dir}')
        
        self.load_rules()
        logger.success('Loaded scanner')

    def load_rules(self) -> bool:
        """Compile all YARA rules in a directory and store in memory.

        Returns True when the compiler is healthy afterwards, False otherwise.
        Always leaves ``self.status``/``self.last_error`` consistent with the
        outcome so callers never have to infer health from ``compiled_rules``
        alone.
        """
        logger.info(f'Loading rules from directory: {self.rules_dir}')
        rules = os.listdir(self.rules_dir)

        yara_paths = {}
        for _file in rules:
            if self.is_yara_file(_file):
                yara_paths[_file] = os.path.join(self.rules_dir, _file)

        if len(yara_paths) == 0:
            logger.warning(f'No YARA rule files found in directory: {self.rules_dir}')
            self.compiled_rules = None
            self.status = 'empty'
            self.last_error = None
            return False

        try:
            self.compiled_rules = yara.compile(filepaths=yara_paths)
        except Exception as err:
            logger.error(f'YARA compilation error: {err}')
            self.compiled_rules = None
            self.status = 'unhealthy'
            self.last_error = str(err)
            return False

        self.status = 'healthy'
        self.last_error = None
        logger.success(f'Compiled {len(yara_paths)} YARA rule file(s)')
        return True

    def reload_rules(self) -> bool:
        """Recompile rules from disk and return the resulting health state.

        Used to recover after a scan error. A failed reload keeps the scanner
        unhealthy instead of silently falling back to the broken compiler.
        """
        logger.info(f'Reloading YARA rules from directory: {self.rules_dir}')
        return self.load_rules()

    @property
    def is_healthy(self) -> bool:
        """True only when a compiled rule set is available for scanning."""
        return self.status == 'healthy' and self.compiled_rules is not None

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

        if not self.is_healthy:
            error = (
                f'YARA scanner is not healthy (status="{self.status}"); '
                f'last_error="{self.last_error}"; id="{scan_id}"'
            )
            logger.error(error)
            raise RuntimeError(error)

        try:
            matches = self.compiled_rules.match(data=scan_obj.prompt)
        except Exception as err:
            logger.error(
                f'Failed to perform yara scan; id="{scan_id}" error="{err}"'
            )
            # Mark the compiler broken and attempt recovery. The empty result
            # list must never be mistaken for a legitimate "no match": raise
            # so the dispatcher records the error instead of a clean scan.
            self.status = 'unhealthy'
            self.last_error = str(err)
            recovered = self.reload_rules()
            logger.info(
                f'YARA reload after scan error; recovered="{recovered}" '
                f'status="{self.status}" id="{scan_id}"'
            )
            raise RuntimeError(
                f'YARA scan failed: {err}; reload_recovered={recovered}; '
                f'status={self.status}; id="{scan_id}"'
            ) from err

        for match in matches:
            m = YaraMatch(rule_name=match.rule, tags=match.tags, category=match.meta.get('category', None))
            logger.warning(f'Matched rule rule="{m.rule_name} tags="{m.tags}" category="{m.category}"')
            scan_obj.results.append(m)

        if len(scan_obj.results) == 0:
            logger.info(f'No matches found; id="{scan_id}"')

        return scan_obj
