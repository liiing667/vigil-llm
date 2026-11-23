import os

from loguru import logger

from typing import List, Dict, Optional, Callable

from vigil.dispatch import Manager
from vigil.schema import BaseScanner

from vigil.core.config import Config
from vigil.core.canary import CanaryTokens
from vigil.core.vectordb import VectorDB
from vigil.core.embedding import Embedder
from vigil.core.cache import compute_generation

from vigil.registry import ScannerRegistry


SECRET_HINTS = ('key', 'secret', 'token', 'password')


def _redact_secrets(section: Dict) -> Dict:
    """Drop secret-looking values so they neither leak into nor needlessly
    churn the generation fingerprint."""
    return {
        k: v for k, v in section.items()
        if not any(hint in k.lower() for hint in SECRET_HINTS)
    }


class Vigil:
    vectordb: Optional[VectorDB] = None
    embedder: Optional[Callable] = None

    def __init__(self, config_path: str):
        self._config = Config(config_path)
        self._initialize_vectordb()
        self._initialize_embedder()
        
        self._input_scanners: List[BaseScanner] = self._setup_scanners(
            self._config.get_scanner_names('input_scanners')
        )
        self._output_scanners: List[BaseScanner] = self._setup_scanners(
            self._config.get_scanner_names('output_scanners')
        )

        self.canary_tokens = CanaryTokens()
        self.input_scanner = self._create_manager(
            name='input',
            scanners=self._input_scanners
        )
        self.output_scanner = self._create_manager(
            name='output',
            scanners=self._output_scanners
        )

    def _initialize_embedder(self):
        full_config = self._config.get_general_config()
        params = full_config.get('embedding', {})
        self.embedder = Embedder(**params)

    def _initialize_vectordb(self):
        full_config = self._config.get_general_config()
        params = full_config.get('vectordb', {})
        params.update(full_config.get('embedding', {}))
        self.vectordb = VectorDB(**params)

    def _setup_scanners(self, scanner_names: List[str]) -> List[BaseScanner]:
        scanners = []

        for name in scanner_names:
            try:
                metadata = ScannerRegistry.get_scanner_metadata(name)
            except ValueError as err:
                logger.error(err)
                raise err

            scanner_config = None
            vectordb = None
            embedder = None

            if metadata.get('requires_config', False):
                scanner_config = self._config.get_scanner_config(name)

            if metadata.get('requires_vectordb', False):
                vectordb = self.vectordb
            
            if metadata.get('requires_embedding', False):
                embedder = self.embedder

            scanner = ScannerRegistry.create_scanner(
                name=name,
                config=scanner_config,
                vectordb=vectordb,
                embedder=embedder
            )
            scanners.append(scanner)

        return scanners

    def _create_manager(self, name: str, scanners: List[BaseScanner]) -> Manager:
        manager_config = self._config.get_general_config()
        auto_update = manager_config.get('auto_update', {}).get('enabled', False)
        update_threshold = int(manager_config.get('auto_update', {}).get('threshold', 3))

        return Manager(
            name=name,
            scanners=scanners,
            auto_update=auto_update,
            update_threshold=update_threshold,
            db_client=self.vectordb if auto_update else None,
            generation=self._manager_generation(name, scanners)
        )

    def _manager_generation(self, name: str, scanners: List[BaseScanner]) -> str:
        """Fingerprint binding this manager's scanner set and the config
        sections that influence scan outcomes. Any config or scanner-set
        change produces a new generation, so stale cache entries keyed under
        an older generation are never read."""
        general = self._config.get_general_config()

        scanner_configs = {}
        for scanner in scanners:
            section = scanner.name  # e.g. 'scanner:yara'
            if self._config.config.has_section(section):
                scanner_configs[section] = _redact_secrets(
                    self._config.get_scanner_config(section.split(':', 1)[1])
                )

        material = {
            'manager': name,
            'scanners': [scanner.name for scanner in scanners],
            'scanner_configs': scanner_configs,
            'embedding': _redact_secrets(general.get('embedding', {})),
            'vectordb': _redact_secrets(general.get('vectordb', {})),
            'auto_update': general.get('auto_update', {}),
        }

        generation = compute_generation(material)
        logger.info(f'dispatch:{name} Pipeline generation: {generation}')
        return generation

    @staticmethod
    def from_config(config_path: str) -> 'Vigil':
        return Vigil(config_path=config_path)
