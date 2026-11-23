"""Configuration-startup tests for the generation-aware registry.

These tests mirror ``Vigil._setup_scanners`` end to end using the real
``Config`` parser and a generation-pinned ``RegistrySnapshot``: names read
from a config file must resolve to the class registered in the pinned
generation, missing sections/requirements must produce ``RegistryError``
instances rather than silently resolving to the wrong class, and built-in
plugins plus empty configurations keep working.
"""
import os
import sys
import tempfile
import unittest

from tests._stubs import install_stubs

install_stubs()

sys.path.insert(0, '.')

from vigil.core.config import Config
from vigil.registry import DuplicateRegistrationError
from vigil.registry import Registration
from vigil.registry import RegistryError
from vigil.registry import StaleRegistrationError
from vigil.registry import ScannerRegistry


CONFIG_TEMPLATE = """
[scanners]
input_scanners = {input_scanners}
output_scanners = {output_scanners}

[scanner:configured]
threshold = 0.7
model = demo-model

[scanner:database]
threshold = 0.4

[scanner:embedding]
threshold = 0.2
"""


class ConfiguredScanner:
    name = 'scanner:configured'

    def __init__(self, threshold=None, model=None):
        self.threshold = threshold
        self.model = model

    def analyze(self, scan_obj, scan_id):
        return scan_obj


class DatabaseScanner:
    name = 'scanner:database'

    def __init__(self, db_client=None, threshold=None):
        self.db_client = db_client
        self.threshold = threshold

    def analyze(self, scan_obj, scan_id):
        return scan_obj


class EmbeddingScanner:
    name = 'scanner:embedding'

    def __init__(self, embedder=None, threshold=None):
        self.embedder = embedder
        self.threshold = threshold

    def analyze(self, scan_obj, scan_id):
        return scan_obj


class NoConfigScanner:
    name = 'scanner:no-config'

    def __init__(self):
        pass

    def analyze(self, scan_obj, scan_id):
        return scan_obj


def write_config(input_scanners='configured', output_scanners=''):
    handle, path = tempfile.mkstemp(suffix='.conf')
    with os.fdopen(handle, 'w') as handle_obj:
        handle_obj.write(
            CONFIG_TEMPLATE.format(
                input_scanners=input_scanners,
                output_scanners=output_scanners
            )
        )
    return path


def build_scanners(config, snapshot, names, vectordb=None, embedder=None):
    """Replicate Vigil._setup_scanners against a pinned registry snapshot."""
    scanners = []
    for name in names:
        metadata = snapshot.get_scanner_metadata(name)

        scanner_config = None
        if metadata.get('requires_config', False):
            scanner_config = config.get_scanner_config(name)

        scanner_vectordb = vectordb if metadata.get('requires_vectordb') else None
        scanner_embedder = embedder if metadata.get('requires_embedding') else None

        scanners.append(
            snapshot.create_scanner(
                name=name,
                config=scanner_config,
                vectordb=scanner_vectordb,
                embedder=scanner_embedder
            )
        )
    return scanners


class ConfigStartupTestBase(unittest.TestCase):
    def setUp(self):
        ScannerRegistry.reset_registry()
        ScannerRegistry.register_scanner(
            'configured', ConfiguredScanner, requires_config=True
        )
        ScannerRegistry.register_scanner(
            'database',
            DatabaseScanner,
            requires_config=True,
            requires_vectordb=True
        )
        ScannerRegistry.register_scanner(
            'embedding',
            EmbeddingScanner,
            requires_config=True,
            requires_embedding=True
        )
        ScannerRegistry.register_scanner('no-config', NoConfigScanner)
        self.config_path = write_config()
        self.config = Config(self.config_path)

    def tearDown(self):
        if os.path.exists(self.config_path):
            os.remove(self.config_path)


class ConfigResolutionTests(ConfigStartupTestBase):
    def test_config_section_resolves_to_registered_class(self):
        snapshot = ScannerRegistry.snapshot()
        names = self.config.get_scanner_names('input_scanners')
        scanners = build_scanners(self.config, snapshot, names)

        self.assertEqual(len(scanners), 1)
        scanner = scanners[0]
        self.assertIsInstance(scanner, ConfiguredScanner)
        self.assertEqual(scanner.threshold, '0.7')
        self.assertEqual(scanner.model, 'demo-model')
        self.assertEqual(
            scanner.registry_generation,
            ScannerRegistry.current_generation()
        )

    def test_missing_config_section_raises_registry_error(self):
        path = write_config(input_scanners='ghost')
        try:
            config = Config(path)
            snapshot = ScannerRegistry.snapshot()
            # Name absent from the registry is rejected first.
            with self.assertRaises(RegistryError):
                build_scanners(
                    config,
                    snapshot,
                    config.get_scanner_names('input_scanners')
                )

            # Name present in the registry but absent from the config file
            # still fails because the scanner requires configuration.
            ScannerRegistry.register_scanner(
                'section-missing', ConfiguredScanner, requires_config=True
            )
            config_without_section = Config(path)
            self.assertEqual(
                config_without_section.get_scanner_config('section-missing'),
                {}
            )
            with self.assertRaises(RegistryError):
                ScannerRegistry.create_scanner(
                    'section-missing',
                    config=config_without_section.get_scanner_config(
                        'section-missing'
                    )
                )
        finally:
            os.remove(path)

    def test_scanner_without_config_accepts_empty_configuration(self):
        path = write_config(input_scanners='no-config')
        try:
            config = Config(path)
            snapshot = ScannerRegistry.snapshot()
            scanners = build_scanners(
                config,
                snapshot,
                config.get_scanner_names('input_scanners')
            )
            self.assertIsInstance(scanners[0], NoConfigScanner)
        finally:
            os.remove(path)

    def test_vectordb_and_embedder_dependencies_are_reported(self):
        path = write_config(
            input_scanners='database', output_scanners='embedding'
        )
        try:
            config = Config(path)
            snapshot = ScannerRegistry.snapshot()

            with self.assertRaises(RegistryError):
                build_scanners(
                    config,
                    snapshot,
                    config.get_scanner_names('input_scanners')
                )
            with self.assertRaises(RegistryError):
                build_scanners(
                    config,
                    snapshot,
                    config.get_scanner_names('output_scanners')
                )

            db_client = object()
            embedder = object()
            db_scanners = build_scanners(
                config,
                snapshot,
                config.get_scanner_names('input_scanners'),
                vectordb=db_client
            )
            embed_scanners = build_scanners(
                config,
                snapshot,
                config.get_scanner_names('output_scanners'),
                embedder=embedder
            )
            self.assertIs(db_scanners[0].db_client, db_client)
            self.assertEqual(db_scanners[0].threshold, '0.4')
            self.assertIs(embed_scanners[0].embedder, embedder)
            self.assertEqual(embed_scanners[0].threshold, '0.2')
        finally:
            os.remove(path)


class ConfigGenerationTests(ConfigStartupTestBase):
    def test_duplicate_registration_does_not_change_resolution(self):
        snapshot = ScannerRegistry.snapshot()
        names = self.config.get_scanner_names('input_scanners')
        first_scanners = build_scanners(self.config, snapshot, names)

        # A second plug-in trying to claim the same name in the same
        # generation must fail instead of hijacking configuration resolution.
        with self.assertRaises(DuplicateRegistrationError):

            @Registration.scanner(name='configured', requires_config=True)
            class ImpostorScanner:
                name = 'scanner:impostor'

        second_scanners = build_scanners(self.config, snapshot, names)
        self.assertIsInstance(second_scanners[0], ConfiguredScanner)
        self.assertIs(type(first_scanners[0]), type(second_scanners[0]))

    def test_second_startup_uses_fresh_generation_not_stale_snapshot(self):
        first_snapshot = ScannerRegistry.snapshot()
        first_names = self.config.get_scanner_names('input_scanners')
        first_scanners = build_scanners(
            self.config, first_snapshot, first_names
        )

        # Simulate a second application process replacing the registry.
        ScannerRegistry.reset_registry()

        class ReplacementScanner:
            name = 'scanner:configured'

            def __init__(self, threshold=None, model=None):
                self.threshold = threshold
                self.model = model

            def analyze(self, scan_obj, scan_id):
                return scan_obj

        ScannerRegistry.register_scanner(
            'configured', ReplacementScanner, requires_config=True
        )

        # The first generation's snapshot must refuse the old registration.
        with self.assertRaises(RegistryError):
            build_scanners(self.config, first_snapshot, first_names)

        # A second startup pins the current generation and gets the
        # replacement class - never the class from the old generation.
        second_snapshot = ScannerRegistry.snapshot()
        second_scanners = build_scanners(
            self.config, second_snapshot, first_names
        )
        self.assertIsInstance(second_scanners[0], ReplacementScanner)
        self.assertNotIsInstance(second_scanners[0], ConfiguredScanner)
        self.assertIsInstance(first_scanners[0], ConfiguredScanner)
        self.assertEqual(
            second_scanners[0].registry_generation,
            ScannerRegistry.current_generation()
        )

    def test_extra_metadata_does_not_override_reserved_fields(self):
        with self.assertRaises(RegistryError):
            ScannerRegistry.register_scanner(
                'metadata-attack',
                NoConfigScanner,
                **{'class': ConfiguredScanner, 'generation': 999}
            )


class VigilSetupPathTests(ConfigStartupTestBase):
    """Exercise Vigil._setup_scanners itself rather than a local replica."""

    def _make_vigil(self):
        from vigil.vigil import Vigil

        app = Vigil.__new__(Vigil)
        app._config = self.config
        app._registry_snapshot = ScannerRegistry.snapshot()
        app.vectordb = object()
        app.embedder = object()
        return app

    def test_vigil_setup_resolves_config_scanner_and_tags_generation(self):
        app = self._make_vigil()
        scanners = app._setup_scanners(['configured'])

        self.assertEqual(len(scanners), 1)
        self.assertIsInstance(scanners[0], ConfiguredScanner)
        self.assertEqual(scanners[0].threshold, '0.7')
        self.assertEqual(
            scanners[0].registry_generation,
            ScannerRegistry.current_generation()
        )

    def test_vigil_setup_propagates_registry_error_for_unknown_name(self):
        app = self._make_vigil()
        with self.assertRaises(RegistryError):
            app._setup_scanners(['not-registered'])

    def test_vigil_setup_injects_vectordb_and_embedder(self):
        app = self._make_vigil()
        db_scanners = app._setup_scanners(['database'])
        embed_scanners = app._setup_scanners(['embedding'])
        self.assertIs(db_scanners[0].db_client, app.vectordb)
        self.assertIs(embed_scanners[0].embedder, app.embedder)

    def test_vigil_snapshot_goes_stale_after_registry_rebuild(self):
        app = self._make_vigil()
        snapshot = app._registry_snapshot
        ScannerRegistry.reset_registry()
        with self.assertRaises(StaleRegistrationError):
            app._setup_scanners(['configured'])
        self.assertEqual(snapshot.generation + 1,
                         ScannerRegistry.current_generation())


class DirectVsConfigParityTests(ConfigStartupTestBase):
    """The two entry paths must report the same errors and classes."""

    def _write_minimal_config(self, contents):
        handle, path = tempfile.mkstemp(suffix='.conf')
        with os.fdopen(handle, 'w') as handle_obj:
            handle_obj.write(contents)
        self.addCleanup(os.remove, path)
        return path

    def test_missing_config_raises_same_error_on_both_paths(self):
        path = self._write_minimal_config('[scanners]\ninput_scanners = x\n')
        config = Config(path)

        direct_error = None
        try:
            ScannerRegistry.create_scanner('configured')
        except RegistryError as err:
            direct_error = type(err)

        config_error = None
        try:
            ScannerRegistry.create_scanner(
                'configured', config=config.get_scanner_config('configured')
            )
        except RegistryError as err:
            config_error = type(err)

        self.assertIs(direct_error, config_error)

    def test_successful_creation_matches_across_both_paths(self):
        path = self._write_minimal_config(
            '[scanner:configured]\nthreshold = 0.7\nmodel = demo\n'
            '[scanners]\ninput_scanners = configured\n'
        )
        config = Config(path)

        direct = ScannerRegistry.create_scanner(
            'configured', config={'threshold': '0.7', 'model': 'demo'}
        )
        via_config = ScannerRegistry.create_scanner(
            'configured', config=config.get_scanner_config('configured')
        )

        self.assertIs(type(direct), type(via_config))
        self.assertIsInstance(direct, ConfiguredScanner)
        self.assertEqual(direct.threshold, via_config.threshold)
        self.assertEqual(direct.model, via_config.model)


if __name__ == '__main__':
    unittest.main(verbosity=2)
