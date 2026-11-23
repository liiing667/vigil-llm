"""Generation-aware scanner registry tests.

These tests exercise the direct registration path: scanners are registered
with ``ScannerRegistry.register_scanner`` / the ``Registration.scanner``
decorator without any third-party dependencies.
"""
import sys
import unittest

from tests._stubs import install_stubs

install_stubs()

sys.path.insert(0, '.')

from vigil.registry import DuplicateRegistrationError
from vigil.registry import Registration
from vigil.registry import RegistryError
from vigil.registry import RegistrySnapshot
from vigil.registry import RESERVED_METADATA_KEYS
from vigil.registry import ScannerRegistry


class SimpleScanner:
    """Stand-in scanner that does not need pydantic/yara/nltk."""

    name = 'scanner:simple'

    def __init__(self, threshold=None):
        self.threshold = threshold

    def analyze(self, scan_obj, scan_id):
        return scan_obj


class ConfigScanner:
    name = 'scanner:config'

    def __init__(self, **config):
        self.config = config

    def analyze(self, scan_obj, scan_id):
        return scan_obj


class DBSanner:
    name = 'scanner:db'

    def __init__(self, db_client=None):
        self.db_client = db_client

    def analyze(self, scan_obj, scan_id):
        return scan_obj


class EmbedScanner:
    name = 'scanner:embed'

    def __init__(self, embedder=None):
        self.embedder = embedder

    def analyze(self, scan_obj, scan_id):
        return scan_obj


class PostInitScanner(SimpleScanner):
    name = 'scanner:post-init'

    def post_init(self):
        self.post_init_called = True


class RegistryTestBase(unittest.TestCase):
    def setUp(self):
        ScannerRegistry.reset_registry()


class RegistrationTests(RegistryTestBase):
    def test_direct_registration_stores_class_and_metadata(self):
        ScannerRegistry.register_scanner(
            'simple',
            SimpleScanner,
            requires_config=False,
            custom_meta='hello'
        )
        metadata = ScannerRegistry.get_scanner_metadata('simple')
        self.assertIs(metadata['class'], SimpleScanner)
        self.assertFalse(metadata['requires_config'])
        self.assertEqual(metadata['custom_meta'], 'hello')
        self.assertEqual(metadata['generation'],
                         ScannerRegistry.current_generation())

    def test_decorator_registers_class_and_extra_metadata(self):
        @Registration.scanner(
            name='decorated', requires_config=False, flavor='v1'
        )
        class DecoratedScanner(SimpleScanner):
            name = 'scanner:decorated'

        metadata = ScannerRegistry.get_scanner_metadata('decorated')
        self.assertIs(metadata['class'], DecoratedScanner)
        self.assertFalse(metadata['requires_config'])
        self.assertEqual(metadata['flavor'], 'v1')
        self.assertEqual(metadata['generation'],
                         ScannerRegistry.current_generation())

    def test_duplicate_name_in_same_generation_raises(self):
        ScannerRegistry.register_scanner('dup', SimpleScanner)
        with self.assertRaises(DuplicateRegistrationError):
            ScannerRegistry.register_scanner('dup', ConfigScanner)
        # The first registration must still resolve to the first class.
        metadata = ScannerRegistry.get_scanner_metadata('dup')
        self.assertIs(metadata['class'], SimpleScanner)

    def test_same_name_allowed_after_new_generation(self):
        ScannerRegistry.register_scanner('swap', SimpleScanner)
        first_generation = ScannerRegistry.current_generation()

        new_generation = ScannerRegistry.reset_registry()
        self.assertGreater(new_generation, first_generation)
        ScannerRegistry.register_scanner('swap', ConfigScanner)

        metadata = ScannerRegistry.get_scanner_metadata('swap')
        self.assertIs(metadata['class'], ConfigScanner)
        self.assertEqual(metadata['generation'], new_generation)

    def test_reserved_metadata_keys_cannot_be_overridden(self):
        # The requires_* flags are explicit keyword arguments; the remaining
        # reserved keys can only arrive through extra metadata.
        extra_only = RESERVED_METADATA_KEYS - {
            'requires_config', 'requires_vectordb', 'requires_embedding'
        }
        for reserved in extra_only:
            with self.subTest(reserved=reserved):
                with self.assertRaises(RegistryError):
                    ScannerRegistry.register_scanner(
                        f'bad-{reserved}',
                        SimpleScanner,
                        **{reserved: 'attacker-controlled'}
                    )

    def test_unknown_name_raises_registry_error(self):
        with self.assertRaises(RegistryError):
            ScannerRegistry.get_scanner_metadata('missing')


class InstanceCreationTests(RegistryTestBase):
    def test_scanner_without_config_works_with_none_and_empty_config(self):
        ScannerRegistry.register_scanner('simple', SimpleScanner)

        none_instance = ScannerRegistry.create_scanner('simple', config=None)
        empty_instance = ScannerRegistry.create_scanner('simple', config={})

        self.assertIsInstance(none_instance, SimpleScanner)
        self.assertIsInstance(empty_instance, SimpleScanner)

    def test_missing_config_raises_registry_error(self):
        ScannerRegistry.register_scanner(
            'config', ConfigScanner, requires_config=True
        )
        with self.assertRaises(RegistryError):
            ScannerRegistry.create_scanner('config')

    def test_config_is_resolved_to_the_registered_class(self):
        ScannerRegistry.register_scanner(
            'config', ConfigScanner, requires_config=True
        )
        config = {'threshold': '0.5', 'model': 'demo'}
        instance = ScannerRegistry.create_scanner('config', config=config)
        self.assertIsInstance(instance, ConfigScanner)
        self.assertEqual(instance.config, config)

        # Mutating the caller's dict after creation must not change the
        # configuration the scanner received.
        config['threshold'] = '9.9'
        self.assertEqual(instance.config['threshold'], '0.5')

    def test_vectordb_and_embedder_requirements_are_validated(self):
        ScannerRegistry.register_scanner(
            'db', DBSanner, requires_vectordb=True
        )
        ScannerRegistry.register_scanner(
            'embed', EmbedScanner, requires_embedding=True
        )

        with self.assertRaises(RegistryError):
            ScannerRegistry.create_scanner('db')
        with self.assertRaises(RegistryError):
            ScannerRegistry.create_scanner('embed')

        db_client = object()
        embedder = object()
        db_instance = ScannerRegistry.create_scanner(
            'db', vectordb=db_client
        )
        embed_instance = ScannerRegistry.create_scanner(
            'embed', embedder=embedder
        )
        self.assertIs(db_instance.db_client, db_client)
        self.assertIs(embed_instance.embedder, embedder)

    def test_post_init_is_invoked(self):
        ScannerRegistry.register_scanner('post', PostInitScanner)
        instance = ScannerRegistry.create_scanner('post')
        self.assertTrue(instance.post_init_called)


class GenerationTests(RegistryTestBase):
    def test_snapshot_creates_instances_from_pinned_generation(self):
        ScannerRegistry.register_scanner('simple', SimpleScanner)
        snapshot = ScannerRegistry.snapshot()
        instance = snapshot.create_scanner('simple')
        self.assertIsInstance(instance, SimpleScanner)
        self.assertEqual(
            instance.registry_generation,
            ScannerRegistry.current_generation()
        )

    def test_stale_snapshot_raises_after_registry_reset(self):
        ScannerRegistry.register_scanner('simple', SimpleScanner)
        stale_snapshot = ScannerRegistry.snapshot()

        ScannerRegistry.reset_registry()
        ScannerRegistry.register_scanner('simple', ConfigScanner)

        # The old snapshot must not silently create the newly registered
        # class; it reports that its generation is stale.
        with self.assertRaises(RegistryError):
            stale_snapshot.create_scanner('simple')
        with self.assertRaises(RegistryError):
            stale_snapshot.get_scanner_metadata('simple')

        # A fresh snapshot uses the current generation and new class.
        fresh_snapshot = ScannerRegistry.snapshot()
        instance = fresh_snapshot.create_scanner(
            'simple', config={'threshold': '1'}
        )
        self.assertIsInstance(instance, ConfigScanner)
        self.assertEqual(
            instance.registry_generation,
            ScannerRegistry.current_generation()
        )

    def test_unregister_advances_generation_and_stales_old_snapshot(self):
        ScannerRegistry.register_scanner('simple', SimpleScanner)
        snapshot = ScannerRegistry.snapshot()
        generation = ScannerRegistry.current_generation()

        ScannerRegistry.unregister_scanner('simple')
        self.assertGreater(ScannerRegistry.current_generation(), generation)
        with self.assertRaises(RegistryError):
            snapshot.get_scanner_names()

    def test_snapshot_lists_names_only_while_current(self):
        ScannerRegistry.register_scanner('a', SimpleScanner)
        ScannerRegistry.register_scanner('b', ConfigScanner)
        snapshot = ScannerRegistry.snapshot()
        self.assertEqual(sorted(snapshot.get_scanner_names()), ['a', 'b'])

        ScannerRegistry.reset_registry()
        with self.assertRaises(RegistryError):
            snapshot.get_scanner_names()


class BuiltinPluginTests(RegistryTestBase):
    @classmethod
    def setUpClass(cls):
        # Importing the package runs the built-in scanner modules and their
        # Registration.scanner decorators exactly once per interpreter.
        import vigil  # noqa: F401

    def setUp(self):
        import vigil
        # Other tests reset the registry; re-run the decorators against the
        # fresh generation so built-in plug-ins keep working as usual.
        ScannerRegistry.reset_registry()
        vigil.load_builtin_scanners(force=True)

    def test_builtin_scanners_self_register_on_package_import(self):
        builtins = {
            'sentiment', 'similarity', 'transformer', 'vectordb', 'yara'
        }
        self.assertTrue(
            builtins.issubset(set(ScannerRegistry.get_scanner_names()))
        )
        for name in builtins:
            metadata = ScannerRegistry.get_scanner_metadata(name)
            self.assertTrue(metadata['requires_config'])
            self.assertEqual(
                metadata['generation'],
                ScannerRegistry.current_generation()
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)
