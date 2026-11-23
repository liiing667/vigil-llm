from typing import Any, Dict, List, Type, Callable, Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vigil.schema import BaseScanner


# Metadata keys managed by the registry itself. Plugins may never supply
# these through ``**additional_metadata`` so that extra metadata cannot
# overwrite the class/requirement information used for configuration
# resolution and instance creation.
RESERVED_METADATA_KEYS = frozenset(
    (
        'class',
        'generation',
        'requires_config',
        'requires_vectordb',
        'requires_embedding',
    )
)


class RegistryError(ValueError):
    """Base class for scanner registry errors."""


class DuplicateRegistrationError(RegistryError):
    """Raised when a scanner name is registered twice within one generation."""


class StaleRegistrationError(RegistryError):
    """Raised when a pinned registry generation is no longer current."""


class RegistrySnapshot:
    """Generation-pinned view of the scanner registry.

    A snapshot captures the registry generation at creation time. Every
    lookup, configuration resolution and instance creation through the
    snapshot verifies that the generation is still current, so a second
    application instance can never silently reuse registrations that were
    replaced after it started.
    """

    def __init__(self, generation: int):
        self.generation = generation

    def _ensure_current(self) -> None:
        current_generation = ScannerRegistry.current_generation()
        if current_generation != self.generation:
            raise StaleRegistrationError(
                'Registry generation changed: snapshot pinned to generation '
                f'{self.generation}, current generation is {current_generation}'
            )

    def get_scanner_metadata(self, name: str) -> dict:
        self._ensure_current()
        return ScannerRegistry.get_scanner_metadata(name)

    def get_scanner_names(self) -> List[str]:
        self._ensure_current()
        return ScannerRegistry.get_scanner_names()

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'generation={self.generation}, '
            f'current={ScannerRegistry.current_generation()})'
        )

    def create_scanner(
        self,
        name: str,
        config: Optional[dict] = None,
        vectordb: Optional[Callable] = None,
        embedder: Optional[Callable] = None,
        **params
    ) -> 'BaseScanner':
        self._ensure_current()
        scanner = ScannerRegistry.create_scanner(
            name,
            config=config,
            vectordb=vectordb,
            embedder=embedder,
            **params
        )
        ScannerRegistry.tag_instance(scanner, self.generation)
        return scanner


class Registration:
    @staticmethod
    def scanner(
        name: str,
        requires_config=False,
        requires_vectordb=False,
        requires_embedding=False,
        **additional_metadata
    ):
        def decorator(scanner_class: Type['BaseScanner']):
            ScannerRegistry.register_scanner(
                name,
                scanner_class,
                requires_config=requires_config,
                requires_vectordb=requires_vectordb,
                requires_embedding=requires_embedding,
                **additional_metadata
            )
            return scanner_class
        return decorator


class ScannerRegistry:
    _registry: Dict[str, dict] = {}
    _generation: int = 0

    @classmethod
    def current_generation(cls) -> int:
        """Return the generation of the currently registered scanners."""
        return cls._generation

    @classmethod
    def snapshot(cls) -> RegistrySnapshot:
        """Pin the current generation for config resolution and creation."""
        return RegistrySnapshot(cls._generation)

    @classmethod
    def reset_registry(cls) -> int:
        """Drop every registration and start a new generation.

        Snapshots pinned to earlier generations fail instead of reading the
        fresh registrations, making the boundary between registry generations
        explicit for configuration resolution, instance creation and reports.
        """
        cls._registry = {}
        cls._generation += 1
        return cls._generation

    @classmethod
    def register_scanner(
        cls,
        name: str,
        scanner_class: Type['BaseScanner'],
        requires_config=False,
        requires_vectordb=False,
        requires_embedding=False,
        **metadata
    ):
        conflicting_keys = RESERVED_METADATA_KEYS.intersection(metadata)
        if conflicting_keys:
            raise RegistryError(
                f"Cannot register scanner '{name}': reserved metadata key(s) "
                f'{sorted(conflicting_keys)} may not be overridden'
            )

        if name in cls._registry:
            raise DuplicateRegistrationError(
                f"Scanner already registered with name: {name} "
                f'(generation {cls._generation})'
            )

        cls._registry[name] = {
            'class': scanner_class,
            'generation': cls._generation,
            'requires_config': requires_config,
            'requires_vectordb': requires_vectordb,
            'requires_embedding': requires_embedding,
            **metadata
        }

    @classmethod
    def create_scanner(
        cls,
        name: str,
        config: Optional[dict] = None,
        vectordb: Optional[Callable] = None,
        embedder: Optional[Callable] = None,
        **params
    ) -> 'BaseScanner':
        scanner_info = cls.get_scanner_metadata(name)
        scanner_class = scanner_info['class']

        init_params = {}
        if scanner_info['requires_config']:
            if not config:
                raise RegistryError(f"Config required for scanner '{name}'")
            # Copy so scanner init cannot mutate the resolved configuration.
            init_params = dict(config)

        if scanner_info['requires_vectordb']:
            if vectordb is None:
                raise RegistryError(f"VectorDB required for scanner '{name}'")

            init_params.update({'db_client': vectordb})

        if scanner_info['requires_embedding']:
            if embedder is None:
                raise RegistryError(f"Embedder required for scanner '{name}'")

            init_params.update({'embedder': embedder})

        scanner_cls = scanner_class(**init_params)
        if hasattr(scanner_cls, 'post_init'):
            scanner_cls.post_init()

        return scanner_cls

    @classmethod
    def get_scanner_names(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def get_scanner_cls(cls) -> List[Type['BaseScanner']]:
        return [info['class'] for info in cls._registry.values()]

    @classmethod
    def get_scanner_metadata(cls, name: str) -> dict:
        if name not in cls._registry:
            raise RegistryError(f'No scanner registered with name: {name}')
        return dict(cls._registry[name])

    @classmethod
    def unregister_scanner(cls, name: str) -> None:
        """Remove one registration and advance to a new generation."""
        if name not in cls._registry:
            raise RegistryError(f'No scanner registered with name: {name}')
        del cls._registry[name]
        cls._generation += 1

    @staticmethod
    def tag_instance(scanner: Any, generation: int) -> Any:
        """Attach the registry generation to a created scanner for reporting."""
        try:
            scanner.registry_generation = generation
        except (AttributeError, TypeError):
            pass
        return scanner
