"""Signal strategy registry for name-based lookup and discovery."""

from __future__ import annotations

from typing import Any, Callable

import structlog

from hedgefund.signals.base import SignalGenerator

log = structlog.get_logger(__name__)

# Type alias for a factory that produces a SignalGenerator from keyword args.
GeneratorFactory = Callable[..., SignalGenerator]


class SignalRegistry:
    """Central registry for signal generation strategies.

    Strategies can be registered either by class or by factory callable, then
    looked up by name at runtime.  This decouples configuration from
    instantiation and makes it straightforward to add new strategies without
    modifying orchestration code.

    Example::

        registry = SignalRegistry()
        registry.register("rule_based", RuleBasedSignalGenerator)
        registry.register("ml", MLSignalGenerator)

        gen = registry.create("rule_based", params={"ema_fast": 5})
    """

    def __init__(self) -> None:
        self._factories: dict[str, GeneratorFactory] = {}

    # -- registration -------------------------------------------------------

    def register(
        self,
        name: str,
        factory: type[SignalGenerator] | GeneratorFactory,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a strategy under *name*.

        Args:
            name: Unique strategy name (case-insensitive internally).
            factory: A class or callable that returns a ``SignalGenerator``.
            overwrite: If *True*, silently replace an existing entry.

        Raises:
            ValueError: If *name* is already registered and *overwrite* is
                *False*.
        """
        key = name.lower()
        if key in self._factories and not overwrite:
            raise ValueError(
                f"Strategy '{name}' is already registered. "
                "Pass overwrite=True to replace it."
            )
        self._factories[key] = factory  # type: ignore[assignment]
        log.debug("strategy_registered", name=key)

    def unregister(self, name: str) -> None:
        """Remove a registered strategy.

        Raises:
            KeyError: If the name is not found.
        """
        key = name.lower()
        if key not in self._factories:
            raise KeyError(f"Strategy '{name}' is not registered")
        del self._factories[key]
        log.debug("strategy_unregistered", name=key)

    # -- lookup / creation --------------------------------------------------

    def get_factory(self, name: str) -> GeneratorFactory:
        """Return the raw factory for *name*.

        Raises:
            KeyError: If the name is not found.
        """
        key = name.lower()
        if key not in self._factories:
            raise KeyError(
                f"Strategy '{name}' not found. "
                f"Available: {', '.join(sorted(self._factories))}"
            )
        return self._factories[key]

    def create(self, name: str, **kwargs: Any) -> SignalGenerator:
        """Instantiate a registered strategy with the given keyword args.

        Raises:
            KeyError: If the name is not found.
        """
        factory = self.get_factory(name)
        instance = factory(**kwargs)
        log.info("strategy_created", name=name.lower())
        return instance

    # -- introspection ------------------------------------------------------

    def list_strategies(self) -> list[str]:
        """Return sorted list of registered strategy names."""
        return sorted(self._factories)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._factories

    def __len__(self) -> int:
        return len(self._factories)

    def __repr__(self) -> str:
        names = ", ".join(self.list_strategies())
        return f"SignalRegistry([{names}])"
