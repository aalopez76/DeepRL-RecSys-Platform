"""Lazy registry — resolves ``"module:Symbol"`` strings via ``importlib``.

Callers interact through :func:`create` and :func:`get_class`.  Imports
happen only when a name is first requested (lazy loading).  Plugins
from ``entry_points`` can be loaded explicitly via :func:`load_plugins`,
which delegates to :mod:`plugin_loader`.
"""

from __future__ import annotations

import importlib
from typing import Any

from deeprl_recsys.core import builtins

# Combined registries (populated from builtins on first access)
_REGISTRY: dict[str, dict[str, str]] = {}


def _ensure_registry() -> None:
    """Populate *_REGISTRY* from :mod:`builtins` if empty (one-time init)."""
    if _REGISTRY:
        return
    _REGISTRY["agents"] = dict(builtins.AGENTS)
    _REGISTRY["environments"] = dict(builtins.ENVIRONMENTS)
    _REGISTRY["simulators"] = dict(builtins.SIMULATORS)
    _REGISTRY["estimators"] = dict(builtins.ESTIMATORS)
    _REGISTRY["metrics"] = dict(builtins.METRICS)


def register(category: str, name: str, target: str) -> None:
    """Register a new ``name → target`` mapping in *category*.

    Args:
        category: One of ``agents``, ``environments``, ``simulators``,
            ``estimators``, ``metrics``.
        name: Lookup name (e.g. ``"dqn"``).
        target: Dotted path ``"module:Symbol"`` (e.g.
            ``"deeprl_recsys.agents.dqn:DQNAgent"``).
    """
    _ensure_registry()
    _REGISTRY.setdefault(category, {})[name] = target


def get_class(category: str, name: str) -> type:
    """Resolve *name* in *category* to its Python class via importlib.

    Args:
        category: Registry category name.
        name: Registered component name.

    Returns:
        The resolved class object.

    Raises:
        KeyError: If *name* is not registered in *category*.
        ImportError: If the target module cannot be imported.
    """
    _ensure_registry()
    cat = _REGISTRY.get(category)
    if cat is None:
        raise KeyError(f"Unknown registry category: {category!r}")
    target = cat.get(name)
    if target is None:
        available = ", ".join(sorted(cat))
        raise KeyError(
            f"Unknown {category} name: {name!r}. Available: [{available}]"
        )
    module_path, symbol = target.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, symbol)


def create(category: str, name: str, **kwargs: Any) -> Any:
    """Instantiate a registered component.

    Args:
        category: Registry category name.
        name: Registered component name.
        **kwargs: Passed to the constructor.

    Returns:
        An instance of the resolved class.
    """
    cls = get_class(category, name)
    return cls(**kwargs)


def load_plugins() -> None:
    """Explicitly load entry-point plugins (opt-in).

    Safe to call multiple times; subsequent calls are no-ops.
    """
    from deeprl_recsys.core.plugin_loader import discover_plugins

    discover_plugins(_REGISTRY)


def list_registered(category: str) -> list[str]:
    """Return sorted names registered under *category*.

    Args:
        category: Registry category name.

    Returns:
        Sorted list of registered names.
    """
    _ensure_registry()
    cat = _REGISTRY.get(category, {})
    return sorted(cat)
