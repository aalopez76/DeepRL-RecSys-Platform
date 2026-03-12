"""Optional plugin loader — discovers entry-point plugins on demand.

This module is NEVER imported at package init time.  It is invoked
only when :func:`registry.load_plugins` is called explicitly (e.g.
from the CLI ``list-plugins`` command).
"""

from __future__ import annotations

import sys
from typing import Any

_LOADED = False


def discover_plugins(registry: dict[str, dict[str, str]]) -> None:
    """Scan ``deeprl_recsys.*`` entry-point groups and register discoveries.

    Args:
        registry: The mutable registry dict from :mod:`registry`.
    """
    global _LOADED  # noqa: PLW0603
    if _LOADED:
        return
    _LOADED = True

    if sys.version_info >= (3, 12):
        from importlib.metadata import entry_points
    else:
        from importlib.metadata import entry_points

    # Entry-point groups follow the pattern:  deeprl_recsys.<category>
    for category in ("agents", "environments", "simulators", "estimators", "metrics"):
        group = f"deeprl_recsys.{category}"
        eps = entry_points().get(group, []) if isinstance(entry_points(), dict) else entry_points(group=group)
        for ep in eps:
            target = f"{ep.value}"
            registry.setdefault(category, {})[ep.name] = target
