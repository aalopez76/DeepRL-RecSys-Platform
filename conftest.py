"""Root conftest — ensures src/ is on sys.path for test discovery."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest as _pytest

# Ensure editable-install resolution works regardless of .pth encoding
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def pytest_collection_modifyitems(items: list) -> None:  # type: ignore[type-arg]
    """Move extras-isolation tests to the very front of the collection.

    The isolation tests verify that importing ``deeprl_recsys`` does NOT
    pull heavy optional deps (torch, transformers, …) into ``sys.modules``.
    Running them first guarantees no other test has had a chance to trigger
    a lazy import of those deps.
    """
    first: list = []
    rest: list = []
    for item in items:
        if "test_extras_isolation" in item.nodeid:
            first.append(item)
        else:
            rest.append(item)
    items[:] = first + rest

