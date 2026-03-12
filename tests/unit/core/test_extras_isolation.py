"""Extras isolation — verify heavy deps are NOT loaded by default.

Importing ``deeprl_recsys`` must NOT trigger import of torch,
transformers, tensorflow, boto3, shap, or lime.  Those are lazy-loaded
and only available when the corresponding extra is installed.
"""

from __future__ import annotations

import sys

import pytest


@pytest.mark.unit
class TestExtrasIsolation:
    """Verify that heavy optional deps are not imported on package load."""

    HEAVY_MODULES = [
        "torch",
        "transformers",
        "tensorflow",
        "tf_agents",
        "boto3",
        "s3fs",
        "shap",
        "lime",
        "weasyprint",
        "reportlab",
    ]

    def test_import_deeprl_recsys_does_not_load_heavy_deps(self) -> None:
        """Importing deeprl_recsys must not pull heavy optional deps."""
        import deeprl_recsys  # noqa: F401

        for mod in self.HEAVY_MODULES:
            # Check that the module is not in sys.modules
            # (unless it was already imported before the test, which
            # shouldn't happen in a clean test environment)
            assert mod not in sys.modules, (
                f"Heavy dependency {mod!r} was loaded by "
                f"'import deeprl_recsys' — lazy loading is broken"
            )

    def test_core_imports_are_clean(self) -> None:
        """Importing core modules must not pull heavy deps."""
        from deeprl_recsys.core import config  # noqa: F401
        from deeprl_recsys.core import artifacts  # noqa: F401
        from deeprl_recsys.core import schema  # noqa: F401
        from deeprl_recsys.core import logging  # noqa: F401
        from deeprl_recsys.core import registry  # noqa: F401

        for mod in self.HEAVY_MODULES:
            assert mod not in sys.modules, (
                f"Heavy dependency {mod!r} was loaded by "
                f"core module import — lazy loading is broken"
            )

    def test_builtins_do_not_eager_import(self) -> None:
        """Builtins module must use lazy loading strings, not real imports."""
        from deeprl_recsys.core import builtins

        # Verify the builtins registry maps to strings, not objects
        for category in ("agents", "environments", "simulators", "estimators", "metrics"):
            entries = getattr(builtins, f"BUILTIN_{category.upper()}", None)
            if entries is None:
                continue
            for name, ref in entries.items():
                assert isinstance(ref, str), (
                    f"Builtin {category}.{name} is not a lazy string ref: {ref!r}"
                )


@pytest.mark.unit
class TestGuardedImports:
    """Verify that optional modules have proper import guards."""

    def test_llm_sim_requires_transformers(self) -> None:
        """LLMSimulator must raise ImportError without [llm] extra."""
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401 
            # If both load smoothly, then transformers is truly installed and working.
            pytest.skip("transformers is installed and working — cannot test guard")
        except (ImportError, OSError):
            pass

        from deeprl_recsys.environment.simulators.llm_sim import LLMSimulator

        with pytest.raises(ImportError, match="\\[llm\\]"):
            LLMSimulator()
