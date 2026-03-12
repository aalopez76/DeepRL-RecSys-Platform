"""Global seeding for reproducibility.

Sets seeds for ``random``, ``numpy``, and optional backends (torch, tf)
so that every run with the same seed produces identical results.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int, *, use_torch: bool = False, use_tf: bool = False) -> None:
    """Set global random seeds for reproducibility.

    Args:
        seed: Integer seed value.
        use_torch: If True, also seed PyTorch (requires torch installed).
        use_tf: If True, also seed TensorFlow (requires tf installed).
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    if use_torch:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    if use_tf:
        try:
            import tensorflow as tf

            tf.random.set_seed(seed)
        except ImportError:
            pass


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Return a NumPy ``Generator`` instance with the given seed.

    Args:
        seed: Optional seed. If ``None``, a non-deterministic generator is returned.

    Returns:
        A ``numpy.random.Generator`` instance.
    """
    return np.random.default_rng(seed)
