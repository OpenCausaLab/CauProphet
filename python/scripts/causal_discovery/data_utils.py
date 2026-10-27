from __future__ import annotations
from typing import Literal

import numpy as np


def detect_data_type(data: np.ndarray) -> Literal["continuous", "discrete"]:
    """Detect whether data is continuous or discrete.

    Returns:
        "continuous" if data appears to be continuous (floats with many unique values).
        "discrete" if data appears to be categorical/discrete.
    """
    is_discrete = True
    for col in range(data.shape[1]):
        col_data = data[:, col]
        unique_vals = np.unique(col_data)

        # Heuristic: If it has floats that are not all integers, and has more than 10 unique values
        if len(unique_vals) > 10 and np.issubdtype(col_data.dtype, np.floating):
            # Check if they are actually fractional
            if not np.all(np.floor(col_data) == col_data):
                is_discrete = False
                break

    return "discrete" if is_discrete else "continuous"
