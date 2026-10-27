from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from causallearn.search.FCMBased.lingam import DirectLiNGAM

from .base import BaseDiscoveryMethod

class DirectLiNGAMAlgorithm(BaseDiscoveryMethod):
    def fit(self, data: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        data_array = data.to_numpy().astype(np.float64)

        model = DirectLiNGAM()
        model.fit(data_array)

        adj = (model.adjacency_matrix_ != 0).astype(int).T
        return adj
