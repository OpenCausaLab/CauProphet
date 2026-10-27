from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from causallearn.search.ScoreBased.GES import ges

from .base import BaseDiscoveryMethod
from .data_utils import detect_data_type

class GESAlgorithm(BaseDiscoveryMethod):
    def fit(self, data: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        score_func = kwargs.get("score_func", "auto")

        data_array = data.to_numpy().astype(np.float64)
        if score_func == "auto":
            dtype = detect_data_type(data_array)
            score_func = "local_score_BIC" if dtype == "continuous" else "local_score_BDeu"

        record = ges(data_array, score_func=score_func)

        raw = record["G"].graph
        d = raw.shape[0]
        adj = np.zeros((d, d), dtype=int)

        for i in range(d):
            for j in range(d):
                if raw[i, j] == -1 and raw[j, i] == 1:
                    adj[i, j] = 1
                elif raw[i, j] == -1 and raw[j, i] == -1:
                    adj[i, j] = 1
                    adj[j, i] = 1

        return adj
