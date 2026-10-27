from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc

from .base import BaseDiscoveryMethod

def _detect_indep_test(data: np.ndarray) -> str:
    n_unique = np.array([len(np.unique(data[:, j])) for j in range(data.shape[1])])
    if np.all(n_unique <= 10):
        return "chisq"
    return "fisherz"

class PCAlgorithm(BaseDiscoveryMethod):
    def fit(self, data: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        alpha = kwargs.get("alpha", 0.05)
        indep_test = kwargs.get("indep_test", "auto")
        uc_rule = kwargs.get("uc_rule", 0)
        uc_priority = kwargs.get("uc_priority", 2)
        stable = kwargs.get("stable", True)

        data_array: np.ndarray = data.to_numpy()

        if indep_test == "auto":
            indep_test = _detect_indep_test(data_array)

        cg = pc(
            data_array,
            alpha=alpha,
            indep_test=indep_test,
            stable=stable,
            uc_rule=uc_rule,
            uc_priority=uc_priority,
            show_progress=False,
        )

        raw_graph: np.ndarray = cg.G.graph
        d = raw_graph.shape[0]
        adj = np.zeros((d, d), dtype=int)

        for i in range(d):
            for j in range(d):
                if raw_graph[i, j] == -1 and raw_graph[j, i] == 1:
                    adj[i, j] = 1
                elif raw_graph[i, j] == -1 and raw_graph[j, i] == -1:
                    adj[i, j] = 1
                    adj[j, i] = 1

        return adj
