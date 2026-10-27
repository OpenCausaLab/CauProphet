from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci

from .base import BaseDiscoveryMethod
from .pc import _detect_indep_test


class FCIAlgorithm(BaseDiscoveryMethod):
    def fit(self, data: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        alpha = kwargs.get("alpha", 0.05)
        indep_test = kwargs.get("indep_test", "auto")

        data_array: np.ndarray = data.to_numpy()

        if indep_test == "auto":
            indep_test = _detect_indep_test(data_array)

        g, _ = fci(
            data_array,
            independence_test_method=indep_test,
            alpha=alpha,
            show_progress=False,
        )

        raw_graph: np.ndarray = g.graph  # type: ignore

        connected = raw_graph != 0
        d = raw_graph.shape[0]
        adj = np.zeros((d, d), dtype=int)

        directed = (raw_graph == -1) & (raw_graph.T == 1)
        adj[directed] = 1

        other = connected & ~directed & ~directed.T
        adj[other] = 1

        # We need to return both raw graph and adj so we can build edge dictionary correctly
        # But BaseDiscoveryMethod fit is supposed to return just adj.
        # However, for causal_discover.py, it expects edges.
        # Let's return just adj, and the caller can format edges? No, FCI requires raw_graph to determine o-> and <->.
        # I'll return raw_graph directly, and handle the rest in the caller! Or return adj.
        # Wait, the reference only returns adj.
        return raw_graph
