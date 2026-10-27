# type: ignore
"""CausalClaw Bridge — Graph Scoring.

Takes an adjacency matrix and a dataset, returns the total score (BIC or BDeu)
of the graph. The score is maximized by causal-learn algorithms, meaning
larger is better (even if the values themselves are negative log-likelihoods).
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import pandas as pd

from utils import load_data, output_result, safe_main
from causal_discovery.data_utils import detect_data_type
from causallearn.search.ScoreBased.GES import local_score_BIC, local_score_BDeu

@safe_main
def main():
    parser = argparse.ArgumentParser(description="Score a causal graph")
    parser.add_argument("--data", required=True, help="Path to data CSV")
    parser.add_argument("--adj", required=True, help="JSON string or path to JSON file containing the adjacency matrix")
    parser.add_argument("--score-func", default="auto", choices=["auto", "bic", "bdeu"], help="Scoring function")

    args = parser.parse_args()

    data = load_data(args.data)
    data_array = data.to_numpy().astype(np.float64)

    try:
        with open(args.adj, "r") as f:
            adj = np.array(json.load(f))
    except Exception:
        adj = np.array(json.loads(args.adj))

    score_func = args.score_func
    if score_func == "auto":
        dtype = detect_data_type(data_array)
        score_func = "bic" if dtype == "continuous" else "bdeu"

    score_fn = local_score_BIC if score_func == "bic" else local_score_BDeu

    total_score = 0.0
    d = data_array.shape[1]
    for i in range(d):
        parents = np.where(adj[:, i] == 1)[0].tolist()
        total_score += score_fn(data_array, i, parents)

    # Output the total score.
    # The score is a "larger is better" metric (the causallearn objective to be maximized).
    # If a strictly positive value is required by some systems, one might apply an offset,
    # but mathematically this represents the raw score.
    result = {
        "score": float(total_score),
        "score_func": score_func,
        "metadata": {
            "n_samples": len(data),
            "n_variables": d,
        }
    }

    output_result(result)

if __name__ == "__main__":
    main()
