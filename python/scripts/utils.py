"""Shared IO utilities for CausalClaw Python bridge scripts.

All bridge scripts use the same protocol:
  - Input:  CLI args (--data, --method, --alpha, etc.)
  - Output: JSON to stdout (parsed by TS handler)
  - Errors: JSON {"error": "..."} to stdout (not stderr, so TS can parse it)
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV dataset. Supports .csv and .tsv."""
    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def adjacency_to_edges(
    adj: np.ndarray,
    var_names: list[str],
    method: str,
    graph_type: str = "dag",
) -> list[dict[str, Any]]:
    """Convert a D×D adjacency matrix to a list of edge dicts for the scratchpad.

    Adjacency matrix conventions:
      - adj[i,j] = 1, adj[j,i] = 0  → directed edge i → j
      - adj[i,j] = 1, adj[j,i] = 1  → undirected edge i — j (CPDAG)
      - For PAG: handled separately by the FCI converter

    Returns a list of CausalEdge-compatible dicts.
    """
    d = adj.shape[0]
    edges: list[dict[str, Any]] = []
    visited: set[tuple[int, int]] = set()

    for i in range(d):
        for j in range(d):
            if adj[i, j] == 0:
                continue
            if (i, j) in visited or (j, i) in visited:
                continue

            if adj[i, j] == 1 and adj[j, i] == 1:
                # Undirected (CPDAG) — record once, smaller index first
                edges.append({
                    "from": var_names[min(i, j)],
                    "to": var_names[max(i, j)],
                    "edgeType": "undirected",
                    "confidence": "high",
                    "method": method,
                })
                visited.add((i, j))
                visited.add((j, i))
            elif adj[i, j] == 1 and adj[j, i] == 0:
                # Directed i → j
                edges.append({
                    "from": var_names[i],
                    "to": var_names[j],
                    "edgeType": "directed",
                    "confidence": "high",
                    "method": method,
                })
                visited.add((i, j))

    return edges


def output_result(result: dict[str, Any]) -> None:
    """Write JSON result to stdout."""
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    sys.stdout.flush()


def output_error(message: str) -> None:
    """Write a structured error to stdout (not stderr)."""
    output_result({"error": message, "traceback": traceback.format_exc()})


def safe_main(func):
    """Decorator that wraps a main function with error handling."""
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            output_error(str(e))
            sys.exit(1)
    return wrapper
