# type: ignore
"""CausalClaw Bridge — Causal Discovery.

Unified CLI entry point for running causal discovery algorithms.
Outputs JSON to stdout for the TypeScript handler to parse.
"""

from __future__ import annotations

import argparse
import contextlib
import signal
import sys
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="causallearn")
warnings.filterwarnings("ignore", category=UserWarning, module="causallearn")

from utils import (
    adjacency_to_edges,
    load_data,
    output_error,
    output_result,
    safe_main,
)
from causal_discovery.data_utils import detect_data_type
from causallearn.search.ScoreBased.GES import local_score_BIC, local_score_BDeu

def calculate_graph_score(data_array: np.ndarray, adj: np.ndarray) -> float:
    """Sum local scores for all nodes in the DAG."""
    dtype = detect_data_type(data_array)
    score_fn = local_score_BIC if dtype == "continuous" else local_score_BDeu
    
    total_score = 0.0
    d = data_array.shape[1]
    for i in range(d):
        # adj[j, i] == 1 means j -> i
        parents = np.where(adj[:, i] == 1)[0].tolist()
        total_score += score_fn(data_array, i, parents)
    return float(total_score)


def run_pc(
    data: pd.DataFrame,
    alpha: float = 0.05,
    indep_test: str = "auto",
    stable: bool = True,
) -> dict[str, Any]:
    from causal_discovery.pc import PCAlgorithm

    algo = PCAlgorithm()
    adj = algo.fit(data, alpha=alpha, indep_test=indep_test, stable=stable)
    var_names = list(data.columns)

    edges = adjacency_to_edges(adj, var_names, "PC", "dag")
    score = calculate_graph_score(data.to_numpy(), adj)

    return {
        "method": "PC",
        "adjacency_matrix": adj.tolist(),
        "variables": var_names,
        "edges": edges,
        "score": score,
        "params": {"alpha": alpha, "indep_test": indep_test, "stable": stable},
    }


def run_fci(
    data: pd.DataFrame,
    alpha: float = 0.05,
    indep_test: str = "auto",
) -> dict[str, Any]:
    from causal_discovery.fci import FCIAlgorithm

    algo = FCIAlgorithm()
    raw = algo.fit(data, alpha=alpha, indep_test=indep_test)
    var_names = list(data.columns)

    d = raw.shape[0]
    edges: list[dict[str, Any]] = []
    visited: set[tuple[int, int]] = set()

    for i in range(d):
        for j in range(d):
            if raw[i, j] == 0 and raw[j, i] == 0:
                continue
            if (i, j) in visited or (j, i) in visited:
                continue

            edge: dict[str, Any] = {
                "from": var_names[i],
                "to": var_names[j],
                "confidence": "high",
                "method": "FCI",
            }

            if raw[i, j] == -1 and raw[j, i] == 1:
                edge["edgeType"] = "directed"
            elif raw[i, j] == 1 and raw[j, i] == -1:
                edge["from"], edge["to"] = var_names[j], var_names[i]
                edge["edgeType"] = "directed"
            elif raw[i, j] == -1 and raw[j, i] == -1:
                edge["edgeType"] = "bidirected"
            elif (raw[i, j] == -1 and raw[j, i] == 2) or (
                raw[i, j] == 2 and raw[j, i] == -1
            ):
                if raw[i, j] == 2 and raw[j, i] == -1:
                    edge["from"], edge["to"] = var_names[i], var_names[j]
                edge["edgeType"] = "partially_directed"
            elif raw[i, j] == 2 and raw[j, i] == 2:
                edge["edgeType"] = "undirected"
                edge["confidence"] = "low"
            else:
                edge["edgeType"] = "undirected"
                edge["confidence"] = "low"

            edges.append(edge)
            visited.add((i, j))
            visited.add((j, i))

    adj = np.zeros((d, d), dtype=int)
    connected = raw != 0
    adj[connected] = 1

    score = calculate_graph_score(data.to_numpy(), adj)

    return {
        "method": "FCI",
        "adjacency_matrix": adj.tolist(),
        "variables": var_names,
        "edges": edges,
        "score": score,
        "params": {"alpha": alpha, "indep_test": indep_test},
    }


def run_ges(data: pd.DataFrame, score_func: str = "local_score_BIC") -> dict[str, Any]:
    from causal_discovery.ges import GESAlgorithm

    algo = GESAlgorithm()
    adj = algo.fit(data, score_func=score_func)
    var_names = list(data.columns)

    edges = adjacency_to_edges(adj, var_names, "GES", "dag")
    score = calculate_graph_score(data.to_numpy(), adj)

    return {
        "method": "GES",
        "adjacency_matrix": adj.tolist(),
        "variables": var_names,
        "edges": edges,
        "score": score,
        "params": {"score_func": score_func},
    }


def run_notears(
    data: pd.DataFrame,
    lambda1: float = 0.1,
    loss_type: str = "l2",
    max_iter: int = 100,
    h_tol: float = 1e-8,
    w_threshold: float = 0.3,
) -> dict[str, Any]:
    from causal_discovery.notears import NOTEARSAlgorithm

    algo = NOTEARSAlgorithm()
    adj = algo.fit(
        data,
        lambda1=lambda1,
        loss_type=loss_type,
        max_iter=max_iter,
        h_tol=h_tol,
        rho_max=1e16,
        w_threshold=w_threshold,
    )
    var_names = list(data.columns)

    edges = adjacency_to_edges(adj, var_names, "NOTEARS", "dag")
    score = calculate_graph_score(data.to_numpy(), adj)

    return {
        "method": "NOTEARS",
        "adjacency_matrix": adj.tolist(),
        "variables": var_names,
        "edges": edges,
        "score": score,
        "params": {"lambda1": lambda1, "w_threshold": w_threshold},
    }


def run_direct_lingam(data: pd.DataFrame) -> dict[str, Any]:
    from causal_discovery.direct_lingam import DirectLiNGAMAlgorithm

    algo = DirectLiNGAMAlgorithm()
    adj = algo.fit(data)
    var_names = list(data.columns)

    edges = adjacency_to_edges(adj, var_names, "DirectLiNGAM", "dag")
    score = calculate_graph_score(data.to_numpy(), adj)

    return {
        "method": "DirectLiNGAM",
        "adjacency_matrix": adj.tolist(),
        "variables": var_names,
        "edges": edges,
        "score": score,
        "params": {},
    }


def run_boss(data: pd.DataFrame, score_func: str = "auto") -> dict[str, Any]:
    from causal_discovery.permutation import BOSSAlgorithm

    algo = BOSSAlgorithm()
    adj = algo.fit(data, score_func=score_func)
    var_names = list(data.columns)

    edges = adjacency_to_edges(adj, var_names, "BOSS", "dag")
    score = calculate_graph_score(data.to_numpy(), adj)

    return {
        "method": "BOSS",
        "adjacency_matrix": adj.tolist(),
        "variables": var_names,
        "edges": edges,
        "score": score,
        "params": {"score_func": score_func},
    }


def run_grasp(
    data: pd.DataFrame, score_func: str = "auto", depth: int = 3
) -> dict[str, Any]:
    from causal_discovery.permutation import GRaSPAlgorithm

    algo = GRaSPAlgorithm()
    adj = algo.fit(data, score_func=score_func, depth=depth)
    var_names = list(data.columns)

    edges = adjacency_to_edges(adj, var_names, "GRaSP", "dag")
    score = calculate_graph_score(data.to_numpy(), adj)

    return {
        "method": "GRaSP",
        "adjacency_matrix": adj.tolist(),
        "variables": var_names,
        "edges": edges,
        "score": score,
        "params": {"score_func": score_func, "depth": depth},
    }


METHOD_DISPATCH = {
    "pc": run_pc,
    "fci": run_fci,
    "ges": run_ges,
    "notears": run_notears,
    "direct_lingam": run_direct_lingam,
    "boss": run_boss,
    "grasp": run_grasp,
}

# Kernel CI tests are roughly O(N²) per test and PC runs many pair/conditioning
# tests. A pure N threshold missed moderate-N, wider graphs, so use a relative
# N² × pair-count proxy. Warn, never block; the wall-clock timeout is the guard.
KERNEL_TESTS = {"kci", "fastkci", "rcit"}
KERNEL_WARN_WORK_UNITS = 25_000_000
GENERALIZED_SCORES = {"local_score_marginal_general", "local_score_cv_general"}

# Degradation ladder surfaced on timeout / compute warnings.
DEGRADATION_LADDER = (
    "Degradation ladder: kci → bounded fastkci/rcit → seeded rows or an "
    "estimand-relevant subset → fisherz sensitivity baseline (loses "
    "nonlinearity and must be labelled)."
)


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise _Timeout()


def _resolve_score_func(score_func: str, data: pd.DataFrame) -> str:
    """Resolve 'auto' to the concrete score for honest params recording."""
    if score_func != "auto":
        return score_func
    dtype = detect_data_type(data.to_numpy())
    return "local_score_BIC" if dtype == "continuous" else "local_score_BDeu"


@safe_main
def main():
    parser = argparse.ArgumentParser(description="CausalClaw — Causal Discovery Bridge")
    parser.add_argument("--method", required=True, choices=list(METHOD_DISPATCH.keys()))
    parser.add_argument("--data", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--indep-test", default="auto")
    parser.add_argument("--lambda1", type=float, default=0.1)
    parser.add_argument("--w-threshold", type=float, default=0.3)
    parser.add_argument("--score-func", default="auto")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Wall-clock budget in seconds for the algorithm run (0 = no limit). "
        "On expiry a structured timeout error is emitted with a degradation "
        "suggestion — treat this as a normal degradation event, not a crash.",
    )

    args = parser.parse_args()

    data = load_data(args.data)

    # Auto-encode non-numeric columns (string / categorical / boolean) so that
    # algorithms like PC + fisherz or KCI don't crash on non-numeric arrays.
    #
    # NOTE: test for "not numeric" rather than enumerating string-ish dtypes.
    # pandas 3.0 infers a dedicated StringDtype for text columns (no longer the
    # legacy `object` dtype), so `dtype == object` misses them and the raw
    # strings would reach the algorithm. `is_numeric_dtype` is version-robust.
    encoded_cols: list[str] = []
    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            data[col], _ = pd.factorize(data[col])
            encoded_cols.append(col)

    kwargs: dict[str, Any] = {}
    if args.method in ("pc", "fci"):
        kwargs["alpha"] = args.alpha
        kwargs["indep_test"] = args.indep_test
    elif args.method == "notears":
        kwargs["lambda1"] = args.lambda1
        kwargs["w_threshold"] = args.w_threshold
    elif args.method == "ges":
        kwargs["score_func"] = args.score_func
    elif args.method == "boss":
        kwargs["score_func"] = args.score_func
    elif args.method == "grasp":
        kwargs["score_func"] = args.score_func
        kwargs["depth"] = args.depth

    # Pre-flight compute warning for kernel-based runs on large N. Advisory
    # only — recorded in metadata, the run still proceeds.
    compute_warning: str | None = None
    n_samples = len(data)
    n_vars = len(data.columns)
    pair_count = max(n_vars * (n_vars - 1) // 2, 1)
    kernel_work_units = int(n_samples**2 * pair_count)
    if args.indep_test in KERNEL_TESTS and args.method in ("pc", "fci"):
        if kernel_work_units >= KERNEL_WARN_WORK_UNITS or n_vars >= 10:
            compute_warning = (
                f"indep_test={args.indep_test} with n={n_samples}, p={n_vars} has "
                f"high relative kernel work ({kernel_work_units:,}); this is not "
                f"an ETA. {DEGRADATION_LADDER}"
            )
    if args.score_func in GENERALIZED_SCORES and args.method in ("ges", "boss", "grasp"):
        if kernel_work_units >= KERNEL_WARN_WORK_UNITS or n_vars >= 10:
            compute_warning = (
                f"score_func={args.score_func} with n={n_samples}, p={n_vars} has "
                f"high relative kernel work ({kernel_work_units:,}); generalized "
                f"scores use kernel regression internally. "
                f"{DEGRADATION_LADDER}"
            )

    if args.timeout > 0:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(args.timeout)
    try:
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            result = METHOD_DISPATCH[args.method](data, **kwargs)
        elapsed = time.perf_counter() - t0
    except _Timeout:
        output_result(
            {
                "error": f"timeout: {args.method} exceeded {args.timeout}s wall-clock budget",
                "timeout": True,
                "method": args.method,
                "suggestion": (
                    f"This is a normal degradation event, not a crash. {DEGRADATION_LADDER}"
                ),
            }
        )
        sys.exit(1)
    finally:
        if args.timeout > 0:
            signal.alarm(0)

    # Honest params recording: surface the CONCRETE score function, not "auto".
    if "score_func" in result.get("params", {}):
        result["params"]["score_func"] = _resolve_score_func(
            result["params"]["score_func"], data
        )
    if args.timeout > 0:
        result["params"]["timeout_seconds"] = args.timeout

    result["metadata"] = {
        "n_samples": len(data),
        "n_variables": n_vars,
        "elapsed_seconds": round(elapsed, 3),
        "data_path": args.data,
    }
    if compute_warning:
        result["metadata"]["compute_warning"] = compute_warning
    if encoded_cols:
        result["metadata"]["auto_encoded_columns"] = encoded_cols

    output_result(result)


if __name__ == "__main__":
    main()
