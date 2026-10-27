# type: ignore
"""CausalClaw Bridge — Causal Inference (ATE/CATE).

Calculates Average Treatment Effect (ATE) and Conditional Average Treatment Effect (CATE)
using the DoWhy library.
"""

from __future__ import annotations

import argparse
import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
from dowhy import CausalModel
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, SplineTransformer, StandardScaler

# Monkey-patch pandas Series to fix DoWhy KeyError on pandas 2.0+ (integer indexing)
_original_series_getitem = pd.Series.__getitem__


def _patched_series_getitem(self, key):
    if isinstance(key, int) and not isinstance(self.index, pd.RangeIndex):
        try:
            return self.iloc[key]
        except Exception:
            pass
    return _original_series_getitem(self, key)


pd.Series.__getitem__ = _patched_series_getitem

# Suppress noisy warnings from dowhy/statsmodels
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# utils is in the same directory
from utils import load_data, output_result, safe_main


def edges_to_gml(edges: list[dict]) -> tuple[str, list[str]]:
    """Convert a scratchpad edge list to a GML graph string for DoWhy.

    - directed / partially_directed  → single directed edge
    - undirected                     → edges in both directions (MEC)
    - bidirected                     → skipped (latent confounder, not encodable
      in GML without extensions; DoWhy will detect missing adjustment)

    Returns (gml, graph_warnings). The warnings make the lossy conversions
    EXPLICIT — a dropped bidirected edge means "there may be an unobserved
    confounder here" and must never vanish silently, because that turns
    "latent confounding present" into "no confounding" downstream.
    """
    nodes: set[str] = set()
    for e in edges:
        nodes.add(e["from"])
        nodes.add(e["to"])

    warnings_out: list[str] = []
    lines = ["graph [", "  directed 1"]
    for node in sorted(nodes):
        lines.append(f'  node [ id "{node}" label "{node}" ]')
    for e in edges:
        etype = e.get("edgeType", "directed")
        src, tgt = e["from"], e["to"]
        if etype == "directed":
            lines.append(f'  edge [ source "{src}" target "{tgt}" ]')
        elif etype == "partially_directed":
            lines.append(f'  edge [ source "{src}" target "{tgt}" ]')
            warnings_out.append(
                f"partially-directed edge {src} o-> {tgt} was treated as CERTAIN "
                "directed — the orientation uncertainty is not represented in "
                "DoWhy's model."
            )
        elif etype == "undirected":
            # Both directions so DoWhy treats it as a Markov-equivalent pair
            lines.append(f'  edge [ source "{src}" target "{tgt}" ]')
            lines.append(f'  edge [ source "{tgt}" target "{src}" ]')
        elif etype == "bidirected":
            warnings_out.append(
                f"bidirected edge {src} <-> {tgt} DROPPED from the DoWhy graph: "
                "this edge marks a possible LATENT CONFOUNDER. The estimate below "
                "assumes NO unobserved confounding between these variables — treat "
                "the result as conditional on that assumption and disclose it."
            )
        else:
            warnings_out.append(
                f"unknown edgeType '{etype}' on {src} -> {tgt} was dropped."
            )
    lines.append("]")
    return "\n".join(lines), warnings_out


def _extract_refutation(res) -> dict[str, Any]:
    """Safely extract new_effect and p_value from a CausalRefutation result.

    DoWhy 0.13/0.14 store results in res.refutation_result as a dict with keys
    'p_value' and 'is_statistically_significant'.
    """
    p_raw = res.refutation_result.get("p_value")
    # p_value may be a numpy scalar or array
    try:
        p_value = float(np.ravel(p_raw)[0]) if p_raw is not None else None
    except Exception:
        p_value = None

    new_effect = res.new_effect
    try:
        new_effect = float(np.ravel(new_effect)[0])
    except Exception:
        new_effect = float(new_effect) if new_effect is not None else None

    # Placebo: p > 0.05 means the placebo effect is NOT significant → passed
    # Random common cause / data subset: p > 0.05 means new effect is not
    # significantly different from original → passed
    passed = bool(p_value > 0.05) if p_value is not None else None

    return {
        "new_effect": new_effect,
        "p_value": p_value,
        "passed": passed,
    }


def _extract_cate_info(
    estimate: Any,
    effect_modifiers: list[str],
    data: pd.DataFrame,
    treatment: str,
    common_causes: list[str] | None = None,
) -> dict[str, Any] | None:
    """Extract CATE information from a DoWhy linear regression estimate.

    Maps index-based statsmodels exog names (x1, x2, ...) to their actual variables
    to extract treatment and interaction coefficients.
    """
    if not effect_modifiers:
        return None
    # Only handle the single-modifier case for now
    modifier = effect_modifiers[0]
    try:
        est = estimate.estimator
        cc_names = getattr(est, "_observed_common_causes_names", []) or []
        if not cc_names and common_causes:
            cc_names = common_causes

        fitted_model = est.model
        params = fitted_model.params
        pvalues = fitted_model.pvalues

        # Build feature map based on DoWhy's _build_features order:
        # Index 0: const
        # Index 1: treatment
        # Index 2 to 2 + len(cc) - 1: common causes
        # Index 2 + len(cc) onwards: interaction terms
        treatment_idx = 1
        interaction_idx = 1 + 1 + len(cc_names)

        treatment_key = f"x{treatment_idx}"
        interaction_key = f"x{interaction_idx}"

        # Fallback in case of unexpected naming or different versions
        if treatment_key not in params.index or interaction_key not in params.index:
            treatment_key = None
            interaction_key = None
            for key in params.index:
                if key == "const":
                    continue
                if key == treatment:
                    treatment_key = key
                elif f"{treatment}:{modifier}" in key or f"{modifier}:{treatment}" in key:
                    interaction_key = key
                elif treatment in key and modifier in key:
                    interaction_key = key
            
            if not treatment_key or not interaction_key:
                return None

        treatment_coef = float(params[treatment_key])
        interaction_coef = float(params[interaction_key])
        interaction_pval = float(pvalues[interaction_key])

        # Human-readable formula (using 6 decimals to prevent small effects rounding to 0)
        sign = "+" if interaction_coef >= 0 else "-"
        formula = (
            f"CATE({modifier}) = {treatment_coef:+.6f} {sign} "
            f"{abs(interaction_coef):.6f} × {modifier}"
        )

        # Q1 / Q2 / Q3 examples
        quantile_estimates = []
        for label, q in [("Q1 (25%)", 0.25), ("Q2 (50%)", 0.50), ("Q3 (75%)", 0.75)]:
            val = float(data[modifier].quantile(q))
            cate_val = treatment_coef + interaction_coef * val
            quantile_estimates.append({
                "label": label,
                "modifier_value": round(val, 2),
                "cate": round(cate_val, 6),
            })

        return {
            "modifier": modifier,
            "formula": formula,
            "treatment_coef": round(treatment_coef, 6),
            "interaction_coef": round(interaction_coef, 6),
            "interaction_p_value": round(interaction_pval, 6) if interaction_pval is not None else None,
            "quantile_estimates": quantile_estimates,
        }
    except Exception:
        return None


def _calculate_contrast_legacy(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    common_causes: list[str] | None = None,
    effect_modifiers: list[str] | None = None,
    method_name: str = "backdoor.linear_regression",
    graph_edges: list[dict] | None = None,
    treatment_value: Any | None = None,
    control_value: Any | None = None,
) -> dict[str, Any]:
    """Calculate ATE (and optionally CATE) using DoWhy.

    If ``graph_edges`` is provided the causal graph is passed to DoWhy and it
    will automatically identify the correct backdoor adjustment set.  The
    ``common_causes`` argument is ignored in that case.

    If neither ``graph_edges`` nor ``common_causes`` is supplied, DoWhy will
    attempt identification without any confounders declared.

    ``treatment_value`` / ``control_value``: explicit treatment/control category
    values. Required when the treatment column contains more than two string
    categories; optional (auto-detected) for binary or numeric treatments.
    """
    original_n_rows = len(data)

    # ── String treatment pre-processing ──────────────────────────────────────
    # DoWhy's _do() replaces the treatment column with numeric 0/1 during
    # counterfactual prediction, which breaks its own OHE encoder that was
    # fitted on the original string values. We resolve this before DoWhy sees
    # the data.
    col = data[treatment]
    is_string_col = col.dtype == object or isinstance(col.dtype, pd.StringDtype)

    if is_string_col:
        unique_vals = sorted(col.dropna().unique().tolist())
        n_unique = len(unique_vals)

        if n_unique == 2:
            # Binary string → auto-map to 0/1, fully transparent to caller.
            # Respect caller-provided treatment_value/control_value if supplied.
            if treatment_value is not None and control_value is not None:
                tv, cv = str(treatment_value), str(control_value)
                if tv not in unique_vals or cv not in unique_vals:
                    raise ValueError(
                        f"treatment_value='{tv}' or control_value='{cv}' not found "
                        f"in column '{treatment}'. Available: {unique_vals}"
                    )
                mapping = {cv: 0, tv: 1}
            else:
                # Lexicographic default: first = 0 (control), second = 1 (treatment)
                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                treatment_value = 1
                control_value = 0
            data = data.copy()
            data[treatment] = data[treatment].map(mapping).astype("int64")
            # After encoding, always use integer 0/1 as DoWhy's treatment/control values
            treatment_value = 1
            control_value = 0

        elif n_unique > 2:
            # Multi-class string treatment: DoWhy cannot handle this directly.
            # Require the caller to specify exactly which two categories to compare.
            if treatment_value is None or control_value is None:
                raise ValueError(
                    f"MULTI_CLASS_TREATMENT: Column '{treatment}' has {n_unique} string "
                    f"categories {unique_vals}. DoWhy's linear regression estimator "
                    f"requires a binary treatment. Please re-call causal_infer with "
                    f"treatment_value=<one category> and control_value=<another category> "
                    f"to compare a specific pair. Example: "
                    f"treatment_value='{unique_vals[-1]}', control_value='{unique_vals[0]}'"
                )
            tv, cv = str(treatment_value), str(control_value)
            if tv not in unique_vals or cv not in unique_vals:
                raise ValueError(
                    f"treatment_value='{tv}' or control_value='{cv}' not found "
                    f"in column '{treatment}'. Available: {unique_vals}"
                )
            # Subset to just the two chosen categories and map to 0/1
            mask = data[treatment].isin([tv, cv])
            data = data[mask].copy()
            data[treatment] = data[treatment].map({cv: 0, tv: 1}).astype("int64")
            treatment_value = 1
            control_value = 0

    # Resolve DoWhy treatment_value / control_value defaults
    if treatment_value is None:
        treatment_value = 1
    if control_value is None:
        control_value = 0

    # Build causal model — graph-based identification takes priority
    graph_warnings: list[str] = []
    if graph_edges:
        gml, graph_warnings = edges_to_gml(graph_edges)
        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            graph=gml,
            effect_modifiers=effect_modifiers,  # was silently ignored before
        )
    else:
        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=common_causes,
            effect_modifiers=effect_modifiers,
        )

    # 2. Identify Effect
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    # Report which backdoor variables were automatically selected (for transparency)
    auto_adjustment_set: list[str] = []
    try:
        auto_adjustment_set = list(identified_estimand.get_backdoor_variables() or [])
    except Exception:
        pass

    # 3. Estimate Effect
    estimate = model.estimate_effect(
        identified_estimand,
        method_name=method_name,
        target_units="ate",
        treatment_value=treatment_value,
        control_value=control_value,
    )

    # 4. Extract ATE point estimate
    ate = float(estimate.value)

    # 5. Confidence interval (shape: (1,2) ndarray in DoWhy 0.13+)
    ci_lower: float | None = None
    ci_upper: float | None = None
    try:
        ci = estimate.get_confidence_intervals()
        ci_arr = np.ravel(ci)
        if len(ci_arr) >= 2:
            ci_lower = float(ci_arr[0])
            ci_upper = float(ci_arr[1])
    except Exception:
        pass

    # 6. p-value of the estimate itself (test that ATE != 0)
    estimate_p_value: float | None = None
    try:
        sig = estimate.test_stat_significance()
        p_raw = sig.get("p_value")
        estimate_p_value = float(np.ravel(p_raw)[0]) if p_raw is not None else None
    except Exception:
        pass

    # 7. Estimator class name
    estimator_name = method_name
    try:
        if hasattr(estimate.params, "get"):
            estimator_class = estimate.params.get("estimator_class")
        else:
            estimator_class = estimate.params["estimator_class"]
        if estimator_class is not None:
            estimator_name = (
                estimator_class.__name__
                if hasattr(estimator_class, "__name__")
                else method_name
            )
    except (KeyError, TypeError, AttributeError):
        pass

    # 8. Refutation tests
    #
    # DoWhy 0.14 method names (verified):
    #   - placebo_treatment_refuter  (was: placebo_treatment_refuter ✓)
    #   - random_common_cause        (was WRONG: add_random_common_cause → ImportError)
    #   - data_subset_refuter        (was: data_subset_refuter ✓)
    refutations: dict[str, Any] = {}

    try:
        res_placebo = model.refute_estimate(
            identified_estimand, estimate, method_name="placebo_treatment_refuter"
        )
        refutations["placebo_treatment"] = _extract_refutation(res_placebo)
    except Exception as e:
        refutations["placebo_treatment"] = {"error": str(e), "passed": None}

    try:
        # FIXED: was "add_random_common_cause" (DoWhy ≤0.11 name) → now "random_common_cause"
        res_random = model.refute_estimate(
            identified_estimand, estimate, method_name="random_common_cause"
        )
        refutations["random_common_cause"] = _extract_refutation(res_random)
    except Exception as e:
        refutations["random_common_cause"] = {"error": str(e), "passed": None}

    try:
        res_subset = model.refute_estimate(
            identified_estimand, estimate, method_name="data_subset_refuter"
        )
        refutations["data_subset"] = _extract_refutation(res_subset)
    except Exception as e:
        refutations["data_subset"] = {"error": str(e), "passed": None}

    # 9. Build result
    result: dict[str, Any] = {
        "treatment": treatment,
        "outcome": outcome,
        "ate": ate,
        "estimator": estimator_name,
        "refutation_results": refutations,
        "_analysis_n_rows": len(data),
        "_analysis_rows_dropped": original_n_rows - len(data),
    }

    if auto_adjustment_set:
        result["auto_adjustment_set"] = auto_adjustment_set

    if graph_warnings:
        result["graph_warnings"] = graph_warnings

    if ci_lower is not None:
        result["ci_lower"] = ci_lower
        result["ci_upper"] = ci_upper

    if estimate_p_value is not None:
        result["p_value"] = estimate_p_value

    # 10. CATE extraction (only when effect_modifiers were requested and the
    #     estimator is linear regression, which adds interaction terms)
    if effect_modifiers and method_name == "backdoor.linear_regression":
        cate_info = _extract_cate_info(estimate, effect_modifiers, data, treatment, auto_adjustment_set)
        if cate_info is not None:
            result["cate"] = cate_info

    return result


_MIN_FLEXIBLE_RMSE_GAIN = 0.05
_MIN_TREATMENT_SHAPE_RMSE_GAIN = 0.01
_DEFAULT_RANDOM_STATE = 20260828
_MAX_CATEGORICAL_CATE_LEVELS = 10
_MAX_CATE_BOOTSTRAP_SAMPLES = 30
_MAX_CATE_ATE_RELATIVE_DISCREPANCY = 0.10
_CATE_ATE_OUTCOME_SD_FLOOR = 0.01


def _relevant_graph_edges(
    graph_edges: list[dict], treatment: str, outcome: str
) -> list[dict]:
    """Restrict identification to graph components touching T or Y.

    Disconnected graph branches cannot affect the estimand and should not
    make DoWhy reject an otherwise valid DAG merely because that branch has an
    uncertain edge. Within either estimand component, however, every uncertain
    endpoint can change a treatment parent, outcome parent, or causal path and
    therefore fails closed.
    """
    adjacency: dict[str, set[str]] = {}
    for edge in graph_edges:
        source = edge.get("from")
        target = edge.get("to")
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError(
                "GRAPH_IDENTIFICATION_INCONCLUSIVE: every edge needs string "
                "'from' and 'to' endpoints."
            )
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)

    relevant_nodes = {treatment, outcome}
    frontier = [treatment, outcome]
    while frontier:
        node = frontier.pop()
        for neighbour in adjacency.get(node, set()):
            if neighbour not in relevant_nodes:
                relevant_nodes.add(neighbour)
                frontier.append(neighbour)

    relevant = [
        edge
        for edge in graph_edges
        if edge["from"] in relevant_nodes and edge["to"] in relevant_nodes
    ]
    represented_nodes = {
        endpoint for edge in relevant for endpoint in (edge["from"], edge["to"])
    }
    missing = [
        node for node in (treatment, outcome) if node not in represented_nodes
    ]
    if missing:
        raise ValueError(
            "GRAPH_IDENTIFICATION_INCONCLUSIVE: the edge-list graph does not "
            f"represent estimand node(s) {missing}; refusing to infer an effect "
            "from an incomplete graph."
        )
    return relevant


def _reject_ambiguous_relevant_edges(graph_edges: list[dict]) -> None:
    ambiguous = []
    for edge in graph_edges:
        edge_type = edge.get("edgeType", "directed")
        if edge_type != "directed":
            ambiguous.append(
                f"{edge['from']} -[{edge_type}]- {edge['to']}"
            )
    if ambiguous:
        raise ValueError(
            "GRAPH_IDENTIFICATION_INCONCLUSIVE: estimand-relevant graph contains "
            "uncertain, latent-confounded, or unsupported edge types: "
            + "; ".join(ambiguous)
            + ". Resolve and update the relevant belief-graph orientations or "
            "latent-confounding assumptions with evidence, otherwise report the "
            "effect as unidentified; no effect was estimated."
        )


def _identify_shift_adjustment_set(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    common_causes: list[str] | None,
    graph_edges: list[dict] | None,
) -> tuple[list[str], list[str]]:
    """Use DoWhy only to identify the backdoor adjustment set."""
    graph_warnings: list[str] = []
    if graph_edges:
        relevant_edges = _relevant_graph_edges(graph_edges, treatment, outcome)
        _reject_ambiguous_relevant_edges(relevant_edges)
        gml, graph_warnings = edges_to_gml(relevant_edges)
        try:
            model = CausalModel(
                data=data,
                treatment=treatment,
                outcome=outcome,
                graph=gml,
            )
        except Exception as exc:
            raise ValueError(
                "CAUSAL_IDENTIFICATION_FAILED: the estimand-relevant directed "
                f"graph is not a valid identifiable DAG: {exc}"
            ) from exc
    else:
        try:
            model = CausalModel(
                data=data,
                treatment=treatment,
                outcome=outcome,
                common_causes=common_causes,
            )
        except Exception as exc:
            raise ValueError(
                "CAUSAL_IDENTIFICATION_FAILED: DoWhy could not construct the "
                f"manual identification model: {exc}"
            ) from exc

    try:
        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=False
        )
    except Exception as exc:
        raise ValueError(
            "CAUSAL_IDENTIFICATION_FAILED: DoWhy could not identify the total "
            f"effect under the supplied graph/common causes: {exc}"
        ) from exc

    estimands = getattr(identified_estimand, "estimands", None)
    if not isinstance(estimands, dict) or estimands.get("backdoor") is None:
        raise ValueError(
            "CAUSAL_IDENTIFICATION_FAILED: no valid backdoor estimand was "
            "identified (the graph may have no directed T->Y path or unresolved "
            "confounding); no effect was estimated."
        )
    try:
        identified_set = list(
            identified_estimand.get_backdoor_variables() or []
        )
    except Exception as exc:
        raise ValueError(
            "CAUSAL_IDENTIFICATION_FAILED: DoWhy did not return a usable "
            "backdoor adjustment set."
        ) from exc

    if graph_edges:
        adjustment_set = identified_set
    elif common_causes is not None:
        # Preserve the caller's explicit identification assumption across
        # DoWhy versions, which expose manual common causes inconsistently.
        adjustment_set = list(common_causes)
    else:
        adjustment_set = identified_set
    return adjustment_set, graph_warnings


def _shift_model_candidates(random_state: int) -> dict[str, Any]:
    """Return the fixed, small candidate family used by auto estimation."""
    return {
        "linear": make_pipeline(StandardScaler(), LinearRegression()),
        "spline_ridge": make_pipeline(
            SplineTransformer(
                n_knots=5,
                degree=3,
                knots="quantile",
                include_bias=False,
                extrapolation="linear",
            ),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-4, 3, 16)),
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=random_state,
        ),
    }


def _treatment_shape_model(n_features: int) -> Any:
    """Spline only T while keeping adjustment variables linear.

    This targeted comparison avoids hiding a treatment-specific curve inside a
    noisy outcome where the global flexible-model RMSE gain can be small.
    """
    treatment_spline = SplineTransformer(
        n_knots=5,
        degree=3,
        knots="quantile",
        include_bias=False,
        extrapolation="linear",
    )
    if n_features == 1:
        features: Any = treatment_spline
    else:
        features = ColumnTransformer(
            [
                ("treatment_spline", treatment_spline, [0]),
                (
                    "linear_adjustment",
                    StandardScaler(),
                    list(range(1, n_features)),
                ),
            ]
        )
    return make_pipeline(
        features,
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-4, 3, 16)),
    )


def _grouped_crossfit_indices(
    groups: np.ndarray, n_splits: int, random_state: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split positions while keeping identical original-row IDs together."""
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Grouped cross-fitting needs at least {n_splits} unique rows; "
            f"got {len(unique_groups)}."
        )
    rng = np.random.default_rng(random_state)
    shuffled_groups = rng.permutation(unique_groups)
    valid_group_folds = np.array_split(shuffled_groups, n_splits)
    return [
        (
            np.flatnonzero(~np.isin(groups, valid_groups)),
            np.flatnonzero(np.isin(groups, valid_groups)),
        )
        for valid_groups in valid_group_folds
    ]


def _cross_fitted_shift(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    shift_delta: float,
    n_splits: int,
    random_state: int,
    groups: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    """Compute held-out fit metrics and per-row T -> T + delta effects."""
    if n_splits < 2 or n_splits > len(y):
        raise ValueError(
            f"n_splits must be between 2 and n_rows ({len(y)}); got {n_splits}."
        )

    oof_prediction = np.empty(len(y), dtype=float)
    individual_effects = np.empty(len(y), dtype=float)
    if groups is None:
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_indices = folds.split(x)
    else:
        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError("groups must have one identity per row.")
        split_indices = _grouped_crossfit_indices(
            groups, n_splits, random_state
        )

    for train_idx, valid_idx in split_indices:
        fitted = clone(model).fit(x[train_idx], y[train_idx])
        x_observed = x[valid_idx].copy()
        x_shifted = x_observed.copy()
        x_shifted[:, 0] += shift_delta
        observed_prediction = fitted.predict(x_observed)
        shifted_prediction = fitted.predict(x_shifted)
        oof_prediction[valid_idx] = observed_prediction
        individual_effects[valid_idx] = shifted_prediction - observed_prediction

    metrics = {
        "oof_r2": float(r2_score(y, oof_prediction)),
        "oof_rmse": float(np.sqrt(mean_squared_error(y, oof_prediction))),
        "effect": float(np.mean(individual_effects)),
    }
    return metrics, individual_effects


def _support_diagnostics(x: np.ndarray, shift_delta: float) -> dict[str, Any]:
    """Surface marginal and approximate joint support for the shift policy."""
    treatment = x[:, 0]
    shifted = treatment + shift_delta
    observed_min = float(np.min(treatment))
    observed_max = float(np.max(treatment))
    q01, q99 = (float(v) for v in np.quantile(treatment, [0.01, 0.99]))
    outside_range = float(
        np.mean((shifted < observed_min) | (shifted > observed_max))
    )
    outside_central = float(np.mean((shifted < q01) | (shifted > q99)))

    nn_unsupported = 0.0
    nn_threshold: float | None = None
    if len(x) >= 3:
        scaler = RobustScaler().fit(x)
        observed_scaled = scaler.transform(x)
        shifted_x = x.copy()
        shifted_x[:, 0] += shift_delta
        shifted_scaled = scaler.transform(shifted_x)
        neighbours = NearestNeighbors(n_neighbors=2).fit(observed_scaled)
        observed_nn = neighbours.kneighbors(observed_scaled)[0][:, 1]
        shifted_nn = neighbours.kneighbors(
            shifted_scaled, n_neighbors=1
        )[0][:, 0]
        nn_threshold = float(np.quantile(observed_nn, 0.95))
        nn_unsupported = float(np.mean(shifted_nn > nn_threshold))

    support_warnings: list[str] = []
    if outside_range > 0:
        support_warnings.append(
            f"{outside_range:.1%} of shifted treatment values lie outside the "
            "observed treatment range and require extrapolation."
        )
    if outside_central > 0.05:
        support_warnings.append(
            f"{outside_central:.1%} of shifted treatment values lie outside the "
            "observed 1%-99% treatment interval."
        )
    if nn_unsupported > 0.10:
        support_warnings.append(
            f"{nn_unsupported:.1%} of shifted (treatment, adjustment) rows have "
            "weak local joint support by the nearest-neighbour diagnostic."
        )

    if outside_range > 0.05 or outside_central > 0.25 or nn_unsupported > 0.25:
        status = "poor"
    elif support_warnings:
        status = "caution"
    else:
        status = "good"

    return {
        "observed_treatment_min": observed_min,
        "observed_treatment_max": observed_max,
        "shifted_treatment_min": float(np.min(shifted)),
        "shifted_treatment_max": float(np.max(shifted)),
        "central_01_99_interval": [q01, q99],
        "outside_observed_range_fraction": outside_range,
        "outside_01_99_fraction": outside_central,
        "nearest_neighbor_unsupported_fraction": nn_unsupported,
        "nearest_neighbor_threshold": nn_threshold,
        "status": status,
        "warnings": support_warnings,
    }


def _effects_agree(first: float, second: float, outcome_sd: float) -> bool:
    near_zero = max(0.02 * outcome_sd, 1e-8)
    if abs(first) <= near_zero and abs(second) <= near_zero:
        return True
    if np.sign(first) != np.sign(second):
        return False
    relative_difference = abs(first - second) / max(
        abs(first), abs(second), 0.10 * outcome_sd, 1e-8
    )
    return bool(relative_difference <= 0.50)


def _bootstrap_shift_ci(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    shift_delta: float,
    bootstrap_samples: int,
    n_splits: int,
    random_state: int,
) -> tuple[dict[str, Any], list[str]]:
    """Bootstrap while keeping duplicate copies of a row in the same fold."""
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative.")
    conditions = [
        "The adjustment set, graph, and selected model family are treated as fixed.",
        "Rows are sampled with replacement; every duplicate copy of one original "
        "row is kept in the same cross-fitting fold to prevent train/validation leakage.",
    ]
    if bootstrap_samples == 0:
        return {
            "method": "not_requested",
            "bootstrap_samples_requested": 0,
            "bootstrap_samples_completed": 0,
            "conditions": conditions,
        }, []

    rng = np.random.default_rng(random_state)
    estimates: list[float] = []
    for index in range(bootstrap_samples):
        sampled = rng.integers(0, len(y), size=len(y))
        try:
            grouped_folds = min(
                3, n_splits, len(np.unique(sampled))
            )
            if grouped_folds < 2:
                continue
            metrics, _ = _cross_fitted_shift(
                model,
                x[sampled],
                y[sampled],
                shift_delta,
                grouped_folds,
                random_state + 10_000 + index,
                groups=sampled,
            )
            estimates.append(metrics["effect"])
        except Exception:
            continue

    uncertainty_warnings: list[str] = []
    minimum_completed = 10
    base = {
        "method": "nonparametric_group_cross_fitted_bootstrap",
        "bootstrap_samples_requested": bootstrap_samples,
        "bootstrap_samples_completed": len(estimates),
        "conditions": conditions,
    }
    if len(estimates) < minimum_completed:
        uncertainty_warnings.append(
            "Too few bootstrap refits completed for a reliable confidence interval."
        )
        return base, uncertainty_warnings

    values = np.asarray(estimates, dtype=float)
    base.update(
        {
            "standard_error": (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ),
            "ci_level": 0.95,
            "ci_lower": float(np.quantile(values, 0.025)),
            "ci_upper": float(np.quantile(values, 0.975)),
        }
    )
    base["conditions"].append(
        "The interval does not include causal-graph or identification uncertainty."
    )
    return base, uncertainty_warnings


def _effect_modifier_graph_status(
    modifier: str,
    treatment: str,
    graph_edges: list[dict] | None,
) -> tuple[str, list[str]]:
    """Reject post-treatment modifiers and disclose unverifiable timing."""
    if not graph_edges:
        return (
            "assumed_pre_treatment_without_graph_validation",
            [
                f"Effect modifier '{modifier}' is assumed to be measured before "
                "treatment; no supplied graph validated that timing assumption."
            ],
        )

    adjacency: dict[str, set[str]] = {}
    represented_nodes: set[str] = set()
    for edge in graph_edges:
        source = edge.get("from")
        target = edge.get("to")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        represented_nodes.update((source, target))
        if edge.get("edgeType", "directed") == "directed":
            adjacency.setdefault(source, set()).add(target)

    descendants: set[str] = set()
    frontier = [treatment]
    while frontier:
        node = frontier.pop()
        for child in adjacency.get(node, set()):
            if child not in descendants:
                descendants.add(child)
                frontier.append(child)
    if modifier in descendants:
        raise ValueError(
            "POST_TREATMENT_EFFECT_MODIFIER: additive-shift CATE conditions on "
            f"'{modifier}', but the supplied graph makes it a descendant of "
            f"treatment '{treatment}'. Conditioning on a post-treatment variable "
            "does not identify a baseline subgroup CATE."
        )

    if modifier not in represented_nodes:
        return (
            "assumed_pre_treatment_not_represented_in_graph",
            [
                f"Effect modifier '{modifier}' is not represented in the supplied "
                "graph, so its pre-treatment status is an explicit assumption."
            ],
        )
    return "validated_non_descendant_in_supplied_graph", []


def _backdoor_moral_path(
    graph_edges: list[dict],
    treatment: str,
    outcome: str,
    conditioned: set[str],
) -> list[str] | None:
    """Return an active path in the conditioned ancestral backdoor graph.

    D-separation in a DAG is equivalent to ordinary separation after taking the
    ancestral graph of X, Y, and Z, moralizing it, and removing Z. Removing all
    outgoing treatment arrows first yields the standard backdoor graph.
    """
    parents: dict[str, set[str]] = {}
    nodes = {treatment, outcome, *conditioned}
    for edge in graph_edges:
        source = edge.get("from")
        target = edge.get("to")
        if (
            not isinstance(source, str)
            or not isinstance(target, str)
            or edge.get("edgeType", "directed") != "directed"
        ):
            continue
        nodes.update((source, target))
        if source == treatment:
            continue
        parents.setdefault(target, set()).add(source)

    ancestors = {treatment, outcome, *conditioned}
    frontier = list(ancestors)
    while frontier:
        node = frontier.pop()
        for parent in parents.get(node, set()):
            if parent not in ancestors:
                ancestors.add(parent)
                frontier.append(parent)

    moral: dict[str, set[str]] = {node: set() for node in ancestors}
    for child in ancestors:
        child_parents = sorted(parents.get(child, set()) & ancestors)
        for parent in child_parents:
            moral[parent].add(child)
            moral[child].add(parent)
        for left_index, left in enumerate(child_parents):
            for right in child_parents[left_index + 1 :]:
                moral[left].add(right)
                moral[right].add(left)

    if treatment in conditioned or outcome in conditioned:
        return [treatment, outcome]
    blocked = set(conditioned)
    queue = [treatment]
    predecessor: dict[str, str | None] = {treatment: None}
    while queue:
        node = queue.pop(0)
        if node == outcome:
            path: list[str] = []
            cursor: str | None = node
            while cursor is not None:
                path.append(cursor)
                cursor = predecessor[cursor]
            return list(reversed(path))
        for neighbour in moral.get(node, set()):
            if neighbour in blocked or neighbour in predecessor:
                continue
            predecessor[neighbour] = node
            queue.append(neighbour)
    return None


def _validate_cate_conditional_exchangeability(
    graph_edges: list[dict],
    treatment: str,
    outcome: str,
    adjustment_set: list[str],
    modifier: str,
) -> dict[str, Any]:
    """Verify Y(t) independent of T given adjustment variables and modifier."""
    conditioning_set = list(dict.fromkeys([*adjustment_set, modifier]))
    active_path = _backdoor_moral_path(
        graph_edges,
        treatment,
        outcome,
        set(conditioning_set),
    )
    if active_path is not None:
        raise ValueError(
            "CATE_CONDITIONAL_EXCHANGEABILITY_FAILED: conditioning on adjustment "
            f"set plus effect modifier {conditioning_set} does not block every "
            f"backdoor path between '{treatment}' and '{outcome}'. An active path "
            "remains in the conditioned ancestral moralized backdoor graph: "
            + " -- ".join(active_path)
            + ". The modifier may be a baseline collider; no subgroup effect was "
            "estimated."
        )
    return {
        "status": "validated_in_supplied_graph",
        "criterion": "d-separation in the backdoor graph after conditioning",
        "conditioning_set": conditioning_set,
        "backdoor_graph_operation": "removed all outgoing treatment edges",
        "decision_grade": True,
    }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _build_cate_strata(
    modifier_values: pd.Series,
) -> tuple[list[tuple[dict[str, Any], np.ndarray]], dict[str, Any]]:
    """Create transparent, deterministic strata for one baseline modifier."""
    unique_count = int(modifier_values.nunique(dropna=True))
    if unique_count < 2:
        raise ValueError(
            "EFFECT_MODIFIER_VARIATION_REQUIRED: the effect modifier needs at "
            "least two observed levels."
        )

    if pd.api.types.is_numeric_dtype(modifier_values) and unique_count > 6:
        codes, bins = pd.qcut(
            modifier_values.astype(float),
            q=3,
            labels=False,
            retbins=True,
            duplicates="drop",
        )
        code_array = np.asarray(codes, dtype=int)
        realized = int(len(bins) - 1)
        if realized < 2:
            raise ValueError(
                "EFFECT_MODIFIER_STRATIFICATION_FAILED: quantile boundaries "
                "collapsed to fewer than two strata."
            )
        strata: list[tuple[dict[str, Any], np.ndarray]] = []
        for index in range(realized):
            lower = float(bins[index])
            upper = float(bins[index + 1])
            label = f"Q{index + 1}: {lower:.6g} to {upper:.6g}"
            strata.append(
                (
                    {
                        "id": f"q{index + 1}",
                        "label": label,
                        "lower": lower,
                        "upper": upper,
                        "lower_inclusive": index == 0,
                        "upper_inclusive": True,
                    },
                    code_array == index,
                )
            )
        return strata, {
            "strategy": "equal_frequency_quantile_strata",
            "requested_groups": 3,
            "realized_groups": realized,
            "boundaries": [float(value) for value in bins],
        }

    if unique_count > _MAX_CATEGORICAL_CATE_LEVELS:
        raise ValueError(
            "HIGH_CARDINALITY_EFFECT_MODIFIER: exact stratification would create "
            f"{unique_count} groups; the supported maximum is "
            f"{_MAX_CATEGORICAL_CATE_LEVELS}. Supply one substantively defined "
            "lower-cardinality baseline modifier instead of data-mining many groups."
        )

    values = sorted(
        modifier_values.drop_duplicates().tolist(), key=lambda value: str(value)
    )
    strata = []
    for index, value in enumerate(values):
        scalar = _json_scalar(value)
        strata.append(
            (
                {
                    "id": f"level_{index + 1}",
                    "label": str(scalar),
                    "value": scalar,
                },
                modifier_values.eq(value).to_numpy(dtype=bool),
            )
        )
    return strata, {
        "strategy": "observed_levels",
        "requested_groups": unique_count,
        "realized_groups": unique_count,
        "levels": [_json_scalar(value) for value in values],
    }


def _select_stratum_shift_model(
    *,
    x: np.ndarray,
    y: np.ndarray,
    shift_delta: float,
    n_splits: int,
    random_state: int,
    method_name: str,
) -> dict[str, Any]:
    """Run the functional-form and effect-agreement gate inside one stratum."""
    candidates = _shift_model_candidates(random_state)
    candidate_metrics: dict[str, dict[str, float]] = {}
    for name, candidate_model in candidates.items():
        metrics, _ = _cross_fitted_shift(
            candidate_model,
            x,
            y,
            shift_delta,
            n_splits,
            random_state,
        )
        candidate_metrics[name] = metrics
    targeted_shape_metrics, _ = _cross_fitted_shift(
        _treatment_shape_model(x.shape[1]),
        x,
        y,
        shift_delta,
        n_splits,
        random_state,
    )

    linear_rmse = candidate_metrics["linear"]["oof_rmse"]
    flexible_names = ["spline_ridge", "hist_gradient_boosting"]
    best_flexible = min(
        flexible_names,
        key=lambda name: candidate_metrics[name]["oof_rmse"],
    )
    best_flexible_rmse = candidate_metrics[best_flexible]["oof_rmse"]
    relative_rmse_gain = (
        (linear_rmse - best_flexible_rmse) / linear_rmse
        if linear_rmse > np.finfo(float).eps
        else 0.0
    )
    targeted_shape_gain = (
        (linear_rmse - targeted_shape_metrics["oof_rmse"]) / linear_rmse
        if linear_rmse > np.finfo(float).eps
        else 0.0
    )
    outcome_sd = float(np.std(y, ddof=1))
    spline_hgb_agreement = _effects_agree(
        candidate_metrics["spline_ridge"]["effect"],
        candidate_metrics["hist_gradient_boosting"]["effect"],
        outcome_sd,
    )
    targeted_spline_agreement = _effects_agree(
        targeted_shape_metrics["effect"],
        candidate_metrics["spline_ridge"]["effect"],
        outcome_sd,
    )
    targeted_hgb_agreement = _effects_agree(
        targeted_shape_metrics["effect"],
        candidate_metrics["hist_gradient_boosting"]["effect"],
        outcome_sd,
    )
    flexible_agreement = (
        spline_hgb_agreement
        and targeted_spline_agreement
        and targeted_hgb_agreement
    )
    linear_spline_agreement = _effects_agree(
        candidate_metrics["linear"]["effect"],
        candidate_metrics["spline_ridge"]["effect"],
        outcome_sd,
    )
    linear_hgb_agreement = _effects_agree(
        candidate_metrics["linear"]["effect"],
        candidate_metrics["hist_gradient_boosting"]["effect"],
        outcome_sd,
    )
    linear_targeted_agreement = _effects_agree(
        candidate_metrics["linear"]["effect"],
        targeted_shape_metrics["effect"],
        outcome_sd,
    )
    flexible_consensus_against_linear = (
        flexible_agreement
        and not linear_spline_agreement
        and not linear_hgb_agreement
    )
    treatment_shape_triggered = (
        targeted_shape_gain >= _MIN_TREATMENT_SHAPE_RMSE_GAIN
        or (
            not linear_targeted_agreement
            and flexible_consensus_against_linear
        )
    )
    global_flexible_gain = relative_rmse_gain >= _MIN_FLEXIBLE_RMSE_GAIN
    flexible_required = global_flexible_gain or treatment_shape_triggered
    linear_flexible_disagreement = not (
        linear_spline_agreement and linear_hgb_agreement
    )

    method_key = (method_name or "auto").lower()
    explicit_methods = {
        "linear": "linear",
        "backdoor.linear_regression": "linear",
        "spline_ridge": "spline_ridge",
        "hist_gradient_boosting": "hist_gradient_boosting",
        "hgb": "hist_gradient_boosting",
    }
    if method_key == "auto":
        selected_name = best_flexible if flexible_required else "linear"
    elif method_key in explicit_methods:
        selected_name = explicit_methods[method_key]
    else:
        raise ValueError(
            "Unsupported additive-shift estimator. Use auto, linear, "
            "spline_ridge, or hist_gradient_boosting."
        )

    warnings_out: list[str] = []
    inconclusive_reasons: list[str] = []
    if global_flexible_gain:
        warnings_out.append(
            "Within this stratum, flexible outcome models materially outperform "
            "the linear model on held-out data."
        )
    elif treatment_shape_triggered:
        warnings_out.append(
            "Within this stratum, the treatment-targeted shape gate detects an "
            "estimand-relevant departure from a linear shift."
        )
    if method_key != "auto" and selected_name == "linear" and flexible_required:
        warnings_out.append(
            "A linear subgroup estimator was explicitly requested despite "
            "stratum-specific nonlinearity."
        )
        inconclusive_reasons.append(
            "explicit_linear_subgroup_estimator_is_misspecified"
        )
    if not flexible_agreement:
        warnings_out.append(
            "Within this stratum, spline, gradient-boosting, and targeted-shape "
            "shift estimates disagree materially."
        )
        inconclusive_reasons.append("subgroup_flexible_estimators_disagree")
    if selected_name == "linear" and linear_flexible_disagreement:
        warnings_out.append(
            "The selected linear subgroup effect disagrees materially with at "
            "least one flexible shift estimate."
        )
        inconclusive_reasons.append(
            "subgroup_linear_flexible_effect_disagreement"
        )

    candidate_output: list[dict[str, Any]] = []
    for name in ["linear", *flexible_names]:
        role = "cross_check"
        if name == "linear" and (
            flexible_required or linear_flexible_disagreement
        ):
            role = "sensitivity_only"
        elif name == selected_name:
            role = "primary"
        candidate_output.append(
            {"name": name, **candidate_metrics[name], "role": role}
        )

    return {
        "model": candidates[selected_name],
        "selected_name": selected_name,
        "metrics": candidate_metrics[selected_name],
        "estimate_role": (
            "sensitivity_only"
            if selected_name == "linear"
            and (flexible_required or linear_flexible_disagreement)
            else "primary"
        ),
        "functional_form": (
            "nonlinear_required"
            if flexible_required
            else (
                "inconclusive"
                if linear_flexible_disagreement or not flexible_agreement
                else "approximately_linear"
            )
        ),
        "model_comparison": {
            "selected": selected_name,
            "recommended": (
                best_flexible
                if flexible_required
                else (
                    "none_inconclusive"
                    if linear_flexible_disagreement or not flexible_agreement
                    else "linear"
                )
            ),
            "selection_basis": f"{n_splits}-fold within-stratum held-out RMSE",
            "minimum_flexible_relative_rmse_gain": _MIN_FLEXIBLE_RMSE_GAIN,
            "relative_rmse_gain": float(relative_rmse_gain),
            "global_flexible_gain_triggered": global_flexible_gain,
            "flexible_effect_agreement": (
                "agree" if flexible_agreement else "disagree"
            ),
            "linear_flexible_effect_agreement": (
                "disagree" if linear_flexible_disagreement else "agree"
            ),
            "treatment_shape_gate": {
                "model": "treatment_spline_ridge",
                **targeted_shape_metrics,
                "minimum_relative_rmse_gain": _MIN_TREATMENT_SHAPE_RMSE_GAIN,
                "relative_rmse_gain_vs_linear": float(targeted_shape_gain),
                "linear_effect_agreement": (
                    "agree" if linear_targeted_agreement else "disagree"
                ),
                "triggered": treatment_shape_triggered,
            },
            "candidates": candidate_output,
        },
        "warnings": warnings_out,
        "inconclusive_reasons": list(dict.fromkeys(inconclusive_reasons)),
    }


def _estimate_stratified_shift_cate(
    *,
    complete: pd.DataFrame,
    modifier: str,
    x: np.ndarray,
    y: np.ndarray,
    population_ate: float,
    estimator_family: str,
    estimate_role: str,
    method_name: str,
    shift_delta: float,
    bootstrap_samples: int,
    n_splits: int,
    random_state: int,
    population_validity: dict[str, Any],
    modifier_graph_status: str,
    identification: dict[str, Any],
    modifier_warnings: list[str],
) -> dict[str, Any]:
    """Estimate a transparent one-modifier stratified additive-shift CATE."""
    strata, stratification = _build_cate_strata(complete[modifier])
    minimum_rows = max(30, n_splits * 5)
    cate_bootstrap_samples = min(
        bootstrap_samples, _MAX_CATE_BOOTSTRAP_SAMPLES
    )
    groups: list[dict[str, Any]] = []

    for index, (definition, mask) in enumerate(strata):
        n_rows = int(np.sum(mask))
        group: dict[str, Any] = {
            **definition,
            "n_rows": n_rows,
            "population_fraction": float(n_rows / len(complete)),
            "population_estimator_family": estimator_family,
        }
        validity_reasons = list(population_validity.get("reasons", []))
        group_warnings = list(modifier_warnings)
        if n_rows < minimum_rows:
            group.update(
                {
                    "effect": None,
                    "support_diagnostics": {"status": "not_assessed"},
                    "uncertainty": {
                        "method": "not_run_insufficient_subgroup_rows",
                        "bootstrap_samples_requested": cate_bootstrap_samples,
                        "bootstrap_samples_completed": 0,
                    },
                    "validity": {
                        "status": "inconclusive",
                        "decision_grade": False,
                        "reasons": [
                            *validity_reasons,
                            "insufficient_subgroup_rows",
                        ],
                    },
                    "warnings": [
                        *group_warnings,
                        f"This stratum has {n_rows} rows; at least {minimum_rows} "
                        "are required for cross-fitted subgroup estimation.",
                    ],
                }
            )
            groups.append(group)
            continue

        group_splits = min(n_splits, max(2, n_rows // 5))
        stratum_gate = _select_stratum_shift_model(
            x=x[mask],
            y=y[mask],
            shift_delta=shift_delta,
            n_splits=group_splits,
            random_state=random_state + 1_000 + index,
            method_name=method_name,
        )
        metrics = stratum_gate["metrics"]
        support = _support_diagnostics(x[mask], shift_delta)
        uncertainty, uncertainty_warnings = _bootstrap_shift_ci(
            stratum_gate["model"],
            x[mask],
            y[mask],
            shift_delta,
            cate_bootstrap_samples,
            group_splits,
            random_state + 20_000 + index,
        )
        uncertainty["scope"] = (
            "conditional subgroup effect; stratum-selected model family and "
            "stratum boundaries fixed"
        )
        if cate_bootstrap_samples > 0:
            uncertainty["interval_role"] = "exploratory_low_precision"
            uncertainty["interval_label"] = (
                "exploratory 95% percentile bootstrap interval"
            )
            uncertainty["decision_grade_evidence"] = False
        group_warnings.extend(stratum_gate["warnings"])
        group_warnings.extend(uncertainty_warnings)
        validity_reasons.extend(stratum_gate["inconclusive_reasons"])

        population_status = str(population_validity.get("status", "unknown"))
        inconclusive = population_status == "inconclusive"
        caution = population_status in {"caution", "sensitivity_only", "unknown"}
        if stratum_gate["inconclusive_reasons"]:
            inconclusive = True
        if estimate_role == "sensitivity_only":
            inconclusive = True
            validity_reasons.append("population_estimator_is_sensitivity_only")
        if support["status"] == "poor":
            inconclusive = True
            validity_reasons.append("poor_subgroup_shift_support")
        elif support["status"] == "caution":
            caution = True
            validity_reasons.append("limited_subgroup_shift_support")
        if identification.get("status") != "validated_in_supplied_graph":
            inconclusive = True
            validity_reasons.append(
                "cate_conditional_exchangeability_not_graph_validated"
            )
            validity_reasons.extend(
                str(reason) for reason in identification.get("reasons", [])
            )
        if cate_bootstrap_samples == 0:
            caution = True
            validity_reasons.append("subgroup_uncertainty_not_requested")
        elif uncertainty.get("bootstrap_samples_completed", 0) < 10:
            inconclusive = True
            validity_reasons.append("insufficient_subgroup_bootstrap_refits")
        elif uncertainty.get("bootstrap_samples_completed", 0) < 100:
            caution = True
            validity_reasons.append(
                "low_precision_exploratory_subgroup_interval"
            )
            group_warnings.append(
                "The subgroup interval uses fewer than 100 completed bootstrap "
                "refits and is exploratory/low-precision; it is not sufficient "
                "by itself for a decision-grade CATE."
            )

        status = "inconclusive" if inconclusive else ("caution" if caution else "valid")
        group.update(
            {
                "effect": metrics["effect"],
                "oof_r2": metrics["oof_r2"],
                "oof_rmse": metrics["oof_rmse"],
                "estimator_family": stratum_gate["selected_name"],
                "estimate_role": stratum_gate["estimate_role"],
                "functional_form": stratum_gate["functional_form"],
                "model_comparison": stratum_gate["model_comparison"],
                "support_diagnostics": support,
                "uncertainty": uncertainty,
                "validity": {
                    "status": status,
                    "decision_grade": status == "valid",
                    "reasons": list(dict.fromkeys(validity_reasons)),
                },
                "warnings": list(dict.fromkeys(group_warnings)),
            }
        )
        if "ci_lower" in uncertainty:
            group["ci_lower"] = uncertainty["ci_lower"]
            group["ci_upper"] = uncertainty["ci_upper"]
        groups.append(group)

    finite_effects = [
        float(group["effect"])
        for group in groups
        if group.get("effect") is not None
    ]
    complete_group_effects = len(finite_effects) == len(groups)
    weighted_cate: float | None = None
    if complete_group_effects:
        weighted_cate = float(
            sum(
                float(group["effect"]) * float(group["population_fraction"])
                for group in groups
            )
        )
    outcome_sd = float(np.std(y, ddof=1))
    absolute_floor = _CATE_ATE_OUTCOME_SD_FLOOR * outcome_sd
    relative_tolerance = _MAX_CATE_ATE_RELATIVE_DISCREPANCY * max(
        abs(population_ate), abs(weighted_cate or 0.0), 1e-8
    )
    effective_tolerance = max(absolute_floor, relative_tolerance, 1e-8)
    discrepancy = (
        float(population_ate - weighted_cate)
        if weighted_cate is not None
        else None
    )
    relative_discrepancy = (
        abs(discrepancy)
        / max(abs(population_ate), abs(weighted_cate or 0.0), 1e-8)
        if discrepancy is not None
        else None
    )
    reconciled = discrepancy is not None and abs(discrepancy) <= effective_tolerance
    reconciliation_status = (
        "reconciled"
        if reconciled
        else ("failed" if discrepancy is not None else "not_computable")
    )
    reconciliation_warning: str | None = None
    if reconciliation_status != "reconciled":
        if discrepancy is None:
            reconciliation_warning = (
                "Population ATE could not be reconciled with subgroup CATEs because "
                "at least one stratum lacks an estimable effect."
            )
            reconciliation_reason = "cate_ate_reconciliation_not_computable"
        else:
            reconciliation_warning = (
                f"The population ATE ({population_ate:.6g}) differs from the "
                f"stratum-size-weighted CATE ({weighted_cate:.6g}) by "
                f"{discrepancy:.6g}, exceeding the tolerance "
                f"{effective_tolerance:.6g}. The stratified CATE is internally "
                "unstable and is not decision-grade."
            )
            reconciliation_reason = "cate_ate_reconciliation_failed"
        for group in groups:
            validity = group.setdefault("validity", {})
            reasons = list(validity.get("reasons", []))
            validity.update(
                {
                    "status": "inconclusive",
                    "decision_grade": False,
                    "reasons": list(dict.fromkeys([*reasons, reconciliation_reason])),
                }
            )
            group["warnings"] = list(
                dict.fromkeys([*group.get("warnings", []), reconciliation_warning])
            )

    group_statuses = {
        str(group.get("validity", {}).get("status", "unknown"))
        for group in groups
    }
    cate_status = (
        "inconclusive"
        if "inconclusive" in group_statuses or reconciliation_status != "reconciled"
        else ("caution" if group_statuses - {"valid"} else "valid")
    )
    cate_reasons = list(
        dict.fromkeys(
            str(reason)
            for group in groups
            for reason in group.get("validity", {}).get("reasons", [])
        )
    )
    cate_warnings = [
        *modifier_warnings,
        "Only one pre-treatment modifier is supported per call to avoid "
        "sparse, data-mined joint strata.",
        "Each stratum runs its own linear/flexible functional-form and effect-"
        "agreement gate; subgroup models are refit separately and their weighted "
        "estimate is checked against the population ATE before use.",
    ]
    if cate_bootstrap_samples > 0:
        cate_warnings.append(
            "Each subgroup bootstrap interval is capped at 30 refits and is "
            "exploratory/low-precision, not standalone decision-grade evidence."
        )
    if reconciliation_warning:
        cate_warnings.append(reconciliation_warning)
    return {
        "estimand": "conditional_total_additive_shift",
        "definition": (
            "Within each baseline stratum G, estimate "
            "E[m_G(T + delta, Z) - m_G(T, Z) | G] after selecting and checking "
            "the functional form independently inside G."
        ),
        "modifier": modifier,
        "modifier_role": modifier_graph_status,
        "identification": identification,
        "stratification": stratification,
        "target_population": "complete observed rows within baseline modifier strata",
        "n_rows": len(complete),
        "estimator_family": "stratum_specific_model_selection",
        "population_estimator_family": estimator_family,
        "population_estimate_role": estimate_role,
        "groups": groups,
        "weighted_cate": weighted_cate,
        "ate_minus_weighted_cate": discrepancy,
        "reconciliation": {
            "status": reconciliation_status,
            "population_ate": population_ate,
            "weighted_cate": weighted_cate,
            "ate_minus_weighted_cate": discrepancy,
            "absolute_discrepancy": (
                abs(discrepancy) if discrepancy is not None else None
            ),
            "relative_discrepancy": relative_discrepancy,
            "maximum_relative_discrepancy": (
                _MAX_CATE_ATE_RELATIVE_DISCREPANCY
            ),
            "outcome_sd_absolute_floor": absolute_floor,
            "effective_absolute_tolerance": effective_tolerance,
            "law_of_total_expectation_target": (
                "population ATE and stratum-size-weighted CATE over the same "
                "complete-row target population"
            ),
        },
        "validity": {
            "status": cate_status,
            "decision_grade": cate_status == "valid",
            "reasons": cate_reasons,
        },
        "heterogeneity_summary": {
            "max_minus_min_effect": (
                max(finite_effects) - min(finite_effects)
                if len(finite_effects) >= 2
                else None
            ),
            "formal_heterogeneity_test": "not_performed",
            "interpretation": (
                "Differences between subgroup point estimates are descriptive; "
                "do not claim statistically established heterogeneity without a "
                "dedicated contrast or simultaneous interval."
            ),
        },
        "uncertainty": {
            "method": (
                "exploratory_per_stratum_nonparametric_group_cross_fitted_bootstrap"
            ),
            "interval_role": (
                "exploratory_low_precision"
                if cate_bootstrap_samples > 0
                else "not_requested"
            ),
            "interval_label": (
                "exploratory 95% percentile bootstrap interval"
                if cate_bootstrap_samples > 0
                else "not available"
            ),
            "decision_grade_evidence": False,
            "bootstrap_samples_requested_per_stratum": cate_bootstrap_samples,
            "bootstrap_cap_per_stratum": _MAX_CATE_BOOTSTRAP_SAMPLES,
            "conditions": [
                "The adjustment set, graph, stratum-selected model family, and "
                "stratum boundaries are treated as fixed within each bootstrap.",
                "Intervals do not include graph-selection, modifier-selection, "
                "model-selection, or multiple-comparison uncertainty.",
                "At most 30 refits are used per stratum for runtime control; the "
                "reported 95% percentile interval is rough/exploratory and cannot "
                "alone make a subgroup estimate decision-grade.",
            ],
        },
        "warnings": cate_warnings,
    }


def _calculate_additive_shift(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    common_causes: list[str] | None,
    graph_edges: list[dict] | None,
    effect_modifiers: list[str] | None,
    method_name: str,
    shift_delta: float,
    bootstrap_samples: int,
    random_state: int,
    n_splits: int,
) -> dict[str, Any]:
    """Identify with DoWhy, then estimate E[m(T+d,Z)-m(T,Z)]."""
    modifiers = [value.strip() for value in (effect_modifiers or []) if value.strip()]
    if len(modifiers) > 1:
        raise ValueError(
            "MULTIPLE_EFFECT_MODIFIERS_NOT_IMPLEMENTED_FOR_ADDITIVE_SHIFT: "
            "request one baseline modifier per call so every reported stratum "
            "has an explicit population and support diagnostic."
        )
    modifier = modifiers[0] if modifiers else None
    if modifier in {treatment, outcome}:
        raise ValueError(
            "INVALID_EFFECT_MODIFIER: the modifier must be a baseline covariate, "
            "not the treatment or outcome itself."
        )
    modifier_graph_status = "not_requested"
    modifier_warnings: list[str] = []
    cate_identification: dict[str, Any] = {"status": "not_requested"}
    if modifier:
        modifier_graph_status, modifier_warnings = _effect_modifier_graph_status(
            modifier, treatment, graph_edges
        )

    adjustment_set, graph_warnings = _identify_shift_adjustment_set(
        data, treatment, outcome, common_causes, graph_edges
    )
    if modifier:
        if (
            graph_edges
            and modifier_graph_status
            == "validated_non_descendant_in_supplied_graph"
        ):
            cate_identification = _validate_cate_conditional_exchangeability(
                graph_edges,
                treatment,
                outcome,
                adjustment_set,
                modifier,
            )
        else:
            missing_graph_reason = (
                "effect modifier is not represented in the supplied graph"
                if graph_edges
                else "no DAG was supplied"
            )
            cate_identification = {
                "status": "not_graph_validated",
                "criterion": (
                    "conditional exchangeability not testable because "
                    + missing_graph_reason
                ),
                "conditioning_set": list(
                    dict.fromkeys([*adjustment_set, modifier])
                ),
                "decision_grade": False,
                "reasons": [
                    "effect_modifier_graph_coverage_missing"
                    if graph_edges
                    else "causal_graph_not_supplied"
                ],
            }
    feature_names = [treatment, *adjustment_set]
    if len(feature_names) != len(set(feature_names)):
        duplicates = sorted(
            {name for name in feature_names if feature_names.count(name) > 1}
        )
        raise ValueError(
            f"Treatment/adjustment features must be unique; duplicates: {duplicates}."
        )

    required = list(dict.fromkeys([*feature_names, outcome, *modifiers]))
    missing_columns = [name for name in required if name not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    non_numeric = [
        name
        for name in [*feature_names, outcome]
        if not pd.api.types.is_numeric_dtype(data[name])
    ]
    if non_numeric:
        raise ValueError(
            "ADDITIVE_SHIFT_REQUIRES_NUMERIC_COLUMNS: treatment, outcome, and "
            f"adjustment variables must be numeric; got {non_numeric}."
        )

    complete = data[required].replace([np.inf, -np.inf], np.nan).dropna()
    minimum_rows = max(30, n_splits * 5)
    if len(complete) < minimum_rows:
        raise ValueError(
            "INSUFFICIENT_COMPLETE_ROWS: additive-shift estimation requires at "
            f"least {minimum_rows} complete rows; got {len(complete)}."
        )
    x = complete[feature_names].to_numpy(dtype=float)
    y = complete[outcome].to_numpy(dtype=float)

    candidates = _shift_model_candidates(random_state)
    candidate_metrics: dict[str, dict[str, float]] = {}
    for name, candidate_model in candidates.items():
        metrics, _ = _cross_fitted_shift(
            candidate_model, x, y, shift_delta, n_splits, random_state
        )
        candidate_metrics[name] = metrics
    targeted_shape_metrics, _ = _cross_fitted_shift(
        _treatment_shape_model(x.shape[1]),
        x,
        y,
        shift_delta,
        n_splits,
        random_state,
    )

    linear_rmse = candidate_metrics["linear"]["oof_rmse"]
    flexible_names = ["spline_ridge", "hist_gradient_boosting"]
    best_flexible = min(
        flexible_names, key=lambda name: candidate_metrics[name]["oof_rmse"]
    )
    best_flexible_rmse = candidate_metrics[best_flexible]["oof_rmse"]
    relative_rmse_gain = (
        (linear_rmse - best_flexible_rmse) / linear_rmse
        if linear_rmse > np.finfo(float).eps
        else 0.0
    )
    targeted_shape_gain = (
        (linear_rmse - targeted_shape_metrics["oof_rmse"]) / linear_rmse
        if linear_rmse > np.finfo(float).eps
        else 0.0
    )
    outcome_sd = float(np.std(y, ddof=1))
    spline_hgb_agreement = _effects_agree(
        candidate_metrics["spline_ridge"]["effect"],
        candidate_metrics["hist_gradient_boosting"]["effect"],
        outcome_sd,
    )
    targeted_spline_agreement = _effects_agree(
        targeted_shape_metrics["effect"],
        candidate_metrics["spline_ridge"]["effect"],
        outcome_sd,
    )
    targeted_hgb_agreement = _effects_agree(
        targeted_shape_metrics["effect"],
        candidate_metrics["hist_gradient_boosting"]["effect"],
        outcome_sd,
    )
    flexible_agreement = (
        spline_hgb_agreement
        and targeted_spline_agreement
        and targeted_hgb_agreement
    )
    linear_spline_agreement = _effects_agree(
        candidate_metrics["linear"]["effect"],
        candidate_metrics["spline_ridge"]["effect"],
        outcome_sd,
    )
    linear_hgb_agreement = _effects_agree(
        candidate_metrics["linear"]["effect"],
        candidate_metrics["hist_gradient_boosting"]["effect"],
        outcome_sd,
    )
    linear_targeted_agreement = _effects_agree(
        candidate_metrics["linear"]["effect"],
        targeted_shape_metrics["effect"],
        outcome_sd,
    )
    flexible_consensus_against_linear = (
        flexible_agreement
        and not linear_spline_agreement
        and not linear_hgb_agreement
    )
    treatment_shape_triggered = (
        targeted_shape_gain >= _MIN_TREATMENT_SHAPE_RMSE_GAIN
        or (
            not linear_targeted_agreement
            and flexible_consensus_against_linear
        )
    )
    global_flexible_gain = relative_rmse_gain >= _MIN_FLEXIBLE_RMSE_GAIN
    flexible_required = global_flexible_gain or treatment_shape_triggered
    linear_flexible_disagreement = not (
        linear_spline_agreement and linear_hgb_agreement
    )

    method_key = (method_name or "auto").lower()
    explicit_methods = {
        "linear": "linear",
        "backdoor.linear_regression": "linear",
        "spline_ridge": "spline_ridge",
        "hist_gradient_boosting": "hist_gradient_boosting",
        "hgb": "hist_gradient_boosting",
    }
    if method_key == "auto":
        selected_name = best_flexible if flexible_required else "linear"
    elif method_key in explicit_methods:
        selected_name = explicit_methods[method_key]
    else:
        raise ValueError(
            "Unsupported additive-shift estimator. Use auto, linear, "
            "spline_ridge, or hist_gradient_boosting."
        )

    support = _support_diagnostics(x, shift_delta)

    candidate_output: list[dict[str, Any]] = []
    for name in ["linear", *flexible_names]:
        role = "cross_check"
        if name == "linear" and (
            flexible_required or linear_flexible_disagreement
        ):
            role = "sensitivity_only"
        elif name == selected_name:
            role = "primary"
        candidate_output.append({"name": name, **candidate_metrics[name], "role": role})

    warnings_out = list(support["warnings"])
    validity_reasons: list[str] = []
    inconclusive_reasons: list[str] = []
    if global_flexible_gain:
        warnings_out.append(
            "Flexible outcome models materially outperform the linear model on "
            "held-out data; the linear shift estimate is sensitivity-only."
        )
    elif treatment_shape_triggered:
        warnings_out.append(
            "The treatment-targeted shape gate detects an estimand-relevant "
            "departure from the linear shift even though global RMSE gain is "
            "small; the linear shift estimate is sensitivity-only."
        )
    if method_key != "auto" and selected_name == "linear" and flexible_required:
        warnings_out.append(
            "A linear estimator was explicitly requested despite estimand-relevant "
            "nonlinearity; do not use it as the decision-grade effect."
        )
        validity_reasons.append("explicit_linear_estimator_is_misspecified")
        inconclusive_reasons.append("explicit_linear_estimator_is_misspecified")
    if not flexible_agreement:
        warnings_out.append(
            "Spline and gradient-boosting shift estimates disagree materially; "
            "no selected effect is decision-grade regardless of predictive RMSE."
        )
        validity_reasons.append("flexible_estimators_disagree")
        inconclusive_reasons.append("flexible_estimators_disagree")
    if selected_name == "linear" and linear_flexible_disagreement:
        warnings_out.append(
            "The selected linear effect disagrees materially with at least one "
            "flexible shift estimate; it is sensitivity-only and not decision-grade."
        )
        validity_reasons.append("linear_flexible_effect_disagreement")
        inconclusive_reasons.append("linear_flexible_effect_disagreement")
    if support["status"] == "poor":
        validity_reasons.append("poor_shift_support")
        inconclusive_reasons.append("poor_shift_support")
    elif support["status"] == "caution":
        validity_reasons.append("limited_shift_support")
    if graph_warnings:
        validity_reasons.append("graph_conversion_warnings")

    if inconclusive_reasons:
        validity_status = "inconclusive"
    elif support["status"] == "caution" or graph_warnings:
        validity_status = "caution"
    else:
        validity_status = "valid"

    uncertainty, uncertainty_warnings = _bootstrap_shift_ci(
        candidates[selected_name],
        x,
        y,
        shift_delta,
        bootstrap_samples,
        n_splits,
        random_state,
    )
    warnings_out.extend(uncertainty_warnings)

    estimator_labels = {
        "linear": "CrossFittedLinearGComputation",
        "spline_ridge": "CrossFittedSplineRidgeGComputation",
        "hist_gradient_boosting": "CrossFittedHistGradientBoostingGComputation",
    }
    result: dict[str, Any] = {
        "treatment": treatment,
        "outcome": outcome,
        "ate": candidate_metrics[selected_name]["effect"],
        "estimand": "total",
        "intervention": {
            "type": "additive_shift",
            "shift_delta": shift_delta,
            "target_population": "complete observed rows",
            "n_rows": len(complete),
            "n_rows_dropped": len(data) - len(complete),
        },
        "estimator": estimator_labels[selected_name],
        "estimator_family": selected_name,
        "estimate_role": (
            "sensitivity_only"
            if selected_name == "linear"
            and (flexible_required or linear_flexible_disagreement)
            else "primary"
        ),
        "auto_adjustment_set": adjustment_set,
        "functional_form": (
            "nonlinear_required"
            if flexible_required
            else (
                "inconclusive"
                if (linear_flexible_disagreement or not flexible_agreement)
                else "approximately_linear"
            )
        ),
        "model_comparison": {
            "selected": selected_name,
            "recommended": (
                best_flexible
                if flexible_required
                else (
                    "none_inconclusive"
                    if (linear_flexible_disagreement or not flexible_agreement)
                    else "linear"
                )
            ),
            "selection_basis": f"{n_splits}-fold held-out RMSE",
            "minimum_flexible_relative_rmse_gain": _MIN_FLEXIBLE_RMSE_GAIN,
            "relative_rmse_gain": float(relative_rmse_gain),
            "global_flexible_gain_triggered": global_flexible_gain,
            "flexible_effect_agreement": (
                "agree" if flexible_agreement else "disagree"
            ),
            "flexible_pairwise_agreement": {
                "spline_vs_hgb": spline_hgb_agreement,
                "targeted_spline_vs_spline": targeted_spline_agreement,
                "targeted_spline_vs_hgb": targeted_hgb_agreement,
            },
            "linear_flexible_effect_agreement": (
                "disagree" if linear_flexible_disagreement else "agree"
            ),
            "treatment_shape_gate": {
                "model": "treatment_spline_ridge",
                **targeted_shape_metrics,
                "minimum_relative_rmse_gain": _MIN_TREATMENT_SHAPE_RMSE_GAIN,
                "relative_rmse_gain_vs_linear": float(targeted_shape_gain),
                "linear_effect_agreement": (
                    "agree" if linear_targeted_agreement else "disagree"
                ),
                "triggered": treatment_shape_triggered,
            },
            "candidates": candidate_output,
        },
        "support_diagnostics": support,
        "validity": {
            "status": validity_status,
            "decision_grade": validity_status != "inconclusive",
            "reasons": validity_reasons,
        },
        "warnings": warnings_out,
        "uncertainty": uncertainty,
        "refutation_results": {
            "dowhy_refuters": {
                "status": "not_applicable",
                "passed": None,
                "reason": (
                    "DoWhy refuters were not run because they would recreate the "
                    "legacy fixed-contrast estimator, not this additive-shift "
                    "g-computation estimator. Model comparison, support, and "
                    "bootstrap diagnostics are reported instead."
                ),
            }
        },
    }
    if graph_warnings:
        result["graph_warnings"] = graph_warnings
    if "ci_lower" in uncertainty:
        result["ci_lower"] = uncertainty["ci_lower"]
        result["ci_upper"] = uncertainty["ci_upper"]
    if modifier:
        result["cate"] = _estimate_stratified_shift_cate(
            complete=complete,
            modifier=modifier,
            x=x,
            y=y,
            population_ate=float(result["ate"]),
            estimator_family=selected_name,
            estimate_role=result["estimate_role"],
            method_name=method_name,
            shift_delta=shift_delta,
            bootstrap_samples=bootstrap_samples,
            n_splits=n_splits,
            random_state=random_state,
            population_validity=result["validity"],
            modifier_graph_status=modifier_graph_status,
            identification=cate_identification,
            modifier_warnings=modifier_warnings,
        )
    return result


def calculate_ate(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    common_causes: list[str] | None = None,
    effect_modifiers: list[str] | None = None,
    method_name: str = "auto",
    graph_edges: list[dict] | None = None,
    treatment_value: Any | None = None,
    control_value: Any | None = None,
    estimand: str = "total",
    intervention_type: str = "auto",
    shift_delta: float | None = None,
    bootstrap_samples: int = 100,
    random_state: int = _DEFAULT_RANDOM_STATE,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Estimate a total effect under an explicit intervention.

    DoWhy performs identification in every path. Fixed treatment/control
    contrasts retain the legacy DoWhy estimator. Continuous additive shifts
    use cross-fitted g-computation for E[m(T + delta, Z) - m(T, Z)].
    """
    missing = [name for name in (treatment, outcome) if name not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    estimand_key = (estimand or "total").lower()
    if estimand_key != "total":
        raise ValueError(
            f"ESTIMAND_NOT_IMPLEMENTED: '{estimand_key}' is not implemented. "
            "causal_infer currently supports total effects only; it will not "
            "silently substitute a total effect for a direct or indirect effect."
        )

    treatment_column = data[treatment]
    is_numeric = pd.api.types.is_numeric_dtype(treatment_column)
    is_continuous = is_numeric and treatment_column.dropna().nunique() > 2
    intervention_key = (intervention_type or "auto").lower()
    if intervention_key not in {"auto", "contrast", "additive_shift"}:
        raise ValueError(
            "intervention_type must be auto, contrast, or additive_shift."
        )

    if intervention_key == "auto":
        if shift_delta is not None:
            intervention_key = "additive_shift"
        elif is_continuous:
            if treatment_value is not None and control_value is not None:
                intervention_key = "contrast"
            else:
                raise ValueError(
                    "CONTINUOUS_INTERVENTION_REQUIRED: continuous treatment "
                    f"'{treatment}' needs an explicit intervention. Use "
                    "intervention_type='additive_shift' with shift_delta=<delta>, "
                    "or intervention_type='contrast' with treatment_value and "
                    "control_value. A continuous treatment is never silently "
                    "interpreted as a 0-to-1 contrast."
                )
        else:
            intervention_key = "contrast"

    if intervention_key == "additive_shift":
        if not is_continuous:
            raise ValueError(
                "ADDITIVE_SHIFT_REQUIRES_CONTINUOUS_TREATMENT: treatment must be "
                "numeric with more than two observed values."
            )
        if shift_delta is None or not np.isfinite(shift_delta) or shift_delta == 0:
            raise ValueError("shift_delta must be a finite, non-zero number.")
        if treatment_value is not None or control_value is not None:
            raise ValueError(
                "treatment_value/control_value cannot be combined with an additive shift."
            )
        return _calculate_additive_shift(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=common_causes,
            graph_edges=graph_edges,
            effect_modifiers=effect_modifiers,
            method_name=method_name,
            shift_delta=float(shift_delta),
            bootstrap_samples=bootstrap_samples,
            random_state=random_state,
            n_splits=n_splits,
        )

    if is_numeric:
        for label, value in (
            ("treatment_value", treatment_value),
            ("control_value", control_value),
        ):
            if value is None or not isinstance(value, str):
                continue
            try:
                parsed = float(value)
            except ValueError as exc:
                raise ValueError(
                    f"{label}='{value}' is not numeric for numeric treatment "
                    f"'{treatment}'."
                ) from exc
            if not np.isfinite(parsed):
                raise ValueError(f"{label} must be finite; got '{value}'.")
            if label == "treatment_value":
                treatment_value = parsed
            else:
                control_value = parsed

    observed_values = sorted(treatment_column.dropna().unique().tolist())
    if len(observed_values) < 2:
        raise ValueError(
            f"TREATMENT_VARIATION_REQUIRED: '{treatment}' has fewer than two "
            "observed non-missing levels."
        )
    if len(observed_values) == 2:
        observed_values = [
            value.item() if isinstance(value, np.generic) else value
            for value in observed_values
        ]
        if (treatment_value is None) != (control_value is None):
            raise ValueError(
                "BINARY_CONTRAST_VALUES_INCOMPLETE: specify both treatment_value "
                "and control_value, or omit both to compare the two observed levels."
            )
        if treatment_value is None and control_value is None:
            control_value, treatment_value = observed_values
        if treatment_value == control_value:
            raise ValueError(
                "BINARY_CONTRAST_VALUES_INVALID: treatment_value and "
                "control_value must be distinct."
            )
        if treatment_value not in observed_values or control_value not in observed_values:
            raise ValueError(
                "BINARY_CONTRAST_VALUES_INVALID: both values must be observed "
                f"levels of '{treatment}'. Available: {observed_values}."
            )

    if is_continuous and (treatment_value is None or control_value is None):
        raise ValueError(
            "CONTINUOUS_CONTRAST_VALUES_REQUIRED: specify both treatment_value "
            "and control_value for a continuous contrast."
        )
    contrast_method = (
        "backdoor.linear_regression"
        if (method_name or "auto").lower() == "auto"
        else method_name
    )
    result = _calculate_contrast_legacy(
        data=data,
        treatment=treatment,
        outcome=outcome,
        common_causes=common_causes,
        effect_modifiers=effect_modifiers,
        method_name=contrast_method,
        graph_edges=graph_edges,
        treatment_value=treatment_value,
        control_value=control_value,
    )
    analysis_n_rows = int(result.pop("_analysis_n_rows", len(data)))
    analysis_rows_dropped = int(result.pop("_analysis_rows_dropped", 0))
    result["estimand"] = "total"
    result["intervention"] = {
        "type": "contrast",
        "treatment_value": treatment_value,
        "control_value": control_value,
        "selected_levels": [control_value, treatment_value],
        "target_population": (
            "rows with the selected treatment levels"
            if analysis_rows_dropped > 0
            else "observed rows"
        ),
        "n_rows": analysis_n_rows,
        "n_rows_dropped": analysis_rows_dropped,
    }
    result["estimator_family"] = "dowhy_contrast"
    result["estimate_role"] = "primary"
    result["functional_form"] = "not_assessed_for_legacy_contrast"
    result["validity"] = {
        "status": "caution",
        "decision_grade": False,
        "reasons": ["legacy_contrast_functional_form_not_assessed"],
    }
    result["warnings"] = [
        "This legacy fixed contrast does not run the continuous-shift "
        "held-out functional-form comparison; interpret it under the stated "
        "DoWhy estimator assumptions."
    ]
    result["uncertainty"] = {
        "method": "dowhy_estimator",
        "bootstrap_samples_requested": 0,
        "bootstrap_samples_completed": 0,
        "conditions": [
            "The confidence interval and p-value, when present, come from the "
            "selected legacy DoWhy estimator."
        ],
    }
    return result


@safe_main
def main():
    parser = argparse.ArgumentParser(
        description="CausalClaw — ATE/CATE Inference Bridge"
    )
    parser.add_argument("--data", required=True, help="Path to data CSV")
    parser.add_argument("--treatment", required=True, help="Treatment variable name")
    parser.add_argument("--outcome", required=True, help="Outcome variable name")
    parser.add_argument(
        "--common-causes", help="Comma-separated list of common causes (confounders)"
    )
    parser.add_argument(
        "--effect-modifiers", help="Comma-separated list of effect modifiers (for CATE)"
    )
    parser.add_argument(
        "--method",
        default="auto",
        help=(
            "Estimator. 'auto' compares linear, spline+ridge, and HGB for an "
            "additive shift; binary/category contrasts retain DoWhy linear regression."
        ),
    )
    parser.add_argument(
        "--estimand",
        default="total",
        help="Causal estimand. Only total is currently implemented.",
    )
    parser.add_argument(
        "--intervention-type",
        default="auto",
        help="One of: auto, contrast, additive_shift.",
    )
    parser.add_argument(
        "--shift-delta",
        type=float,
        help="Additive intervention amount for continuous treatment (T -> T + delta).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=100,
        help="Bootstrap refits for additive-shift confidence intervals (0 disables).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=_DEFAULT_RANDOM_STATE,
        help="Fixed seed for folds, models, and bootstrap.",
    )
    parser.add_argument(
        "--graph-edges",
        help=(
            "JSON array of edge objects from the causal scratchpad "
            "({from, to, edgeType}). When supplied, DoWhy will automatically "
            "identify the backdoor adjustment set from the graph structure. "
            "Takes priority over --common-causes."
        ),
    )
    parser.add_argument(
        "--treatment-value",
        help=(
            "The value of the treatment variable representing the 'treated' group. "
            "Required when the treatment column has more than 2 string categories."
        ),
    )
    parser.add_argument(
        "--control-value",
        help=(
            "The value of the treatment variable representing the 'control' group. "
            "Required when the treatment column has more than 2 string categories."
        ),
    )

    args = parser.parse_args()
    data = load_data(args.data)

    graph_edges: list[dict] | None = None
    if args.graph_edges:
        try:
            graph_edges = json.loads(args.graph_edges)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--graph-edges is not valid JSON: {exc}") from exc

    common_causes = (
        [c.strip() for c in args.common_causes.split(",")]
        if args.common_causes and not graph_edges
        else None
    )
    effect_modifiers = (
        [m.strip() for m in args.effect_modifiers.split(",")]
        if args.effect_modifiers
        else None
    )

    result = calculate_ate(
        data=data,
        treatment=args.treatment,
        outcome=args.outcome,
        common_causes=common_causes,
        effect_modifiers=effect_modifiers,
        method_name=args.method,
        graph_edges=graph_edges,
        treatment_value=args.treatment_value,
        control_value=args.control_value,
        estimand=args.estimand,
        intervention_type=args.intervention_type,
        shift_delta=args.shift_delta,
        bootstrap_samples=args.bootstrap_samples,
        random_state=args.random_state,
    )

    output_result(result)


if __name__ == "__main__":
    main()
