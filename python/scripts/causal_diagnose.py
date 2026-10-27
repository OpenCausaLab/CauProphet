# type: ignore
"""CausalClaw Bridge — Data Diagnosis.

Profiles a dataset BEFORE causal discovery so the agent can pick the right
algorithm and parameters. Outputs JSON to stdout for the TypeScript handler.

Reports, per dataset and per column:
  - column data type (continuous / discrete / binary)
  - overall dataset type verdict (all-continuous / all-discrete / mixed)
  - Gaussianity of continuous columns (skew / kurtosis / normaltest)
  - pairwise relationship shape: nonlinear-detected / approximately-linear /
    inconclusive, using dependence statistics plus held-out spline-vs-linear fit
  - sample size, missing values, constant / ID-like columns
  - a concrete algorithm recommendation

Zero new dependencies — uses numpy / pandas / scipy / scikit-learn only.
"""

from __future__ import annotations

import argparse
import warnings
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

from utils import load_data, output_result, safe_main

# --- Heuristic thresholds (kept explicit so the report can cite them) ---------
DISCRETE_MAX_UNIQUE = 10       # ≤ this many distinct values ⇒ treat as discrete
STRONG_DEP = 0.30              # |spearman| or normalized-MI above this ⇒ dependent
WEAK_LINEAR = 0.20             # |pearson| below this while dependent ⇒ nonlinear
NONLINEAR_GAP = 0.25           # (rank/MI strength − |pearson|) above this ⇒ nonlinear
SPLINE_MIN_SAMPLES = 80        # below this, held-out shape comparison is too unstable
SPLINE_MAX_SAMPLES = 2000      # deterministic cap so diagnosis stays lightweight
SPLINE_CV_FOLDS = 5
SPLINE_KNOTS = 6
SPLINE_MIN_R2 = 0.10           # spline must explain non-trivial held-out variance
SPLINE_GAIN = 0.05             # minimum held-out R² gain over a straight line
MAX_PAIRS = 60                 # cap pairwise checks so wide tables stay fast
TOP_ASSOCIATIONS = 10          # how many strongest pairs to surface for EDA
NEAR_DETERMINISTIC_R = 0.95    # |pearson| ≥ this ⇒ near-deterministic pair warning
DEFAULT_DISCOVERY_TIMEOUT = 90
KERNEL_WARN_WORK_UNITS = 25_000_000
NONLINEAR_SUBSAMPLE_ROWS = 500


def _discovery_cost_profile(n_samples: int, n_vars: int) -> dict[str, Any]:
    """Return a deliberately rough, monotone discovery-cost profile.

    Kernel CI search cost depends on the learned skeleton and conditioning sets,
    so pretending to predict wall time is misleading.  The proxy below only
    estimates the *minimum screening surface*: pair count times N-squared kernel
    work.  It is used to choose a bounded default, never to claim an ETA.
    """
    n_pairs = max(n_vars * (n_vars - 1) // 2, 1)
    work_units = int(n_pairs * n_samples**2)
    if work_units >= KERNEL_WARN_WORK_UNITS or n_vars >= 10:
        tier = "high"
    elif work_units >= KERNEL_WARN_WORK_UNITS // 5 or n_vars >= 7:
        tier = "medium"
    else:
        tier = "low"
    return {
        "tier": tier,
        "kernel_screen_work_units": work_units,
        "default_timeout_seconds": DEFAULT_DISCOVERY_TIMEOUT,
        "recommended_initial_rows": (
            min(n_samples, NONLINEAR_SUBSAMPLE_ROWS) if tier == "high" else n_samples
        ),
        "wall_time_note": (
            "This is a relative cost tier, not an ETA; density and conditioning "
            "depth can dominate runtime."
        ),
    }

# --- Gaussianity: effect-size PRIMARY, p-value SECONDARY ----------------------
# Rationale: D'Agostino-Pearson p-values are hyper-sensitive at large N — clean
# Gaussian data at N=5000 with skew≈-0.09 still gets p<0.05. Skew/kurtosis are
# consistent estimators whose magnitudes do NOT drift to significance with N, so
# we gate on the *magnitude* of departure and use the p-value only as support.
GAUSSIAN_SKEW_NEAR = 0.25      # |skew| ≤ this ⇒ near-zero (Gaussian-ish)
GAUSSIAN_KURT_NEAR = 0.50      # |excess_kurtosis| ≤ this ⇒ near-zero
GAUSSIAN_SKEW_STRONG = 0.75    # |skew| ≥ this ⇒ strong departure
GAUSSIAN_KURT_STRONG = 1.00    # |excess_kurtosis| ≥ this ⇒ strong departure
GAUSSIAN_ALPHA_STRONG = 0.01   # p ≤ this REQUIRED to confirm a strong departure
# LiNGAM gate: only when non-Gaussianity is strong AND broad across columns.
LINGAM_MIN_STRONG_COLS = 2
LINGAM_MIN_STRONG_RATIO = 0.60


def _classify_column(col: pd.Series) -> dict[str, Any]:
    """Classify a single column's data type."""
    non_null = col.dropna()
    n_unique = int(non_null.nunique())
    n_non_null = int(len(non_null))
    is_numeric = pd.api.types.is_numeric_dtype(col)

    fractional = False
    if is_numeric and n_non_null > 0:
        arr = non_null.to_numpy()
        fractional = bool(np.any(np.floor(arr) != arr))

    # ID-like: every value distinct AND high cardinality AND integer-valued or
    # non-numeric (e.g. user_id, uuid). Continuous floats are NOT id-like even
    # though they are typically all-distinct.
    id_like = (
        n_unique == n_non_null
        and n_unique > DISCRETE_MAX_UNIQUE
        and (not is_numeric or not fractional)
    )

    if id_like:
        col_type = "id"
    elif n_unique <= 2:
        col_type = "binary"
    elif not is_numeric:
        col_type = "discrete"
    elif n_unique <= DISCRETE_MAX_UNIQUE and not fractional:
        col_type = "discrete"
    else:
        col_type = "continuous"

    return {
        "name": str(col.name),
        "type": col_type,
        "n_unique": n_unique,
        "is_numeric": is_numeric,
        "missing": int(col.isna().sum()),
        "constant": n_unique <= 1,
        "id_like": id_like,
    }


def _gaussianity(col: pd.Series) -> dict[str, Any]:
    """Graded Gaussianity verdict, effect-size primary and p-value secondary.

    Returns a per-column dict with skew, excess kurtosis, normaltest p-value,
    an effect-size band, a graded verdict, a short evidence string, and a
    backward-compatible tri-state `is_gaussian`.

    Note: this inspects the *marginal* distribution of an observed variable. It
    is only a SCREENING signal for LiNGAM's assumption (independent non-Gaussian
    structural errors) — it does NOT verify that assumption directly.
    """
    arr = col.dropna().to_numpy(dtype=np.float64)
    result: dict[str, Any] = {
        "n_observed": int(len(arr)),
        "skew": None,
        "excess_kurtosis": None,
        "normaltest_p": None,
        "effect_size_band": "insufficient",
        "verdict": "insufficient",
        "evidence": "fewer than 8 observations or zero variance",
        "is_gaussian": None,
    }
    if len(arr) < 8 or np.std(arr) == 0:
        return result

    sk = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr))  # Fisher (excess): 0 == Gaussian
    result["skew"] = round(sk, 4)
    result["excess_kurtosis"] = round(kurt, 4)

    p: float | None = None
    try:
        _, p_val = stats.normaltest(arr)
        p = float(p_val)
        result["normaltest_p"] = round(p, 4)
    except Exception:
        pass

    near = abs(sk) <= GAUSSIAN_SKEW_NEAR and abs(kurt) <= GAUSSIAN_KURT_NEAR
    strong = abs(sk) >= GAUSSIAN_SKEW_STRONG or abs(kurt) >= GAUSSIAN_KURT_STRONG

    if near:
        result["effect_size_band"] = "near-zero"
        result["verdict"] = "gaussian-ish"
        result["is_gaussian"] = True
        p_note = f", normaltest p={p:.4g}" if p is not None else ""
        result["evidence"] = (
            f"|skew|={abs(sk):.3f}≤{GAUSSIAN_SKEW_NEAR}, "
            f"|excess_kurt|={abs(kurt):.3f}≤{GAUSSIAN_KURT_NEAR} "
            f"⇒ Gaussian-ish (magnitude near zero{p_note}; "
            "a small normaltest p at large N does NOT override near-zero moments)"
        )
    elif strong and p is not None and p <= GAUSSIAN_ALPHA_STRONG:
        result["effect_size_band"] = "strong"
        result["verdict"] = "strongly-non-gaussian"
        result["is_gaussian"] = False
        result["evidence"] = (
            f"|skew|={abs(sk):.3f} / |excess_kurt|={abs(kurt):.3f} large "
            f"and normaltest p={p:.4g}≤{GAUSSIAN_ALPHA_STRONG} ⇒ strong departure"
        )
    elif strong:
        # Large moments but p not confirming (rare) — stay cautious.
        result["effect_size_band"] = "strong"
        result["verdict"] = "ambiguous"
        result["is_gaussian"] = None
        result["evidence"] = (
            f"large moments (|skew|={abs(sk):.3f}, |excess_kurt|={abs(kurt):.3f}) "
            "but normaltest did not confirm ⇒ ambiguous"
        )
    else:
        result["effect_size_band"] = "moderate"
        result["verdict"] = "ambiguous"
        result["is_gaussian"] = None
        result["evidence"] = (
            f"moderate departure (|skew|={abs(sk):.3f}, "
            f"|excess_kurt|={abs(kurt):.3f}) ⇒ neither clearly Gaussian nor strong"
        )
    return result


def _cross_validated_shape(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Compare linear and flexible univariate fits out of sample, both ways.

    Pearson-vs-rank/MI gaps catch symmetric or strongly curved relationships
    such as y=x² but miss monotone nonlinear mechanisms such as tanh and sin
    over a restricted range. A held-out spline gain catches that second family
    without declaring nonlinearity from an in-sample flexible fit.

    This remains a marginal screening diagnostic. It does not establish a
    structural edge or its direction.
    """
    result: dict[str, Any] = {
        "available": False,
        "cv_samples": 0,
        "linear_cv_r2": None,
        "spline_cv_r2": None,
        "spline_gain": None,
        "screening_direction": None,
    }
    if len(x) < SPLINE_MIN_SAMPLES:
        return result

    try:
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.model_selection import KFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import SplineTransformer

        if len(x) > SPLINE_MAX_SAMPLES:
            rng = np.random.default_rng(0)
            indices = np.sort(
                rng.choice(len(x), size=SPLINE_MAX_SAMPLES, replace=False)
            )
            x_eval = x[indices]
            y_eval = y[indices]
        else:
            x_eval = x
            y_eval = y

        cv = KFold(n_splits=SPLINE_CV_FOLDS, shuffle=True, random_state=0)

        def directional_scores(
            predictor: np.ndarray, target: np.ndarray, direction: str
        ) -> dict[str, Any]:
            predictor_2d = predictor.reshape(-1, 1)
            linear_r2 = float(
                np.mean(
                    cross_val_score(
                        LinearRegression(),
                        predictor_2d,
                        target,
                        cv=cv,
                        scoring="r2",
                    )
                )
            )
            spline_model = make_pipeline(
                SplineTransformer(
                    n_knots=SPLINE_KNOTS,
                    degree=3,
                    include_bias=False,
                ),
                Ridge(alpha=1e-3),
            )
            spline_r2 = float(
                np.mean(
                    cross_val_score(
                        spline_model,
                        predictor_2d,
                        target,
                        cv=cv,
                        scoring="r2",
                    )
                )
            )
            return {
                "linear_cv_r2": linear_r2,
                "spline_cv_r2": spline_r2,
                "spline_gain": spline_r2 - linear_r2,
                "screening_direction": direction,
            }

        candidates = [
            directional_scores(x_eval, y_eval, "x->y"),
            directional_scores(y_eval, x_eval, "y->x"),
        ]
        best = max(candidates, key=lambda item: item["spline_gain"])
        result.update(
            {
                "available": True,
                "cv_samples": int(len(x_eval)),
                "linear_cv_r2": round(float(best["linear_cv_r2"]), 4),
                "spline_cv_r2": round(float(best["spline_cv_r2"]), 4),
                "spline_gain": round(float(best["spline_gain"]), 4),
                "screening_direction": best["screening_direction"],
            }
        )
    except Exception:
        # Diagnosis remains usable if sklearn's optional machinery fails.
        # The correlation/rank/MI screen below still provides legacy evidence.
        pass
    return result


def _pair_relationship(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Screen a continuous pair for nonlinear marginal shape.

    Logic: Pearson captures *linear* strength; Spearman (rank) and mutual
    information capture *any monotone/general* dependence. That legacy signal
    is combined with a held-out spline-vs-linear comparison so strong-Pearson
    monotone nonlinearities are not silently called linear.
    """
    pearson = float(stats.pearsonr(x, y)[0])
    spearman = float(stats.spearmanr(x, y)[0])
    pearson_abs, spearman_abs = abs(pearson), abs(spearman)

    mi_norm = 0.0
    try:
        from sklearn.feature_selection import mutual_info_regression

        mi = float(mutual_info_regression(x.reshape(-1, 1), y, random_state=0)[0])
        # Normalize MI to ~[0,1] against the marginal entropy of a Gaussian proxy.
        # A monotone squashing keeps it comparable to correlation magnitudes.
        mi_norm = float(1.0 - np.exp(-2.0 * max(mi, 0.0)))
    except Exception:
        mi_norm = spearman_abs  # fall back to rank correlation

    dep_strength = max(spearman_abs, mi_norm)
    dependent = dep_strength >= STRONG_DEP
    legacy_nonlinear = dependent and (
        pearson_abs < WEAK_LINEAR
        or (dep_strength - pearson_abs) >= NONLINEAR_GAP
    )
    shape = (
        _cross_validated_shape(x, y)
        if dependent
        else {
            "available": False,
            "cv_samples": 0,
            "linear_cv_r2": None,
            "spline_cv_r2": None,
            "spline_gain": None,
            "screening_direction": None,
        }
    )
    spline_r2 = shape.get("spline_cv_r2")
    spline_gain = shape.get("spline_gain")
    spline_nonlinear = bool(
        dependent
        and shape["available"]
        and isinstance(spline_r2, (int, float))
        and isinstance(spline_gain, (int, float))
        and spline_r2 >= SPLINE_MIN_R2
        and spline_gain >= SPLINE_GAIN
    )
    nonlinear = bool(legacy_nonlinear or spline_nonlinear)

    if nonlinear:
        shape_verdict = "nonlinear-detected"
    elif dependent and shape["available"]:
        shape_verdict = "approximately-linear"
    else:
        shape_verdict = "inconclusive"

    evidence: list[str] = []
    if legacy_nonlinear:
        evidence.append("rank/MI dependence is much stronger than Pearson")
    if spline_nonlinear:
        evidence.append(
            f"held-out spline ΔR²={shape['spline_gain']:.3f} "
            f"(linear={shape['linear_cv_r2']:.3f}, "
            f"spline={shape['spline_cv_r2']:.3f})"
        )
    if not evidence and shape_verdict == "approximately-linear":
        evidence.append(
            f"no material held-out spline gain (ΔR²={shape['spline_gain']:.3f})"
        )
    if not evidence:
        evidence.append("dependence signal or sample size is insufficient")

    return {
        "pearson": round(pearson, 4),
        "spearman": round(spearman, 4),
        "pearson_abs": round(pearson_abs, 4),
        "spearman_abs": round(spearman_abs, 4),
        "mi_norm": round(mi_norm, 4),
        "dependent": dependent,
        "nonlinear": nonlinear,
        "shape_verdict": shape_verdict,
        "shape_evidence": evidence,
        **shape,
    }


def _association_anchor(
    data: pd.DataFrame, treatment: str, outcome: str
) -> dict[str, Any]:
    """Unadjusted marginal association between treatment and outcome.

    This is the agent's SIGN ANCHOR: the final causal estimate should be
    reported side-by-side with this number, and any sign/magnitude divergence
    must be explained (confounding / mediation / collider). It is an
    associational quantity — E[Y|X] — NOT a causal effect.
    """
    anchor: dict[str, Any] = {
        "treatment": treatment,
        "outcome": outcome,
        "estimand_note": (
            "Marginal association E[Y|X], NOT a causal effect E[Y|do(X)]. "
            "Report the adjusted causal estimate side-by-side with this anchor; "
            "a sign flip between them is the signature of confounding and must "
            "be explained, never silently picked."
        ),
    }
    missing = [c for c in (treatment, outcome) if c not in data.columns]
    if missing:
        anchor["error"] = f"column(s) not found: {missing}"
        return anchor

    sub = data[[treatment, outcome]].dropna()
    x = sub[treatment].to_numpy(dtype=np.float64)
    y = sub[outcome].to_numpy(dtype=np.float64)
    anchor["n_observed"] = int(len(sub))
    if len(sub) < 10 or np.std(x) == 0 or np.std(y) == 0:
        anchor["error"] = "fewer than 10 complete observations or zero variance"
        return anchor

    lr = stats.linregress(x, y)
    anchor["unadjusted_slope"] = round(float(lr.slope), 6)
    anchor["slope_ci95"] = [
        round(float(lr.slope - 1.96 * lr.stderr), 6),
        round(float(lr.slope + 1.96 * lr.stderr), 6),
    ]
    anchor["slope_sign"] = "positive" if lr.slope > 0 else ("negative" if lr.slope < 0 else "zero")
    anchor["pearson"] = round(float(stats.pearsonr(x, y)[0]), 4)
    anchor["spearman"] = round(float(stats.spearmanr(x, y)[0]), 4)
    return anchor


def _effect_estimation_profile(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    linearity_verdict: str,
) -> dict[str, Any]:
    """Describe the intervention contract required by downstream inference.

    Dataset-wide shape screening is useful for discovery routing, but it cannot
    decide whether the *adjusted* treatment-response surface is linear.  This
    profile therefore never chooses an effect model from marginal evidence.  It
    tells the agent which intervention values must be made explicit and leaves
    the final linear-vs-flexible decision to ``causal_infer(method=\"auto\")``
    after the graph has supplied an adjustment set.
    """
    missing = [name for name in (treatment, outcome) if name not in data.columns]
    if missing:
        return {"error": f"column(s) not found: {missing}"}

    treatment_series = data[treatment].dropna()
    outcome_series = data[outcome].dropna()
    treatment_numeric = pd.api.types.is_numeric_dtype(treatment_series)
    outcome_numeric = pd.api.types.is_numeric_dtype(outcome_series)
    treatment_unique = int(treatment_series.nunique())

    if treatment_numeric and treatment_unique > 2:
        treatment_type = "continuous"
        supported_interventions = ["additive_shift", "contrast"]
        requires_explicit_intervention = True
        intervention_guidance = (
            "For wording such as 'increase T by delta from its current level', "
            "call causal_infer with intervention_type=additive_shift and an "
            "explicit shift_delta. A fixed treatment_value/control_value contrast "
            "is a different estimand."
        )
    else:
        treatment_type = "binary" if treatment_unique == 2 else "categorical"
        supported_interventions = ["contrast"]
        requires_explicit_intervention = treatment_unique != 2
        intervention_guidance = (
            "Use an explicit treatment_value/control_value contrast when the "
            "treatment is not an unambiguous binary variable."
        )

    return {
        "treatment": treatment,
        "outcome": outcome,
        "treatment_type": treatment_type,
        "outcome_numeric": bool(outcome_numeric),
        "requires_explicit_intervention": requires_explicit_intervention,
        "supported_interventions": supported_interventions,
        "recommended_method": "auto",
        "post_adjustment_shape_check": "required",
        "linear_estimator_policy": (
            "sensitivity-only when a held-out flexible outcome model materially "
            "outperforms the adjusted linear model"
        ),
        "dataset_linearity_screen": linearity_verdict,
        "guidance": intervention_guidance,
    }


def _summarize_gaussianity(per_col: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-column verdicts into a dataset-level summary + LiNGAM gate.

    Replaces the old `all(gaussian_flags)` one-vote-veto. A single column with a
    significant normaltest p-value can no longer flip the whole dataset.
    """
    assessed = [c for c in per_col if c["verdict"] != "insufficient"]
    n_assessed = len(assessed)
    n_gaussianish = sum(1 for c in assessed if c["verdict"] == "gaussian-ish")
    n_ambiguous = sum(1 for c in assessed if c["verdict"] == "ambiguous")
    n_strong = sum(1 for c in assessed if c["verdict"] == "strongly-non-gaussian")

    gaussianish_ratio = (n_gaussianish / n_assessed) if n_assessed else None
    strong_ratio = (n_strong / n_assessed) if n_assessed else None

    if n_assessed == 0:
        verdict = "insufficient"
    elif n_strong == 0 and n_gaussianish >= n_assessed / 2:
        verdict = "gaussian-ish"
    elif (
        n_strong >= LINGAM_MIN_STRONG_COLS
        and strong_ratio is not None
        and strong_ratio >= LINGAM_MIN_STRONG_RATIO
    ):
        verdict = "strongly-non-gaussian"
    else:
        verdict = "ambiguous"

    gate_eligible = verdict == "strongly-non-gaussian" and n_assessed >= 2
    gate_reasons: list[str] = []
    if gate_eligible:
        gate_reasons.append(
            f"{n_strong}/{n_assessed} continuous columns show strong non-Gaussian "
            f"effect sizes (≥{int(LINGAM_MIN_STRONG_RATIO*100)}%) — broad enough to "
            "consider DirectLiNGAM as a cross-check candidate."
        )
    else:
        gate_reasons.append(
            "Non-Gaussianity is not strong and broad across columns "
            f"(assessed={n_assessed}, strong={n_strong}, gaussian-ish={n_gaussianish}); "
            "DirectLiNGAM is NOT recommended — marginal moments do not verify LiNGAM's "
            "structural-error assumption."
        )

    # Backward-compatible tri-state: True only if every assessed col is gaussian-ish;
    # False only if a strong departure exists; else None. Recommendation no longer
    # depends on this field.
    if n_assessed == 0:
        all_gaussian: bool | None = None
    elif n_strong > 0:
        all_gaussian = False
    elif n_gaussianish == n_assessed:
        all_gaussian = True
    else:
        all_gaussian = None

    return {
        "verdict": verdict,
        "assessed_columns": n_assessed,
        "gaussianish_count": n_gaussianish,
        "ambiguous_count": n_ambiguous,
        "strong_non_gaussian_count": n_strong,
        "gaussianish_ratio": round(gaussianish_ratio, 4) if gaussianish_ratio is not None else None,
        "strong_non_gaussian_ratio": round(strong_ratio, 4) if strong_ratio is not None else None,
        "lingam_gate": {"eligible": gate_eligible, "reasons": gate_reasons},
        "all_gaussian": all_gaussian,
    }


def _recommend(
    dataset_type: str,
    linearity_verdict: str,
    gauss_summary: dict[str, Any],
    n_samples: int,
    n_vars: int,
) -> dict[str, Any]:
    """Produce a concrete algorithm recommendation from the profile.

    Routing is data-type-first, then nonlinearity, then a conservative
    Gaussianity gate. DirectLiNGAM is only offered when non-Gaussianity is
    strong AND broad across an all-continuous dataset, and even then only as a
    cross-check candidate — never because it "outputs a full DAG".
    """
    recs: list[str] = []
    params: dict[str, Any] = {}
    reasons: list[str] = []
    requires_cross_check = False
    optional_cross_checks: list[str] = []
    cost_profile = _discovery_cost_profile(n_samples, n_vars)

    if dataset_type == "unknown":
        reasons.append(
            "No modelable columns (all ID / constant). Drop ID columns, remove "
            "constants, and confirm variable definitions before discovery."
        )
        return {
            "algorithms": recs,
            "suggested_params": params,
            "reasons": reasons,
            "requires_cross_check": requires_cross_check,
            "optional_cross_checks": optional_cross_checks,
            "cost_profile": cost_profile,
            "run_policy": "one-primary",
        }

    if dataset_type == "all-discrete":
        recs = ["pc"]
        params = {"indep_test": "chisq", "timeout": DEFAULT_DISCOVERY_TIMEOUT}
        optional_cross_checks = ["ges + local_score_BDeu", "boss"]
        reasons.append(
            "All columns discrete → chisq / BDeu are non-parametric and appropriate. "
            "Start with one PC run; GES/BOSS are optional checks, not a checklist. "
            "Do NOT use DirectLiNGAM or NOTEARS (they assume continuous data)."
        )
    elif dataset_type == "mixed":
        # Explicit branch — mixed data must NOT fall through to continuous logic.
        recs = ["pc"]
        params = {"timeout": DEFAULT_DISCOVERY_TIMEOUT}
        optional_cross_checks = ["ges sensitivity run", "boss sensitivity run"]
        reasons.append(
            "Mixed continuous + discrete/categorical columns. `causal_discover` "
            "factorizes non-numeric columns (arbitrary integer codes; ordinal "
            "order NOT preserved), so there is no true mixed-data model here — "
            "treat PC / GES / BOSS as SENSITIVITY baselines, not a guarantee. "
            "Do NOT use DirectLiNGAM or NOTEARS (they assume all-continuous). "
            "For ordered categories (e.g. Low/Medium/High) apply an explicit "
            "ordinal encoding before discovery instead of relying on factorize."
        )
    elif linearity_verdict == "nonlinear-detected":
        recs = ["pc"]
        params = {
            "indep_test": "rcit" if cost_profile["tier"] == "high" else "fastkci",
            "timeout": DEFAULT_DISCOVERY_TIMEOUT,
        }
        optional_cross_checks = [
            "targeted PC with the other approximate kernel test",
            "linear PC/GES sensitivity baseline (label the lost nonlinearity)",
        ]
        if cost_profile["tier"] == "high":
            reasons.append(
                "Nonlinear continuous relationships were detected, but a global kernel "
                f"search is high-cost at n={n_samples}, p={n_vars}. Start with ONE "
                f"bounded PC+rcit run on a seeded sample of about "
                f"{cost_profile['recommended_initial_rows']} rows (or an estimand-relevant "
                "subset). Do not launch generalized-score GES by default."
            )
        else:
            reasons.append(
                "Nonlinear continuous relationships were detected. Start with ONE "
                "bounded PC+fastkci run. Reserve full KCI and generalized-score GES "
                "for a small targeted confirmation only."
            )
    elif linearity_verdict == "inconclusive":
        recs = ["pc"]
        params = {"indep_test": "fisherz", "timeout": DEFAULT_DISCOVERY_TIMEOUT}
        optional_cross_checks = [
            "targeted PC+rcit/fastkci if an estimand-relevant pair remains suspicious",
            "GES/BOSS only if a core orientation is unresolved",
        ]
        lingam_gate = gauss_summary.get("lingam_gate", {})
        if lingam_gate.get("eligible"):
            requires_cross_check = True
            optional_cross_checks.append("direct_lingam after the primary baseline")
        reasons.append(
            "Linearity screening is INCONCLUSIVE: no reliable nonlinear shape was "
            "detected, but the available dependent pairs are too weak, unresolved, "
            "or truncated to establish approximate linearity. Treat the primary "
            "fisher-z run as a sensitivity baseline; add a targeted nonlinear check "
            "only where it can change the estimand-relevant graph. Do not describe "
            "the dataset as linear."
        )
        if requires_cross_check:
            reasons.append(
                "Strong and broad non-Gaussianity keeps DirectLiNGAM available only "
                "as a cross-check after PC/GES/BOSS; the inconclusive linearity "
                "screen does not justify leading with it."
            )
    else:
        # all-continuous + approximately-linear screening evidence.
        lingam_gate = gauss_summary.get("lingam_gate", {})
        if lingam_gate.get("eligible"):
            recs = ["pc"]
            params = {"indep_test": "fisherz", "timeout": DEFAULT_DISCOVERY_TIMEOUT}
            optional_cross_checks = [
                "direct_lingam",
                "GES/BOSS only if the estimand-relevant core is unresolved",
            ]
            requires_cross_check = True
            reasons.append(
                "Continuous, approximately linear in held-out screening, with "
                "STRONG and BROAD non-Gaussianity across "
                "columns → run one PC baseline first; DirectLiNGAM is an optional "
                "different-family check. A full DAG from LiNGAM is NOT higher confidence; "
                "LiNGAM's ICA orientation is unidentifiable on Gaussian/near-Gaussian "
                "noise."
            )
        else:
            recs = ["pc"]
            params = {"indep_test": "fisherz", "timeout": DEFAULT_DISCOVERY_TIMEOUT}
            optional_cross_checks = [
                "GES/BOSS if a core orientation is unresolved",
                "NOTEARS as a supplementary sensitivity run",
            ]
            verdict = gauss_summary.get("verdict", "ambiguous")
            reasons.append(
                f"Continuous, approximately linear in held-out screening, "
                f"Gaussianity verdict = '{verdict}' → start with one PC run. "
                "GES/BOSS/NOTEARS are optional checks, not a default sequence. Do NOT rely on "
                "DirectLiNGAM: its ICA orientation degrades on Gaussian/near-Gaussian "
                "data. A small normaltest p-value at large N is sample-size noise, "
                "not evidence for LiNGAM."
            )

    if n_vars >= 8:
        reasons.append(
            "Do not add BOSS/GRaSP merely because p≥8. Run a second family only when "
            "a disputed treatment parent, backdoor path, or T→Y path can change the answer."
        )
    if n_samples < 5 * n_vars:
        reasons.append(
            f"Low sample-to-variable ratio (n={n_samples}, p={n_vars}) → results may be "
            "unstable; prefer sparser settings (larger alpha penalty) and cross-check."
        )

    return {
        "algorithms": recs,
        "suggested_params": params,
        "reasons": reasons,
        "requires_cross_check": requires_cross_check,
        "optional_cross_checks": optional_cross_checks,
        "cost_profile": cost_profile,
        "run_policy": "one-primary",
    }


def diagnose(
    data: pd.DataFrame,
    treatment: str | None = None,
    outcome: str | None = None,
) -> dict[str, Any]:
    n_samples, n_vars = data.shape

    columns = [_classify_column(data[c]) for c in data.columns]

    id_cols = [c["name"] for c in columns if c["type"] == "id"]
    constant_cols = [c["name"] for c in columns if c["constant"]]

    # ID and constant columns carry no causal meaning — exclude both from the
    # type verdict (a constant numeric column must not flip the dataset to mixed).
    modeled = [c for c in columns if c["type"] != "id" and not c["constant"]]
    types = {c["type"] for c in modeled}
    if not types:
        dataset_type = "unknown"
    elif types <= {"discrete", "binary"}:
        dataset_type = "all-discrete"
    elif types <= {"continuous"}:
        dataset_type = "all-continuous"
    else:
        dataset_type = "mixed"

    # Gaussianity for continuous columns (graded, effect-size primary)
    gaussian_report: dict[str, Any] = {}
    per_col_gauss: list[dict[str, Any]] = []
    for c in columns:
        if c["type"] == "continuous":
            g = _gaussianity(data[c["name"]])
            gaussian_report[c["name"]] = g
            per_col_gauss.append(g)
    gauss_summary = _summarize_gaussianity(per_col_gauss)
    all_gaussian = gauss_summary["all_gaussian"]

    # Pairwise marginal shape screening among continuous columns.
    cont_cols = [c["name"] for c in columns if c["type"] == "continuous"]
    pair_reports: list[dict[str, Any]] = []
    any_nonlinear = False
    pairs = list(combinations(cont_cols, 2))[:MAX_PAIRS]
    truncated = len(list(combinations(cont_cols, 2))) > MAX_PAIRS
    for a, b in pairs:
        sub = data[[a, b]].dropna()
        if len(sub) < 10:
            continue
        rel = _pair_relationship(
            sub[a].to_numpy(dtype=np.float64), sub[b].to_numpy(dtype=np.float64)
        )
        rel["pair"] = [a, b]
        pair_reports.append(rel)
        if rel["nonlinear"]:
            any_nonlinear = True

    nonlinear_pair_count = sum(
        1 for report in pair_reports if report["shape_verdict"] == "nonlinear-detected"
    )
    approximately_linear_pair_count = sum(
        1 for report in pair_reports if report["shape_verdict"] == "approximately-linear"
    )
    dependent_pair_count = sum(1 for report in pair_reports if report["dependent"])
    unresolved_dependent_pair_count = sum(
        1
        for report in pair_reports
        if report["dependent"] and report["shape_verdict"] == "inconclusive"
    )

    if nonlinear_pair_count > 0:
        linearity_verdict = "nonlinear-detected"
        linearity_reason = (
            f"{nonlinear_pair_count} pair(s) show nonlinear marginal shape via "
            "rank/MI discrepancy or held-out spline gain."
        )
    elif (
        dependent_pair_count > 0
        and unresolved_dependent_pair_count == 0
        and not truncated
    ):
        linearity_verdict = "approximately-linear"
        linearity_reason = (
            f"{approximately_linear_pair_count} dependent pair(s) were screened "
            "without material held-out spline gain. This is approximate and "
            "marginal, not proof of a linear SCM."
        )
    else:
        linearity_verdict = "inconclusive"
        linearity_reason = (
            "No nonlinear shape was detected, but dependence was absent/weak, "
            "some dependent pairs were unresolved, or the pair scan was truncated; "
            "absence of detection is not evidence of linearity."
        )

    linearity_summary = {
        "verdict": linearity_verdict,
        "reason": linearity_reason,
        "pairs_screened": len(pair_reports),
        "dependent_pairs": dependent_pair_count,
        "nonlinear_pairs": nonlinear_pair_count,
        "approximately_linear_pairs": approximately_linear_pair_count,
        "unresolved_dependent_pairs": unresolved_dependent_pair_count,
        "pairwise_truncated": truncated,
        "spline_gain_threshold": SPLINE_GAIN,
        "spline_min_r2": SPLINE_MIN_R2,
    }

    # --- EDA surfacing (the agent must SEE the data, not just a routing verdict)
    top_associations = sorted(
        pair_reports,
        key=lambda r: max(r["spearman_abs"], r["mi_norm"]),
        reverse=True,
    )[:TOP_ASSOCIATIONS]

    near_deterministic_pairs = [
        {
            "pair": r["pair"],
            "pearson": r["pearson"],
            "warning": (
                "near-deterministic relationship — this breaks/distorts CI tests, "
                "LiNGAM and regression (rank deficiency, inflated orientations). "
                "Consider merging, dropping, or explicitly modelling one of the pair."
            ),
        }
        for r in pair_reports
        if r["pearson_abs"] >= NEAR_DETERMINISTIC_R
    ]

    duplicate_rows = int(data.duplicated().sum())
    missing_rate = (
        round(float(data.isna().sum().sum()) / (n_samples * n_vars), 4)
        if n_samples and n_vars
        else 0.0
    )

    # --- LiNGAM gate must respect linearity: nonlinear data kills LiNGAM even
    # when non-Gaussianity is strong and broad. Override the narrative field so
    # the report never shows "nonlinear: YES" next to "gate: eligible".
    if (
        linearity_verdict == "nonlinear-detected"
        and gauss_summary["lingam_gate"]["eligible"]
    ):
        gauss_summary["lingam_gate"]["eligible"] = False
        gauss_summary["lingam_gate"]["reasons"] = [
            "Nonlinear relationship(s) detected → DirectLiNGAM's LINEARITY "
            "assumption is violated regardless of non-Gaussianity. Gate closed; "
            "use kernel methods (PC + fastkci/rcit, GES + generalized score)."
        ]

    recommendation = _recommend(
        dataset_type, linearity_verdict, gauss_summary, n_samples, n_vars
    )
    if id_cols:
        recommendation["reasons"].insert(
            0,
            f"Drop ID-like column(s) {id_cols} before discovery — they carry no "
            "causal meaning.",
        )

    # Non-numeric (string / categorical) columns, excluding ID columns.
    # causal_discover auto-encodes these via pd.factorize, but some direct
    # algorithm calls (or manual pipelines) can fail on object-dtype arrays,
    # so surface them explicitly.
    non_numeric_cols = [
        c["name"] for c in columns if not c["is_numeric"] and c["type"] != "id"
    ]
    if non_numeric_cols:
        recommendation["reasons"].insert(
            0,
            f"Non-numeric (string/categorical) column(s) {non_numeric_cols} detected. "
            "`causal_discover` auto-encodes them with pd.factorize (arbitrary integer "
            "codes — ordinal order NOT preserved). Some methods may fail on raw "
            "string data; verify encoding is meaningful for ordered categories "
            "(e.g. low/medium/high).",
        )

    # Compute warning: use N, P, and pair count. N alone missed an n=800, p=11
    # run that spent almost an hour in a generalized score.
    compute_warning: str | None = None
    kernel_test = recommendation["suggested_params"].get("indep_test")
    kernel_score = "general" in str(recommendation["suggested_params"].get("score_func", ""))
    cost_profile = recommendation["cost_profile"]
    if (kernel_test or kernel_score) and cost_profile["tier"] == "high":
        compute_warning = (
            f"High-cost nonlinear search profile at n={n_samples}, p={n_vars} "
            f"(relative work proxy={cost_profile['kernel_screen_work_units']:,}). "
            f"Use the recommended timeout={cost_profile['default_timeout_seconds']}s and "
            f"start with about {cost_profile['recommended_initial_rows']} seeded rows or "
            "an estimand-relevant subset. Do not start global generalized-score GES by default."
        )
        recommendation["reasons"].append(compute_warning)

    result: dict[str, Any] = {
        "dataset_type": dataset_type,
        "n_samples": int(n_samples),
        "n_variables": int(n_vars),
        "id_columns": id_cols,
        "constant_columns": constant_cols,
        "non_numeric_columns": non_numeric_cols,
        "duplicate_rows": duplicate_rows,
        "missing_rate": missing_rate,
        "columns": columns,
        "gaussianity": gaussian_report,
        "gaussianity_summary": gauss_summary,
        "all_gaussian": all_gaussian,
        "pairwise": pair_reports,
        "pairwise_truncated": truncated,
        "any_nonlinear": any_nonlinear,
        "linearity_verdict": linearity_verdict,
        "linearity_summary": linearity_summary,
        "top_associations": top_associations,
        "near_deterministic_pairs": near_deterministic_pairs,
        "compute_warning": compute_warning,
        "recommendation": recommendation,
    }

    if treatment or outcome:
        if not (treatment and outcome):
            result["association_anchor"] = {
                "error": "both --treatment and --outcome are required for an anchor"
            }
            result["effect_estimation_profile"] = {
                "error": "both --treatment and --outcome are required for inference routing"
            }
        else:
            result["association_anchor"] = _association_anchor(data, treatment, outcome)
            result["effect_estimation_profile"] = _effect_estimation_profile(
                data,
                treatment,
                outcome,
                linearity_verdict,
            )

    return result


@safe_main
def main():
    parser = argparse.ArgumentParser(description="CausalClaw — Data Diagnosis Bridge")
    parser.add_argument("--data", required=True)
    parser.add_argument("--treatment", default=None)
    parser.add_argument("--outcome", default=None)
    args = parser.parse_args()

    data = load_data(args.data)
    result = diagnose(data, treatment=args.treatment, outcome=args.outcome)
    output_result(result)


if __name__ == "__main__":
    main()
