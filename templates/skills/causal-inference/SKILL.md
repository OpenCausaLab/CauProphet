---
name: causal-inference
description: Provides rules and guidelines for running causal inference (ATE/CATE estimation), selecting adjustment sets (common causes) based on the causal graph, and interpreting refutation tests.
---

# Skill: Causal Inference

> **When to use this skill:** The user asks you to estimate the causal effect of X on Y (ATE or CATE), or you are preparing to call the `causal_infer` tool.

## 1. Prerequisites: The Causal Graph

Before estimating an effect, you **MUST** have a directed acyclic graph (DAG) in your Causal Scratchpad. If you don't have one, you must perform Causal Discovery first or ask the user to provide the causal links.

You cannot just pass all other variables as `common_causes`. You must explicitly identify the correct adjustment set using the **Backdoor Criterion**.

> **String treatment handling:**
> - **Binary string** (e.g., `AB_Test_Group`: A/B): the tool auto-encodes to 0/1. No action needed.
> - **Multi-class string** (e.g., `Content_Category` with 5 values): the tool returns a `MULTI_CLASS_TREATMENT` error listing the available categories. Re-call with `treatment_value=<treated category>` and `control_value=<control category>` to compare a specific pair.

## 2. Identifying the Estimand (MANDATORY — decide BEFORE calling the tool)

> [!CAUTION]
> **You MUST decide which estimand the question asks for before calling `causal_infer`.**
> The current tool implements only the `total` effect. If the question asks for a direct
> or indirect effect, state that the tool cannot estimate it yet; never return a total
> effect under a different label.

| Estimand | What it measures | `estimand` param | Adjustment set rule |
|---|---|---|---|
| **Total Effect (TE)** | E[Y\|do(X=x+δ)] - E[Y\|do(X=x)] — all causal pathways | `"total"` (default) | Only adjust for **confounders** (non-descendant common causes of T and Y). **NEVER** include mediators or treatment descendants. |
| **Direct Effect (DE)** | Effect of T→Y with mediator paths blocked | Not implemented | Do not call `causal_infer`; a dedicated mediation estimator and stronger assumptions are required. |
| **Indirect Effect (IE)** | Effect mediated through intermediate pathways | Not implemented | Do not subtract an unsupported direct estimate from the total. |

**If the user asks "what happens if we increase X?", they want the total effect (default).**
**If the user asks "what is the direct impact of X on Y?", acknowledge that they want a direct effect and explain the current limitation.**

> [!CAUTION]
> **HARD RULE: The `causal_infer` tool will REJECT your call if `common_causes`
> contains ANY descendant of the treatment variable when `estimand="total"` (or omitted).**
> This is enforced programmatically — the tool checks the scratchpad DAG before running.
> Direct and indirect estimands are rejected rather than silently replaced by a total effect.

### 2.1 Define the intervention contrast separately from the causal pathway estimand

The pathway estimand says **which causal pathways** are included; the intervention
says which treatment values change. The current tool estimates total effects only,
but the intervention must still be defined separately:

| User wording | Tool contract | Quantity |
|---|---|---|
| "increase continuous X by δ from its current level" | `intervention_type="additive_shift"`, `shift_delta=δ` | $E[m(X+δ,Z)-m(X,Z)]$ |
| "compare X=a with X=b" | `intervention_type="contrast"`, `treatment_value=a`, `control_value=b` | $E[m(a,Z)-m(b,Z)]$ |
| unambiguous binary treatment | `intervention_type="contrast"` | treated versus control |

> [!CAUTION]
> For a continuous treatment, a fixed `1 versus 0` contrast is **not** the same
> estimand as adding 1 to every observed treatment level when the response is
> nonlinear. If neither a shift nor a contrast is specified, stop and resolve the
> intervention from the user's wording; never silently guess one.

### 2.2 Functional-form gate (MANDATORY for continuous effects)

Use `method="auto"` unless the user explicitly requests a sensitivity estimator.
For an additive shift, the tool compares an adjusted linear outcome model with
held-out flexible models after the adjustment set is known, then evaluates
$m(X+δ,Z)-m(X,Z)$ record by record.

- A linear treatment coefficient is a valid shift effect only when the adjusted
  response is sufficiently linear. Dataset-wide or marginal linearity is not enough.
- If a flexible model materially improves held-out fit, an explicit linear estimate
  is labelled `sensitivity_only`; do not promote it because its p-value is small or
  its refutation tests pass.
- Review `model_comparison`, `support_diagnostics`, `validity`, and `warnings` before
  reporting. If strong flexible models disagree in sign, or the shifted values have
  poor support, report the result as inconclusive rather than falling back to OLS.
- For total effects, the outcome model uses the treatment plus the valid backdoor
  adjustment set. Do not add mediators merely to improve prediction.
- Refutation tests evaluate stability of the selected estimator. They do not repair
  functional-form misspecification and cannot override `sensitivity_only` or
  `inconclusive` validity.

## 3. Identifying the Adjustment Set (`common_causes`)

To estimate the true causal effect of a Treatment (T) on an Outcome (O), you must close all "backdoor paths" (spurious correlations) without closing any directed paths (causal mechanisms).

1. **Find Backdoor Paths:** Look for any path from T to O that has an arrow pointing *into* T (e.g., T ← C → O).
2. **Identify Confounders:** Variables like C that cause both T and O are common causes. These **must** be included in the `common_causes` argument.
3. **Do NOT adjust for Mediators (for total effect):** If a variable M is on the causal path (T → M → O), do **NOT** include it in `common_causes`. Adjusting for a mediator blocks the true causal effect (over-adjustment bias).
4. **Do NOT adjust for Colliders:** If two variables point to C (A → C ← B), C is a collider. Conditioning on a collider *opens* a spurious path. Do not include it unless necessary to block another path.

### 3.1 Empty adjustment set — legitimate, but verify the root-node claim

**An empty adjustment set can be perfectly correct.** It is the right answer when the
treatment genuinely has no parents — e.g. a randomized/experimentally assigned treatment,
an exogenous shock, or a true source node in the DAG. When that is the case, report the
unadjusted estimate with confidence and say *why* T has no parents.

The failure mode to avoid is a *silently wrong* empty set: if discovery dropped a
confounder edge (Z→T), the graph will claim T is a root, and unadjusted regression walks
straight into Simpson's paradox (this has produced a sign-flipped ATE in practice).

So the goal is **confirming the claim, not overturning it.** A short check:

1. Confirm **why** T has no parents — by design (randomized/exogenous/first in time), or
   only because one algorithm said so? If it rests on a single algorithm, confirm with a
   second algorithm family (see the causal-discovery skill's root-node rule). A functional
   method like DirectLiNGAM is unidentifiable on Gaussian data and can invent false roots.
2. Sanity-check for a **missed** confounder: is some `Z` strongly associated with both T
   and Y, or does the T coefficient move materially when Z is added? Treat this only as an
   **alarm that triggers classification**, never as a verdict.
3. **Association is NOT proof of confounding.** A `Z` correlated with both T and Y may be:
   - a **pre-treatment common cause** (adjust for it),
   - a **mediator** T→Z→Y (adjusting blocks the real effect — over-adjustment),
   - a **collider** T→Z←Y or M-bias (adjusting *creates* bias),
   - a **descendant of T** or an **outcome proxy** (do not adjust).
4. Classify each candidate using the **DAG + temporal order + domain knowledge** *before*
   deciding. **Never auto-add correlated variables to `common_causes`** just to make the
   set non-empty — an unnecessary adjustment can bias a correct estimate.
5. If a variable's role genuinely cannot be determined, report sensitivity results under
   the plausible alternatives instead of silently picking one.

**If the check passes, an empty adjustment set is a valid result — report it as such.**

## 4. Estimating CATE (Conditional Average Treatment Effect)

If the user asks "How does the effect vary by Age?" or "What is the effect on males vs females?", you are looking for CATE.
- For a continuous additive-shift intervention, pass exactly **one baseline,
  pre-treatment** grouping variable in `effect_modifiers`. The tool reports a
  stratified conditional total effect for each observed categorical level or each
  equal-frequency numeric stratum. Each stratum independently runs the linear,
  spline, gradient-boosting, and treatment-shape gates before selecting its estimator;
  this prevents opposite subgroup nonlinearities from cancelling in the population
  gate. The tool reports the stratum definition, row count, selected model/form,
  local shift support, bootstrap interval, and validity separately. Material
  disagreement among subgroup estimators makes that stratum inconclusive.
- Never use a mediator or any other treatment descendant as an effect modifier. The
  tool rejects descendants found in the active DAG. Non-descendant is necessary but
  not sufficient: a baseline collider can open a backdoor path when conditioned on.
  The tool therefore verifies $T \perp Y\mid Z,M$ in the backdoor graph for the
  identified adjustment set $Z$ and modifier $M$. If that check fails, it returns no
  CATE. If the modifier is absent from the DAG, conditional exchangeability is not
  graph-validated and every subgroup result remains inconclusive.
- Treat differences between subgroup point estimates as descriptive unless a formal
  heterogeneity contrast is reported. The current output explicitly says that no
  simultaneous/formal heterogeneity test was performed.
- Check `reconciliation`: over the same complete-row target population, the
  stratum-size-weighted CATE should agree with the population ATE by the law of total
  expectation. If the discrepancy exceeds the reported tolerance, every subgroup is
  downgraded to `inconclusive`; do not rationalize the mismatch after the fact.
- The per-stratum bootstrap is capped at 30 refits for runtime. Its percentile interval
  is explicitly **exploratory/low-precision**, not standalone decision-grade evidence.
- Request one modifier per call. This keeps the target subgroup and support auditable
  and avoids sparse, data-mined combinations.
- The legacy fixed-contrast path still uses DoWhy's single linear interaction: disclose
  that assumption and do not present it as nonlinear CATE.

## 5. Interpreting Refutation Tests

A causal estimate is only as good as its assumptions. The legacy fixed-contrast path
runs the three DoWhy refutation tests below. The additive-shift path instead reports
held-out model comparison, flexible-estimator agreement, support diagnostics, and
bootstrap uncertainty because the legacy refuters would recreate a different estimator.
Review the diagnostics that actually apply to the returned estimator.

| Test | Mechanism | What it means if it PASSES (✅) | What to do if it FAILS (⚠️) |
|---|---|---|---|
| **Placebo Treatment** | Replaces the actual treatment with a random variable. | The model correctly identifies that a random variable has no effect (ATE ≈ 0). | Your model might be forcing an effect where none exists. Check for severe model misspecification. |
| **Random Common Cause** | Adds an independent random variable to the dataset as a confounder. | The estimate is robust. Adding a irrelevant variable didn't change the ATE. | Your estimator is highly sensitive to noise. Try a different `method` (e.g., propensity_score_weighting). |
| **Data Subset** | Re-estimates the effect on a random subset of the data. | The effect is consistent across the dataset. | The effect might be driven by outliers or specific subgroups. Check for heterogeneity. |

*Note: A "Pass" usually means the p-value is > 0.05, indicating the new effect is not statistically significantly different from the original effect (or is not significantly different from zero in the placebo case).*

> [!CAUTION]
> **An applicable failed refuter is a downgrade, not a footnote.** If `placebo_treatment` fails,
> do NOT rationalize it after the fact ("the new effect was close to zero anyway",
> "tool-level noise"). The required response is:
> 1. Do not treat the estimate as decision-grade evidence.
> 2. Inspect what the refuter's null hypothesis and "failure" actually mean numerically.
> 3. State explicitly: "this estimate did not pass all diagnostics and should not be
>    the basis for a decision" — then investigate or stop.

> [!CAUTION]
> **"Edge not selected" ≠ "no edge".** A small-sample or kernel-based run that does
> NOT select an edge has NOT accepted independence and does NOT support "no causal
> path". Power at n=100–500 with heavy tails, near-deterministic pairs, or numerical
> warnings is too low for that reading. If the run shows numerical warnings, ghost
> variables, or degenerate output, label the whole verification **unreliable** and
> say the kernel pass "provided no usable structural evidence" — never cite it as
> corroboration for a near-zero effect.

> [!WARNING]
> **What these three tests CANNOT do.** They check the stability of the model *you
> already specified*. They do **not** validate the DAG and **cannot detect an omitted
> confounder** — in particular `Random Common Cause` adds an *independent random* variable,
> so it passes whether or not a real confounder is missing. All three refutations passing
> is therefore **not** evidence that the adjustment set is complete or that an empty
> adjustment set is correct. If omitted confounding is plausible, say so explicitly and
> treat the estimate as conditional on the graph.

> [!WARNING]
> **Statistical stability is NOT causal correctness.** A small p-value, narrow confidence
> interval, and passing placebo test only mean your model is stable *given the adjustment
> set you chose*. They CANNOT verify:
> 1. Whether the adjustment set contains descendants/mediators it shouldn't
> 2. Whether a real confounder was omitted
> 3. Whether the requested estimand is the supported total effect
>
> **Before reporting statistically stable results, you MUST first verify the causal
> correctness of the adjustment set by checking the DAG.**

## 6. The Association Anchor — always report marginal vs. adjusted side-by-side

Before inference, get the **unadjusted marginal association** between T and Y —
the cheapest way is `causal_diagnose(data_path=..., treatment=T, outcome=Y)`,
which returns an `association_anchor` (unadjusted slope, sign, 95% CI, Pearson/
Spearman).

Your final report **MUST present both numbers together**:

- **Marginal association** E[Y|T] — what the raw data shows;
- **Adjusted causal estimate** E[Y|do(T)] — what your DAG + adjustment set implies.

If the two **differ in sign or by an order of magnitude**, that is the signature of
confounding (or mediation/collider structure) — explain *which mechanism* produces
the divergence using the DAG. Never report only one number, and never let a
sophisticated pipeline output an estimate whose sign contradicts the marginal
association without an explicit structural explanation. A sign flip you cannot
explain means the graph is probably wrong — go back, don't report.

## 7. Path Cancellation Detection

> [!IMPORTANT]
> **If the estimated total ATE is close to zero but individual path coefficients (or the
> coefficient on treatment in unadjusted regression) are large, suspect path cancellation.**
>
> Path cancellation occurs when the direct effect (T→Y) and indirect effects (T→M→...→Y)
> have opposite signs and similar magnitudes, causing them to approximately cancel.
>
> When you detect this pattern, report path cancellation as a structural hypothesis and
> identify the mediator paths that would need a dedicated mediation analysis. Do not ask
> `causal_infer` for unsupported direct/indirect estimates or fabricate a numerical
> decomposition. A near-zero total effect can still be meaningful, but the component
> magnitudes remain unestimated until that analysis is implemented.

## 8. Practical Checklist Before Reporting

1. [ ] Is the requested estimand the supported **total effect**? If not, did I state the limitation instead of relabelling a total effect?
2. [ ] Did I explicitly select `common_causes` based on the DAG (Backdoor Criterion)?
3. [ ] Did I exclude mediators and treatment descendants (for total effect)?
4. [ ] **If the adjustment set is empty**, did I confirm *why* T has no parents (by design, or cross-family confirmed)? An empty set is a valid result once verified — but did I avoid adding variables just to make it non-empty?
5. [ ] Did I review the estimator-compatible diagnostics? For a fixed contrast, did I handle failed refuters; for an additive shift, did I inspect model agreement, support, and bootstrap uncertainty?
6. [ ] Did I report the 95% Confidence Interval alongside the ATE?
7. [ ] **Did I report the marginal association anchor next to the adjusted estimate?** If the estimate changes sign or magnitude drastically vs. the anchor, did I explain the structural mechanism — or go back and re-examine the graph?
8. [ ] Did I state the adjustment set, its causal justification, and which DAG/algorithm it came from (the tool prints a provenance banner — carry it into the report)?
9. [ ] **If the graph contains latent-confounding or unresolved-orientation markers**, did I avoid presenting an unidentified estimate as decision-grade?
10. [ ] **If total ATE ≈ 0 but path coefficients look large**, did I flag possible path cancellation without inventing a direct/indirect decomposition?
11. [ ] **Did I state the intervention itself** (additive shift δ or treatment/control contrast), separately from the total pathway estimand?
12. [ ] For a continuous treatment, did I use `method="auto"` and inspect held-out linear-vs-flexible model comparison?
13. [ ] If the tool returned `sensitivity_only` or `inconclusive`, did I avoid presenting the number as a decision-grade causal effect?

---

## Example Workflow

```
User: "What is the effect of Medication on Recovery?"
Scratchpad Graph: Age → Medication, Age → Recovery, Medication → BloodPressure → Recovery.

1. Analyze DAG: 
   - Treatment: Medication
   - Outcome: Recovery
   - Backdoor path: Medication ← Age → Recovery. (Age is a common cause).
   - Directed path: Medication → BloodPressure → Recovery. (BloodPressure is a mediator).

2. Decide estimand: User asks "effect" → total effect (default).

3. Action: Call `causal_infer`
   - treatment: "Medication"
   - outcome: "Recovery"
   - common_causes: "Age" (Do NOT include BloodPressure for total effect!)
   - estimand: "total"

4. Review Results: 
   - ATE = 2.4. All refutation tests pass.

5. Report: Tell the user the effect is 2.4, adjusting for Age, and the estimate is robust to refutations.
   State that BloodPressure is a mediator (part of the causal mechanism, not adjusted for).
```
