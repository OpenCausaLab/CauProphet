---
name: causal-discovery
description: Provides rules and guidelines for running causal discovery, interpreting graph types (CPDAG, PAG, DAG), and orienting ambiguous edges using domain knowledge or web search.
---

# Skill: Causal Discovery

> **When to use this skill:** The user asks you to discover causal structure from data,
> or you are about to call the `causal_discover` tool.

## Algorithm Output Types

Each algorithm outputs a different graph class. You **must** understand what you receive before reporting results or running causal inference.

| Algorithm | Output class | Contains undirected edges? | Contains latent-confounder edges? |
|-----------|-------------|---------------------------|-----------------------------------|
| `direct_lingam` | **DAG** | ❌ No | ❌ No |
| `ges`     | **CPDAG** (Completed Partially Directed Acyclic Graph)| ✅ Yes — unresolved MEC edges | ❌ No |
| `pc`      | **CPDAG** | ✅ Yes — unresolved MEC edges | ❌ No |
| `fci`     | **PAG** (Partial Ancestral Graph) | ✅ Yes — circle endpoints (o) | ✅ Yes — bidirected edges (↔) |
| `notears` | **DAG** | ❌ No | ❌ No |
| `boss`    | **CPDAG** | ✅ Yes — unresolved MEC edges | ❌ No |
| `grasp`   | **CPDAG** | ✅ Yes — unresolved MEC edges | ❌ No |

### Edge type legend

| Symbol | `edgeType` value | Meaning |
|--------|-----------------|---------| 
| A → B  | `directed` | Confirmed causal direction |
| A — B  | `undirected` | Markov-equivalent; direction unresolved (CPDAG) |
| A ↔ B  | `bidirected` | Hidden common cause (PAG) |
| A o→ B | `partially_directed` | One endpoint uncertain (PAG) |

---

## Data Preprocessing

The `causal_discover` tool **automatically encodes** non-numeric columns (string / categorical) using `pd.factorize` before running the algorithm. The encoded column names are reported in the result metadata. You do NOT need to manually label-encode string columns before calling this tool.

However, you should still:
- **Drop ID / index columns** that have no causal meaning (e.g. `user_id`, `row_number`).
- **Be aware** that factorize assigns arbitrary integer codes — ordinal relationships in the original categories are not preserved.

### Auto-Write-Back to Scratchpad

When an active scratchpad exists, `causal_discover` automatically writes the discovered edges, variables, and graph score to the scratchpad. You do **not** need to manually call `write_scratchpad` after discovery.

> **⚠️ The write-back REPLACES the current graph.** It records the full prior edge list in Notes.
> The belief remains your scratchpad: use `write_causal_belief` to merge, replace,
> remove, or clear any state before or after discovery.


## ⚠️ Verification Gate — read before choosing a non-default algorithm

The diagnosis is **evidence, not a decision**. A wrong algorithm choice here has
produced a completely wrong DAG and a sign-flipped ATE in practice. These rules are
mandatory, not advisory.

**1. Large-sample p-value trap.** Normality tests (D'Agostino/Shapiro) are hyper-sensitive
at large N. At `N=5000`, clean Gaussian data with `skew≈-0.06` still gets `p<0.05`. Read
the **`Gaussianity verdict`** and the per-column **skew / excess_kurtosis** magnitudes —
never treat a small p-value alone as "non-Gaussian".

**2. LiNGAM is unidentifiable on Gaussian data.** DirectLiNGAM's ICA orientation requires
*non-Gaussian* independent structural errors. On Gaussian/near-Gaussian data the component
ordering collapses, producing pseudo-random orientations, flipped edges, and **false root
nodes**. A marginal-distribution check is only a screening signal — it does **not** verify
the structural-error assumption. Requirements: all-continuous, linear, no hidden
confounding, and strong+broad non-Gaussianity (the diagnosis reports a `DirectLiNGAM gate`).
Never choose it on `mixed` / discrete / factorized-string data.

**3. Budget-first rule.** Run the diagnosis-recommended **single primary method** with a
wall-clock budget. Do not automatically run a linear baseline before every nonlinear method,
and do not treat the list of available algorithms as a checklist. Add a second family only
when it can change a treatment parent, backdoor path, T→Y path, or identification decision.
A linear run on nonlinear data is a sensitivity analysis and must be labelled as such.
Once nonlinearity is detected, a precision matrix, partial correlation, or linear-regression
p-value cannot prove that an edge is absent. In particular, a quadratic or other symmetric
response can have a near-zero linear coefficient while retaining strong conditional
dependence. Preserve a core adjacency as disputed and use a targeted nonlinear conditional
check; do not delete it merely because OLS says `p>0.05`.

**4. Red flags — STOP and re-examine, do not report side-by-side as "cross-validation":**

| Red flag | What it usually means |
|---|---|
| Two algorithms give **opposite directions** on a core edge, or edge counts differ several-fold | An assumption is violated — not corroboration |
| **Dense DAG** (e.g. 30 variables → 100+ edges) | LiNGAM/optimization is unidentifiable; orientations are arbitrary |
| Two variables strongly correlated but **no path** between them in the graph | The graph is missing a real edge (a confounder may be dropped) |
| An algorithm declares the treatment a **root node with no parents** | See the root-node rule below — verify before trusting |

> [!CAUTION]
> **MANDATORY STOP CONDITION:** If two algorithms exhibit ANY of the following divergences,
> you MUST stop. Do NOT "average", "vote", or "splice" their edges into a composite graph.
> 
> - **Edge count ratio > 2×** (e.g., PC: 40 edges, GES: 112 edges)
> - **Treatment → Outcome core edge direction is reversed** between algorithms
> - **Treatment in-degree differs by > 2** (e.g., one says root node, another says 3 parents)
> 
> **Correct response to divergence:**
> 1. Report the **skeleton** (undirected edges) that both algorithms agree on
> 2. Mark disputed edges as "uncertain" with both algorithms' opinions
> 3. Resolve key disputes ONLY with verifiable evidence — temporal order, study
>    design, user confirmation, citable references. **Disguised/anonymized variable
>    names are not evidence** (see the Orientation Protocol's warning).
> 4. Optionally, arbitrate statistically: re-run one algorithm on 3–5 subsamples
>    (seeded) and keep edges by **selection frequency** (≥3/5) to stabilize the
>    skeleton before arguing about direction. See "Subsampling / resampling".
> 5. Never combine one algorithm's backdoor set with another algorithm's edge orientations

**5. Do not silently tune parameters to manufacture agreement.** Tightening `alpha`
(e.g. 0.05 → 0.01) changes sparsity and can amplify disagreement. If you change a
parameter, say so and report both results.

**6. Root-node / no-parent claims deserve a second algorithm family.** A root node is a
perfectly legitimate finding — randomized, exogenous, and upstream-most variables really do
have no parents. The problem is *inheriting* that claim from a single algorithm when it
determines whether any confounder gets adjusted for. Confirm with a *different family* —
constraint-based (PC/FCI) vs. score/permutation (GES/BOSS) vs. functional (LiNGAM). GES and
BOSS are both score/permutation-based, so their agreement is weaker evidence than PC vs. GES.
If families disagree on T's in-degree, resolve that before running inference; if they agree,
report it as a confirmed root node.

---

## Step 0 — Diagnose the Data Before Choosing an Algorithm

**Before calling `causal_discover`, you must understand the data's distribution and type.**
Do not guess the algorithm from column names or intuition — the wrong choice (e.g. a
linear method on nonlinear data) silently produces a wrong graph.

Call **`causal_diagnose(data_path=...)`** first. It reports:

- **Per-column type:** `continuous` / `discrete` / `binary` / `id`.
- **Dataset verdict:** `all-continuous` / `all-discrete` / `mixed`.
- **Gaussianity verdict** (`gaussian-ish` / `ambiguous` / `strongly-non-gaussian` /
  `insufficient`) plus a **per-column table of skew / excess_kurtosis / normaltest_p**,
  and a **`DirectLiNGAM gate`** flag. The verdict is effect-size-driven; the p-value is
  only supporting evidence.
- **Linear vs. NONLINEAR** pairwise relationships (strong Spearman/MI but weak Pearson ⇒ nonlinear).
- **Sample size, missing values, and ID-like columns** to drop.
- A concrete **algorithm + parameter recommendation**.

Then map the diagnosis to your call:

| Diagnosis | Use |
|---|---|
| `all-continuous`, linear, `gaussian-ish` or `ambiguous` | `pc` / `ges` / `boss` (+ `notears` as a supplementary check). **Not** LiNGAM. |
| `all-continuous`, linear, `strongly-non-gaussian` **and** gate eligible | Run `pc`/`ges`/`boss` first, then `direct_lingam` as a cross-check candidate |
| **any nonlinear relationship** | One bounded `pc` run with the diagnosis-recommended `rcit` or `fastkci`; use a seeded sample/estimand-relevant subset when the compute tier is high. |
| `all-discrete` | `pc` (`chisq`) / `ges` (`local_score_BDeu`) / `boss` |
| `mixed` (continuous + discrete/string) | `pc` / `ges` / `boss` as **sensitivity baselines** only. **Never** `direct_lingam` or `notears` — they assume all-continuous. Prefer an explicit ordinal encoding over factorize for ordered categories. |
| A core ambiguity can change the answer | Run one different-family check targeted at that ambiguity; do not add BOSS/GRaSP routinely. |

The diagnosis output is a run plan: **primary method + timeout + compute tier + optional
conditional checks**. Start with the primary only. A cross-check is triggered by an
estimand-relevant uncertainty, not by graph size or the mere availability of another method.

Always **drop the `id` columns** the diagnosis flags before discovery.

> **⚠️ The diagnosis is a heuristic, not ground truth.** It can be wrong — treat it as
> a starting point, not a verdict. Known failure modes:
> - **ID / index columns may be missed.** A discrete-looking integer key (e.g. small
>   category codes, zip-like codes, or an ID that repeats) will *not* be flagged as `id`.
>   Judge ID-ness from the column's *meaning*, not just the diagnosis.
> - **Type classification is threshold-based.** An integer-coded ordinal with many levels
>   can be called `continuous`, and a coarsely-binned continuous variable can be called
>   `discrete`.
> - **Linearity detection needs signal.** Weak or noisy relationships, tiny samples, or
>   confounded pairs can hide a nonlinear shape (reported "linear") or fabricate one.
> - It only inspects **pairwise, marginal** relationships — it says nothing about
>   conditional or higher-order structure.
>
> If the diagnosis conflicts with what you know about the data (column semantics, the
> user's description, domain knowledge), **trust your judgment over the tool** and pick
> the algorithm accordingly. When unsure, run more than one algorithm and compare.

---

## Algorithm Selection & Parameters

When calling `causal_discover`, choose the algorithm and parameters based on the data characteristics.

### Algorithm Selection Cheat-Sheet

| Data / situation | Recommended method(s) | Notes |
|---|---|---|
| Continuous, linear-Gaussian (or weakly non-Gaussian at large N) | `pc`, `ges`, `boss`, `notears` | **Standard default baselines.** Robust to sample-size noise. |
| Continuous, linear, **strongly + broadly non-Gaussian** (LiNGAM gate eligible) | `direct_lingam` — *after* a `pc`/`ges`/`boss` baseline | Outputs a full DAG, but that is **not** higher confidence. Unidentifiable on Gaussian/near-Gaussian noise. |
| **Continuous, non-linear** | One bounded `pc`/`fci` run with `fastkci` or `rcit` | Use `kci` or generalized-score GES only for a small targeted confirmation whose value justifies the cost. |
| Discrete / categorical | `pc` (`chisq`/`gsq`), `ges` (`local_score_BDeu`), `boss`/`grasp` (`auto`) | These tests/scores are non-parametric — they do **not** assume linearity. |
| **Mixed** continuous + discrete/string | `pc`, `ges`, `boss` as sensitivity baselines | **Never** `direct_lingam` / `notears`. factorize gives arbitrary codes and loses ordinal order. |
| Possible hidden confounders | `fci` | Outputs a PAG (may contain ↔ / o→ edges). |
| Estimand-relevant GES ambiguity | `boss`, `grasp` | Optional permutation-based sensitivity check; not part of the default path. |

> **Concept — "discrete" ≠ "non-linear".** Data type (continuous vs. discrete) and
> functional form (linear vs. non-linear) are orthogonal. PC/GES on discrete data use
> chi-squared / G² / BDeu, which are already non-parametric. The genuine blind spot is
> **continuous + non-linear**, where the default `fisherz` / `local_score_BIC` / NOTEARS
> all assume linear-Gaussian relationships and can mis-orient or miss edges.

### Handling Non-linear Continuous Data

If a continuous variable is generated by a non-linear mechanism (e.g. `Y = tanh(wX) + ε`,
`Y = X² + ε`), the linear defaults can miss edges. Use a kernel-based path instead:

#### Choosing the kernel CI test

For `pc` / `fci`, all three kernel-based tests are available:

| Test | Speed / cost | When to use |
|---|---|---|
| `fastkci` | Approximate kernel CI; still expensive across many conditioning sets | Use for low/medium compute tiers with a wall-clock budget. |
| `rcit` | Random-feature approximation with Monte Carlo variability | Cost-aware primary for a high compute tier, normally on seeded rows or a relevant subset. |
| `kci` | **Exact/full kernel test; slowest by far**. Cost grows steeply with sample size and with the number of CI tests in PC/FCI | Use only for small graphs/datasets or final confirmation. A 15-node graph can require thousands of conditional-independence tests; with `n≈5,000`, a single run can take many days. Do not start this blindly on a large graph. |

There is no universal runtime formula because PC/FCI's number of tests depends on the
skeleton, conditioning depth, kernel bandwidth, hardware, and early stopping. As a
rough planning model, kernel matrix work is at least quadratic in `n` per test:

- Halving `n` from 5,000 to 2,500 reduces the per-test kernel work by roughly **4×**.
- Reducing to 1,000 reduces it by roughly **25×** (and can reduce memory pressure much more).
- The total runtime is approximately `number_of_CI_tests × cost_per_test`; the number
  of tests can grow rapidly with node count and graph density. Therefore a 15-node,
  5,000-row KCI job taking tens of hours or days is plausible, not necessarily a hang.

Use:

- **PC / FCI:** `indep_test="fastkci"`, `"rcit"`, or `"kci"` as selected above.
- **GES:** `score_func="local_score_marginal_general"` or `"local_score_cv_general"`
  uses expensive kernel regression. Do not launch it as the global default; reserve it
  for a small targeted confirmation with an explicit timeout.

#### Subsampling / resampling for large nonlinear data

**This is an optional strategy, not a required workflow.** The agent should decide whether
it is worthwhile based on graph size, sample size, expected number of CI tests, available
runtime, and the importance of exact full-data results. For a large nonlinear graph, it
may create a temporary downsampled CSV and run the nonlinear method on `n≈500–1,000`
rows as a speed/variance tradeoff:

1. Keep the original data untouched and record the random seed.
2. Run one bounded primary sample first.
3. Only if an estimand-relevant edge remains unstable, run additional independent samples
   and compare by **edge selection frequency** rather than trusting one sample.
4. Compare stable nonlinear edges with any linear sensitivity baseline; do not claim that
   a downsampled result has the same certainty as a full-data result.

There is no obligation to downsample a dataset merely because it is nonlinear. Prefer
full-data approximate tests when their runtime is acceptable. Use subsampling only when
it provides a meaningful computational benefit and explicitly report the tradeoff.

#### Optional divide-and-conquer strategy

The agent may also choose to combine a fast global baseline with targeted nonlinear
discovery when a global kernel search would be unnecessarily expensive. This is an
**optional optimization, not a mandatory sequence**:

1. Optionally run a fast linear baseline (`pc`/`ges`, or `direct_lingam` when non-Gaussian
   linear) on the full dataset to obtain a broad skeleton and identify suspicious pairs,
   residuals, or clusters.
2. If the diagnosis, domain knowledge, or baseline suggests a localized nonlinear region,
   optionally select a relevant subset, including plausible confounders, mediators, and
   neighboring variables — not only the two endpoints.
3. Run `fastkci` or `rcit` on that subset, optionally after subsampling; use full `kci`
   only for a small graph where confirmation justifies the cost.
4. Compare the targeted nonlinear result against the baseline and preserve the method,
   subset, sample size, seed, and test choice in the rationale.

Do not use this strategy automatically, and do not infer that variables outside the
subset are irrelevant. If targeted and global results disagree, report the disagreement
and investigate rather than silently merging them.


### Parameters Summary

Note that you can use the default value by just simply skipping the parameter in tool calls.

| Parameter | Applicable Methods | Default | Description |
|-----------|-------------------|---------|-------------|
| `alpha` | `pc`, `fci` | `0.05` | Significance level for independence tests. Smaller values → sparser graphs. |
| `indep_test` | `pc`, `fci` | `"auto"` | Test method. `"fisherz"` (linear continuous), `"chisq"`/`"gsq"` (discrete), `"fastkci"` (fast approximate nonlinear continuous), `"rcit"` (randomized approximate nonlinear continuous), `"kci"` (full nonlinear continuous, very slow). `"auto"` detects type (picks chisq/fisherz — never a kernel test). |
| `score_func` | `ges`, `boss`, `grasp` | `"auto"` | Scoring function. `"local_score_BIC"` (linear continuous), `"local_score_BDeu"` (discrete), `"local_score_marginal_general"` / `"local_score_cv_general"` (**non-linear continuous, slow**). `"auto"` detects type (BIC/BDeu). |
| `lambda1` | `notears` | `0.1` | L1 regularization penalty. Higher values → sparser graphs. |
| `w_threshold` | `notears` | `0.3` | Weight threshold. Edges with absolute weights below this are removed. |
| `depth` | `grasp` | `3` | DFS search depth. Larger → more thorough search but slower. |
| `timeout` | all | cost-aware: 90s nonlinear, 120s otherwise | Wall-clock budget in **seconds**. Set a larger value only when the expected information gain justifies it. Every kernel/generalized run must remain bounded, regardless of N. |

#### Timeouts and the degradation ladder

A timeout is a **normal degradation event**, not a failure. When a run times out,
step down the ladder instead of retrying the same configuration:

`kci` → bounded `fastkci`/`rcit` → seeded rows or estimand-relevant subset →
`fisherz` sensitivity baseline (loses nonlinearity — say so if you land here)

or reduce the work: subsample to `n≈500–1,000` (record the seed), or split a large
graph and discover communities separately. Report which rung you ended on and what
was traded away.

---

## Rule: Orient the Estimand-Relevant Core — or Declare Non-Identifiability

Causal inference needs a DAG **only on the estimand-relevant core subgraph**: the
treatment's parents, every T→Y path, and every backdoor path between them. Edges
outside that core may stay undirected — label them as such and move on.

Within the core:

1. **Orient only with evidence** — Meek rules, verifiable temporal order, study
   design, or confirmed domain knowledge (see the Orientation Protocol).
2. **NEVER "orient" a `bidirected` (`↔`) edge.** It does not mean "direction
   unknown" — it means *there is probably a latent common cause*. Turning A↔B
   into A→B fabricates certainty and destroys the single most important warning
   for the adjustment set. Keep it bidirected and disclose it at inference time
   (the inference tool will warn that it was dropped from DoWhy's graph).
3. **If a core edge cannot be oriented by evidence, STOP and declare
   non-identifiability.** Report the consensus skeleton, mark the disputed
   edges with each algorithm's opinion, and state plainly: *"the causal effect
   of T on Y is not identifiable from the current discovery results."* This is
   a **valid, often correct deliverable** — far better than a fully-directed
   DAG whose orientations were invented to satisfy a checklist. Do NOT pick a
   direction to "complete" the graph, and do NOT blend one algorithm's edges
   with another's orientations.

---

## Orientation Protocol (for undirected / partially-directed edges)

Apply these steps **in order**, stopping as soon as an edge is oriented **by evidence**:

### Step 1 — Apply Meek Rules (automatic, no data needed)
Many undirected edges in a valid CPDAG/PDAG can be oriented deterministically by Meek's rules. These rules assume that the skeleton and already-oriented unshielded colliders/non-colliders are valid; do not apply them blindly to arbitrary LLM uncertainty:
- **R1 (avoid a new unshielded collider):** If C→A, C is not adjacent to B, and A—B, orient A→B.
- **R2 (avoid a directed cycle):** If A→C→B and A—B, orient A→B.
- **R3:** If A—B, A—C, C→B, D→B, and C is not adjacent to D, orient A→B.
- **R4 (background-knowledge extension):** If A—B, A—D, D→C→B, and B is not adjacent to D, orient A→B.

### Step 2 — Use Verifiable Priors (temporal order, study design)
Apply temporal ordering or study-design logic:
- Causes must precede effects in time. If A is *measured* before B by design, orient A→B.
- Demographics (Age, Sex) are usually causes, not effects.

> [!CAUTION]
> **Variable names are NOT domain knowledge.** When names are anonymized, disguised,
> or you cannot verify what a variable physically is, inferring direction from what
> a name *seems* to mean is fabrication, not expertise. A "prior" counts only if you
> can point to a verifiable source: measurement timestamps, the data-generating
> protocol, the user's confirmation, or a citable reference. Otherwise leave the
> edge unoriented.

### Step 3 — Web Search for Prior Knowledge
Call `web_search` with a query like `"causal direction [variable A] [variable B] [domain]"`. Cite the source. Only meaningful when the variables' real-world identity is known and verifiable.

### Step 4 — Ask the User (Last Resort)
Ask: *"I found an undirected edge between X and Y. Does X cause Y, or Y cause X?"*

### If nothing resolves a core edge
Leave it unoriented and declare the effect non-identifiable (see the rule above).
A smaller honest answer beats a larger invented one.

---

## Cycle Detection Rule

After orientation, **verify the graph is acyclic**. If a directed cycle exists (A→B→C→A):
1. Identify the edge with the lowest confidence.
2. Re-apply the orientation protocol to that edge.
3. If unresolvable, ask the user. **Never report a cyclic graph as a DAG.**

---

## Practical Checklist Before Reporting

- [ ] Every edge in the **estimand-relevant core subgraph** (treatment's parents, T→Y paths, backdoor paths) is directed **by evidence** — or non-identifiability is explicitly declared instead. Non-core edges may remain undirected if labelled as such.
- [ ] No `bidirected` edge was "oriented" or silently dropped; latent-confounder warnings are carried into the inference step.
- [ ] No directed cycles among the directed edges.
- [ ] Each orientation decision is justified in the rationale by a verifiable source (Meek rule, temporal order, study design, user, citation) — never by a variable name's apparent meaning.
- [ ] Score and metadata are correctly updated in the Scratchpad.
- [ ] **No red flag left unresolved** (opposite directions on a core edge, dense DAG, strong correlation with no path in the graph). If one fired, say how it was resolved — "declared non-identifiable" is a legitimate resolution.
- [ ] **Root-node / no-parent claims cross-checked** with a second algorithm family before they are used to justify an empty adjustment set.
- [ ] Any non-default parameter change (e.g. `alpha`) is disclosed.
- [ ] Final report structure: **Method → Discovery Result → Orientation Logic → Final DAG (or non-identifiability statement)**.

---

## Example Workflow

```
User: "Run causal discovery on data.csv"

1. Call causal_discover(method="pc", data_path="data.csv")
   → Returns CPDAG with edges: A—B, B→C, A→C

2. Check for Meek R1: A→C and B→C, and A—B.
   A and C are adjacent → R1 does not apply.
   Try R2: A→C, C and B non-adjacent? Check graph. R2 doesn't fire.
   No Meek rule resolves A—B.

3. Web search: "causal direction between [A] and [B] in [domain]"
   → Source suggests A temporally precedes B in study protocol.
   → Orient A→B with confidence: "medium", rationale: "temporal ordering from study design"

4. Verify: A→B, B→C, A→C — acyclic DAG. ✅

5. Report the DAG with all edge rationales.
```
