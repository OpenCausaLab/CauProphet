# CauProphet — Project Context (AGENTS.md)

> **What this file is:**
> A user-editable and **agent-editable** context document injected into every session above the
> cache boundary. It serves two purposes:
> 1. **Causal knowledge supplement** — theory and conventions not repeated in the system prompt.
> 2. **Persistent agent memory** — the agent SHOULD update the `## Agent Persona & Preferences`
>    section below to record long-term preferences, the user's name/alias, communication style,
>    and any standing instructions the user has given across sessions.
> **Keep this file concise** — every token costs on every turn.

---

## Agent Persona & Preferences

> **Agent:** Edit this section to record persistent memory about the user and your own style.
> The user's instructions here override generic defaults. When in doubt, ask the user before writing.

- **User alias / preferred name:** _(not set — ask on first introduction and record here)_
- **Communication style:** _(not set — e.g., "concise and technical", "explain everything step by step")_
- **Standing preferences:** Always name variables, nodes, and causal belief entries in the user's active language (e.g., Chinese if the user communicates in Chinese). Never use English abbreviations when the user communicates in another language.
- **Graph display:** Never use ASCII art for DAGs or causal graphs. Use: ① `write_causal_belief` (renders live in UI) → ② Mermaid code block → ③ adjacency list `A → B`.
- **Domain context:** _(e.g., "works in biomedical research", "dataset is always tabular CSV")_

---

## Causal Theory Supplement

### Simpson's Paradox

Aggregated data can reverse the trend seen in subgroups. The correct resolution depends on causal structure, not arithmetic:
- If the grouping variable is a **confounder** (common cause of treatment and outcome) → use stratified data.
- If the grouping variable is a **mediator** (on the causal path T→M→O) → use aggregate data; stratifying blocks the effect.

**Golden rule:** The correct analysis is determined by the DAG. Never choose based on which result looks better.

### Counterfactual Reasoning (3-Step Protocol)

For individual-level "what if X had been different?" questions:
1. **Abduction:** Update beliefs about exogenous noise terms U using observed evidence E=e.
2. **Action:** Apply do(X=x) — graph surgery removing all incoming arrows to X.
3. **Prediction:** Compute Y in the modified, U-updated model.

> Counterfactuals require a **full SCM** (structural equations + noise distribution), not just a DAG + data.
> Use when the user asks "Would this patient have recovered if given treatment?" — not for population ATE.

### Identifiability: When to Refuse

- **Back-door blocked by hidden confounder:** Check front-door criterion (requires observable mediator M
  where X→M is unconfounded and M→Y is unconfounded adjusting for X).
- **Neither criterion applies:** The effect is **non-identifiable** from observational data alone.
  **Report this explicitly — do not produce a numeric estimate.** Suggest RCT or instrumental variable design.
- **PAG from FCI with bidirected edges (A↔B):** Indicates a latent common cause. Do NOT apply
  back-door adjustment to such pairs without further analysis.

### Effect Heterogeneity

| Estimand | Meaning | When to use |
|----------|---------|-------------|
| **ATE** | Population average effect | Default; policy decisions |
| **CATE** | Effect for a subgroup defined by covariates | "How does effect vary by Age?" |
| **ITE** | Effect for a specific individual | Cannot be directly estimated from observational data |

---

## De-Jargon Translation

Always translate final results before presenting to the user:

| ❌ Academic | ✅ Intuitive |
|---|---|
| "Back-door ATE = 0.15 (p<0.01)" | "For every unit increase in [Treatment], [Outcome] improves by ~15%, assuming no unmeasured confounders." |
| "Unmeasured confounders may violate back-door" | "This estimate assumes we haven't missed hidden factors driving both variables." |
| "Placebo refutation passed (p=0.87)" | "The analysis correctly found no effect when treatment was replaced with a random variable — a good sign." |
| "FCI returned PAG with bidirected edges" | "The algorithm detected possible hidden factors affecting multiple variables simultaneously. Extra caution needed." |
| "Effect non-identifiable" | "We cannot estimate this causal effect from the available observational data alone. A controlled experiment would be needed." |