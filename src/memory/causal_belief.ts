/**
 * causal_belief.ts
 *
 * Defines the typed state for the Causal Belief (formerly Causal Scratchpad) —
 * the structured working SCM hypothesis that tracks variables, DAG edges,
 * and effect estimates across the agent's reasoning iterations.
 *
 * Design:
 *   - The causal belief is programmatically managed by the orchestrator.
 *   - It is injected into the system prompt before every LLM call.
 *   - The LLM has FULL write access via `write_causal_belief` (or `write_scratchpad`).
 *   - Fields are minimal and each directly supports causal analysis tools.
 */

import { randomUUID } from "node:crypto";

// ---------------------------------------------------------------------------
// Core Types
// ---------------------------------------------------------------------------

/** A causal variable identified by the agent or user. */
export interface CausalVariable {
	/** Short identifier used in DAG edges (e.g., "X", "age"). */
	name: string;
	/** Human-readable description of what this variable measures. */
	description: string;
	/** Statistical type of the variable. Canonical values: "continuous", "binary", "categorical", "unknown". Custom strings are accepted for non-causal domains. */
	type: "continuous" | "categorical" | "binary" | "unknown" | (string & {});
	/**
	 * Role in the current causal graph.
	 * Canonical values: treatment, outcome, confounder, mediator, instrument, other.
	 * Custom strings are accepted for non-causal domains (e.g. "target", "intervention").
	 */
	role: "treatment" | "outcome" | "confounder" | "mediator" | "instrument" | "other" | (string & {});
	/** Free-text note on where this variable comes from or how to compute it. */
	source: string;
}

/**
 * An edge in the causal graph.
 *
 * Edge types reflect the output of different discovery algorithms:
 *   - directed:            A → B  (certain causal direction)
 *   - undirected:          A — B  (in MEC / CPDAG: direction unresolved)
 *   - bidirected:          A ↔ B  (PAG: latent common cause)
 *   - partially_directed:  A o→ B (PAG: one end uncertain)
 */
export interface CausalEdge {
	from: string;
	to: string;
	edgeType: "directed" | "undirected" | "bidirected" | "partially_directed";
	/** Confidence level assigned by the discovery algorithm or agent reasoning. */
	confidence: "high" | "medium" | "low";
	/** Algorithm or method that produced this edge. */
	method?: string;
	/** Optional note on why this edge is believed to exist. */
	rationale?: string;
}

/** Explicit intervention semantics for an effect estimate. */
export interface EffectIntervention {
	type: string;
	shiftDelta?: number;
	treatmentValue?: string | number | boolean;
	controlValue?: string | number | boolean;
	targetPopulation?: string;
	nRows?: number;
	nRowsDropped?: number;
}

/** Whether an estimate is suitable for the user's causal decision. */
export interface EffectValidity {
	status: "valid" | "caution" | "sensitivity_only" | "inconclusive" | "unknown";
	decisionGrade: boolean;
	reasons: string[];
}

/** A treatment effect estimate produced by the causal inference pipeline. */
export interface EffectEstimate {
	treatment: string;
	outcome: string;
	/** Average causal effect for the intervention below, or null when unavailable. */
	ate: number | null;
	ci_lower?: number;
	ci_upper?: number;
	p_value?: number;
	estimator?: string;
	/** Requested method (for example auto or an explicit DoWhy method). */
	requestedMethod?: string;
	/** Selected estimator family after any automatic model comparison. */
	estimatorFamily?: string;
	/** Whether this is the primary estimate or a sensitivity-only result. */
	estimateRole?: "primary" | "sensitivity_only" | (string & {});
	/** Functional-form assumption used by the selected estimator. */
	functionalForm?: string;
	/** Total/direct/indirect effect requested for this estimate. */
	estimand?: string;
	/** Fully resolved intervention semantics. */
	intervention?: EffectIntervention;
	/** Adjustment variables actually used by the estimator. */
	adjustmentSet?: string[];
	/** Optional effect modifiers used for CATE estimation. */
	effectModifiers?: string[];
	/** Structured subgroup CATE estimates, definitions, support, and uncertainty. */
	cate?: Record<string, unknown>;
	/** Functional-form and estimator comparison retained for later reasoning. */
	modelComparison?: Record<string, unknown> | Array<Record<string, unknown>>;
	/** Positivity, overlap, and extrapolation diagnostics. */
	supportDiagnostics?: Record<string, unknown>;
	/** Decision-grade validity, separate from graph freshness in status. */
	validity?: EffectValidity;
	/** Estimator-aware uncertainty metadata (for example bootstrap completion). */
	uncertainty?: Record<string, unknown>;
	/** Mandatory caveats that must remain visible in the active belief. */
	warnings?: string[];
	/** Whether refutation tests flagged this estimate. */
	refuted: boolean;
	refutation_note?: string;
	/** Full refutation payload retained for later reasoning. */
	refutations?: Record<string, unknown>;
}

function effectIdentityValue(value: unknown): string {
	if (value === undefined || value === null || value === "") return "unspecified";
	return String(value);
}

/**
 * Stable identity for effect upserts.
 *
 * Legacy records remain addressable: missing semantic fields are represented as
 * "unspecified" instead of being assigned a potentially false intervention.
 */
export function effectEstimateKey(
	effect: Pick<EffectEstimate, "treatment" | "outcome"> & Partial<EffectEstimate>,
): string {
	const estimand = effectIdentityValue(effect.estimand ?? "total").toLowerCase();
	const intervention = effectIdentityValue(effect.intervention?.type).toLowerCase();
	const interventionDefinition = intervention === "additive_shift"
		? `delta=${effectIdentityValue(effect.intervention?.shiftDelta)}`
		: intervention === "contrast"
			? `control=${effectIdentityValue(effect.intervention?.controlValue)};treatment=${effectIdentityValue(effect.intervention?.treatmentValue)}`
			: "unspecified";
	const estimatorIdentity = effectIdentityValue(
		effect.requestedMethod ?? effect.estimatorFamily ?? effect.estimator,
	).toLowerCase();
	const adjustmentIdentity = effect.adjustmentSet?.length
		? [...new Set(effect.adjustmentSet.map((value) => value.trim()).filter(Boolean))]
			.sort((left, right) => left.localeCompare(right))
			.join(",")
		: "unspecified";
	const modifierIdentity = effect.effectModifiers?.length
		? [...new Set(effect.effectModifiers.map((value) => value.trim()).filter(Boolean))]
			.sort((left, right) => left.localeCompare(right))
			.join(",")
		: "none";
	return [
		effect.treatment,
		effect.outcome,
		estimand,
		intervention,
		interventionDefinition,
		estimatorIdentity,
		adjustmentIdentity,
		modifierIdentity,
	].join("\0");
}

/**
 * The full causal belief state (SCM hypothesis working memory).
 * Every field is either required for identity/storage or directly consumed
 * by causal analysis tools and the system prompt.
 */
export interface CausalBelief {
	/** Unique identifier for this causal belief state. */
	id: string;
	/** Short title summarizing the task. */
	title: string;
	/** The high-level research question or optimization goal. */
	objective: string;
	/** Path to the dataset associated with this analysis. Used by causal tools. */
	dataPath?: string;
	/** Identified causal variables. Used by causal_discover and causal_infer tools. */
	variables: CausalVariable[];
	/** Edges of the working causal graph. Used by causal_infer for adjustment sets. */
	edges: CausalEdge[];
	/** Effect estimates computed so far. */
	effects: EffectEstimate[];
	/** Free-form working notes. The LLM can track hypotheses, observations, next steps, etc. */
	notes: string;
	/** Graph score indicating quality of the causal structure (larger is better). */
	score?: number;
	/** ISO 8601 timestamp of the last update. */
	lastUpdated: string;
}

/** Backward compatibility alias */
export type CausalScratchpad = CausalBelief;

// ---------------------------------------------------------------------------
// Factory & Utilities
// ---------------------------------------------------------------------------

/** Creates a fresh, empty CausalBelief with default values. */
export function createCausalBelief(objective = ""): CausalBelief {
	return {
		id: randomUUID(),
		title: "",
		objective,
		variables: [],
		edges: [],
		effects: [],
		notes: "",
		lastUpdated: new Date().toISOString(),
	};
}

/** Backward compatibility alias */
export const createScratchpad = createCausalBelief;

function finiteNumberOrNull(value: unknown): number | null {
	if (value === null || value === undefined || value === "") return null;
	const number = Number(value);
	return Number.isFinite(number) ? number : null;
}

function formatFiniteNumber(value: unknown, digits: number): string {
	const number = finiteNumberOrNull(value);
	return number === null ? "N/A" : number.toFixed(digits);
}

function formatEffectIntervention(effect: EffectEstimate): string {
	const intervention = effect.intervention;
	if (!intervention?.type) return "unspecified (legacy)";
	if (intervention.type === "additive_shift") {
		const population = intervention.targetPopulation
			? ` on ${intervention.targetPopulation}`
			: "";
		const rows = intervention.nRows !== undefined ? ` (n=${intervention.nRows})` : "";
		return `additive shift Δ=${formatFiniteNumber(intervention.shiftDelta, 4)}${population}${rows}`;
	}
	if (intervention.type === "contrast") {
		const control = effectIdentityValue(intervention.controlValue);
		const treatment = effectIdentityValue(intervention.treatmentValue);
		return `contrast ${control}→${treatment}`;
	}
	return intervention.type;
}

function compactStructuredValue(value: unknown, maxLength = 600): string {
	if (value === undefined) return "";
	let rendered = "";
	try {
		rendered = JSON.stringify(value);
	} catch {
		rendered = String(value);
	}
	return rendered.length <= maxLength
		? rendered
		: `${rendered.slice(0, maxLength - 1)}…`;
}

function renderCateBeliefSummary(
	label: string,
	cate: Record<string, unknown>,
): string[] {
	const lines: string[] = [];
	const groups = Array.isArray(cate.groups)
		? cate.groups.map(storedRecord).filter((value): value is Record<string, unknown> => value !== null)
		: [];
	const isAdditiveShift = cate.estimand === "conditional_total_additive_shift"
		|| Array.isArray(cate.groups);
	if (!isAdditiveShift) {
		const modifier = String(cate.modifier ?? "unknown");
		const interactionCoefficient = finiteNumberOrNull(cate.interaction_coef);
		const interactionPValue = finiteNumberOrNull(cate.interaction_p_value);
		lines.push(
			`- **${label} legacy linear-interaction CATE (by ${modifier}):** \`${String(cate.formula ?? "not available")}\``,
		);
		lines.push(
			`- **${label} CATE interaction:** coefficient=${interactionCoefficient === null ? "N/A" : interactionCoefficient.toFixed(4)}; p-value=${interactionPValue === null ? "N/A" : interactionPValue.toFixed(4)}`,
		);
		const quantiles = Array.isArray(cate.quantile_estimates)
			? cate.quantile_estimates.map(storedRecord).filter((value): value is Record<string, unknown> => value !== null)
			: [];
		if (quantiles.length > 0) {
			lines.push(`- **${label} legacy CATE examples:**`);
			lines.push(`  | ${modifier} point | CATE |`);
			lines.push("  |---|---:|");
			for (const quantile of quantiles) {
				lines.push(
					`  | ${String(quantile.label ?? "unknown")} = ${String(quantile.modifier_value ?? "?")} | ${formatFiniteNumber(quantile.cate, 4)} |`,
				);
			}
		}
		return lines;
	}
	const identification = storedRecord(cate.identification);
	const reconciliation = storedRecord(cate.reconciliation);
	const conditioningSet = Array.isArray(identification?.conditioning_set)
		? identification.conditioning_set.map(String).join(", ")
		: "unknown";
	lines.push(
		`- **${label} CATE identification:** ${String(identification?.status ?? "unknown")}; conditioning set: ${conditioningSet}`,
	);
	lines.push(
		`- **${label} CATE population estimator:** ${String(cate.population_estimator_family ?? "unknown")} (${String(cate.population_estimate_role ?? "unknown")})`,
	);
	if (reconciliation) {
		lines.push(
			`- **${label} CATE reconciliation:** ${String(reconciliation.status ?? "unknown")}; population ATE=${formatFiniteNumber(reconciliation.population_ate, 4)}; weighted CATE=${formatFiniteNumber(reconciliation.weighted_cate, 4)}; ATE−weighted=${formatFiniteNumber(reconciliation.ate_minus_weighted_cate, 4)}; tolerance=${formatFiniteNumber(reconciliation.effective_absolute_tolerance, 4)}`,
		);
	}
	if (groups.length > 0) {
		lines.push(`- **${label} CATE strata:**`);
		lines.push("  | Stratum | n | Effect | Estimator | Form | Role | Exploratory 95% bootstrap interval | Support | Validity / reasons | Warnings |");
		lines.push("  |---|---:|---:|---|---|---|---:|---|---|---|");
		for (const group of groups) {
			const support = storedRecord(group.support_diagnostics);
			const validity = storedRecord(group.validity);
			const lower = finiteNumberOrNull(group.ci_lower);
			const upper = finiteNumberOrNull(group.ci_upper);
			const interval = lower === null || upper === null
				? "not available"
				: `[${lower.toFixed(4)}, ${upper.toFixed(4)}]`;
			const reasons = Array.isArray(validity?.reasons)
				? validity.reasons.map(String).join("; ")
				: "none";
			const warnings = Array.isArray(group.warnings)
				? group.warnings.map(String).join("; ")
				: "none";
			lines.push(
				`  | ${String(group.label ?? group.id ?? "unknown")} | ${String(group.n_rows ?? "?")} | ${formatFiniteNumber(group.effect, 4)} | ${String(group.estimator_family ?? "unknown")} | ${String(group.functional_form ?? "unknown")} | ${String(group.estimate_role ?? "unknown")} | ${interval} | ${String(support?.status ?? "unknown")} | ${String(validity?.status ?? "unknown")}: ${reasons} | ${warnings} |`,
			);
		}
	}
	return lines;
}

/**
 * Renders the causal belief as a Markdown block for injection into the system prompt.
 * Compact format optimized for LLM consumption — every line carries information.
 */
export function renderBeliefBlock(belief: CausalBelief): string {
	const lines: string[] = [];

	lines.push("## Active Causal Belief (SCM State)");
	lines.push("> This is your editable causal working memory. Rewrite, upsert, remove, or clear entries whenever your hypothesis changes; keep enough provenance to understand why.");
	lines.push("");
	lines.push(`**ID:** \`${belief.id}\`  |  **Updated:** ${belief.lastUpdated}`);
	if (belief.title) lines.push(`**Title:** ${belief.title}`);
	if (belief.objective) lines.push(`**Objective:** ${belief.objective}`);
	if (belief.dataPath) lines.push(`**Data:** \`${belief.dataPath}\``);
	if (belief.score !== undefined) {
		lines.push(`**Graph Score:** ${formatFiniteNumber(belief.score, 2)}`);
	}
	// Variables
	if (belief.variables.length > 0) {
		lines.push("", "### Variables");
		lines.push("| Name | Role | Type | Description |");
		lines.push("|------|------|------|-------------|");
		for (const v of belief.variables) {
			lines.push(`| \`${v.name}\` | ${v.role} | ${v.type} | ${v.description} |`);
		}
	} else {
		lines.push("", "### Variables", "_No variables identified yet._");
	}

	// Causal Graph
	if (belief.edges.length > 0) {
		lines.push("", "### Causal Graph");
		const arrowMap: Record<string, string> = {
			directed: "→", undirected: "—", bidirected: "↔", partially_directed: "o→",
		};
		const edgeStr = belief.edges.map((e) => `\`${e.from}\` ${arrowMap[e.edgeType] ?? "→"} \`${e.to}\` _(${e.method || "manual"})_`).join(", ");
		lines.push(`**Edges:** ${edgeStr}`);

		const undirected = belief.edges.filter((e) => e.edgeType === "undirected");
		if (undirected.length > 0) {
			lines.push(`**⚠ Undirected (need orientation):** ${undirected.map((e) => `${e.from}—${e.to}`).join(", ")}`);
		}
		const bidirected = belief.edges.filter((e) => e.edgeType === "bidirected");
		if (bidirected.length > 0) {
			lines.push(`**⚠ Bidirected (possible latent confounders — never "orient" these):** ${bidirected.map((e) => `${e.from}↔${e.to}`).join(", ")}`);
		}
	} else {
		lines.push("", "### Causal Graph", "_No edges discovered yet._");
	}

	// Effect Estimates
	if (belief.effects.length > 0) {
		lines.push("", "### Effect Estimates");
		lines.push("| Treatment | Outcome | Effect | Estimand / Intervention | Estimator | Role | Validity | Adjustment | Refuted? |");
		lines.push("|-----------|---------|--------|-------------------------|-----------|------|----------|------------|----------|");
		for (const e of belief.effects) {
			const refuted = e.refuted ? `⚠ Yes: ${e.refutation_note ?? ""}` : "No";
			const adjustment = e.adjustmentSet?.length ? e.adjustmentSet.join(", ") : "∅ / unknown";
			const estimator = e.estimatorFamily ?? e.estimator ?? e.requestedMethod ?? "unknown";
			const estimateRole = e.estimateRole ?? "unknown (legacy)";
			const validity = e.validity
				? `${e.validity.decisionGrade ? "decision-grade" : "⚠ not decision-grade"} (${e.validity.status})`
				: "⚠ unknown (legacy)";
			const modifierLabel = e.effectModifiers?.length
				? ` / CATE by ${e.effectModifiers.join(", ")}`
				: "";
			lines.push(
				`| ${e.treatment} | ${e.outcome} | ${formatFiniteNumber(e.ate, 4)} | ${e.estimand ?? "total"} / ${formatEffectIntervention(e)}${modifierLabel} | ${estimator} | ${estimateRole} | ${validity} | ${adjustment} | ${refuted} |`,
			);

			const label = `${e.treatment}→${e.outcome}`;
			if (e.validity?.reasons?.length) {
				lines.push(`- **${label} validity reasons:** ${e.validity.reasons.join("; ")}`);
			}
			if (e.warnings?.length) {
				lines.push(`- **⚠ ${label} warnings:** ${e.warnings.join("; ")}`);
			}
			if (e.modelComparison !== undefined) {
				lines.push(`- **${label} model comparison:** \`${compactStructuredValue(e.modelComparison)}\``);
			}
			if (e.supportDiagnostics !== undefined) {
				lines.push(`- **${label} support diagnostics:** \`${compactStructuredValue(e.supportDiagnostics)}\``);
			}
			if (e.cate !== undefined) {
				lines.push(...renderCateBeliefSummary(label, e.cate));
			}
		}
	}

	// Notes
	if (belief.notes.trim()) {
		lines.push("", "### Notes", belief.notes);
	}

	return lines.join("\n");
}

/** Backward compatibility alias */
export const renderScratchpadBlock = renderBeliefBlock;

/** Serialises to JSON for persistence. */
export function serialiseBelief(belief: CausalBelief): string {
	return JSON.stringify(belief, null, 2);
}

/** Backward compatibility alias */
export const serialiseScratchpad = serialiseBelief;

function storedRecord(value: unknown): Record<string, unknown> | null {
	return typeof value === "object" && value !== null && !Array.isArray(value)
		? value as Record<string, unknown>
		: null;
}

function storedScalar(value: unknown): string | number | boolean | undefined {
	return typeof value === "string" || typeof value === "number" || typeof value === "boolean"
		? value
		: undefined;
}

function normalizeStoredIntervention(record: Record<string, unknown>): EffectIntervention | undefined {
	const raw = storedRecord(record.intervention);
	const legacyType = typeof record.intervention === "string"
		? record.intervention
		: record.intervention_type;
	const type = String(raw?.type ?? raw?.intervention_type ?? legacyType ?? "").trim();
	if (!type) return undefined;
	const shiftDelta = finiteNumberOrNull(
		raw?.shiftDelta ?? raw?.shift_delta ?? record.shiftDelta ?? record.shift_delta,
	);
	const nRows = finiteNumberOrNull(raw?.nRows ?? raw?.n_rows);
	const nRowsDropped = finiteNumberOrNull(raw?.nRowsDropped ?? raw?.n_rows_dropped);
	const targetPopulation = raw?.targetPopulation ?? raw?.target_population;
	return {
		type,
		...(shiftDelta === null ? {} : { shiftDelta }),
		...(storedScalar(raw?.treatmentValue ?? raw?.treatment_value ?? record.treatmentValue ?? record.treatment_value) === undefined
			? {}
			: { treatmentValue: storedScalar(raw?.treatmentValue ?? raw?.treatment_value ?? record.treatmentValue ?? record.treatment_value) }),
		...(storedScalar(raw?.controlValue ?? raw?.control_value ?? record.controlValue ?? record.control_value) === undefined
			? {}
			: { controlValue: storedScalar(raw?.controlValue ?? raw?.control_value ?? record.controlValue ?? record.control_value) }),
		...(typeof targetPopulation === "string" ? { targetPopulation } : {}),
		...(nRows === null ? {} : { nRows }),
		...(nRowsDropped === null ? {} : { nRowsDropped }),
	};
}

function normalizeStoredValidity(value: unknown): EffectValidity {
	const raw = storedRecord(value);
	if (raw) {
		const candidate = String(raw.status ?? "unknown");
		const status: EffectValidity["status"] = ["valid", "caution", "sensitivity_only", "inconclusive", "unknown"].includes(candidate)
			? candidate as EffectValidity["status"]
			: "unknown";
		return {
			status,
			decisionGrade: raw.decisionGrade === true || raw.decision_grade === true,
			reasons: Array.isArray(raw.reasons) ? raw.reasons.map(String) : [],
		};
	}
	if (typeof value === "string") {
		if (value === "decision_ready" || value === "valid") {
			return { status: "valid", decisionGrade: true, reasons: [] };
		}
		if (value === "sensitivity_only") {
			return { status: "sensitivity_only", decisionGrade: false, reasons: [] };
		}
		if (value === "caution") {
			return { status: "caution", decisionGrade: false, reasons: [] };
		}
		if (value === "inconclusive") {
			return { status: "inconclusive", decisionGrade: false, reasons: [] };
		}
	}
	return { status: "unknown", decisionGrade: false, reasons: [] };
}

/** Deserialises from JSON. Returns a fresh belief on parse failure. */
export function deserialiseBelief(raw: string): CausalBelief {
	try {
		const parsed = JSON.parse(raw) as Record<string, unknown>;
		// Drop legacy freshness metadata when loading older persisted beliefs.
		delete parsed.graphRevision;
		if (!("notes" in parsed)) {
			(parsed as Record<string, unknown>).notes = "";
		}
		if (Array.isArray(parsed.effects)) {
			parsed.effects = parsed.effects.map((effect) => {
				if (typeof effect !== "object" || effect === null) return effect;
				const record = effect as Record<string, unknown>;
				const ciLower = finiteNumberOrNull(record.ci_lower);
				const ciUpper = finiteNumberOrNull(record.ci_upper);
				const pValue = finiteNumberOrNull(record.p_value);
				const modelComparison = record.modelComparison ?? record.model_comparison;
				const supportDiagnostics = storedRecord(record.supportDiagnostics ?? record.support_diagnostics) ?? undefined;
				const uncertainty = storedRecord(record.uncertainty) ?? undefined;
				const cate = storedRecord(record.cate) ?? undefined;
				const modifiers = record.effectModifiers ?? record.effect_modifiers;
				const recordWithoutFreshness = { ...record };
				delete recordWithoutFreshness.graphRevision;
				delete recordWithoutFreshness.status;
				delete recordWithoutFreshness.staleReason;
				return {
					...recordWithoutFreshness,
					ate: finiteNumberOrNull(record.ate),
					estimator: typeof record.estimator === "string" ? record.estimator : undefined,
					requestedMethod: typeof (record.requestedMethod ?? record.requested_method) === "string"
						? String(record.requestedMethod ?? record.requested_method)
						: undefined,
					estimatorFamily: typeof (record.estimatorFamily ?? record.estimator_family) === "string"
						? String(record.estimatorFamily ?? record.estimator_family)
						: undefined,
					estimateRole: typeof (record.estimateRole ?? record.estimate_role) === "string"
						? String(record.estimateRole ?? record.estimate_role)
						: undefined,
					functionalForm: typeof (record.functionalForm ?? record.functional_form) === "string"
						? String(record.functionalForm ?? record.functional_form)
						: undefined,
					intervention: normalizeStoredIntervention(record),
					effectModifiers: Array.isArray(modifiers) ? modifiers.map(String) : undefined,
					modelComparison: Array.isArray(modelComparison) || storedRecord(modelComparison)
						? modelComparison
						: undefined,
					supportDiagnostics,
					cate,
					validity: normalizeStoredValidity(record.validity),
					warnings: Array.isArray(record.warnings) ? record.warnings.map(String) : [],
					uncertainty,
					refuted: record.refuted === true,
					...(ciLower === null ? { ci_lower: undefined } : { ci_lower: ciLower }),
					...(ciUpper === null ? { ci_upper: undefined } : { ci_upper: ciUpper }),
					...(pValue === null ? { p_value: undefined } : { p_value: pValue }),
				};
			});
		}
		return parsed as unknown as CausalBelief;
	} catch {
		return createCausalBelief();
	}
}

/** Backward compatibility alias */
export const deserialiseScratchpad = deserialiseBelief;
