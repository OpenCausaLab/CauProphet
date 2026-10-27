/**
 * causal_belief_tools.ts
 *
 * Tool definitions and handlers for managing Causal Beliefs (formerly Causal Scratchpads).
 * The LLM has full write access to the causal belief via write_causal_belief (or write_scratchpad).
 *
 * Tools exposed:
 *   - list_causal_beliefs / list_scratchpads    → list all causal beliefs in the workspace
 *   - open_causal_belief / open_scratchpad      → switch the active causal belief
 *   - create_causal_belief / create_scratchpad  → start a new causal analysis task
 *   - write_causal_belief / write_scratchpad    → full write access to update any belief fields
 *   - export_belief_adj_matrix / export_scratchpad_adj_matrix → export graph as CSV adjacency matrix
 */

import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import {
	createCausalBelief,
	deserialiseBelief,
	effectEstimateKey,
	serialiseBelief,
	renderBeliefBlock,
	type CausalBelief,
	type CausalEdge,
	type CausalVariable,
	type EffectEstimate,
} from "@/memory/causal_belief";
import {
	loadActiveSession,
	persistSession,
} from "./session_tools";
import { atomicWriteFileSync, withFileLockSync } from "@/core/resource_lock";

// ---------------------------------------------------------------------------
// Storage helpers
// ---------------------------------------------------------------------------

/** Returns the causal belief storage directory for the current workspace. */
export function getBeliefDir(): string {
	// Support both .cauprophet/causal_beliefs and legacy .causalclaw/scratchpads
	const newDir = join(process.cwd(), ".cauprophet", "causal_beliefs");
	const legacyDir = join(process.cwd(), ".causalclaw", "scratchpads");
	if (existsSync(legacyDir) && !existsSync(newDir)) {
		return legacyDir;
	}
	return newDir;
}

/** Backward compatibility alias */
export const getScratchpadDir = getBeliefDir;

/**
 * Loads the active causal belief for the current session.
 * The canonical pointer is session.activeBeliefId || session.activeScratchpadId.
 */
export function loadActiveBelief(): CausalBelief | null {
	const session = loadActiveSession();
	const activeId = (session as any)?.activeBeliefId ?? session?.activeScratchpadId;
	if (!activeId) return null;

	const padPath = join(getBeliefDir(), `${activeId}.json`);
	if (!existsSync(padPath)) {
		// Fallback check in legacy directory if exists
		const legacyPath = join(process.cwd(), ".causalclaw", "scratchpads", `${activeId}.json`);
		if (existsSync(legacyPath)) {
			try {
				return deserialiseBelief(readFileSync(legacyPath, "utf-8"));
			} catch {
				return null;
			}
		}
		return null;
	}

	try {
		return deserialiseBelief(readFileSync(padPath, "utf-8"));
	} catch {
		return null;
	}
}

/** Backward compatibility alias */
export const loadActiveScratchpad = loadActiveBelief;

/**
 * Loads a causal belief by ID from disk.
 */
export function loadBeliefById(id: string): CausalBelief | null {
	const padPath = join(getBeliefDir(), `${id}.json`);
	if (existsSync(padPath)) {
		try {
			return deserialiseBelief(readFileSync(padPath, "utf-8"));
		} catch {
			return null;
		}
	}
	const legacyPath = join(process.cwd(), ".causalclaw", "scratchpads", `${id}.json`);
	if (existsSync(legacyPath)) {
		try {
			return deserialiseBelief(readFileSync(legacyPath, "utf-8"));
		} catch {
			return null;
		}
	}
	return null;
}

/** Backward compatibility alias */
export const loadScratchpadById = loadBeliefById;

/**
 * Persists a causal belief to disk and updates the active session's pointer.
 */
export function persistBelief(belief: CausalBelief): void {
	try {
		const dir = getBeliefDir();
		mkdirSync(dir, { recursive: true });
		const beliefPath = join(dir, `${belief.id}.json`);
		withFileLockSync(beliefPath, () => {
			belief.lastUpdated = new Date().toISOString();
			atomicWriteFileSync(beliefPath, `${serialiseBelief(belief)}\n`);

			// Update the worker-scoped session pointer.  A worker's session lookup
			// never touches the browser's active_session.txt pointer.
			const session = loadActiveSession();
			if (session) {
				(session as any).activeBeliefId = belief.id;
				session.activeScratchpadId = belief.id;
				persistSession(session);
			}
		});
	} catch {
		// Non-fatal
	}
}

/** Backward compatibility alias */
export const persistScratchpad = persistBelief;

// ---------------------------------------------------------------------------
// Tool: list_causal_beliefs / list_scratchpads
// ---------------------------------------------------------------------------

export const listCausalBeliefsTool = {
	type: "function" as const,
	function: {
		name: "list_causal_beliefs",
		description:
			"List all causal analysis beliefs (SCM hypotheses) saved in this workspace. " +
			"Returns a summary (id, title, objective, last updated) for each past task. " +
			"Use this to check if a relevant past analysis exists before starting a new one.",
		parameters: {
			type: "object",
			properties: {},
			required: [],
		},
	},
};

export const listScratchpadsTool = {
	type: "function" as const,
	function: {
		name: "list_scratchpads",
		description: "Alias for list_causal_beliefs.",
		parameters: {
			type: "object",
			properties: {},
			required: [],
		},
	},
};

export async function listCausalBeliefs(): Promise<string> {
	const dir = getBeliefDir();
	if (!existsSync(dir)) {
		return "No causal beliefs found in this workspace. Use `create_causal_belief` to start a new analysis.";
	}

	const session = loadActiveSession();
	const activeId = (session as any)?.activeBeliefId ?? session?.activeScratchpadId ?? "";

	const files = readdirSync(dir).filter((f) => /^[0-9a-f-]{36}\.json$/.test(f));
	if (files.length === 0) {
		return "No causal beliefs found in this workspace.";
	}

	const rows: string[] = [
		"# Causal Beliefs in this workspace",
		"",
		"| ID (first 8) | Title | Objective | Updated | Active? |",
		"|---|---|---|---|---|",
	];

	for (const file of files) {
		try {
			const pad = deserialiseBelief(readFileSync(join(dir, file), "utf-8"));
			const shortId = pad.id.slice(0, 8);
			const isActive = pad.id === activeId ? "✓" : "";
			rows.push(
				`| \`${shortId}...\` | ${pad.title || "—"} | ${pad.objective || "—"} | ${pad.lastUpdated.slice(0, 10)} | ${isActive} |`,
			);
		} catch {
			// Skip malformed files
		}
	}

	rows.push(
		"",
		"To inspect a specific causal belief, call `open_causal_belief` with its full ID.",
		"To start a fresh analysis, call `create_causal_belief`.",
	);

	return rows.join("\n");
}

/** Backward compatibility alias */
export const listScratchpads = listCausalBeliefs;

// ---------------------------------------------------------------------------
// Tool: open_causal_belief / open_scratchpad
// ---------------------------------------------------------------------------

export const openCausalBeliefTool = {
	type: "function" as const,
	function: {
		name: "open_causal_belief",
		description:
			"Open a specific causal belief by ID and make it active for this session. " +
			"Returns the full causal belief contents. " +
			"Use this to resume a past analysis or reference findings from a similar task.",
		parameters: {
			type: "object",
			properties: {
				id: {
					type: "string",
					description: "The full or partial UUID of the causal belief to open.",
				},
			},
			required: ["id"],
		},
	},
};

export const openScratchpadTool = {
	type: "function" as const,
	function: {
		name: "open_scratchpad",
		description: "Alias for open_causal_belief.",
		parameters: {
			type: "object",
			properties: {
				id: {
					type: "string",
					description: "The full or partial UUID of the causal belief to open.",
				},
			},
			required: ["id"],
		},
	},
};

export async function openCausalBelief(id: string): Promise<string> {
	const dir = getBeliefDir();
	let resolvedId = id.trim();

	// Support partial ID matching
	if (!existsSync(join(dir, `${resolvedId}.json`))) {
		try {
			const files = readdirSync(dir).filter((f) => f.endsWith(".json"));
			const match = files.find((f) => f.startsWith(resolvedId));
			if (match) {
				resolvedId = match.replace(".json", "");
			} else {
				return `Error: No causal belief found with ID starting with "${id}". Call \`list_causal_beliefs\` to see available IDs.`;
			}
		} catch {
			return `Error: Could not read causal belief directory.`;
		}
	}

	try {
		const padPath = join(dir, `${resolvedId}.json`);
		const pad = deserialiseBelief(readFileSync(padPath, "utf-8"));

		// Update session pointer
		const session = loadActiveSession();
		if (session) {
			(session as any).activeBeliefId = pad.id;
			session.activeScratchpadId = pad.id;
			persistSession(session);
		}

		return [
			`✓ Opened causal belief "${pad.title || pad.id}" — now active for this session.`,
			"",
			renderBeliefBlock(pad),
		].join("\n");
	} catch {
		return `Error: Failed to read causal belief ${resolvedId}.`;
	}
}

/** Backward compatibility alias */
export const openScratchpad = openCausalBelief;

// ---------------------------------------------------------------------------
// Tool: create_causal_belief / create_scratchpad
// ---------------------------------------------------------------------------

export const createCausalBeliefTool = {
	type: "function" as const,
	function: {
		name: "create_causal_belief",
		description:
			"Create a new causal analysis belief (SCM state) and make it active for this session. " +
			"Call this when starting a genuinely new task unrelated to any existing causal belief.",
		parameters: {
			type: "object",
			properties: {
				title: {
					type: "string",
					description: "Short, descriptive title for this analysis task.",
				},
				objective: {
					type: "string",
					description: "The research question or optimization goal.",
				},
				data_path: {
					type: "string",
					description: "Optional: path to the CSV data file.",
				},
			},
			required: ["title", "objective"],
		},
	},
};

export const createScratchpadTool = {
	type: "function" as const,
	function: {
		name: "create_scratchpad",
		description: "Alias for create_causal_belief.",
		parameters: {
			type: "object",
			properties: {
				title: {
					type: "string",
					description: "Short, descriptive title for this analysis task.",
				},
				objective: {
					type: "string",
					description: "The research question or optimization goal.",
				},
				data_path: {
					type: "string",
					description: "Optional: path to the CSV data file.",
				},
			},
			required: ["title", "objective"],
		},
	},
};

export async function createNewCausalBelief(title: string, objective: string, dataPath?: string): Promise<string> {
	const pad = createCausalBelief(objective);
	pad.title = title;
	if (dataPath) pad.dataPath = dataPath;
	persistBelief(pad);
	return [
		`✓ Created new causal belief: "${title}"`,
		`  ID: \`${pad.id}\``,
		`  Objective: ${objective}`,
		"",
		"The causal belief is now active. Use `write_causal_belief` to populate variables, edges, etc.",
	].join("\n");
}

/** Backward compatibility alias */
export const createNewScratchpad = createNewCausalBelief;

// ---------------------------------------------------------------------------
// Tool: write_causal_belief / write_scratchpad
// ---------------------------------------------------------------------------

const beliefWriteParameters = {
	type: "object",
	properties: {
		mode: { type: "string", enum: ["merge", "replace"], description: "merge upserts collection entries; replace replaces each supplied collection." },
		title: { type: "string" },
		objective: { type: "string" },
		data_path: { type: ["string", "null"], description: "Set or clear the associated data path." },
		score: { type: ["number", "null"], description: "Set or clear the graph score." },
		variables: { type: "array", items: { type: ["object", "string"], additionalProperties: true } },
		edges: { type: "array", items: { type: ["object", "array", "string"], additionalProperties: true } },
		effects: { type: "array", items: { type: "object", additionalProperties: true } },
		notes: { type: "string", description: "Replace working notes." },
		append_notes: { type: "string", description: "Append to working notes without resending them." },
		remove_variables: { type: "array", items: { type: "string" }, description: "Remove variables by name; incident edges/effects are removed too." },
		remove_edges: { type: "array", items: { type: "object", properties: { from: { type: "string" }, to: { type: "string" }, edgeType: { type: "string" } }, required: ["from", "to"] } },
		remove_effects: {
			type: "array",
			description:
				"Remove every estimate for a treatment/outcome pair, or add estimand/intervention/estimator/effect-modifier selectors to remove one variant.",
			items: {
				type: "object",
				properties: {
					treatment: { type: "string" },
					outcome: { type: "string" },
					estimand: { type: "string" },
					intervention: { type: ["object", "string"], additionalProperties: true },
					requestedMethod: { type: "string" },
					estimatorFamily: { type: "string" },
					estimator: { type: "string" },
					effectModifiers: { type: "array", items: { type: "string" } },
				},
				required: ["treatment", "outcome"],
			},
		},
		clear_fields: { type: "array", items: { type: "string", enum: ["variables", "edges", "effects", "notes", "score", "data_path"] } },
		content: { type: "string", description: "Legacy JSON-string payload; supplied arrays retain replace semantics." },
	},
	required: [],
};

export const writeCausalBeliefTool = {
	type: "function" as const,
	function: {
		name: "write_causal_belief",
		description:
		"Directly edit your active causal working memory. Pass fields normally; no JSON-inside-a-string is needed. " +
		"Default mode=merge upserts variables by name, edges by endpoints/type, and effects by treatment/outcome/estimand/intervention/estimator/adjustment/effect-modifier identity. " +
		"Use mode=replace for a complete supplied collection; use remove_* or clear_fields for deletion; append_notes for incremental reasoning. " +
		"Graph edits invalidate the old score, but never launch a hidden discovery/scoring job.",
		parameters: beliefWriteParameters,
	},
};

export const writeScratchpadTool = {
	type: "function" as const,
	function: {
		name: "write_scratchpad",
		description: "Alias for write_causal_belief with the same merge/replace/remove/clear controls.",
		parameters: beliefWriteParameters,
	},
};

function record(value: unknown): Record<string, any> | null {
	return typeof value === "object" && value !== null && !Array.isArray(value)
		? value as Record<string, any>
		: null;
}

function hasAny(value: Record<string, any>, keys: string[]): boolean {
	return keys.some((key) => key in value);
}

function normalizeVariable(value: unknown): CausalVariable | null {
	const v = record(value);
	const name = typeof value === "string"
		? value.trim()
		: String(v?.name ?? v?.variable ?? v?.id ?? v?.var_name ?? "").trim();
	if (!name) return null;
	let type = String(v?.type ?? v?.dataType ?? v?.data_type ?? "unknown").trim() || "unknown";
	if (type.toLowerCase() === "ordinal") type = "categorical";
	const role = String(v?.role ?? v?.variable_role ?? v?.var_role ?? "other").trim() || "other";
	return {
		name,
		description: String(v?.description ?? v?.desc ?? v?.definition ?? v?.notes ?? v?.note ?? "").trim(),
		type: type.toLowerCase(),
		role: role.toLowerCase(),
		source: String(v?.source ?? v?.origin ?? "").trim(),
	};
}

function normalizeEdge(value: unknown): CausalEdge | null {
	let from = "";
	let to = "";
	let edgeType = "directed";
	let confidence = "high";
	let rationale = "";
	let method = "";
	if (typeof value === "string") {
		const parts = value.trim().split(/\s*(?:->|→|-|—|↔|o->|o→)\s*/);
		[from, to] = [parts[0] ?? "", parts[1] ?? ""];
		const arrow = value.match(/(?:->|→|-|—|↔|o->|o→)/)?.[0] ?? "→";
		edgeType = arrow === "↔" ? "bidirected" : arrow === "-" || arrow === "—" ? "undirected" : arrow.startsWith("o") ? "partially_directed" : "directed";
	} else if (Array.isArray(value)) {
		from = String(value[0] ?? "");
		to = String(value[1] ?? "");
		edgeType = String(value[2] ?? "directed");
		confidence = String(value[3] ?? "high");
	} else {
		const e = record(value);
		if (!e) return null;
		from = String(e.from ?? e.source ?? e.from_node ?? e.fromNode ?? "");
		to = String(e.to ?? e.target ?? e.to_node ?? e.toNode ?? "");
		edgeType = String(e.edgeType ?? e.edge_type ?? e.type ?? e.direction ?? "directed");
		confidence = String(e.confidence ?? e.confidence_level ?? "high");
		rationale = String(e.rationale ?? "");
		method = String(e.method ?? "");
	}
	from = from.trim();
	to = to.trim();
	if (!from || !to) return null;
	const cleanType = edgeType.toLowerCase().trim();
	const finalType: CausalEdge["edgeType"] =
		["-", "—", "undirected"].includes(cleanType) ? "undirected" :
		["↔", "bidirected"].includes(cleanType) ? "bidirected" :
		["o->", "o→", "partially_directed", "partially-directed"].includes(cleanType) ? "partially_directed" : "directed";
	const cleanConfidence = confidence.toLowerCase().trim();
	return {
		from,
		to,
		edgeType: finalType,
		confidence: (["high", "medium", "low"].includes(cleanConfidence) ? cleanConfidence : "high") as CausalEdge["confidence"],
		method: method.trim(),
		rationale: rationale.trim(),
	};
}

function finiteOptional(value: unknown): number | undefined {
	if (value === undefined || value === null || value === "") return undefined;
	const parsed = Number(value);
	return Number.isFinite(parsed) ? parsed : undefined;
}

function scalarOptional(value: unknown): string | number | boolean | undefined {
	return typeof value === "string" || typeof value === "number" || typeof value === "boolean"
		? value
		: undefined;
}

function normalizeIntervention(value: Record<string, any>): EffectEstimate["intervention"] {
	const nested = record(value.intervention);
	const legacyType = typeof value.intervention === "string"
		? value.intervention
		: value.intervention_type;
	const type = String(nested?.type ?? nested?.intervention_type ?? legacyType ?? "").trim();
	if (!type) return undefined;
	const treatmentValue = scalarOptional(
		nested?.treatmentValue ?? nested?.treatment_value ?? value.treatmentValue ?? value.treatment_value,
	);
	const controlValue = scalarOptional(
		nested?.controlValue ?? nested?.control_value ?? value.controlValue ?? value.control_value,
	);
	const targetPopulation = nested?.targetPopulation ?? nested?.target_population;
	return {
		type,
		...(finiteOptional(nested?.shiftDelta ?? nested?.shift_delta ?? value.shiftDelta ?? value.shift_delta) === undefined
			? {}
			: { shiftDelta: finiteOptional(nested?.shiftDelta ?? nested?.shift_delta ?? value.shiftDelta ?? value.shift_delta) }),
		...(treatmentValue === undefined ? {} : { treatmentValue }),
		...(controlValue === undefined ? {} : { controlValue }),
		...(typeof targetPopulation === "string" ? { targetPopulation } : {}),
		...(finiteOptional(nested?.nRows ?? nested?.n_rows) === undefined
			? {}
			: { nRows: finiteOptional(nested?.nRows ?? nested?.n_rows) }),
		...(finiteOptional(nested?.nRowsDropped ?? nested?.n_rows_dropped) === undefined
			? {}
			: { nRowsDropped: finiteOptional(nested?.nRowsDropped ?? nested?.n_rows_dropped) }),
	};
}

function normalizeValidity(value: unknown): EffectEstimate["validity"] {
	const nested = record(value);
	if (nested) {
		const candidate = String(nested.status ?? "unknown");
		const status = ["valid", "caution", "sensitivity_only", "inconclusive", "unknown"].includes(candidate)
			? candidate as NonNullable<EffectEstimate["validity"]>["status"]
			: "unknown";
		return {
			status,
			decisionGrade: nested.decisionGrade === true || nested.decision_grade === true,
			reasons: Array.isArray(nested.reasons) ? nested.reasons.map(String) : [],
		};
	}
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
	return { status: "unknown", decisionGrade: false, reasons: [] };
}

function structuredOptional(value: unknown): Record<string, unknown> | Array<Record<string, unknown>> | undefined {
	if (Array.isArray(value)) {
		const records = value.map(record).filter((item): item is Record<string, any> => item !== null);
		return records.length === value.length ? records : undefined;
	}
	return record(value) ?? undefined;
}

function normalizeEffect(value: unknown): EffectEstimate | null {
	const e = record(value);
	if (!e) return null;
	const treatment = String(e.treatment ?? e.treatment_node ?? e.cause ?? e.from ?? "").trim();
	const outcome = String(e.outcome ?? e.outcome_node ?? e.effect ?? e.to ?? "").trim();
	if (!treatment || !outcome) return null;
	const ate = finiteOptional(e.ate ?? e.value ?? e.effect_size ?? e.coefficient);
	const adjustment = e.adjustmentSet ?? e.adjustment_set ?? e.common_causes;
	const modifiers = e.effectModifiers ?? e.effect_modifiers;
	return {
		treatment,
		outcome,
		ate: ate ?? null,
		ci_lower: finiteOptional(e.ci_lower ?? e.ciLower ?? e.lower_ci),
		ci_upper: finiteOptional(e.ci_upper ?? e.ciUpper ?? e.upper_ci),
		p_value: finiteOptional(e.p_value ?? e.pValue ?? e.p),
		estimator: e.estimator ? String(e.estimator) : undefined,
		requestedMethod: e.requestedMethod || e.requested_method
			? String(e.requestedMethod ?? e.requested_method)
			: undefined,
		estimatorFamily: e.estimatorFamily || e.estimator_family
			? String(e.estimatorFamily ?? e.estimator_family)
			: undefined,
		estimateRole: e.estimateRole || e.estimate_role
			? String(e.estimateRole ?? e.estimate_role)
			: undefined,
		functionalForm: e.functionalForm || e.functional_form
			? String(e.functionalForm ?? e.functional_form)
			: undefined,
		estimand: e.estimand ? String(e.estimand) : undefined,
		intervention: normalizeIntervention(e),
		adjustmentSet: Array.isArray(adjustment) ? adjustment.map(String) : undefined,
		effectModifiers: Array.isArray(modifiers) ? modifiers.map(String) : undefined,
		cate: record(e.cate) ?? undefined,
		modelComparison: structuredOptional(e.modelComparison ?? e.model_comparison),
		supportDiagnostics: record(e.supportDiagnostics ?? e.support_diagnostics) ?? undefined,
		validity: normalizeValidity(e.validity),
		warnings: Array.isArray(e.warnings) ? e.warnings.map(String) : [],
		uncertainty: record(e.uncertainty) ?? undefined,
		refuted: e.refuted === true,
		refutation_note: e.refutation_note ? String(e.refutation_note) : undefined,
		refutations: record(e.refutations) ?? undefined,
	};
}

function upsertByKey<T>(existing: T[], incoming: T[], key: (value: T) => string): T[] {
	const result = [...existing];
	const positions = new Map(result.map((value, index) => [key(value), index]));
	for (const value of incoming) {
		const position = positions.get(key(value));
		if (position === undefined) {
			positions.set(key(value), result.length);
			result.push(value);
		} else {
			result[position] = value;
		}
	}
	return result;
}

function effectMatchesRemoval(effect: EffectEstimate, target: Record<string, any>): boolean {
	if (
		effect.treatment !== String(target.treatment ?? "") ||
		effect.outcome !== String(target.outcome ?? "")
	) {
		return false;
	}
	if (
		"estimand" in target &&
		(effect.estimand ?? "total").toLowerCase() !== String(target.estimand).toLowerCase()
	) {
		return false;
	}

	const targetIntervention = normalizeIntervention(target);
	if (targetIntervention) {
		if (effect.intervention?.type !== targetIntervention.type) return false;
		const rawIntervention = record(target.intervention);
		const hasShift = hasAny(target, ["shiftDelta", "shift_delta"]) ||
			(rawIntervention ? hasAny(rawIntervention, ["shiftDelta", "shift_delta"]) : false);
		const hasTreatmentValue = hasAny(target, ["treatmentValue", "treatment_value"]) ||
			(rawIntervention ? hasAny(rawIntervention, ["treatmentValue", "treatment_value"]) : false);
		const hasControlValue = hasAny(target, ["controlValue", "control_value"]) ||
			(rawIntervention ? hasAny(rawIntervention, ["controlValue", "control_value"]) : false);
		if (hasShift && effect.intervention?.shiftDelta !== targetIntervention.shiftDelta) return false;
		if (hasTreatmentValue && effect.intervention?.treatmentValue !== targetIntervention.treatmentValue) return false;
		if (hasControlValue && effect.intervention?.controlValue !== targetIntervention.controlValue) return false;
	}

	const requestedMethod = target.requestedMethod ?? target.requested_method;
	if (requestedMethod !== undefined && effect.requestedMethod !== String(requestedMethod)) return false;
	const estimatorFamily = target.estimatorFamily ?? target.estimator_family;
	if (estimatorFamily !== undefined && effect.estimatorFamily !== String(estimatorFamily)) return false;
	if (target.estimator !== undefined && effect.estimator !== String(target.estimator)) return false;
	const targetModifiers = target.effectModifiers ?? target.effect_modifiers;
	if (Array.isArray(targetModifiers)) {
		const expected = [...new Set(targetModifiers.map(String).map((value) => value.trim()).filter(Boolean))].sort();
		const actual = [...new Set((effect.effectModifiers ?? []).map((value) => value.trim()).filter(Boolean))].sort();
		if (expected.join("\0") !== actual.join("\0")) return false;
	}
	return true;
}

export async function writeCausalBelief(input: unknown): Promise<string> {
	const pad = loadActiveBelief();
	if (!pad) return "Error: No active causal belief. Call `create_causal_belief` first.";

	let updates: Record<string, any>;
	let legacyPayload = typeof input === "string";
	try {
		if (typeof input === "string") {
			updates = JSON.parse(input) as Record<string, any>;
		} else {
			const direct = record(input) ?? {};
			if (typeof direct.content === "string") {
				legacyPayload = true;
				updates = { ...(JSON.parse(direct.content) as Record<string, any>), ...direct };
				delete updates.content;
			} else {
				updates = direct;
			}
		}
	} catch (error) {
		return `Error: Failed to parse causal belief update. Details: ${String(error)}`;
	}

	const mode = updates.mode === "replace" || (legacyPayload && updates.mode !== "merge") ? "replace" : "merge";
	if (typeof updates.title === "string") pad.title = updates.title;
	if (typeof updates.objective === "string") pad.objective = updates.objective;
	const dataPath = "data_path" in updates ? updates.data_path : updates.dataPath;
	if (dataPath === null) delete pad.dataPath;
	else if (typeof dataPath === "string") pad.dataPath = dataPath;
	if (updates.score === null) delete pad.score;
	else if (typeof updates.score === "number" && Number.isFinite(updates.score)) pad.score = updates.score;

	const clearFields = new Set(Array.isArray(updates.clear_fields) ? updates.clear_fields.map(String) : []);
	let graphChanged = false;
	if (clearFields.has("variables")) {
		pad.variables = [];
		pad.edges = [];
		pad.effects = [];
		graphChanged = true;
	}
	if (clearFields.has("edges")) {
		pad.edges = [];
		graphChanged = true;
	}
	if (clearFields.has("effects")) pad.effects = [];
	if (clearFields.has("notes")) pad.notes = "";
	if (clearFields.has("score")) delete pad.score;
	if (clearFields.has("data_path")) delete pad.dataPath;

	if (Array.isArray(updates.variables)) {
		const variables = updates.variables.map((input: unknown) => {
			const normalized = normalizeVariable(input);
			if (!normalized || mode === "replace") return normalized;
			const previous = pad.variables.find((value) => value.name === normalized.name);
			const raw = record(input);
			if (!previous || !raw) return normalized;
			return {
				...previous,
				name: normalized.name,
				...(hasAny(raw, ["description", "desc", "definition", "notes", "note"])
					? { description: normalized.description }
					: {}),
				...(hasAny(raw, ["type", "dataType", "data_type"])
					? { type: normalized.type }
					: {}),
				...(hasAny(raw, ["role", "variable_role", "var_role"])
					? { role: normalized.role }
					: {}),
				...(hasAny(raw, ["source", "origin"])
					? { source: normalized.source }
					: {}),
			};
		}).filter((value): value is CausalVariable => value !== null);
		pad.variables = mode === "replace" ? variables : upsertByKey(pad.variables, variables, (value) => value.name);
	}
	const removedVariables = new Set(Array.isArray(updates.remove_variables) ? updates.remove_variables.map(String) : []);
	if (removedVariables.size > 0) {
		pad.variables = pad.variables.filter((value) => !removedVariables.has(value.name));
		const previousEdgeCount = pad.edges.length;
		pad.edges = pad.edges.filter((edge) => !removedVariables.has(edge.from) && !removedVariables.has(edge.to));
		pad.effects = pad.effects.filter((effect) => !removedVariables.has(effect.treatment) && !removedVariables.has(effect.outcome));
		graphChanged ||= pad.edges.length !== previousEdgeCount;
	}

	if (Array.isArray(updates.edges)) {
		const beforeTopology = new Set(pad.edges.map((value) => `${value.from}\0${value.to}\0${value.edgeType}`));
		const edges = updates.edges.map((input: unknown) => {
			const normalized = normalizeEdge(input);
			if (!normalized || mode === "replace") return normalized;
			const previous = pad.edges.find((value) =>
				value.from === normalized.from && value.to === normalized.to && value.edgeType === normalized.edgeType
			);
			const raw = record(input);
			if (!previous || !raw) return normalized;
			return {
				...previous,
				...(hasAny(raw, ["confidence", "confidence_level"])
					? { confidence: normalized.confidence }
					: {}),
				...("method" in raw ? { method: normalized.method } : {}),
				...("rationale" in raw ? { rationale: normalized.rationale } : {}),
			};
		}).filter((value): value is CausalEdge => value !== null);
		pad.edges = mode === "replace"
			? edges
			: upsertByKey(pad.edges, edges, (value) => `${value.from}\0${value.to}\0${value.edgeType}`);
		const afterTopology = new Set(pad.edges.map((value) => `${value.from}\0${value.to}\0${value.edgeType}`));
		graphChanged ||= beforeTopology.size !== afterTopology.size || [...beforeTopology].some((key) => !afterTopology.has(key));
	}
	if (Array.isArray(updates.remove_edges)) {
		for (const candidate of updates.remove_edges) {
			const target = record(candidate);
			if (!target) continue;
			const before = pad.edges.length;
			pad.edges = pad.edges.filter((edge) => !(
				edge.from === String(target.from ?? "") &&
				edge.to === String(target.to ?? "") &&
				(!target.edgeType || edge.edgeType === target.edgeType)
			));
			graphChanged ||= pad.edges.length !== before;
		}
	}

	if (graphChanged) {
		if (!("score" in updates)) delete pad.score;
	}

	if (Array.isArray(updates.effects)) {
		const effects = updates.effects.map((input: unknown) => {
			const normalized = normalizeEffect(input);
			if (!normalized || mode === "replace") return normalized;
			const exactPrevious = pad.effects.find((value) =>
				effectEstimateKey(value) === effectEstimateKey(normalized)
			);
			const pairMatches = pad.effects.filter((value) =>
				value.treatment === normalized.treatment && value.outcome === normalized.outcome
			);
			const previous = exactPrevious ?? (pairMatches.length === 1 ? pairMatches[0] : undefined);
			const raw = record(input);
			if (!previous || !raw) return normalized;
			return {
				...previous,
				treatment: normalized.treatment,
				outcome: normalized.outcome,
				...(hasAny(raw, ["ate", "value", "effect_size", "coefficient"])
					? { ate: normalized.ate }
					: {}),
				...(hasAny(raw, ["ci_lower", "ciLower", "lower_ci"])
					? { ci_lower: normalized.ci_lower }
					: {}),
				...(hasAny(raw, ["ci_upper", "ciUpper", "upper_ci"])
					? { ci_upper: normalized.ci_upper }
					: {}),
				...(hasAny(raw, ["p_value", "pValue", "p"])
					? { p_value: normalized.p_value }
					: {}),
				...("estimator" in raw ? { estimator: normalized.estimator } : {}),
				...(hasAny(raw, ["requestedMethod", "requested_method"])
					? { requestedMethod: normalized.requestedMethod }
					: {}),
				...(hasAny(raw, ["estimatorFamily", "estimator_family"])
					? { estimatorFamily: normalized.estimatorFamily }
					: {}),
				...(hasAny(raw, ["estimateRole", "estimate_role"])
					? { estimateRole: normalized.estimateRole }
					: {}),
				...(hasAny(raw, ["functionalForm", "functional_form"])
					? { functionalForm: normalized.functionalForm }
					: {}),
				...("estimand" in raw ? { estimand: normalized.estimand } : {}),
				...(hasAny(raw, [
					"intervention",
					"intervention_type",
					"shiftDelta",
					"shift_delta",
					"treatmentValue",
					"treatment_value",
					"controlValue",
					"control_value",
				]) ? { intervention: normalized.intervention } : {}),
				...(hasAny(raw, ["adjustmentSet", "adjustment_set", "common_causes"])
					? { adjustmentSet: normalized.adjustmentSet }
					: {}),
				...(hasAny(raw, ["effectModifiers", "effect_modifiers"])
					? { effectModifiers: normalized.effectModifiers }
					: {}),
				...("cate" in raw ? { cate: normalized.cate } : {}),
				...("refuted" in raw ? { refuted: normalized.refuted } : {}),
				...("refutation_note" in raw ? { refutation_note: normalized.refutation_note } : {}),
				...("refutations" in raw ? { refutations: normalized.refutations } : {}),
				...(hasAny(raw, ["modelComparison", "model_comparison"])
					? { modelComparison: normalized.modelComparison }
					: {}),
				...(hasAny(raw, ["supportDiagnostics", "support_diagnostics"])
					? { supportDiagnostics: normalized.supportDiagnostics }
					: {}),
				...("validity" in raw ? { validity: normalized.validity } : {}),
				...("warnings" in raw ? { warnings: normalized.warnings } : {}),
				...("uncertainty" in raw ? { uncertainty: normalized.uncertainty } : {}),
			};
		}).filter((value): value is EffectEstimate => value !== null);
		pad.effects = mode === "replace"
			? effects
			: upsertByKey(pad.effects, effects, effectEstimateKey);
	}
	if (Array.isArray(updates.remove_effects)) {
		for (const candidate of updates.remove_effects) {
			const target = record(candidate);
			if (!target) continue;
			pad.effects = pad.effects.filter((effect) => !effectMatchesRemoval(effect, target));
		}
	}
	if (typeof updates.notes === "string") pad.notes = updates.notes;
	if (typeof updates.append_notes === "string" && updates.append_notes.trim()) {
		pad.notes = pad.notes ? `${pad.notes}\n${updates.append_notes}` : updates.append_notes;
	}

	persistBelief(pad);
	return ["✓ Causal belief updated.", "", renderBeliefBlock(pad)].join("\n");
}

/** Backward compatibility alias */
export const writeScratchpad = writeCausalBelief;

// ---------------------------------------------------------------------------
// Tool: export_belief_adj_matrix / export_scratchpad_adj_matrix
// ---------------------------------------------------------------------------

export const exportBeliefAdjMatrixTool = {
	type: "function" as const,
	function: {
		name: "export_belief_adj_matrix",
		description:
			"Export the active causal belief's graph as a CSV adjacency matrix.",
		parameters: {
			type: "object",
			properties: {
				path: {
					type: "string",
					description: "File path where the CSV should be saved.",
				},
			},
			required: ["path"],
		},
	},
};

export const exportScratchpadAdjMatrixTool = {
	type: "function" as const,
	function: {
		name: "export_scratchpad_adj_matrix",
		description: "Alias for export_belief_adj_matrix.",
		parameters: {
			type: "object",
			properties: {
				path: {
					type: "string",
					description: "File path where the CSV should be saved.",
				},
			},
			required: ["path"],
		},
	},
};

export async function exportBeliefAdjMatrix(path: string): Promise<string> {
	const pad = loadActiveBelief();
	if (!pad) return "Error: No active causal belief.";

	let varNames = pad.variables.map((v) => v.name);
	if (varNames.length === 0) {
		if (pad.edges.length === 0) {
			return "Error: No variables or edges found in the active causal belief.";
		}
		const uniqueNames = new Set<string>();
		for (const edge of pad.edges) {
			uniqueNames.add(edge.from);
			uniqueNames.add(edge.to);
		}
		varNames = Array.from(uniqueNames);
	}

	const n = varNames.length;
	const matrix: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
	const nameToIndex = new Map<string, number>();
	varNames.forEach((name, i) => nameToIndex.set(name, i));

	for (const edge of pad.edges) {
		const fromIdx = nameToIndex.get(edge.from);
		const toIdx = nameToIndex.get(edge.to);
		if (fromIdx !== undefined && toIdx !== undefined) {
			const fromRow = matrix[fromIdx];
			if (!fromRow) continue;
			if (edge.edgeType === "directed") {
				fromRow[toIdx] = 1;
			} else if (edge.edgeType === "undirected") {
				const toRow = matrix[toIdx];
				if (!toRow) continue;
				fromRow[toIdx] = 1;
				toRow[fromIdx] = 1;
			}
		}
	}

	const csvLines: string[] = [];
	csvLines.push("," + varNames.join(","));
	for (let i = 0; i < n; i++) {
		const row = matrix[i];
		if (row) csvLines.push(`${varNames[i]},${row.join(",")}`);
	}

	try {
		writeFileSync(path, csvLines.join("\n"), "utf-8");
		return `✓ Successfully exported adjacency matrix to \`${path}\`.`;
	} catch (err) {
		return `Error: Failed to write to \`${path}\`: ${String(err)}`;
	}
}

/** Backward compatibility alias */
export const exportScratchpadAdjMatrix = exportBeliefAdjMatrix;
