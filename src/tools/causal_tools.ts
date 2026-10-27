/**
 * causal_tools.ts
 *
 * Tool definitions and handlers for Python-bridged causal analysis.
 * Each tool spawns a Python script via `uv run` and parses JSON stdout.
 *
 * Architecture:
 *   TS tool handler → spawn `uv run --project python/ python scripts/xxx.py`
 *   → Python script runs algorithm → JSON stdout
 *   → TS handler parses → returns result string to LLM
 *   → orchestrator can also parse result to update scratchpad
 */

import { join } from "node:path";
import type {
	CausalEdge,
	EffectEstimate,
	EffectIntervention,
	EffectValidity,
} from "@/memory/causal_belief";
import {
	effectEstimateKey,
	renderScratchpadBlock,
} from "@/memory/causal_belief";
import {
	loadActiveScratchpad,
	persistScratchpad,
} from "@/tools/causal_belief_tools";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Resolves the python/ directory (now the workspace root). */
function getPythonDir(): string {
	return process.cwd();
}

/** Resolves the scripts/ directory. */
function getScriptsDir(): string {
	return join(getPythonDir(), "scripts");
}

/**
 * Runs a Python bridge script via uv and returns parsed JSON result.
 * Falls back to system python if uv is not available.
 */
async function runPythonScript(
	scriptName: string,
	args: string[],
	timeoutMs: number = 120_000,
): Promise<{ success: boolean; data?: Record<string, unknown>; error?: string }> {
	const { $ } = await import("bun");
	const scriptPath = join(getScriptsDir(), scriptName);

	// Build the command: prefer uv, fall back to python3
	let stdout: string;
	let stderr: string;

	// Wrap execution with a timeout to prevent algorithms like GES from running indefinitely.
	const withTimeout = <T>(promise: Promise<T>, ms: number): Promise<T> =>
		Promise.race([
			promise,
			new Promise<never>((_, reject) =>
				setTimeout(() => reject(new Error(`Script timed out after ${ms / 1000}s. Use a smaller or estimand-relevant subset, or choose a cheaper method that preserves the assumptions you need.`)), ms),
			),
		]);

	// On timeout, KILL the child process — a rejected promise alone leaves an
	// orphaned Python process burning CPU (this has happened in practice).
	const runWithKill = async <T>(
		procPromise: Promise<T> & { kill?: () => void },
		ms: number,
	): Promise<T> => {
		try {
			return await withTimeout(procPromise, ms);
		} catch (e) {
			try {
				procPromise.kill?.();
			} catch {
				// best-effort
			}
			throw e;
		}
	};

	try {
		// Try uv first (assumes pyproject.toml is in workspace root). `nothrow`
		// is important: Python uses exit 1 for a *structured* algorithm timeout.
		// Treating that as a missing-uv failure used to rerun the same expensive
		// job with system Python.
		const result = await runWithKill(
			$`uv run python ${scriptPath} ${args}`.quiet().nothrow(),
			timeoutMs,
		);
		stdout = result.stdout.toString();
		stderr = result.stderr.toString();
		if (result.exitCode !== 0 && !stdout.trim().startsWith("{")) {
			throw new Error(`uv exited ${result.exitCode}: ${stderr.slice(0, 500)}`);
		}
	} catch (uvError) {
		// Check if it was a timeout
		if (String(uvError).includes("timed out")) {
			return { success: false, error: String(uvError) };
		}
		// Fall back to python directly on Windows, python3 on other platforms
		try {
			const pyCmd = process.platform === "win32" ? "python" : "python3";
			const result = await runWithKill(
				(pyCmd === "python"
					? $`python ${scriptPath} ${args}`.quiet().nothrow()
					: $`python3 ${scriptPath} ${args}`.quiet().nothrow()),
				timeoutMs,
			);
			stdout = result.stdout.toString();
			stderr = result.stderr.toString();
			if (result.exitCode !== 0 && !stdout.trim().startsWith("{")) {
				throw new Error(`${pyCmd} exited ${result.exitCode}: ${stderr.slice(0, 500)}`);
			}
		} catch (pyError) {
			const fallbackCmd = process.platform === "win32" ? "python" : "python3";
			return {
				success: false,
				error: `Failed to run Python script. uv error: ${String(uvError)}. ${fallbackCmd} error: ${String(pyError)}`,
			};
		}
	}

	// Parse JSON from stdout
	try {
		const data = JSON.parse(stdout.trim()) as Record<string, unknown>;
		if (data.error) {
			return { success: false, error: String(data.error), data };
		}
		return { success: true, data };
	} catch {
		return {
			success: false,
			error: `Failed to parse Python output as JSON.\nstdout: ${stdout.slice(0, 500)}\nstderr: ${stderr.slice(0, 500)}`,
		};
	}
}

/**
 * Computes all descendants of a node given a list of directed edges.
 * Used for the hard descendant-check before causal inference.
 */
function getDescendants(
	edges: Array<{ from: string; to: string; edgeType?: string }>,
	node: string,
): Set<string> {
	const children = new Map<string, string[]>();
	for (const e of edges) {
		if (e.edgeType === "directed" || !e.edgeType) {
			if (!children.has(e.from)) children.set(e.from, []);
			children.get(e.from)!.push(e.to);
		}
	}

	const visited = new Set<string>();
	const stack = [...(children.get(node) ?? [])];
	while (stack.length > 0) {
		const current = stack.pop()!;
		if (visited.has(current)) continue;
		visited.add(current);
		for (const child of children.get(current) ?? []) {
			stack.push(child);
		}
	}
	return visited;
}

/**
 * Finds unresolved or latent-confounding edges in the undirected connected
 * components containing the treatment or outcome. Identification must fail
 * closed on these edges even when the caller supplies common_causes manually.
 */
export function getEstimandRelevantGraphBlockers(
	edges: CausalEdge[],
	treatment: string,
	outcome: string,
): CausalEdge[] {
	const adjacency = new Map<string, Set<string>>();
	for (const edge of edges) {
		if (!adjacency.has(edge.from)) adjacency.set(edge.from, new Set());
		if (!adjacency.has(edge.to)) adjacency.set(edge.to, new Set());
		adjacency.get(edge.from)?.add(edge.to);
		adjacency.get(edge.to)?.add(edge.from);
	}

	const relevantNodes = new Set<string>();
	const queue = [treatment, outcome];
	while (queue.length > 0) {
		const current = queue.shift();
		if (!current || relevantNodes.has(current)) continue;
		relevantNodes.add(current);
		for (const neighbour of adjacency.get(current) ?? []) {
			if (!relevantNodes.has(neighbour)) queue.push(neighbour);
		}
	}

	return edges.filter(
		(edge) =>
			edge.edgeType !== "directed" &&
			(relevantNodes.has(edge.from) || relevantNodes.has(edge.to)),
	);
}

export interface CausalIdentificationPlan {
	args: string[];
	usedGraph: boolean;
	manualAdjustmentSet: string[];
	manualSetIgnored: boolean;
}

export interface ResolvedCausalAdjustmentSet {
	adjustmentSet: string[];
	manualSetWarning?: string;
}

function canonicalAdjustmentSet(values: string[]): string[] {
	return [...new Set(values.map((value) => value.trim()).filter(Boolean))]
		.sort((left, right) => left.localeCompare(right));
}

/**
 * The active graph is the canonical identification source. A caller-supplied
 * adjustment set is retained only for mismatch reporting and cannot override
 * graph-based identification.
 */
export function buildCausalIdentificationPlan(
	edges: CausalEdge[] | undefined,
	commonCauses?: string,
): CausalIdentificationPlan {
	const manualAdjustmentSet = canonicalAdjustmentSet(
		commonCauses?.split(",") ?? [],
	);
	if (edges && edges.length > 0) {
		return {
			args: [
				"--graph-edges",
				JSON.stringify(
					edges.map((edge) => ({
							from: edge.from,
							to: edge.to,
							edgeType: edge.edgeType,
						})),
				),
			],
			usedGraph: true,
			manualAdjustmentSet,
			manualSetIgnored: manualAdjustmentSet.length > 0,
		};
	}
	return {
		args: manualAdjustmentSet.length > 0
			? ["--common-causes", manualAdjustmentSet.join(",")]
			: [],
		usedGraph: false,
		manualAdjustmentSet,
		manualSetIgnored: false,
	};
}

/**
 * Resolve the adjustment set that was actually used by Python. In graph mode,
 * a manually supplied set is never persisted as though it drove the estimate.
 */
export function resolveCausalAdjustmentSet(
	plan: CausalIdentificationPlan,
	reportedAdjustmentSet: unknown,
): ResolvedCausalAdjustmentSet {
	const pythonReportedSet = Array.isArray(reportedAdjustmentSet)
		? canonicalAdjustmentSet(reportedAdjustmentSet.map(String))
		: undefined;
	const adjustmentSet = pythonReportedSet ?? (
		plan.usedGraph ? [] : plan.manualAdjustmentSet
	);

	if (!plan.manualSetIgnored) return { adjustmentSet };

	const manualLabel = plan.manualAdjustmentSet.length > 0
		? plan.manualAdjustmentSet.join(", ")
		: "∅";
	const actualLabel = adjustmentSet.length > 0 ? adjustmentSet.join(", ") : "∅";
	const setsAgree = plan.manualAdjustmentSet.length === adjustmentSet.length &&
		plan.manualAdjustmentSet.every((value, index) => value === adjustmentSet[index]);
	return {
		adjustmentSet,
		manualSetWarning:
			`Supplied manual adjustment set [${manualLabel}] was ignored because the active directed graph ` +
			`is the canonical identification source. Python identified [${actualLabel}] from that graph. ` +
			(setsAgree
				? "The two sets agree, but the graph-derived set was used."
				: "The sets differ; the manual set was not used for fitting or saved as the estimate's adjustment set."),
	};
}

/**
 * Formats a causal discovery result into a human-readable string for the LLM.
 */
function formatDiscoveryResult(data: Record<string, unknown>): string {
	const lines: string[] = [];
	const method = data.method as string;
	const variables = data.variables as string[];
	const edges = data.edges as Array<Record<string, unknown>>;
	const metadata = data.metadata as Record<string, unknown> | undefined;
	const params = data.params as Record<string, unknown> | undefined;

	lines.push(`## Causal Discovery Result: ${method}`);
	lines.push(`**Variables (${variables.length}):** ${variables.join(", ")}`);

	if (metadata) {
		lines.push(
			`**Data:** ${metadata.n_samples} samples × ${metadata.n_variables} variables | ` +
			`**Runtime:** ${metadata.elapsed_seconds}s`,
		);
		const encodedCols = metadata.auto_encoded_columns as string[] | undefined;
		if (encodedCols && encodedCols.length > 0) {
			lines.push(`**Auto-encoded columns:** ${encodedCols.join(", ")} _(string → integer via factorize)_`);
		}
		if (metadata.compute_warning) {
			lines.push(`**⚠️ Compute warning:** ${String(metadata.compute_warning)}`);
		}
	}

	if (data.score !== undefined) {
		lines.push(`**Graph Score (normalized BIC/BDeu):** ${Number(data.score).toFixed(2)}`);
	}

	if (params && Object.keys(params).length > 0) {
		const paramStr = Object.entries(params)
			.map(([k, v]) => `${k}=${v}`)
			.join(", ");
		lines.push(`**Parameters:** ${paramStr}`);
	}

	lines.push("");

	// Edge summary
	const directed = edges.filter((e) => e.edgeType === "directed");
	const undirected = edges.filter((e) => e.edgeType === "undirected");
	const bidirected = edges.filter((e) => e.edgeType === "bidirected");
	const partial = edges.filter((e) => e.edgeType === "partially_directed");

	lines.push(`### Edges (${edges.length} total)`);

	if (directed.length > 0) {
		lines.push(`**Directed (${directed.length}):** ${directed.map((e) => `${e.from} → ${e.to}`).join(", ")}`);
	}
	if (undirected.length > 0) {
		lines.push(`**Undirected (${undirected.length}):** ${undirected.map((e) => `${e.from} — ${e.to}`).join(", ")}`);
		lines.push("_These edges are in the Markov equivalence class and need orientation via domain knowledge._");
	}
	if (bidirected.length > 0) {
		lines.push(`**Bidirected (${bidirected.length}):** ${bidirected.map((e) => `${e.from} ↔ ${e.to}`).join(", ")}`);
		lines.push("_These suggest latent common causes._");
	}
	if (partial.length > 0) {
		lines.push(`**Partially directed (${partial.length}):** ${partial.map((e) => `${e.from} o→ ${e.to}`).join(", ")}`);
	}

	if (edges.length === 0) {
		lines.push("_No edges discovered. The variables may be independent given this method and significance level._");
	}

	// Adjacency matrix (compact)
	const adj = data.adjacency_matrix as number[][] | undefined;
	if (adj && adj.length <= 15) {
		lines.push("");
		lines.push("### Adjacency Matrix");
		const header = "     " + variables.map((v) => v.slice(0, 4).padStart(5)).join("");
		lines.push("```");
		lines.push(header);
		for (let i = 0; i < adj.length; i++) {
			const varName = variables[i] ?? `V${i}`;
			const rowData = adj[i] ?? [];
			const row = varName.slice(0, 4).padEnd(5) + rowData.map((v: number) => String(v).padStart(5)).join("");
			lines.push(row);
		}
		lines.push("```");
	}

	return lines.join("\n");
}

/**
 * Formats a causal inference result into a human-readable string for the LLM.
 */
function finiteNumberOrNull(value: unknown): number | null {
	if (value === null || value === undefined || value === "") return null;
	const number = Number(value);
	return Number.isFinite(number) ? number : null;
}

function resultRecord(value: unknown): Record<string, any> | undefined {
	return typeof value === "object" && value !== null && !Array.isArray(value)
		? value as Record<string, any>
		: undefined;
}

function resultScalar(value: unknown): string | number | boolean | undefined {
	return typeof value === "string" || typeof value === "number" || typeof value === "boolean"
		? value
		: undefined;
}

function parseEffectIntervention(value: unknown): EffectIntervention | undefined {
	const raw = resultRecord(value);
	if (!raw) return undefined;
	const type = String(raw.type ?? raw.intervention_type ?? "").trim();
	if (!type) return undefined;
	const shiftDelta = finiteNumberOrNull(raw.shiftDelta ?? raw.shift_delta);
	const nRows = finiteNumberOrNull(raw.nRows ?? raw.n_rows);
	const nRowsDropped = finiteNumberOrNull(raw.nRowsDropped ?? raw.n_rows_dropped);
	const targetPopulation = raw.targetPopulation ?? raw.target_population;
	const treatmentValue = resultScalar(raw.treatmentValue ?? raw.treatment_value);
	const controlValue = resultScalar(raw.controlValue ?? raw.control_value);
	return {
		type,
		...(shiftDelta === null ? {} : { shiftDelta }),
		...(treatmentValue === undefined ? {} : { treatmentValue }),
		...(controlValue === undefined ? {} : { controlValue }),
		...(typeof targetPopulation === "string" ? { targetPopulation } : {}),
		...(nRows === null ? {} : { nRows }),
		...(nRowsDropped === null ? {} : { nRowsDropped }),
	};
}

function parseEffectValidity(value: unknown): EffectValidity {
	const raw = resultRecord(value);
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
		if (value === "valid" || value === "decision_ready") {
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

function formatIntervention(intervention: EffectIntervention | undefined): string {
	if (!intervention) return "unspecified";
	if (intervention.type === "additive_shift") {
		const population = intervention.targetPopulation
			? ` over ${intervention.targetPopulation}`
			: "";
		const rows = intervention.nRows !== undefined ? ` (n=${intervention.nRows})` : "";
		return `additive shift T := T + ${intervention.shiftDelta ?? "?"}${population}${rows}`;
	}
	if (intervention.type === "contrast") {
		return `contrast ${String(intervention.controlValue ?? "?")} → ${String(intervention.treatmentValue ?? "?")}`;
	}
	return intervention.type;
}

export function formatInferenceResult(data: Record<string, unknown>): string {
	const lines: string[] = [];
	const treatment = String(data.treatment ?? "");
	const outcome = String(data.outcome ?? "");
	const ate = finiteNumberOrNull(data.ate);
	const estimator = String(data.estimator ?? data.estimator_family ?? "unknown");
	const estimatorFamily = typeof data.estimator_family === "string" ? data.estimator_family : undefined;
	const estimateRole = typeof data.estimate_role === "string" ? data.estimate_role : "unknown";
	const functionalForm = typeof data.functional_form === "string" ? data.functional_form : undefined;
	const estimand = String(data.estimand ?? "total");
	const intervention = parseEffectIntervention(data.intervention);
	const validity = parseEffectValidity(data.validity);
	const warnings = Array.isArray(data.warnings) ? data.warnings.map(String) : [];
	const refutations = resultRecord(data.refutation_results) ?? {};
	const autoAdjSet = data.auto_adjustment_set as string[] | undefined;
	const cate = data.cate as Record<string, any> | undefined;

	lines.push(`## Causal Inference Result`);
	lines.push(`**Relationship:** ${treatment} → ${outcome}`);
	lines.push(`**Estimand:** ${estimand}`);
	lines.push(`**Intervention:** ${formatIntervention(intervention)}`);
	lines.push(`**Average causal effect:** ${ate === null ? "N/A" : ate.toFixed(4)}`);
	lines.push(`**Estimator:** ${estimator}`);
	if (estimatorFamily) lines.push(`**Estimator family:** ${estimatorFamily}`);
	lines.push(`**Estimate role:** ${estimateRole === "sensitivity_only" ? "⚠️ sensitivity_only" : estimateRole}`);
	if (functionalForm) lines.push(`**Functional-form assessment:** ${functionalForm}`);
	lines.push(
		`**Decision validity:** ${validity.decisionGrade ? "✅ decision-grade" : "⚠️ NOT decision-grade"} (${validity.status})`,
	);
	for (const reason of validity.reasons) lines.push(`- Validity reason: ${reason}`);
	for (const warning of warnings) lines.push(`- ⚠️ ${warning}`);

	if (data.ci_lower !== undefined && data.ci_upper !== undefined) {
		lines.push(`**95% Confidence Interval:** [${Number(data.ci_lower).toFixed(4)}, ${Number(data.ci_upper).toFixed(4)}]`);
	}
	if (data.p_value !== undefined) {
		lines.push(`**p-value (ATE ≠ 0):** ${Number(data.p_value).toExponential(3)}`);
	}

	if (autoAdjSet && autoAdjSet.length > 0) {
		lines.push(`**Auto-identified adjustment set:** ${autoAdjSet.join(", ")}`);
	} else if (autoAdjSet && autoAdjSet.length === 0) {
		lines.push(`**Adjustment set:** _(empty — no confounders detected in graph)_`);
	}

	const modelComparison = resultRecord(data.model_comparison);
	const candidates = Array.isArray(modelComparison?.candidates)
		? modelComparison.candidates.filter((value: unknown) => resultRecord(value)) as Array<Record<string, unknown>>
		: [];
	if (modelComparison) {
		lines.push("", "### Functional-form model comparison");
		if (modelComparison.selected !== undefined) {
			lines.push(`**Selected:** ${String(modelComparison.selected)}`);
		}
		if (candidates.length > 0) {
			lines.push("| Candidate | Role | OOF R² | OOF RMSE | Effect |", "|---|---|---:|---:|---:|");
			for (const candidate of candidates) {
				lines.push(
					`| ${String(candidate.name ?? "unknown")} | ${String(candidate.role ?? "")} | ${String(candidate.oof_r2 ?? "N/A")} | ${String(candidate.oof_rmse ?? "N/A")} | ${String(candidate.effect ?? "N/A")} |`,
				);
			}
		}
		if (modelComparison.relative_rmse_gain !== undefined) {
			lines.push(`- **Flexible-model relative RMSE gain:** ${String(modelComparison.relative_rmse_gain)}`);
		}
		if (modelComparison.flexible_effect_agreement !== undefined) {
			lines.push(`- **Flexible-effect agreement:** ${String(modelComparison.flexible_effect_agreement)}`);
		}
	}

	const support = resultRecord(data.support_diagnostics);
	if (support) {
		lines.push("", "### Intervention support diagnostics");
		lines.push(`- **Status:** ${String(support.status ?? "unknown")}`);
		for (const [label, key] of [
			["Outside observed range", "outside_observed_range_fraction"],
			["Outside 1%–99% range", "outside_01_99_fraction"],
			["Nearest-neighbor unsupported", "nearest_neighbor_unsupported_fraction"],
		] as const) {
			if (support[key] !== undefined) lines.push(`- **${label}:** ${String(support[key])}`);
		}
		if (Array.isArray(support.warnings)) {
			for (const warning of support.warnings) lines.push(`- ⚠️ ${String(warning)}`);
		}
	}

	const uncertainty = resultRecord(data.uncertainty);
	if (uncertainty) {
		lines.push("", "### Uncertainty");
		lines.push(
			`- **Method:** ${String(uncertainty.method ?? "unknown")}; ` +
			`bootstrap ${String(uncertainty.bootstrap_samples_completed ?? 0)}/${String(uncertainty.bootstrap_samples_requested ?? 0)}; ` +
			`SE ${String(uncertainty.standard_error ?? "N/A")}`,
		);
		if (Array.isArray(uncertainty.conditions)) {
			for (const condition of uncertainty.conditions) lines.push(`- Condition: ${String(condition)}`);
		}
	}

	// Lossy graph conversions (dropped bidirected = latent confounder, uncertain
	// orientations treated as certain) — mandatory disclosure, never silent.
	const graphWarnings = data.graph_warnings as string[] | undefined;
	if (graphWarnings && graphWarnings.length > 0) {
		lines.push("", "### ⚠️ Graph conversion warnings (MUST be disclosed in your report)");
		for (const w of graphWarnings) {
			lines.push(`- ${w}`);
		}
	}

	// CATE section — additive-shift results use explicit strata; legacy contrasts
	// retain the historical single linear-interaction representation.
	if (cate) {
		const groups = Array.isArray(cate.groups)
			? cate.groups.filter((value: unknown) => resultRecord(value)) as Array<Record<string, any>>
			: [];
		const isAdditiveShift = cate.estimand === "conditional_total_additive_shift"
			|| Array.isArray(cate.groups);
		if (isAdditiveShift) {
			const stratification = resultRecord(cate.stratification);
			const identification = resultRecord(cate.identification);
			const reconciliation = resultRecord(cate.reconciliation);
			const conditioningSet = Array.isArray(identification?.conditioning_set)
				? identification.conditioning_set.map(String).join(", ")
				: "unknown";
			lines.push("", `### Stratified additive-shift CATE (by ${String(cate.modifier ?? "unknown")})`);
			if (cate.definition) lines.push(`**Definition:** ${String(cate.definition)}`);
			lines.push(`**Modifier role:** ${String(cate.modifier_role ?? "unknown")}`);
			lines.push(`**Conditional identification:** ${String(identification?.status ?? "unknown")} (conditioning set: ${conditioningSet})`);
			lines.push(`**Stratification:** ${String(stratification?.strategy ?? "unknown")}`);
			lines.push(
				`**Population estimator:** ${String(cate.population_estimator_family ?? "unknown")} ` +
				`(${String(cate.population_estimate_role ?? "unknown")})`,
			);
			if (reconciliation) {
				lines.push(
					`**ATE/CATE reconciliation:** ${String(reconciliation.status ?? "unknown")}; population ATE=${String(reconciliation.population_ate ?? "N/A")}; weighted CATE=${String(reconciliation.weighted_cate ?? "N/A")}; ATE−weighted=${String(reconciliation.ate_minus_weighted_cate ?? "N/A")}; tolerance=${String(reconciliation.effective_absolute_tolerance ?? "N/A")}`,
				);
			}
			if (groups.length > 0) {
				lines.push(
					"",
					"| Stratum | n | Effect | Estimator | Form | Role | Exploratory 95% bootstrap interval | Support | Validity / reasons | Warnings |",
					"|---|---:|---:|---|---|---|---:|---|---|---|",
				);
				for (const group of groups) {
					const effect = finiteNumberOrNull(group.effect);
					const lower = finiteNumberOrNull(group.ci_lower);
					const upper = finiteNumberOrNull(group.ci_upper);
					const interval = lower === null || upper === null
						? "not available"
						: `[${lower.toFixed(4)}, ${upper.toFixed(4)}]`;
					const supportStatus = resultRecord(group.support_diagnostics)?.status ?? "unknown";
					const groupValidity = resultRecord(group.validity);
					const reasons = Array.isArray(groupValidity?.reasons)
						? groupValidity.reasons.map(String).join("; ")
						: "none";
					const groupWarnings = Array.isArray(group.warnings)
						? group.warnings.map(String).join("; ")
						: "none";
					lines.push(
						`| ${String(group.label ?? group.id ?? "unknown")} | ${String(group.n_rows ?? "?")} | ${effect === null ? "N/A" : effect.toFixed(4)} | ${String(group.estimator_family ?? "unknown")} | ${String(group.functional_form ?? "unknown")} | ${String(group.estimate_role ?? "unknown")} | ${interval} | ${String(supportStatus)} | ${String(groupValidity?.status ?? "unknown")}: ${reasons} | ${groupWarnings} |`,
					);
				}
			} else {
				lines.push("- No additive-shift CATE strata were returned.");
			}
			const heterogeneity = resultRecord(cate.heterogeneity_summary);
			if (heterogeneity) {
				lines.push(
					`- **Heterogeneity inference:** ${String(heterogeneity.formal_heterogeneity_test ?? "not_performed")}. ${String(heterogeneity.interpretation ?? "")}`,
				);
			}
			if (Array.isArray(cate.warnings)) {
				for (const warning of cate.warnings) lines.push(`- ⚠️ ${String(warning)}`);
			}
		} else {
			const interactionP = finiteNumberOrNull(cate.interaction_p_value);
			const interactionCoefficient = finiteNumberOrNull(cate.interaction_coef);
			const pSig = interactionP === null
				? ""
				: (interactionP < 0.05 ? " ⭐ significant" : " (not significant)");
			lines.push("", `### Legacy linear-interaction CATE (by ${String(cate.modifier ?? "unknown")})`);
			lines.push(`**Formula:** \`${String(cate.formula ?? "not available")}\``);
			lines.push(
				`**Interaction coefficient:** ${interactionCoefficient === null ? "N/A" : interactionCoefficient.toFixed(4)}, ` +
				`p-value: ${interactionP === null ? "N/A" : interactionP.toFixed(4)}${pSig}`,
			);
			const quantiles = Array.isArray(cate.quantile_estimates) ? cate.quantile_estimates : [];
			if (quantiles.length > 0) {
				lines.push("", `| ${String(cate.modifier ?? "Modifier")} | CATE |`, "|---|---:|");
				for (const q of quantiles) {
					const row = resultRecord(q);
					const value = finiteNumberOrNull(row?.cate);
					lines.push(`| ${String(row?.label ?? "")} = ${String(row?.modifier_value ?? "?")} | ${value === null ? "N/A" : value.toFixed(4)} |`);
				}
			}
		}
	}

	lines.push("", "### Estimator-aware refutation tests");
	if (Object.keys(refutations).length === 0) {
		lines.push("- No compatible refutation result was reported; this does not increase confidence.");
	} else {
		for (const [name, res] of Object.entries(refutations)) {
			if (!res || typeof res !== "object") {
				lines.push(`- **${name}**: not available`);
				continue;
			}
			if (res.error) {
				lines.push(`- **${name}**: ❌ Failed to run (${res.error})`);
				continue;
			}
			if (res.passed === null || res.passed === undefined) {
				lines.push(`- **${name}**: not applicable${res.reason ? ` (${res.reason})` : ""}`);
				continue;
			}
			const status = res.passed ? "✅ Passed" : "⚠️ Failed";
			lines.push(`- **${name}**: ${status} (p-value: ${res.p_value?.toFixed?.(4) ?? "N/A"}, new effect: ${res.new_effect?.toFixed?.(4) ?? "N/A"})`);
		}
	}

	return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Tool: causal_diagnose
// ---------------------------------------------------------------------------

export const causalDiagnoseTool = {
	type: "function" as const,
	function: {
		name: "causal_diagnose",
		description:
			"Profile a CSV dataset BEFORE choosing a causal discovery algorithm. " +
			"Reports per-column data type (continuous / discrete / binary / id), " +
			"Gaussianity of continuous columns, whether pairwise relationships are " +
			"LINEAR or NONLINEAR, sample size, missing values, and a concrete algorithm " +
			"recommendation. Always run this first when you are unsure of the data's " +
			"distribution or type. Please read the Skill: causal-discovery (Step 0).",
		parameters: {
			type: "object",
			properties: {
				data_path: {
					type: "string",
					description: "Path to the CSV data file (relative to workspace).",
				},
				treatment: {
					type: "string",
					description:
						"Optional: treatment variable name. When given together with 'outcome', " +
						"the diagnosis includes an ASSOCIATION ANCHOR — the unadjusted marginal " +
						"slope/correlation of T→Y. Use it as the sign reference that your final " +
						"causal estimate must be compared against.",
				},
				outcome: {
					type: "string",
					description: "Optional: outcome variable name (requires 'treatment').",
				},
			},
			required: ["data_path"],
		},
	},
};

/**
 * Formats a data-diagnosis result into a human-readable string for the LLM.
 */
export function formatDiagnosisResult(data: Record<string, unknown>): string {
	const lines: string[] = ["## Data Diagnosis"];

	const datasetType = String(data.dataset_type ?? "unknown");
	const nSamples = data.n_samples as number | undefined;
	const nVars = data.n_variables as number | undefined;
	lines.push(
		`- **Dataset type:** ${datasetType}  |  **Samples:** ${nSamples ?? "?"}  |  **Variables:** ${nVars ?? "?"}`,
	);

	const anyNonlinear = data.any_nonlinear as boolean | undefined;

	// Graded Gaussianity summary (falls back to legacy all_gaussian if absent).
	const gsummary = data.gaussianity_summary as Record<string, unknown> | undefined;
	if (gsummary) {
		const verdict = String(gsummary.verdict ?? "n/a");
		const assessed = gsummary.assessed_columns as number | undefined;
		const gish = gsummary.gaussianish_count as number | undefined;
		const strong = gsummary.strong_non_gaussian_count as number | undefined;
		lines.push(
			`- **Gaussianity verdict:** ${verdict}  (gaussian-ish ${gish ?? "?"} / strong-non-Gaussian ${strong ?? "?"} of ${assessed ?? "?"} assessed)  |  **Nonlinear relationship:** ${anyNonlinear ? "⚠️ YES" : "no"}`,
		);
		const gate = gsummary.lingam_gate as Record<string, unknown> | undefined;
		if (gate) {
			lines.push(
				`- **DirectLiNGAM gate:** ${gate.eligible ? "eligible (strong+broad non-Gaussianity)" : "NOT eligible — do not lead with LiNGAM"}. Normality p-values are supporting evidence only, not the decision.`,
			);
		}
	} else {
		const allGaussian = data.all_gaussian as boolean | null | undefined;
		lines.push(
			`- **Gaussian (continuous cols):** ${allGaussian === null || allGaussian === undefined ? "n/a" : allGaussian ? "yes" : "no"}  |  **Nonlinear relationship detected:** ${anyNonlinear ? "⚠️ YES" : "no"}`,
		);
	}

	const idCols = (data.id_columns as string[] | undefined) ?? [];
	if (idCols.length > 0) {
		lines.push(`- **ID-like columns (drop before discovery):** ${idCols.join(", ")}`);
	}

	const constCols = (data.constant_columns as string[] | undefined) ?? [];
	if (constCols.length > 0) {
		lines.push(`- **Constant columns (excluded from analysis):** ${constCols.join(", ")}`);
	}

	const dupRows = data.duplicate_rows as number | undefined;
	const missRate = data.missing_rate as number | undefined;
	if ((dupRows ?? 0) > 0 || (missRate ?? 0) > 0) {
		lines.push(
			`- **Data quality:** duplicate rows = ${dupRows ?? 0}, overall missing rate = ${((missRate ?? 0) * 100).toFixed(1)}%`,
		);
	}

	const nonNumericCols = (data.non_numeric_columns as string[] | undefined) ?? [];
	if (nonNumericCols.length > 0) {
		lines.push(
			`- **⚠️ Non-numeric (string/categorical) columns:** ${nonNumericCols.join(", ")} — auto-encoded via factorize; some methods may fail on raw strings, and ordinal order (e.g. low/medium/high) is NOT preserved.`,
		);
	}

	// Column breakdown
	const columns = (data.columns as Array<Record<string, unknown>>) ?? [];
	if (columns.length > 0) {
		lines.push("", "### Columns", "| Column | Type | Unique | Missing |", "|---|---|---|---|");
		for (const c of columns) {
			lines.push(
				`| ${String(c.name)} | ${String(c.type)} | ${String(c.n_unique)} | ${String(c.missing)} |`,
			);
		}
	}

	// Per-column Gaussianity detail (anti-blackbox: never report a bare yes/no).
	const gauss = data.gaussianity as Record<string, Record<string, unknown>> | undefined;
	if (gauss && Object.keys(gauss).length > 0) {
		lines.push(
			"",
			"### Gaussianity detail (continuous columns)",
			"| Column | skew | excess_kurtosis | normaltest_p | verdict |",
			"|---|---|---|---|---|",
		);
		for (const [name, g] of Object.entries(gauss)) {
			lines.push(
				`| ${name} | ${g.skew ?? "n/a"} | ${g.excess_kurtosis ?? "n/a"} | ${g.normaltest_p ?? "n/a"} | ${String(g.verdict ?? "n/a")} |`,
			);
		}
		lines.push(
			"_At large N a small normaltest p-value is expected even for clean Gaussian data; trust the skew/kurtosis magnitude (the verdict already does)._",
		);
	}

	// Nonlinear pairs (only surface the flagged ones)
	const pairwise = (data.pairwise as Array<Record<string, unknown>>) ?? [];
	const nonlinearPairs = pairwise.filter((p) => p.nonlinear);
	if (nonlinearPairs.length > 0) {
		lines.push("", "### Nonlinear relationships");
		for (const p of nonlinearPairs) {
			const pair = (p.pair as string[]) ?? [];
			lines.push(
				`- ${pair.join(" ~ ")}: |pearson|=${String(p.pearson_abs)}, |spearman|=${String(p.spearman_abs)}, MI=${String(p.mi_norm)} → nonlinear`,
			);
		}
	}

	// Association anchor (T→Y unadjusted marginal slope — the sign reference).
	const anchor = data.association_anchor as Record<string, unknown> | undefined;
	if (anchor) {
		lines.push("", "### Association anchor (marginal, NOT causal)");
		if (anchor.error) {
			lines.push(`- ⚠️ ${String(anchor.error)}`);
		} else {
			const ci = anchor.slope_ci95 as [number, number] | undefined;
			lines.push(
				`- **Unadjusted slope ${String(anchor.treatment)} → ${String(anchor.outcome)}:** ${String(anchor.unadjusted_slope)} ` +
					`(${String(anchor.slope_sign)}${ci ? `, 95% CI [${ci[0]}, ${ci[1]}]` : ""})`,
			);
			lines.push(
				`- **Pearson:** ${String(anchor.pearson)}  |  **Spearman:** ${String(anchor.spearman)}`,
			);
			lines.push(
				`_${String(anchor.estimand_note ?? "Compare your final adjusted estimate against this anchor; explain any sign flip.")}_`,
			);
		}
	}

	// Top associations (EDA — see the strongest signals in the data).
	const topAssoc = (data.top_associations as Array<Record<string, unknown>>) ?? [];
	if (topAssoc.length > 0) {
		lines.push("", "### Strongest associations (top pairs)");
		lines.push("| Pair | pearson | spearman | MI | nonlinear |");
		lines.push("|---|---|---|---|---|");
		for (const r of topAssoc) {
			const pair = (r.pair as string[]) ?? [];
			lines.push(
				`| ${pair.join(" ~ ")} | ${String(r.pearson)} | ${String(r.spearman)} | ${String(r.mi_norm)} | ${r.nonlinear ? "⚠️ yes" : "no"} |`,
			);
		}
	}

	// Near-deterministic pairs (a hard modelling hazard).
	const nearDet = (data.near_deterministic_pairs as Array<Record<string, unknown>>) ?? [];
	if (nearDet.length > 0) {
		lines.push("", "### ⚠️ Near-deterministic pairs");
		for (const r of nearDet) {
			const pair = (r.pair as string[]) ?? [];
			lines.push(`- **${pair.join(" ~ ")}** (pearson = ${String(r.pearson)}): ${String(r.warning)}`);
		}
	}

	// Compute warning (kernel methods on large N).
	if (data.compute_warning) {
		lines.push("", `> ⚠️ **Compute warning:** ${String(data.compute_warning)}`);
	}

	const effectProfile = resultRecord(data.effect_estimation_profile);
	if (effectProfile) {
		lines.push("", "### Effect-estimation profile");
		if (effectProfile.treatment_type !== undefined) {
			lines.push(`- **Treatment type:** ${String(effectProfile.treatment_type)}`);
		}
		if (effectProfile.requires_explicit_intervention !== undefined) {
			lines.push(
				`- **Explicit intervention required:** ${effectProfile.requires_explicit_intervention ? "YES — define a contrast or additive shift" : "no"}`,
			);
		}
		if (effectProfile.recommended_method !== undefined) {
			lines.push(`- **Recommended inference method:** ${String(effectProfile.recommended_method)}`);
		}
		if (Array.isArray(effectProfile.supported_interventions)) {
			lines.push(`- **Supported interventions:** ${effectProfile.supported_interventions.map(String).join(", ")}`);
		}
		if (effectProfile.post_adjustment_shape_check !== undefined) {
			lines.push(`- **Post-adjustment shape check:** ${String(effectProfile.post_adjustment_shape_check)}`);
		}
		if (effectProfile.linear_estimator_policy !== undefined) {
			lines.push(`- **Linear-estimator policy:** ${String(effectProfile.linear_estimator_policy)}`);
		}
	}

	// Recommendation
	const rec = data.recommendation as Record<string, unknown> | undefined;
	if (rec) {
		lines.push("", "### Recommendation");
		const algos = (rec.algorithms as string[]) ?? [];
		lines.push(`- **Primary run (run one):** ${algos.length > 0 ? algos.join(", ") : "(none — resolve data issues first)"}`);
		if (rec.run_policy === "one-primary") {
			lines.push("- **Run policy:** one bounded primary run; add a cross-check only if it can change the estimand-relevant conclusion.");
		}
		if (rec.requires_cross_check) {
			lines.push(
				"- **⚠️ Cross-check required:** run the baseline methods first and confirm agreement before trusting DirectLiNGAM; a full DAG is not higher confidence.",
			);
		}
		const params = (rec.suggested_params as Record<string, unknown>) ?? {};
		const paramKeys = Object.keys(params);
		if (paramKeys.length > 0) {
			lines.push(
				`- **Suggested parameters:** ${paramKeys.map((k) => `${k}=${String(params[k])}`).join(", ")}`,
			);
		}
		const cost = rec.cost_profile as Record<string, unknown> | undefined;
		if (cost) {
			lines.push(
				`- **Compute profile:** ${String(cost.tier)} (relative kernel work ${String(cost.kernel_screen_work_units)}; default budget ${String(cost.default_timeout_seconds)}s; initial rows ${String(cost.recommended_initial_rows)}). This is not an ETA.`,
			);
		}
		const optionalChecks = (rec.optional_cross_checks as string[]) ?? [];
		if (optionalChecks.length > 0) {
			lines.push(`- **Optional, conditional checks:** ${optionalChecks.join("; ")}`);
		}
		const reasons = (rec.reasons as string[]) ?? [];
		for (const r of reasons) {
			lines.push(`  - ${r}`);
		}
	}

	return lines.join("\n");
}

export async function causalDiagnose(
	dataPath: string,
	treatment?: string,
	outcome?: string,
): Promise<string> {
	const args = ["--data", dataPath];
	if (treatment) args.push("--treatment", treatment);
	if (outcome) args.push("--outcome", outcome);
	const result = await runPythonScript("causal_diagnose.py", args);

	if (!result.success || !result.data) {
		return `Error running data diagnosis: ${result.error}`;
	}

	return formatDiagnosisResult(result.data);
}

// ---------------------------------------------------------------------------
// Tool: causal_discover
// ---------------------------------------------------------------------------

export const causalDiscoverTool = {
	type: "function" as const,
	function: {
		name: "causal_discover",
		description:
			"Run a causal discovery algorithm on a CSV dataset file. " +
			"Returns the discovered causal graph. " +
			"Available methods: pc, fci, ges, notears, direct_lingam, boss, grasp. " +
			"Please read the Skill: causal-discovery for more details. " +
			"Run one budgeted primary method by default; do not interpret a diagnosis list as a checklist. " +
			"The result includes an adjacency matrix, a graph score, and typed edge list that will be written to the Causal Scratchpad.",
		parameters: {
			type: "object",
			properties: {
				method: {
					type: "string",
					enum: ["pc", "fci", "ges", "notears", "direct_lingam", "boss", "grasp"],
					description: "The causal discovery algorithm to use.",
				},
				data_path: {
					type: "string",
					description: "Path to the CSV data file (relative to workspace).",
				},
				alpha: {
					type: "number",
					description:
						"Significance level for independence tests (PC/FCI only). " +
						"Smaller values → sparser graphs. Default: 0.05.",
				},
				indep_test: {
					type: "string",
					enum: ["auto", "fisherz", "chisq", "gsq", "kci", "fastkci", "rcit"],
					description:
						"Independence test method (PC/FCI only). 'auto' detects data type. " +
						"'fisherz' = linear-Gaussian continuous; 'chisq'/'gsq' = discrete; " +
						"'kci' = exact kernel test for NONLINEAR continuous data (VERY slow — see skill); " +
						"'fastkci' / 'rcit' = fast approximate kernel tests for nonlinear data (much faster than kci). " +
						"Default: auto.",
				},
				lambda1: {
					type: "number",
					description: "L1 regularization penalty (NOTEARS only). Default: 0.1.",
				},
				w_threshold: {
					type: "number",
					description: "Edge weight threshold for NOTEARS. Weights below this are zeroed. Default: 0.3.",
				},
				score_func: {
					type: "string",
					enum: [
						"auto",
						"local_score_BIC",
						"local_score_BIC_from_cov",
						"local_score_BDeu",
						"local_score_marginal_general",
						"local_score_cv_general",
					],
					description:
						"Score function for GES / BOSS / GRaSP. " +
						"'auto' picks BIC (continuous) or BDeu (discrete). " +
						"'local_score_BIC' = linear-Gaussian continuous; 'local_score_BDeu' = discrete; " +
						"'local_score_marginal_general' / 'local_score_cv_general' = RKHS kernel scores for " +
						"NONLINEAR continuous data (slow). Default: auto.",
				},
				depth: {
					type: "number",
					description: "DFS search depth for GRaSP (larger = more thorough but slower). Default: 3.",
				},
				timeout: {
					type: "number",
					description:
						"Wall-clock budget in SECONDS. Omit for the cost-aware default: 90s for nonlinear kernel/generalized runs, 120s otherwise. " +
						"On expiry the run stops and returns a structured timeout with a degradation " +
						"suggestion. Set a larger value only when the expected information gain justifies it.",
				},
			},
			required: ["method", "data_path"],
		},
	},
};

// ---------------------------------------------------------------------------
// Tool: causal_infer
// ---------------------------------------------------------------------------

export const causalInferTool = {
	type: "function" as const,
	function: {
		name: "causal_infer",
		description:
			"Identify and estimate a causal effect for an explicit intervention. " +
			"Uses the DAG/DoWhy for identification and method=auto for functional-form-aware estimation. " +
			"For a continuous treatment, define either a fixed contrast or an additive shift; never silently treat 0→1 as 'increase by one'. " +
			"The tool fails closed when the active treatment/outcome graph component contains undirected, partially directed, or bidirected edges; explicit common_causes cannot bypass this. " +
			"For additive-shift CATE, pass one baseline effect modifier; the tool validates conditional exchangeability for adjustment∪modifier, rejects collider-opened backdoors, and reports each stratum plus ATE/weighted-CATE reconciliation. Subgroup bootstrap intervals are exploratory. " +
			"⚠️ HARD RULE: this tool will REJECT your call if common_causes contains ANY " +
			"descendant of the treatment in the current scratchpad DAG. " +
			"Please read the Skill: causal-inference for rules on selecting common causes.",
		parameters: {
			type: "object",
			properties: {
				data_path: {
					type: "string",
					description: "Path to the CSV data file.",
				},
				treatment: {
					type: "string",
					description: "The name of the treatment variable.",
				},
				outcome: {
					type: "string",
					description: "The name of the outcome variable.",
				},
				common_causes: {
					type: "string",
					description:
						"Comma-separated manual adjustment set, used only when there is no active causal graph. " +
						"When the active scratchpad has a graph, that graph is the canonical identification source; " +
						"a supplied manual set cannot override it and any mismatch is reported.",
				},
				graph_edges: {
					type: "array",
					description:
						"Optional explicit directed graph for stateless calls. Each item must contain from, to, " +
						"and optional edgeType/method/confidence. When supplied, this graph is the identification source " +
						"and no active Belief is consulted.",
					items: {
						type: "object",
						properties: {
							from: { type: "string" },
							to: { type: "string" },
							edgeType: {
								type: "string",
								enum: ["directed", "undirected", "bidirected", "partially_directed"],
							},
							method: { type: "string" },
							confidence: { type: "string" },
						},
						required: ["from", "to"],
					},
				},
				estimand: {
					type: "string",
					enum: ["total"],
					default: "total",
					description:
						"Total effect including all causal pathways (the only currently supported estimand). Adjustment variables must not include treatment descendants or mediators. Direct and indirect effects are not yet implemented and are rejected rather than silently mislabeled.",
				},
				effect_modifiers: {
					type: "string",
					description:
						"Optional baseline effect modifier for CATE. Additive-shift CATE accepts exactly one modifier per call, creates transparent quantile or observed-level strata, and requires graph-validated conditional exchangeability after conditioning on adjustment∪modifier; being merely non-descendant is insufficient. Legacy contrast CATE retains its linear-interaction behavior.",
				},
				method: {
					type: "string",
					default: "auto",
					description:
						"Estimator selection policy. Use 'auto' (default) to compare linear and flexible outcome models and apply the functional-form gate. Explicit legacy DoWhy methods remain available as sensitivity analyses.",
				},
				intervention_type: {
					type: "string",
					enum: ["auto", "contrast", "additive_shift"],
					default: "auto",
					description:
						"Intervention semantics. additive_shift estimates mean[m(T+delta,Z)-m(T,Z)]. contrast compares treatment_value against control_value. auto infers only when the supplied values or treatment type make the intervention unambiguous.",
				},
				shift_delta: {
					type: "number",
					description:
						"Delta for intervention_type='additive_shift' (for example 1 means increase every observed treatment value by one unit).",
				},
				treatment_value: {
					type: ["string", "number"],
					description:
						"The treatment value for intervention_type='contrast'. " +
						"Required when the treatment column has more than 2 string categories. " +
						"For binary string treatments this is auto-detected.",
				},
				control_value: {
					type: ["string", "number"],
					description:
						"The control value for intervention_type='contrast'. " +
						"Required when the treatment column has more than 2 string categories.",
				},
			},
			required: ["data_path", "treatment", "outcome"],
		},
	},
};

export type CausalInferHandlerArguments = [
	dataPath: string,
	treatment: string,
	outcome: string,
	commonCauses: string,
	effectModifiers: string,
	method: string,
	treatmentValue: string,
	controlValue: string,
	estimand: string,
	interventionType: string,
	shiftDelta: string,
];

export interface CausalToolRuntime {
	/** When false, handlers must not load or persist the active Belief. */
	useCausalBelief?: boolean;
	/** Explicit graph supplied by the caller for stateless identification. */
	graphEdges?: CausalEdge[];
}

function toolArgumentString(value: unknown): string {
	return value === undefined || value === null ? "" : String(value);
}

export function normalizeCausalEstimand(value: unknown): string {
	return toolArgumentString(value).trim().toLowerCase() || "total";
}

/** Single source of truth for orchestrator → causalInfer positional mapping. */
export function mapCausalInferToolArgs(
	args: Record<string, unknown>,
): CausalInferHandlerArguments {
	return [
		toolArgumentString(args.data_path),
		toolArgumentString(args.treatment),
		toolArgumentString(args.outcome),
		toolArgumentString(args.common_causes),
		toolArgumentString(args.effect_modifiers),
		toolArgumentString(args.method),
		toolArgumentString(args.treatment_value),
		toolArgumentString(args.control_value),
		normalizeCausalEstimand(args.estimand),
		toolArgumentString(args.intervention_type),
		toolArgumentString(args.shift_delta),
	];
}

export async function causalInfer(
	dataPath: string,
	treatment: string,
	outcome: string,
	commonCauses?: string,
	effectModifiers?: string,
	method?: string,
	treatmentValue?: string | number,
	controlValue?: string | number,
	estimand?: string,
	interventionType?: string,
	shiftDelta?: string | number,
	runtime: CausalToolRuntime = {},
): Promise<string> {
	const effectType = estimand?.trim().toLowerCase() || "total";
	const modifierNames = (effectModifiers ?? "")
		.split(",")
		.map((value) => value.trim())
		.filter(Boolean);

	// ── HARD CHECK: Descendant guard (total effect only) ────────────────────
	// When estimating TOTAL effect, the adjustment set must NOT contain any
	// descendant of treatment. This is enforced programmatically to prevent
	// the agent from accidentally adjusting for mediators/descendants.
	const pad = runtime.useCausalBelief === false
		? undefined
		: loadActiveScratchpad();
	const graphEdges = runtime.graphEdges ?? pad?.edges;
	if (graphEdges && graphEdges.length > 0) {
		const graphBlockers = getEstimandRelevantGraphBlockers(
			graphEdges,
			treatment,
			outcome,
		);
		if (graphBlockers.length > 0) {
			const arrowMap: Record<CausalEdge["edgeType"], string> = {
				directed: "→",
				undirected: "—",
				bidirected: "↔",
				partially_directed: "o→",
			};
			const blockerList = graphBlockers
				.map((edge) => `${edge.from} ${arrowMap[edge.edgeType]} ${edge.to} (${edge.edgeType})`)
				.join(", ");
			const hasBidirected = graphBlockers.some((edge) => edge.edgeType === "bidirected");
			return (
				`⛔ **BLOCKED — Estimand-Relevant Graph Ambiguity**\n\n` +
				`The active Causal Belief contains unresolved or latent-confounding edges in the connected component of ` +
				`the treatment/outcome: ${blockerList}.\n\n` +
				`A manually supplied \`common_causes\` list cannot bypass this identification failure. ` +
				`Undirected or partially directed edges must first be resolved in the Belief using defensible evidence. ` +
				(hasBidirected
					? `A bidirected edge represents possible latent confounding; adding observed covariates does not by itself remove that threat. ` +
						`Revise the identification strategy or record why the latent-confounding claim is no longer supported before estimating.\n\n`
					: "\n\n") +
				`No effect estimate was computed or saved.`
			);
		}
	}
	if (graphEdges && graphEdges.length > 0 && commonCauses && commonCauses.trim().length > 0 && effectType === "total") {
		const descendants = getDescendants(graphEdges, treatment);
		const causes = commonCauses.split(",").map((s) => s.trim()).filter(Boolean);
		const invalidCauses = causes.filter((c) => descendants.has(c));
		if (invalidCauses.length > 0) {
			return (
				`⛔ **BLOCKED — Descendant Adjustment Violation**\n\n` +
				`Cannot include [${invalidCauses.join(", ")}] in the adjustment set for **total effect** estimation — ` +
				`they are descendants of treatment "${treatment}" in the current DAG.\n\n` +
				`**Why this is wrong:** Adjusting for a treatment descendant (mediator or downstream node) ` +
				`blocks causal pathways and biases the total effect estimate. ` +
				`The resulting estimate would be a partial/direct effect, not the total effect the user asked for.\n\n` +
				`**How to fix:**\n` +
				`1. Remove [${invalidCauses.join(", ")}] from \`common_causes\` and retry for the **total effect**.\n` +
				`2. If you need a direct effect, state that the current tool does not implement it; do not relabel this adjusted estimate as direct.`
			);
		}
	}
	if (graphEdges && graphEdges.length > 0 && modifierNames.length > 0 && effectType === "total") {
		const descendants = getDescendants(graphEdges, treatment);
		const invalidModifiers = modifierNames.filter((name) => descendants.has(name));
		if (invalidModifiers.length > 0) {
			return (
				`⛔ **BLOCKED — Post-Treatment Effect Modifier**\n\n` +
				`Cannot condition a baseline total-effect CATE on [${invalidModifiers.join(", ")}]: ` +
				`the active DAG makes ${invalidModifiers.length === 1 ? "it a descendant" : "them descendants"} of treatment "${treatment}".\n\n` +
				`Choose one genuinely pre-treatment subgroup variable. No effect estimate was computed or saved.`
			);
		}
	}

	const args: string[] = [
		"--data", dataPath,
		"--treatment", treatment,
		"--outcome", outcome,
	];

	const identificationPlan = buildCausalIdentificationPlan(
		graphEdges,
		commonCauses,
	);
	args.push(...identificationPlan.args);

	const requestedMethod = method?.trim() || "auto";
	const requestedInterventionType = interventionType?.trim() || "auto";
	if (effectModifiers) args.push("--effect-modifiers", effectModifiers);
	args.push("--method", requestedMethod);
	if (treatmentValue !== undefined && String(treatmentValue) !== "") {
		args.push("--treatment-value", String(treatmentValue));
	}
	if (controlValue !== undefined && String(controlValue) !== "") {
		args.push("--control-value", String(controlValue));
	}
	args.push("--estimand", effectType);
	args.push("--intervention-type", requestedInterventionType);
	if (shiftDelta !== undefined && String(shiftDelta) !== "") {
		args.push("--shift-delta", String(shiftDelta));
	}

	const result = await runPythonScript("causal_infer.py", args);

	if (!result.success || !result.data) {
		return `Error running causal inference: ${result.error}`;
	}
	const reportedAdjustmentSet = result.data.auto_adjustment_set ??
		result.data.adjustment_set;
	const resolvedAdjustment = resolveCausalAdjustmentSet(
		identificationPlan,
		reportedAdjustmentSet,
	);

	// ── Provenance banner: the adjustment set is only as valid as the graph it
	// came from. State which algorithm(s) produced that graph, so the estimate
	// is never presented as graph-agnostic truth.
	let provenanceBanner = "";
	const usedGraph = identificationPlan.usedGraph;
	if (usedGraph && (pad || graphEdges)) {
		const methodCounts = new Map<string, number>();
		for (const e of graphEdges ?? []) {
			const m = e.method || "manual";
			methodCounts.set(m, (methodCounts.get(m) ?? 0) + 1);
		}
		const desc = [...methodCounts.entries()].map(([m, n]) => `${m} (${n})`).join(", ");
		const graphSource = pad ? "current Causal Belief graph" : "explicit graph_edges input";
		provenanceBanner =
			`> ℹ️ **Provenance:** the auto-identified adjustment set below was derived from the ` +
			`${graphSource} — edges by: ${desc}. Its validity is CONDITIONAL on that ` +
			`graph being correct. If the graph came from a single algorithm or was hand-spliced, ` +
			`say so in your report.\n\n`;
	}
	if (resolvedAdjustment.manualSetWarning) {
		provenanceBanner +=
			`> ⚠️ **Identification source:** ${resolvedAdjustment.manualSetWarning}\n\n`;
	}
	const displayData = usedGraph
		? { ...result.data, auto_adjustment_set: resolvedAdjustment.adjustmentSet }
		: result.data;

	// ── Auto-persist result into the active scratchpad ──────────────────────
	// Check whether any refutation flagged the estimate as unreliable.
	const refutations = result.data.refutation_results as Record<string, any> | undefined;
	const anyFailed = refutations
		? Object.values(refutations).some(
				(r) =>
					typeof r === "object" &&
					r !== null &&
					"passed" in r &&
					r.passed === false,
		  )
		: false;

	const refutationNote = anyFailed
		? Object.entries(refutations ?? {})
				.filter(([, r]) => typeof r === "object" && r !== null && r.passed === false)
				.map(([name]) => name)
				.join(", ")
		: undefined;
	const reportedIntervention = parseEffectIntervention(result.data.intervention);
	const fallbackShiftDelta = finiteNumberOrNull(shiftDelta);
	const intervention = reportedIntervention ?? (requestedInterventionType === "auto"
		? undefined
		: {
			type: requestedInterventionType,
			...(requestedInterventionType === "additive_shift" && fallbackShiftDelta !== null
				? { shiftDelta: fallbackShiftDelta }
				: {}),
			...(requestedInterventionType === "contrast" && treatmentValue !== undefined && String(treatmentValue) !== ""
				? { treatmentValue }
				: {}),
			...(requestedInterventionType === "contrast" && controlValue !== undefined && String(controlValue) !== ""
				? { controlValue }
				: {}),
		} satisfies EffectIntervention);
	const validity = parseEffectValidity(result.data.validity);

	const estimate: EffectEstimate = {
		treatment,
		outcome,
		ate: finiteNumberOrNull(result.data.ate),
		ci_lower: result.data.ci_lower as number | undefined,
		ci_upper: result.data.ci_upper as number | undefined,
		p_value: result.data.p_value as number | undefined,
		estimator: result.data.estimator as string | undefined,
		requestedMethod,
		estimatorFamily: result.data.estimator_family as string | undefined,
		estimateRole: result.data.estimate_role as string | undefined,
		functionalForm: result.data.functional_form as string | undefined,
		estimand: typeof result.data.estimand === "string" ? result.data.estimand : effectType,
		intervention,
		adjustmentSet: resolvedAdjustment.adjustmentSet,
		effectModifiers: modifierNames,
		cate: resultRecord(result.data.cate),
		modelComparison: resultRecord(result.data.model_comparison),
		supportDiagnostics: resultRecord(result.data.support_diagnostics),
		validity,
		warnings: [
			...(Array.isArray(result.data.warnings) ? result.data.warnings.map(String) : []),
			...(resolvedAdjustment.manualSetWarning
				? [resolvedAdjustment.manualSetWarning]
				: []),
		],
		uncertainty: resultRecord(result.data.uncertainty),
		refuted: anyFailed,
		refutation_note: refutationNote,
		refutations,
	};

	// Reuse `pad` from the descendant check above (already loaded). Stateless
	// calls intentionally return only the estimate and never persist it.
	if (pad && runtime.useCausalBelief !== false) {
		// Preserve estimates for distinct estimands, interventions, and estimators.
		const existingIdx = pad.effects.findIndex(
			(e) => effectEstimateKey(e) === effectEstimateKey(estimate),
		);
		if (existingIdx >= 0) {
			pad.effects[existingIdx] = estimate;
		} else {
			pad.effects.push(estimate);
		}
		persistScratchpad(pad);

		return [
			provenanceBanner + formatInferenceResult(displayData),
			"",
			"---",
			"_Effect estimate saved to active scratchpad. Updated state:_",
			"",
			renderScratchpadBlock(pad),
		].join("\n");
	}

	return provenanceBanner + formatInferenceResult(displayData);
}

export async function causalDiscover(
	method: string,
	dataPath: string,
	alpha?: string,
	indepTest?: string,
	lambda1?: string,
	wThreshold?: string,
	scoreFunc?: string,
	depth?: string,
	timeout?: string,
	runtime: CausalToolRuntime = {},
): Promise<string> {
	const args: string[] = ["--method", method, "--data", dataPath];

	if (alpha) args.push("--alpha", alpha);
	if (indepTest) args.push("--indep-test", indepTest);
	if (lambda1) args.push("--lambda1", lambda1);
	if (wThreshold) args.push("--w-threshold", wThreshold);
	if (scoreFunc) args.push("--score-func", scoreFunc);
	if (depth) args.push("--depth", depth);
	const nonlinearRun =
		["kci", "fastkci", "rcit"].includes(indepTest ?? "") ||
		(scoreFunc ?? "").includes("general");
	const requestedTimeout = timeout ? Number.parseInt(timeout, 10) : Number.NaN;
	const timeoutSec = Number.isFinite(requestedTimeout) && requestedTimeout > 0
		? requestedTimeout
		: nonlinearRun
			? 90
			: 120;
	args.push("--timeout", String(timeoutSec));

	// Python owns the algorithm budget. TypeScript only supplies a short startup
	// and JSON-serialization grace period instead of silently adding 90 seconds.
	const tsTimeoutMs = (timeoutSec + 15) * 1000;

	const result = await runPythonScript("causal_discover.py", args, tsTimeoutMs);

	if (!result.success || !result.data) {
		// A Python-side timeout arrives as {error, timeout: true} — surface the
		// degradation ladder instead of a bare failure string.
		const d = result.data;
		if (d?.timeout) {
			return (
				`⏱️ **Discovery timed out (${method})** — ${String(d.error)}\n\n` +
				`${String(d.suggestion ?? "Use a faster test/score or subsample the data.")}`
			);
		}
		return `Error running causal discovery (${method}): ${result.error}`;
	}

	// ── Auto-persist result into the active scratchpad ──────────────────────
	const pad = runtime.useCausalBelief === false
		? undefined
		: loadActiveScratchpad();
	if (pad) {
		const extracted = extractEdgesFromResult(result.data);

		// ── Overwrite guardrail: the edge list is REPLACED, not merged. Before
		// destroying the previous graph, record what is being overwritten (and
		// warn in the tool response) so cross-algorithm comparison is never
		// silently lost — a cross-check that overwrites its own baseline is no
		// cross-check at all.
		let overwriteWarning = "";
		if (pad.edges.length > 0) {
			const prevMethods = new Map<string, number>();
			for (const e of pad.edges) {
				const m = e.method || "manual";
				prevMethods.set(m, (prevMethods.get(m) ?? 0) + 1);
			}
			const prevDesc = [...prevMethods.entries()]
				.map(([m, n]) => `${m} (${n} edges)`)
				.join(", ");
			const prevEdgeSet = new Set(pad.edges.map((e) => `${e.from}|${e.to}|${e.edgeType}`));
			const newEdgeSet = new Set(extracted.edges.map((e) => `${e.from}|${e.to}|${e.edgeType}`));
			const kept = [...newEdgeSet].filter((k) => prevEdgeSet.has(k)).length;
			const previousEdges = pad.edges.map((edge) =>
				`${edge.from}${edge.edgeType === "directed" ? "→" : edge.edgeType === "undirected" ? "—" : edge.edgeType === "bidirected" ? "↔" : "o→"}${edge.to}`,
			).join(", ");
			const noteLine =
				`[graph-history] ${method} run replaced previous graph ` +
				`(${prevDesc}); ${kept}/${extracted.edges.length} new edges also present before. ` +
				`Previous edges: ${previousEdges || "(none)"}.`;
			pad.notes = pad.notes ? `${pad.notes}\n${noteLine}` : noteLine;

			overwriteWarning =
				`⚠️ **Scratchpad graph OVERWRITTEN:** the previous graph (${prevDesc}) was replaced ` +
				`by this ${method} run (${extracted.edges.length} edges; ${kept} overlap). ` +
				`The previous edge list is summarized in scratchpad Notes — use it for your ` +
				`cross-algorithm comparison; do NOT treat the latest run as "the" graph.\n\n`;
		}

		// Replace edges with the newly discovered ones
		pad.edges = extracted.edges;

		// Update score
		if (extracted.score !== undefined) {
			pad.score = extracted.score;
		}

		// Register variables that are not already in the scratchpad
		const vars = result.data.variables as string[] | undefined;
		if (vars) {
			for (const v of vars) {
				if (!pad.variables.find((pv) => pv.name === v)) {
					pad.variables.push({
						name: v,
						description: "",
						type: "unknown",
						role: "other",
						source: method,
					});
				}
			}
		}

		// Record the data path if not already set
		if (!pad.dataPath && extracted.dataPath) {
			pad.dataPath = extracted.dataPath;
		}

		persistScratchpad(pad);

		return [
			overwriteWarning + formatDiscoveryResult(result.data),
			"",
			"---",
			"_Discovery result saved to active scratchpad. Updated state:_",
			"",
			renderScratchpadBlock(pad),
		].join("\n");
	}

	return formatDiscoveryResult(result.data);
}

/**
 * Extracts typed CausalEdge objects from a causal_discover result.
 * Used by the orchestrator to programmatically update the scratchpad.
 */
export function extractEdgesFromResult(
	data: Record<string, unknown>,
): { edges: CausalEdge[]; score?: number; dataPath?: string } {
	const rawEdges = data.edges as Array<Record<string, unknown>>;
	const score = data.score as number | undefined;
	const metadata = data.metadata as Record<string, unknown> | undefined;
	const dataPath = metadata?.data_path as string | undefined;

	const edges: CausalEdge[] = rawEdges.map((e) => ({
		from: String(e.from),
		to: String(e.to),
		edgeType: (e.edgeType as CausalEdge["edgeType"]) || "directed",
		confidence: (e.confidence as CausalEdge["confidence"]) || "high",
		method: String(e.method || data.method || "unknown"),
	}));

	return {
		edges,
		score,
		dataPath,
	};
}

/**
 * Recalculates the score for the current scratchpad edges using the scoring bridge.
 */
export async function calculateScore(dataPath: string, edges: CausalEdge[], variables: string[]): Promise<number | undefined> {
	const n = variables.length;
	const adj = Array.from({ length: n }, () => Array(n).fill(0));
	const nameToIndex = new Map(variables.map((v, i) => [v, i]));

	for (const edge of edges) {
		const fromIdx = nameToIndex.get(edge.from);
		const toIdx = nameToIndex.get(edge.to);
		if (fromIdx !== undefined && toIdx !== undefined) {
			const fromRow = adj[fromIdx];
			const toRow = adj[toIdx];
			if (fromRow && toRow) {
				fromRow[toIdx] = 1;
				if (edge.edgeType === "undirected") {
					toRow[fromIdx] = 1;
				}
			}
		}
	}

	const result = await runPythonScript("score_graph.py", ["--data", dataPath, "--adj", JSON.stringify(adj)]);
	if (result.success && result.data) {
		return result.data.score as number;
	}
	return undefined;
}
