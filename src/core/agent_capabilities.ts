import { existsSync, realpathSync } from "node:fs";
import {
	basename,
	dirname,
	isAbsolute,
	relative,
	resolve,
	sep,
} from "node:path";

export type CausalBeliefAccess = "read-write" | "read-only" | "none";
export type CausalCognitionAccess = "read-write" | "read-only" | "none";
/**
 * Reproducible capability baselines.  `full` is the historical default;
 * `wo_causal_state` keeps the causal calculators but removes the persistent
 * Belief/Cognition/Scout surface; `wo_causal` is a general assistant.
 */
export type AgentMode = "full" | "wo_causal_state" | "wo_causal";
export const AGENT_MODES: readonly AgentMode[] = [
	"full",
	"wo_causal_state",
	"wo_causal",
];
const GENERAL_TOOL_NAMES = new Set([
	"read_file",
	"write_file",
	"execute_command",
	"web_search",
	"list_directory",
	"search_files",
]);

export function isAgentMode(value: unknown): value is AgentMode {
	return typeof value === "string" && AGENT_MODES.includes(value as AgentMode);
}

export interface AgentCapabilities {
	/** Session-persisted baseline. Omission preserves the historical full mode. */
	agentMode?: AgentMode;
	/** Selects the system-prompt identity/workflow. Defaults to CauProphet. */
	agentProfile?: "cauprophet" | "generic-react";
	/** Defaults to the complete registered tool set. */
	allowTools?: readonly string[];
	/** Applied after allowTools. */
	denyTools?: readonly string[];
	/** Defaults to read-write for backward compatibility. */
	causalBeliefAccess?: CausalBeliefAccess;
	/** Cross-task memory access; defaults to causalBeliefAccess when supplied. */
	causalCognitionAccess?: CausalCognitionAccess;
	/** Optional workspace-relative allowlist for read_file. */
	readPaths?: readonly string[];
	/** Optional workspace-relative allowlist for write_file. */
	writePaths?: readonly string[];
	/** Silent pre-compaction writes are disabled unless explicitly enabled. */
	allowSilentWrite?: boolean;
	/** Optional per-run execution budgets, keyed by exact tool name. */
	maxToolCalls?: Readonly<Record<string, number>>;
}

export const CAUSAL_BELIEF_LIST_TOOLS = new Set([
	"list_causal_beliefs",
	"list_scratchpads",
]);

/** Read-only causal analysis tools that inspect the active Belief. */
export const CAUSAL_BELIEF_READ_TOOLS = new Set(["causal_scout"]);

export const CAUSAL_BELIEF_POINTER_TOOLS = new Set([
	"open_causal_belief",
	"open_scratchpad",
]);

export const CAUSAL_BELIEF_WRITE_TOOLS = new Set([
	"create_causal_belief",
	"create_scratchpad",
	"write_causal_belief",
	"write_scratchpad",
	"instantiate_cognition_note",
	"export_belief_adj_matrix",
	"export_scratchpad_adj_matrix",
	// These analysis tools auto-persist into an active belief.
	"causal_discover",
	"causal_infer",
]);

export const CAUSAL_COGNITION_READ_TOOLS = new Set(["search_causal_cognition", "instantiate_cognition_note"]);

export const CAUSAL_COGNITION_WRITE_TOOLS = new Set([
	"update_causal_cognition",
	"write_cognition_note",
]);

/** Tools which belong to the coupled Causal Belief/Cognition/Scout group. */
export const CAUSAL_STATE_TOOLS = new Set([
	...CAUSAL_BELIEF_LIST_TOOLS,
	...CAUSAL_BELIEF_READ_TOOLS,
	...CAUSAL_BELIEF_POINTER_TOOLS,
	...CAUSAL_BELIEF_WRITE_TOOLS,
	...CAUSAL_COGNITION_READ_TOOLS,
	...CAUSAL_COGNITION_WRITE_TOOLS,
]);

// The calculators are historically grouped with Belief writes because their
// full-mode handlers auto-persist.  The named `wo_causal_state` baseline keeps
// them as pure per-call calculators, so remove only the persistence surface.
CAUSAL_STATE_TOOLS.delete("causal_discover");
CAUSAL_STATE_TOOLS.delete("causal_infer");

/**
 * Returns the canonical capability configuration for a named baseline.
 * Callers may spread this and add a bounded allow/deny list for a particular
 * test, while the baseline's invariants remain explicit and auditable.
 */
export function capabilitiesForAgentMode(mode: AgentMode = "full"): AgentCapabilities {
	if (mode === "wo_causal") {
		return {
			agentMode: mode,
			agentProfile: "generic-react",
			allowTools: [
				"read_file",
				"write_file",
				"execute_command",
				"web_search",
				"list_directory",
				"search_files",
			],
			causalBeliefAccess: "none",
			causalCognitionAccess: "none",
		};
	}
	if (mode === "wo_causal_state") {
		return {
			agentMode: mode,
			agentProfile: "cauprophet",
			causalBeliefAccess: "none",
			causalCognitionAccess: "none",
		};
	}
	return { agentMode: "full" };
}

/** Merge explicit per-run overrides onto a named baseline. */
export function resolveAgentCapabilities(
	mode: AgentMode = "full",
	overrides: AgentCapabilities = {},
): AgentCapabilities {
	return { ...capabilitiesForAgentMode(mode), ...overrides, agentMode: mode };
}

export function resolveAllowedToolNames(
	allToolNames: readonly string[],
	capabilities: AgentCapabilities = {},
): Set<string> {
	const allowed = new Set(capabilities.allowTools ?? allToolNames);
	for (const denied of capabilities.denyTools ?? []) allowed.delete(denied);
	for (const [name, limit] of Object.entries(capabilities.maxToolCalls ?? {})) {
		if (!Number.isFinite(limit) || limit <= 0) allowed.delete(name);
	}

	// The named state ablation is stronger than a prompt-only omission: remove
	// every state/scout/cognition entry from both the advertised surface and the
	// dispatch allowlist.  Causal calculators are intentionally not in this set.
	if (capabilities.agentMode === "wo_causal_state") {
		for (const name of CAUSAL_STATE_TOOLS) allowed.delete(name);
	}
	if (capabilities.agentMode === "wo_causal") {
		for (const name of allToolNames) {
			if (!GENERAL_TOOL_NAMES.has(name)) allowed.delete(name);
		}
	}

	const beliefAccess = capabilities.causalBeliefAccess ?? "read-write";
	// Preserve the historical read-only/none contract.  The named stateless
	// baseline is handled above and must keep causal_discover/infer callable.
	if (beliefAccess !== "read-write" && capabilities.agentMode !== "wo_causal_state") {
		for (const name of CAUSAL_BELIEF_POINTER_TOOLS) allowed.delete(name);
		for (const name of CAUSAL_BELIEF_WRITE_TOOLS) allowed.delete(name);
	}
	if (beliefAccess === "none") {
		for (const name of CAUSAL_BELIEF_LIST_TOOLS) allowed.delete(name);
		for (const name of CAUSAL_BELIEF_READ_TOOLS) allowed.delete(name);
	}

	const cognitionAccess =
		capabilities.causalCognitionAccess ??
		capabilities.causalBeliefAccess ??
		"read-write";
	if (cognitionAccess !== "read-write") {
		for (const name of CAUSAL_COGNITION_WRITE_TOOLS) allowed.delete(name);
	}
	if (cognitionAccess === "none") {
		for (const name of CAUSAL_COGNITION_READ_TOOLS) allowed.delete(name);
	}

	return allowed;
}

export function filterToolDefinitions<T extends { function: { name: string } }>(
	allTools: readonly T[],
	capabilities: AgentCapabilities = {},
): T[] {
	const allowed = resolveAllowedToolNames(
		allTools.map((tool) => tool.function.name),
		capabilities,
	);
	return allTools.filter((tool) => allowed.has(tool.function.name));
}

/**
 * Rewrites descriptions/schema hints for the stateless causal baseline.  The
 * callable tool set is still the same calculator set; only references that
 * would instruct the model to read or persist the removed state are removed.
 */
export function adaptToolDefinitionsForCapabilities<
	T extends { function: { name: string; description?: string; parameters?: unknown } },
>(tools: readonly T[], capabilities: AgentCapabilities = {}): T[] {
	if (capabilities.agentMode !== "wo_causal_state") return [...tools];
	return tools.map((tool) => {
		if (tool.function.name === "causal_discover") {
			return {
				...tool,
				function: {
					...tool.function,
					description:
						"Run a causal discovery algorithm on a CSV dataset and return the discovered graph. " +
						"This stateless baseline returns results directly and does not read or write persistent task state. " +
						"Available methods: pc, fci, ges, notears, direct_lingam, boss, grasp. " +
						"Run one bounded primary method by default. Please read the causal-discovery skill.",
				},
			} as T;
		}
		if (tool.function.name === "causal_infer") {
			const parameters = (tool.function.parameters ?? {}) as Record<string, unknown>;
			const properties = { ...((parameters.properties ?? {}) as Record<string, unknown>) };
			if (properties.common_causes && typeof properties.common_causes === "object") {
				properties.common_causes = {
					...(properties.common_causes as Record<string, unknown>),
					description:
						"Comma-separated adjustment set used when no explicit graph_edges are supplied. " +
						"There is no hidden active graph in this baseline.",
				};
			}
			if (properties.graph_edges && typeof properties.graph_edges === "object") {
				properties.graph_edges = {
					...(properties.graph_edges as Record<string, unknown>),
					description:
						"Optional explicit graph for this call. Each item has from, to, and optional edgeType/method/confidence. " +
						"When supplied, it is the identification source.",
				};
			}
			return {
				...tool,
				function: {
					...tool.function,
					description:
						"Identify and estimate a causal effect for an explicit intervention using only the " +
						"dataset plus graph_edges/common_causes supplied in this call. " +
						"This stateless baseline neither reads nor writes persistent task state. " +
						"Please read the causal-inference skill.",
						parameters: { ...parameters, properties },
				},
			} as T;
		}
		return tool;
	});
}

export function allowsSilentWrite(
	capabilities: AgentCapabilities | undefined,
	allowedToolNames: ReadonlySet<string>,
): boolean {
	return (
		(capabilities === undefined || capabilities.allowSilentWrite === true || capabilities.agentMode === "full") &&
		allowedToolNames.has("write_file")
	);
}

function isInsideOrEqual(parent: string, child: string): boolean {
	const rel = relative(parent, child);
	return (
		rel === "" ||
		(!rel.startsWith(`..${sep}`) && rel !== ".." && !isAbsolute(rel))
	);
}

/**
 * Resolves symlinks for the nearest existing ancestor. This also protects
 * not-yet-created write targets whose parent directory is a symlink.
 */
function canonicalPolicyPath(path: string): string {
	let cursor = resolve(path);
	const missingSegments: string[] = [];
	while (!existsSync(cursor)) {
		const parent = dirname(cursor);
		if (parent === cursor) break;
		missingSegments.unshift(basename(cursor));
		cursor = parent;
	}
	const realAncestor = existsSync(cursor) ? realpathSync(cursor) : cursor;
	return resolve(realAncestor, ...missingSegments);
}

export function isWorkspacePathAllowed(
	workspaceDir: string,
	requestedPath: string,
	allowedPaths?: readonly string[],
): boolean {
	if (!allowedPaths) return true;
	const workspace = canonicalPolicyPath(workspaceDir);
	const requested = canonicalPolicyPath(resolve(workspaceDir, requestedPath));
	if (!isInsideOrEqual(workspace, requested)) return false;
	return allowedPaths.some((allowedPath) =>
		isInsideOrEqual(
			canonicalPolicyPath(resolve(workspaceDir, allowedPath)),
			requested,
		),
	);
}

export function capabilityViolation(
	toolName: string,
	args: Record<string, string>,
	workspaceDir: string,
	allowedToolNames: ReadonlySet<string>,
	capabilities: AgentCapabilities = {},
	toolCallCounts?: ReadonlyMap<string, number>,
): string | null {
	if (!allowedToolNames.has(toolName)) {
		return `tool '${toolName}' is not available in this task`;
	}

	if (
		toolName === "read_file" &&
		!isWorkspacePathAllowed(
			workspaceDir,
			args["filePath"] ?? "",
			capabilities.readPaths,
		)
	) {
		return `read path '${args["filePath"] ?? ""}' is outside the task allowlist`;
	}

	if (
		toolName === "write_file" &&
		!isWorkspacePathAllowed(
			workspaceDir,
			args["filePath"] ?? "",
			capabilities.writePaths,
		)
	) {
		return `write path '${args["filePath"] ?? ""}' is outside the task allowlist`;
	}

	const maxCalls = capabilities.maxToolCalls?.[toolName];
	if (maxCalls !== undefined) {
		if (!Number.isInteger(maxCalls) || maxCalls < 0) {
			return `tool '${toolName}' has an invalid call limit`;
		}
		const used = toolCallCounts?.get(toolName) ?? 0;
		if (used >= maxCalls) {
			return `tool '${toolName}' exceeded its call limit (${maxCalls})`;
		}
	}

	return null;
}

/**
 * Performs the dispatch-time capability check and consumes one budget unit
 * only for calls that are actually authorized.
 */
export function authorizeToolCall(
	toolName: string,
	args: Record<string, string>,
	workspaceDir: string,
	allowedToolNames: ReadonlySet<string>,
	capabilities: AgentCapabilities,
	toolCallCounts: Map<string, number>,
): string | null {
	const violation = capabilityViolation(
		toolName,
		args,
		workspaceDir,
		allowedToolNames,
		capabilities,
		toolCallCounts,
	);
	if (violation) return violation;
	toolCallCounts.set(toolName, (toolCallCounts.get(toolName) ?? 0) + 1);
	return null;
}
