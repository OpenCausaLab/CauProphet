import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { createModelCompletion, describeModelError, getConfiguredModel } from "@/core/llm";
import {
	allowsSilentWrite,
	adaptToolDefinitionsForCapabilities,
	authorizeToolCall,
	filterToolDefinitions,
	resolveAllowedToolNames,
	capabilitiesForAgentMode,
	type AgentCapabilities,
	type AgentMode,
} from "@/core/agent_capabilities";
import { createCausalBelief, type CausalBelief } from "@/memory/causal_belief";
import { GlobalState } from "@/core/state";
import {
	buildCauProphetSystemPrompt,
	loadContextFiles,
	toolDefsToEntries,
} from "@/core/system-prompt";
import {
	loadActiveBelief,
	loadActiveScratchpad,
	persistBelief,
} from "@/tools/causal_belief_tools";
import {
	createSession,
	persistSession,
	loadActiveSession,
	loadSessionById,
	appendTurnToSession,
	updateLastTurnInSession,
	buildWorkingHistory,
	type Session,
	type PersistedToolCallRecord,
	type PersistedChatTurn,
	type MessageBlock,
} from "@/tools/session_tools";
import { toolHandlers, tools } from "@/tools/builtin_tools";
import { mapCausalInferToolArgs, type CausalToolRuntime } from "@/tools/causal_tools";
import {
	microcompact,
	sanitizeToolMessages,
	shouldAutocompact,
	autocompact,
	SILENT_TURN_CONTENT,
} from "@/core/compactor";
import {
	cognitionReflectionDecision,
	COGNITION_COMPLETION_ADVISORY_PROMPT,
	isSuccessfulCognitionMemoryResult,
	isSuccessfulToolResult,
} from "@/core/causal_cognition_completion";

const MAX_ITERATIONS = 1000;

const CAUSAL_BELIEF_ENTRY_TOOLS = new Set([
	"create_causal_belief",
	"create_scratchpad",
	"write_causal_belief",
	"write_scratchpad",
	"instantiate_cognition_note",
]);

const CAUSAL_REFLECTION_BLOCKED_TOOLS = new Set([
	"web_search",
	"execute_command",
	"create_causal_belief",
	"create_scratchpad",
	"open_causal_belief",
	"open_scratchpad",
]);

function parseExplicitGraphEdges(raw: unknown): CausalToolRuntime["graphEdges"] {
	let value: unknown = raw;
	if (typeof value === "string" && value.trim()) {
		try {
			value = JSON.parse(value);
		} catch {
			return undefined;
		}
	}
	if (!Array.isArray(value)) return undefined;
	const edges = value.flatMap((edge) => {
		if (!edge || typeof edge !== "object") return [];
		const record = edge as Record<string, unknown>;
		const from = String(record.from ?? "").trim();
		const to = String(record.to ?? "").trim();
		if (!from || !to) return [];
		const edgeType = String(record.edgeType ?? "directed") as NonNullable<CausalToolRuntime["graphEdges"]>[number]["edgeType"];
		return [{
			from,
			to,
			edgeType,
			method: String(record.method ?? "explicit"),
			confidence: (String(record.confidence ?? "medium") as "high" | "medium" | "low"),
		}];
	});
	return edges.length > 0 ? edges : undefined;
}

/**
 * Per-model reasoning/thinking request options.
 *
 * - GPT-series endpoints (e.g. gpt-5.6-terra) reject `reasoning_effort` when
 *   function tools are present ("400 ... use /v1/responses instead"). Since we
 *   always send tools, GPT models must get NO reasoning params at all.
 * - Claude models take an explicit adaptive-thinking directive.
 * - Everything else (deepseek, minimax, o-series, …) is left to think at its
 *   own default strength — we deliberately do NOT pin `reasoning_effort`.
 */
function buildReasoningRequestOptions(model: string): Record<string, unknown> {
	const normalized = model.toLowerCase();
	if (normalized.startsWith("gpt")) {
		return {};
	}
	if (normalized.startsWith("claude-")) {
		return { thinking: { type: "adaptive", display: "summarized" } };
	}
	if (normalized.startsWith("deepseek") || normalized.startsWith("minimax")) {
		return { reasoning_effort: "high" };
	}
	return {};
}

// ---------------------------------------------------------------------------
// In-memory state (per process)
// ---------------------------------------------------------------------------

/** Conversation history excluding the system message (rebuilt each turn). */
const conversationHistory: ChatCompletionMessageParam[] = [];

/** The live causal belief (SCM state) — null if none active. */
let _causalBelief: CausalBelief | null = null;
let _scratchpad: CausalBelief | null = null;

/** The active conversation session. */
let _session: Session | null = null;

/**
 * Context window token occupancy from the last API call.
 * Used to determine whether to trigger auto-compact (not a cumulative spend counter).
 */
let _contextTokens = 0;

function ensureSession(): Session {
	if (!_session) {
		_session = loadActiveSession() ?? createSession();
		persistSession(_session);
		// Restore LLM history from session using pointer-aware buildWorkingHistory
		conversationHistory.length = 0;
		conversationHistory.push(...buildWorkingHistory(_session));
		_contextTokens = _session.contextTokens ?? 0;
	}
	return _session;
}

/** Returns the active causal belief (may be null if none created yet). */
export function getActiveBelief(): CausalBelief | null {
	if (_causalBelief === undefined) _causalBelief = null;
	if (_causalBelief) return _causalBelief;
	_causalBelief = loadActiveBelief();
	_scratchpad = _causalBelief;
	return _causalBelief;
}

/** Backward compatibility alias */
export const getActiveScratchpad = getActiveBelief;

/** Returns a read-only snapshot of the current causal belief (may be null). */
export function getCausalBelief(): Readonly<CausalBelief> | null {
	return getActiveBelief();
}

/** Backward compatibility alias */
export const getScratchpad = getCausalBelief;

// ---------------------------------------------------------------------------
// Skills registry
// ---------------------------------------------------------------------------

/**
 * Parses the YAML frontmatter of a skill Markdown file and returns the
 * `name` and `description` fields.  Falls back to the provided defaults
 * when either field is absent.
 */
export function parseSkillFrontmatter(
	content: string,
	defaultName: string,
): { name: string; description: string } {
	const frontmatterMatch = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
	let name = defaultName;
	let description = "No description provided.";
	if (!frontmatterMatch) {
		return { name, description };
	}
	const yamlStr = frontmatterMatch[1] ?? "";
	const lines = yamlStr.split(/\r?\n/);
	const data: Record<string, string> = {};
	let currentKey = "";
	let blockMode: ">" | "|" | "" = "";
	let blockLines: string[] = [];
	let blockIndent = -1;

	const finalizeBlock = () => {
		if (!currentKey || !blockMode) return;
		if (blockMode === ">") {
			let result = "";
			let paragraph = "";
			for (const line of blockLines) {
				if (line.trim() === "") {
					if (paragraph) {
						result += (result ? "\n" : "") + paragraph;
						paragraph = "";
					}
					result += "\n";
				} else {
					paragraph += (paragraph ? " " : "") + line.trim();
				}
			}
			if (paragraph) {
				result += (result ? "\n" : "") + paragraph;
			}
			data[currentKey] = result.trim();
		} else if (blockMode === "|") {
			data[currentKey] = blockLines.join("\n").trim();
		}
		blockMode = "";
		currentKey = "";
	};

	for (const line of lines) {
		const trimmed = line.trim();

		if (currentKey && blockMode) {
			if (trimmed === "") {
				blockLines.push("");
				continue;
			}
			const indent = line.search(/\S/);
			if (blockIndent === -1) {
				blockIndent = indent;
			}
			if (indent >= blockIndent) {
				blockLines.push(line.slice(blockIndent));
				continue;
			} else {
				finalizeBlock();
			}
		}

		if (trimmed === "") continue;

		const match = line.match(/^([a-zA-Z0-9_-]+):\s*(.*)$/);
		if (match) {
			const key = match[1];
			const rawValue = match[2];
			if (!key || rawValue === undefined) continue;
			const val = rawValue.trim();
			currentKey = key;

			if (val === ">" || val === "|") {
				blockMode = val;
				blockLines = [];
				blockIndent = -1;
			} else {
				blockMode = "";
				data[key] = val.replace(/^['"]([\s\S]*?)['"]$/, "$1");
			}
		}
	}

	finalizeBlock();

	if (data.name) {
		name = data.name.trim();
	}
	if (data.description) {
		description = data.description.trim();
	}

	return { name, description };
}

/**
 * Scans the workspace `skills/` directory for skills in the standard
 * directory-based format.
 * Each skill folder MUST contain a `SKILL.md` file; directories without
 * one are silently skipped.  The returned string is injected into the
 * system prompt's Skills Registry section so the LLM knows which skills
 * are available and where to read them.
 */
export function adaptStatelessSkillContent(content: string): string {
	const lines = content.split(/\r?\n/);
	const kept = lines.filter((line) => {
		const lower = line.toLowerCase();
		return !(
			lower.includes("causal belief") ||
			lower.includes("causal cognition") ||
			lower.includes("causal scout") ||
			lower.includes("scratchpad") ||
			lower.includes("write_causal_belief") ||
			lower.includes("search_causal_cognition") ||
			lower.includes("update_causal_cognition") ||
			lower.includes("write_cognition_note") ||
			lower.includes("instantiate_cognition_note")
		);
	});
	return [
		"<!-- Stateless causal baseline: provide all graph/adjustment inputs in each tool call; do not rely on persistent agent state. -->",
		...kept,
	].join("\n");
}

function adaptStatelessContextContent(content: string): string {
	return content
		.split(/\r?\n/)
		.filter((line) => {
			const lower = line.toLowerCase();
			return !(
				lower.includes("causal belief") ||
				lower.includes("causal cognition") ||
				lower.includes("causal scout") ||
				lower.includes("scratchpad") ||
				lower.includes("write_causal_belief")
			);
		})
		.join("\n");
}

function adaptContextFilesForCapabilities(
	files: ReturnType<typeof loadContextFiles>,
	capabilities?: AgentCapabilities,
): ReturnType<typeof loadContextFiles> {
	if (capabilities?.agentMode !== "wo_causal_state") return files;
	return files.map((file) => ({ ...file, content: adaptStatelessContextContent(file.content) }));
}

function isSkillPath(filePath: string): boolean {
	const normalized = filePath.replaceAll("\\", "/").replace(/^\.\//, "");
	return normalized.startsWith("skills/") && normalized.endsWith("/SKILL.md");
}

function buildSkillsRegistry(capabilities?: AgentCapabilities): string {
	const skillsDir = join(process.cwd(), "skills");
	if (!existsSync(skillsDir)) return "";

	const entries: string[] = [];

	try {
		const dirEntries = readdirSync(skillsDir, { withFileTypes: true });
		for (const entry of dirEntries) {
			if (!entry.isDirectory()) continue;
			if (capabilities?.agentMode === "wo_causal_state" && entry.name === "causal-cognition") {
				continue;
			}

			const skillMdPath = join(skillsDir, entry.name, "SKILL.md");
			if (!existsSync(skillMdPath)) continue;

			try {
				const content = readFileSync(skillMdPath, "utf-8");
				const visibleContent = capabilities?.agentMode === "wo_causal_state"
					? adaptStatelessSkillContent(content)
					: content;
				const { name, description } = parseSkillFrontmatter(
					visibleContent,
					entry.name,
				);
				const visibleDescription = capabilities?.agentMode === "wo_causal_state"
					? description.replace(/causal\s+(belief|cognition|scout)|scratchpad/gi, "per-call causal inputs")
					: description;
				entries.push(
					`- **${name}** (Location: \`skills/${entry.name}/SKILL.md\`): ${visibleDescription}`,
				);
			} catch (e) {
				console.error(`[skills] Failed to read ${skillMdPath}: ${e}`);
			}
		}
	} catch (e) {
		console.error(`[skills] Failed to scan skills directory: ${e}`);
	}

	return entries.join("\n");
}

// ---------------------------------------------------------------------------
// System prompt construction
// ---------------------------------------------------------------------------

function buildSystemMessage(
	availableTools: typeof tools = tools,
	capabilities?: AgentCapabilities,
	runtimeModel?: string,
): ChatCompletionMessageParam {
	const config = GlobalState.getInstance().config;
	const agentProfile = capabilities?.agentProfile ?? "cauprophet";
	const isGeneric = agentProfile === "generic-react";
	const isStatelessCausal = capabilities?.agentMode === "wo_causal_state";
	const webSearchLimit = capabilities?.maxToolCalls?.["web_search"];
	const genericBudgetDirective =
		isGeneric && Number.isInteger(webSearchLimit)
			? "Call `web_search` at most " +
				webSearchLimit +
				" times. After that budget, do not retry search; make the best supported estimate from available evidence and always return the requested terminal answer."
			: undefined;
	const beliefAccess = isStatelessCausal
		? "none"
		: capabilities?.causalBeliefAccess ?? "read-write";
	const cognitionAccess =
		capabilities?.causalCognitionAccess ??
		capabilities?.causalBeliefAccess ??
		"read-write";
	const belief = beliefAccess === "none" ? null : getActiveBelief();

	const agentContextFiles = adaptContextFilesForCapabilities(
		isGeneric ? [] : loadContextFiles(process.cwd(), ["AGENTS.md"]),
		capabilities,
	);
	const visibleTools = adaptToolDefinitionsForCapabilities(
		filterToolDefinitions(availableTools, capabilities ?? {}),
		capabilities ?? {},
	);

	const systemContent = buildCauProphetSystemPrompt({
		tools: toolDefsToEntries(visibleTools),
		contextFiles: agentContextFiles,
		workspaceDir: config.workspace ?? process.cwd(),
		runtimeInfo: {
			model: runtimeModel ?? getConfiguredModel(),
			os: process.platform,
			node: process.version,
			shell: process.env["SHELL"] ?? "sh",
		},
		userTimezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
		userTime: new Date().toLocaleString(),
		skillsPrompt: isGeneric ? undefined : buildSkillsRegistry(capabilities),
			agentProfile,
		agentMode: capabilities?.agentMode,
		extraDirectives: genericBudgetDirective,
		causalBeliefAccess: beliefAccess,
		causalCognitionAccess: cognitionAccess,
		// Pass null-safe causal belief — buildBeliefSection handles undefined
		causalBelief: belief ?? undefined,
	});

	return { role: "system", content: systemContent };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface AgentLoopOptions {
	onUpdate: (status: string) => void;
	/** Optional observation hook for isolated benchmarks; it cannot alter dispatch. */
	onToolResult?: (event: {
		tool: string;
		arguments: Readonly<Record<string, unknown>>;
		result: string;
		capabilityError?: string;
	}) => void;
	objective?: string;
	/**
	 * Optional per-run capability boundary. Omission preserves the historical
	 * full-tool, read-write behavior.
	 */
	capabilities?: AgentCapabilities;
	/** When aborted, the loop stops before the next LLM call and any in-flight request is cancelled. */
	signal?: AbortSignal;
	/** Optional model override for a whole run; never switches silently mid-run. */
	model?: string;
}

/**
 * Starts a fresh conversation session (clears LLM history).
 * Does NOT create a causal belief — the LLM creates one via create_causal_belief when needed.
 */
export function resetSession(title?: string, agentMode: AgentMode = "full"): void {
	conversationHistory.length = 0;
	_session = createSession(title, agentMode);
	_causalBelief = null;
	_scratchpad = null;
	_contextTokens = 0;
	persistSession(_session);
}

/**
 * Switches to an existing session by ID, restoring its LLM conversation history
 * and loading its associated causal belief.
 */
export function switchSession(sessionId: string): Session | null {
	conversationHistory.length = 0;
	const session = loadSessionById(sessionId);
	if (!session) return null;
	_session = session;
	persistSession(_session); // updates active pointer
	conversationHistory.push(...buildWorkingHistory(_session));

	// Load the causal belief linked to this session (may be null)
	_causalBelief = loadActiveBelief();
	_scratchpad = _causalBelief;
	// Restore persisted context token occupancy (fixes display after session switch)
	_contextTokens = session.contextTokens ?? 0;
	return _session;
}

/**
 * Returns the current session (creates/loads one if needed).
 */
export function getSession(): Readonly<Session> {
	return ensureSession();
}

/** Resolve the persisted baseline for the current conversation. */
export function getSessionCapabilities(): AgentCapabilities {
	return capabilitiesForAgentMode(ensureSession().agentMode ?? "full");
}

/**
 * Patches the in-memory session cache when the session is renamed via the REST API.
 * Without this, the stale title in `_session` would be written back to disk on the
 * next call to appendTurnToSession(), silently reverting the rename.
 */
export function notifySessionRenamed(id: string, title: string): void {
	if (_session && _session.id === id) {
		_session = { ..._session, title };
	}
}

/**
 * Manually triggers a full autocompact of the current session.
 * Includes the Silent Turn (user has set a long proxy timeout).
 * Returns { ran: true } if compaction ran, { ran: false } if skipped.
 */
export async function triggerManualCompact(
	onStatus: (s: string) => void = () => {},
	capabilities?: AgentCapabilities,
): Promise<{ ran: boolean }> {
	const session = ensureSession();
	return runCompaction(
		session.turns.length,
		onStatus,
		/* runSilentTurn */ true,
		capabilities,
	);
}

function estimateTokensOfHistory(capabilities?: AgentCapabilities): number {
	if (conversationHistory.length === 0) {
		return 0;
	}
	let charCount = 0;
	try {
		const sysMsg = buildSystemMessage(tools, capabilities);
		if (typeof sysMsg.content === "string") {
			charCount += sysMsg.content.length;
		}
	} catch {
		/* ignore */
	}

	for (const msg of conversationHistory) {
		if (typeof msg.content === "string") {
			charCount += msg.content.length;
		}
	}

	// 1 token ≈ 3 characters for mixed text
	return Math.round(charCount / 3);
}

/**
 * Returns the context window token occupancy from the last API call.
 * Used by the server to include token stats in WebSocket 'done' messages
 * and to determine whether to trigger auto-compact.
 */
export function getContextTokens(capabilities?: AgentCapabilities): number {
	if (_contextTokens > 0) {
		return _contextTokens;
	}
	return estimateTokensOfHistory(capabilities);
}

/**
 * Returns the configured context window size from config.
 */
export function getContextWindow(): number {
	return GlobalState.getInstance().config.contextWindow ?? 128_000;
}

// ---------------------------------------------------------------------------
// Internal: compaction runner (shared by auto and manual paths)
// ---------------------------------------------------------------------------

/**
 * Runs the two-phase compaction.
 *
 * @param persistedTurnCount  session.turns.length at the time of invocation
 * @param onStatus            status callback
 * @param runSilentTurn       whether to execute the pre-compact Silent Turn
 *                            (true for auto-compact, false for manual/button)
 */
async function runCompaction(
	persistedTurnCount: number,
	onStatus: (s: string) => void,
	runSilentTurn = true,
	capabilities?: AgentCapabilities,
	availableTools: typeof tools = adaptToolDefinitionsForCapabilities(
		filterToolDefinitions(tools, capabilities ?? {}),
		capabilities ?? {},
	),
	allowedToolNames: ReadonlySet<string> = resolveAllowedToolNames(
		tools.map((tool) => tool.function.name),
		capabilities ?? {},
	),
	toolCallCounts: Map<string, number> = new Map(),
	model?: string,
): Promise<{ ran: boolean }> {
	// ── Phase 2a: Silent Turn ─────────────────────────────────────────────
	// Only runs during auto-compact (not manual): injects a hidden prompt
	// asking the agent to flush key facts to AGENTS.md before the history
	// is destroyed. Manual compact skips this to keep the HTTP round-trip fast.
	const mayRunSilentWrite =
		runSilentTurn && allowsSilentWrite(capabilities, allowedToolNames);
	if (mayRunSilentWrite) {
		try {
			const silentMessages: ChatCompletionMessageParam[] = [
				buildSystemMessage(availableTools, capabilities, model),
				...sanitizeToolMessages(microcompact(conversationHistory)),
				{ role: "user", content: SILENT_TURN_CONTENT },
			];

			const silentResp = await createModelCompletion({
				model: model ?? getConfiguredModel(),
				messages: silentMessages,
				tools: availableTools,
				tool_choice: "auto",
			});

			const silentMsg = silentResp.choices[0]?.message;
			if (silentMsg?.tool_calls?.length) {
				// Execute only write_file calls — the only safe action here
				for (const tc of silentMsg.tool_calls) {
					if (tc.type !== "function") continue;
					if (tc.function.name !== "write_file") continue;
					try {
						const args = JSON.parse(tc.function.arguments) as Record<
							string,
							string
						>;
						const violation = authorizeToolCall(
							tc.function.name,
							args,
							process.cwd(),
							allowedToolNames,
							capabilities ?? {},
							toolCallCounts,
						);
						if (violation) continue;
						await toolHandlers["write_file"]?.(
							args["filePath"] ?? "",
							args["content"] ?? "",
						);
					} catch {
						/* non-fatal */
					}
				}
			}
		} catch {
			/* silent turn failure is non-fatal */
		}
	}

	// ── Phase 2b: LLM Autocompact ─────────────────────────────────────────
	const result = await autocompact(
		conversationHistory,
		persistedTurnCount,
		onStatus,
		model,
	);

	if (result && _session) {
		_session = {
			..._session,
			compactSummary: result.summary,
			historyStartIndex: result.newHistoryStartIndex,
			updatedAt: new Date().toISOString(),
		};
		persistSession(_session);

		// Rebuild the in-memory history from the compact result
		conversationHistory.length = 0;
		conversationHistory.push(...result.newConversationHistory);
		_contextTokens = 0;
	}
	return { ran: result !== null };
}

// ---------------------------------------------------------------------------
// Main agent loop
// ---------------------------------------------------------------------------

export async function runAgentLoop(
	userPrompt: string,
	onUpdate: (status: string) => void,
): Promise<void>;

export async function runAgentLoop(
	userPrompt: string,
	optionsOrCallback: AgentLoopOptions | ((status: string) => void),
): Promise<void>;

export async function runAgentLoop(
	userPrompt: string,
	optionsOrCallback: AgentLoopOptions | ((status: string) => void),
): Promise<void> {
	const opts: AgentLoopOptions =
		typeof optionsOrCallback === "function"
			? { onUpdate: optionsOrCallback }
			: optionsOrCallback;
	const { onUpdate, onToolResult, objective, signal, capabilities, model } = opts;
	// A session-selected baseline is sticky across turns.  Explicit per-run
	// capabilities remain available for tests/legacy callers, but omission now
	// resolves from the persisted session instead of silently reverting to full.
	const sessionForMode = ensureSession();
	const effectiveCapabilities =
		capabilities ?? capabilitiesForAgentMode(sessionForMode.agentMode ?? "full");
	const stateEnabled =
		effectiveCapabilities.agentMode !== "wo_causal_state" &&
		effectiveCapabilities.agentMode !== "wo_causal";
	const allowedToolNames = resolveAllowedToolNames(
		tools.map((tool) => tool.function.name),
		effectiveCapabilities,
	);
	const availableTools = adaptToolDefinitionsForCapabilities(
		filterToolDefinitions(tools, effectiveCapabilities),
		effectiveCapabilities,
	);
	const toolCallCounts = new Map<string, number>();
	const beliefAccess = effectiveCapabilities.causalBeliefAccess ?? "read-write";
	const cognitionAccess =
		effectiveCapabilities.causalCognitionAccess ??
		effectiveCapabilities.causalBeliefAccess ??
		"read-write";
	const cognitionCompletionRequired =
		beliefAccess === "read-write" &&
		cognitionAccess === "read-write" &&
		(allowedToolNames.has("update_causal_cognition") || allowedToolNames.has("write_cognition_note"));
	let causalWorkflowEntered = false;
	let successfulCognitionMemory = false;
	let cognitionAdvisoryOffered = false;
	let cognitionAdvisoryHistoryStart: number | null = null;

	/** True once the caller has requested an interruption. */
	const isAborted = () => signal?.aborted === true;

	// Seed causal belief objective if provided and none active
	if (
		stateEnabled &&
		beliefAccess === "read-write" &&
		objective &&
		!getActiveBelief()?.objective
	) {
		_causalBelief = createCausalBelief(objective);
		_scratchpad = _causalBelief;
		persistBelief(_causalBelief);
	}

	// Restore/create the session before appending the current prompt. On a
	// fresh worker, ensureSession() rebuilds conversationHistory from disk; if
	// the prompt were pushed first it would be erased during that restoration,
	// causing the model to answer the system prompt as if no user task existed.
	const session = ensureSession();
	// Append user turn to in-memory history and persist
	conversationHistory.push({ role: "user", content: userPrompt });
	_session = appendTurnToSession(session, {
		role: "user",
		content: userPrompt,
		timestamp: new Date().toISOString(),
		blocks: [{ kind: "content", content: userPrompt }],
	});

	let iterations = 0;

	// Sequential blocks preserving exact chronological order of assistant outputs (thinking, content, tools)
	const turnBlocks: MessageBlock[] = [];

	// Accumulators for think blocks and tool calls across all LLM iterations in this turn.
	// Persisted at the end for frontend display restoration on session reload.
	let accumulatedThink = "";
	const accumulatedToolCalls: PersistedToolCallRecord[] = [];
	let preservedCognitionAnswer: {
		message: ChatCompletionMessageParam;
		content: string;
		reasoning: string;
	} | null = null;
	let cognitionCuratorStatusActive = false;

	// ---------------------------------------------------------------------------
	// Incremental persistence: create an in-flight assistant turn immediately
	// so that any work done before a crash is preserved on disk.
	// ---------------------------------------------------------------------------
	const inFlightTurn: PersistedChatTurn = {
		role: "assistant",
		content: null,
		timestamp: new Date().toISOString(),
		blocks: turnBlocks,
		status: "streaming",
	};
	_session = appendTurnToSession(_session, inFlightTurn);

	/** Helper: persist the current state of the in-flight turn to disk. */
	const checkpointTurn = () => {
		const updatedTurn: PersistedChatTurn = {
			...inFlightTurn,
			blocks: [...turnBlocks],
			...(accumulatedThink ? { think: accumulatedThink } : {}),
			...(accumulatedToolCalls.length > 0
				? { toolCalls: [...accumulatedToolCalls] }
				: {}),
		};
		const freshSession = loadActiveSession();
		if (freshSession) {
			_session = updateLastTurnInSession(freshSession, updatedTurn);
		}
	};

	const finalizeCompletedTurn = (content: string | null) => {
		if (stateEnabled) _scratchpad = loadActiveScratchpad();
		const hasAnyContent = turnBlocks.some((block) => block.kind === "content");
		const hasAnyTools = accumulatedToolCalls.length > 0;
		const hasAnyThink = turnBlocks.some((block) => block.kind === "think");

		if (hasAnyContent || hasAnyTools || hasAnyThink) {
			const finalTurn: PersistedChatTurn = {
				role: "assistant",
				content,
				timestamp: inFlightTurn.timestamp,
				status: "done",
				...(turnBlocks.length > 0 ? { blocks: [...turnBlocks] } : {}),
				...(accumulatedThink ? { think: accumulatedThink } : {}),
				...(hasAnyTools ? { toolCalls: [...accumulatedToolCalls] } : {}),
			};
			const freshSession = loadActiveSession();
			if (freshSession) {
				_session = updateLastTurnInSession(
					freshSession,
					finalTurn,
					_contextTokens,
				);
			}
			return;
		}

		const freshSession = loadActiveSession();
		if (!freshSession || freshSession.turns.length === 0) return;
		const lastTurn = freshSession.turns[freshSession.turns.length - 1];
		if (lastTurn?.status !== "streaming") return;
		const cleaned: Session = {
			...freshSession,
			turns: freshSession.turns.slice(0, -1),
			updatedAt: new Date().toISOString(),
		};
		persistSession(cleaned);
		_session = cleaned;
	};

	const restorePreservedCognitionAnswer = () => {
		const preserved = preservedCognitionAnswer;
		if (!preserved) return null;
		if (cognitionAdvisoryHistoryStart !== null) {
			conversationHistory.splice(cognitionAdvisoryHistoryStart);
		}
		conversationHistory.push(preserved.message);
		preservedCognitionAnswer = null;
		cognitionAdvisoryHistoryStart = null;
		if (cognitionCuratorStatusActive) {
			onUpdate("EVENT:END:causal_cognition_curator");
			cognitionCuratorStatusActive = false;
		}
		return preserved;
	};

	const emitAndFinalizePreservedCognitionAnswer = (): boolean => {
		const preserved = restorePreservedCognitionAnswer();
		if (!preserved) return false;
		const trimmedReasoning = preserved.reasoning.trim();
		if (trimmedReasoning) {
			onUpdate(`THINK:${trimmedReasoning}`);
			accumulatedThink +=
				(accumulatedThink ? "\n\n---\n\n" : "") + trimmedReasoning;
			turnBlocks.push({ kind: "think", content: trimmedReasoning });
		}
		onUpdate(`CONTENT:${preserved.content}`);
		turnBlocks.push({ kind: "content", content: preserved.content });
		checkpointTurn();
		finalizeCompletedTurn(preserved.content);
		return true;
	};

	/**
	 * Handle a user-requested interruption. Persists whatever the assistant produced
	 * so far while leaving the turn in `streaming` status — the frontend renders this
	 * as an "interrupted" turn (see turnsToChatMessages) and the user can continue.
	 */
	const finalizeInterrupted = () => {
		if (stateEnabled) _scratchpad = loadActiveScratchpad();
		checkpointTurn();
		onUpdate("WARN:__INTERRUPTED__");
	};

	while (iterations < MAX_ITERATIONS) {
		iterations++;

		// ── Interruption check ─────────────────────────────────────────────
		// Stop before starting a new LLM call if the client requested an abort.
		if (isAborted()) {
			if (emitAndFinalizePreservedCognitionAnswer()) return;
			finalizeInterrupted();
			return;
		}

		// ── Auto-compact check ─────────────────────────────────────────────
		// shouldAutocompact uses the cumulative token counter set after each
		// API response. We check here (before the next call) so that if
		// threshold is crossed we compact first, then proceed with a fresh window.
		if (
			!preservedCognitionAnswer &&
			shouldAutocompact(_contextTokens) &&
			_session
		) {
			await runCompaction(
				_session.turns.length,
				onUpdate,
				/* runSilentTurn */ true,
				effectiveCapabilities,
				availableTools,
				allowedToolNames,
				toolCallCounts,
				model,
			);
		}

		// ── Build messages for this call ─────────────────────────────────────
		// Apply microcompact to avoid sending huge tool outputs unnecessarily.
		const systemMessage = buildSystemMessage(
			availableTools,
			effectiveCapabilities,
			model,
		);
		const messages: ChatCompletionMessageParam[] = [
			systemMessage,
			...sanitizeToolMessages(microcompact(conversationHistory)),
		];

		let response: Awaited<ReturnType<typeof createModelCompletion>>;
		const requestStartedAt = Date.now();
		try {
			const configuredModel = model ?? getConfiguredModel();
			response = await createModelCompletion(
				{
					model: configuredModel,
					messages,
					tools: availableTools,
					tool_choice: "auto",
					...buildReasoningRequestOptions(configuredModel),
				},
				signal ? { signal } : undefined,
			);
		} catch (apiErr) {
			// An abort surfaces here as a rejected request — treat as a clean interruption.
			if (isAborted()) {
				finalizeInterrupted();
				return;
			}
			onUpdate(`WARN:LLM API error: ${describeModelError(apiErr)} elapsedMs=${Date.now() - requestStartedAt}`);
			if (emitAndFinalizePreservedCognitionAnswer()) return;
			break;
		}

		// ── Accumulate context token occupancy ─────────────────────────────
		if (response.usage?.total_tokens) {
			_contextTokens = response.usage.total_tokens;
		}

		if (
			!response ||
			!Array.isArray(response.choices) ||
			response.choices.length === 0
		) {
			const raw = JSON.stringify(
				(response as unknown as Record<string, unknown>)?.["choices"] ??
					"missing",
			);
			onUpdate(
				`WARN:Empty or malformed API response (choices=${raw}). Check your model name, API key, and Base URL.`,
			);
			if (emitAndFinalizePreservedCognitionAnswer()) return;
			break;
		}

		const choice = response.choices[0];
		if (!choice) {
			if (emitAndFinalizePreservedCognitionAnswer()) return;
			break;
		}

		const message = choice.message;
		conversationHistory.push(message);

		let reasoning = "";
		const rawReasoning =
			(message as any).reasoning_content ??
			(message as any).reasoning ??
			(message as any).thinking;
		if (typeof rawReasoning === "string") {
			reasoning = rawReasoning;
		}

		let finalContent: string | null = null;
		if (message.content) {
			let content = (message as any).content;
			if (Array.isArray(content)) {
				content = content
					.map((part) => {
						if (typeof part === "string") return part;
						if (!part || typeof part !== "object") return "";
						const block = part as Record<string, unknown>;
						if (
							block.type === "thinking" &&
							typeof block.thinking === "string" &&
							!reasoning
						) {
							reasoning = block.thinking;
							return "";
						}
						return typeof block.text === "string" ? block.text : "";
					})
					.join("");
			}
			if (typeof content === "string") {
				// Strip think/reasoning blocks from various model formats:
				//   <think>...</think>        (DeepSeek, QwQ, etc.)
				//   <mm:think>...</mm:think>  (MiniMax)
				//   <|thinking|>...</|thinking|>  (other providers)
				content = content
					.replace(
						/^(?:\s*<(?:[\w:-]*think(?:ing)?)[^>]*>)([\s\S]*?)<\/(?:[\w:-]*think(?:ing)?)>/i,
						(_, think: string) => {
							const trimmed = think.trim();
							if (trimmed && !reasoning) {
								reasoning = trimmed;
							}
							return "";
						},
					)
					// Strip any residual closing think tags that appear without a matching open tag
					// e.g. "</mm:think>\n\nActual content" when the open tag was already in a prior chunk
					.replace(/^\s*<\/(?:[\w:-]*think(?:ing)?)>\s*/i, "")
					.trim();

				if (content) {
					finalContent = content;
				}
			}
		}

		const hasTools = message.tool_calls && message.tool_calls.length > 0;
		const hasContent = !!finalContent;
		const isTrailingReasoning = !hasTools && !hasContent;

		if (
			!hasTools &&
			finalContent !== null &&
			cognitionCompletionRequired &&
			causalWorkflowEntered
		) {
			const reflectionDecision = cognitionReflectionDecision({
				required: cognitionCompletionRequired,
				workflowEntered: causalWorkflowEntered,
				lifted: successfulCognitionMemory,
				advisoryOffered: cognitionAdvisoryOffered,
			});
			if (reflectionDecision === "offer") {
				cognitionAdvisoryOffered = true;
				cognitionAdvisoryHistoryStart = conversationHistory.length - 1;
				preservedCognitionAnswer = {
					message,
					content: finalContent,
					reasoning,
				};
				conversationHistory.push({
					role: "user",
					content: COGNITION_COMPLETION_ADVISORY_PROMPT,
				});
				cognitionCuratorStatusActive = true;
				onUpdate(
					"EVENT:START:causal_cognition_curator:Reviewing long-term task experience and causal knowledge",
				);
				continue;
			}
			const preserved = restorePreservedCognitionAnswer();
			if (preserved) {
				finalContent = preserved.content;
				reasoning = preserved.reasoning;
			}
		}

		if (reasoning && !isTrailingReasoning) {
			const trimmedReasoning = reasoning.trim();
			if (trimmedReasoning) {
				onUpdate(`THINK:${trimmedReasoning}`);
				accumulatedThink +=
					(accumulatedThink ? "\n\n---\n\n" : "") + trimmedReasoning;
				turnBlocks.push({ kind: "think", content: trimmedReasoning });
				checkpointTurn();
			}
		}

		if (finalContent) {
			onUpdate(`CONTENT:${finalContent}`);
			turnBlocks.push({ kind: "content", content: finalContent });
			checkpointTurn();
		}

		// No tool calls → final reply, exit loop
		if (!hasTools) {
			finalizeCompletedTurn(finalContent);
			break;
		}

		// Execute all tool calls in this turn
		for (const toolCall of message.tool_calls ?? []) {
			// Abort between tool calls so a long tool chain stops promptly.
			if (isAborted()) {
				if (emitAndFinalizePreservedCognitionAnswer()) return;
				finalizeInterrupted();
				return;
			}

			// Every declared tool_call MUST get a matching `tool` response pushed
			// onto conversationHistory, or the assistant/tool pairing breaks and
			// the API rejects every subsequent request. Non-function calls and
			// malformed-argument calls therefore emit a stub result instead of
			// being skipped.
			if (toolCall.type !== "function") {
				conversationHistory.push({
					role: "tool",
					tool_call_id: toolCall.id,
					content: `Error: Unsupported tool call type '${toolCall.type}'.`,
				});
				continue;
			}

			const name = toolCall.function.name;
			let args: Record<string, string>;
			try {
				args = JSON.parse(toolCall.function.arguments) as Record<
					string,
					string
				>;
			} catch (e) {
				conversationHistory.push({
					role: "tool",
					tool_call_id: toolCall.id,
					content: `Error: Could not parse arguments for '${name}' as JSON: ${String(e)}. Raw arguments: ${toolCall.function.arguments}`,
				});
				continue;
			}

			// Build a human-readable detail string for display
			let detail: string | undefined;
			if (name === "execute_command") {
				detail = args["command"];
			} else if (name === "read_file" || name === "write_file") {
				detail = args["filePath"];
			} else if (name === "web_search") {
				detail = args["query"];
			} else if (name === "list_directory") {
				detail = args["dirPath"];
			} else if (name === "search_files") {
				detail = args["keyword"];
			} else if (name === "open_causal_belief" || name === "open_scratchpad") {
				detail = args["id"];
			} else if (
				name === "create_causal_belief" ||
				name === "create_scratchpad"
			) {
				detail = args["title"];
			} else if (name === "causal_discover") {
				detail = `${args["method"] ?? ""} on ${args["data_path"] ?? ""}`;
			} else if (name === "causal_infer") {
				detail = `${args["treatment"] ?? ""} -> ${args["outcome"] ?? ""}`;
			} else if (name === "causal_scout") {
				detail = "Enumerate local causal hypotheses";
			} else if (
				name === "export_belief_adj_matrix" ||
				name === "export_scratchpad_adj_matrix"
			) {
				detail = args["path"];
			} else if (name === "search_causal_cognition") {
				detail = "Index search / frontier expand";
			} else if (name === "update_causal_cognition") {
				detail = "Save experience / Lift task graph";
			} else if (name === "write_cognition_note") {
				detail = args["title"] ?? args["id"];
			} else if (name === "instantiate_cognition_note") {
				detail = args["id"];
			}

			if (detail) {
				onUpdate(`EVENT:START:${name}:${detail}`);
			} else {
				onUpdate(`EVENT:START:${name}`);
			}

			// Record for display persistence (no result stored)
			accumulatedToolCalls.push({ tool: name, ...(detail ? { detail } : {}) });
			turnBlocks.push({
				kind: "tool",
				tool: name,
				state: "done",
				...(detail ? { detail } : {}),
			});
			checkpointTurn();

			const handler = toolHandlers[name];
			let result: string;
			const capabilityError =
				cognitionAdvisoryOffered && CAUSAL_REFLECTION_BLOCKED_TOOLS.has(name)
					? "Causal Cognition reflection cannot restart research or switch Beliefs"
					: authorizeToolCall(
							name,
							args,
							process.cwd(),
							allowedToolNames,
							effectiveCapabilities,
							toolCallCounts,
						);

			if (capabilityError) {
				result = `Error: Capability denied: ${capabilityError}.`;
				onUpdate(`WARN:Capability denied for ${name}: ${capabilityError}`);
			} else if (handler) {
				try {
						switch (name) {
						case "read_file":
							result = await handler(
								args["filePath"] ?? "",
								args["startLine"] !== undefined
									? Number(args["startLine"])
									: undefined,
								args["endLine"] !== undefined
									? Number(args["endLine"])
									: undefined,
							);
							if (
								effectiveCapabilities.agentMode === "wo_causal_state" &&
								isSkillPath(args["filePath"] ?? "")
							) {
								const marker = "\n";
								const markerIndex = result.indexOf(marker);
								const header = markerIndex >= 0 ? result.slice(0, markerIndex + marker.length) : "";
								const content = markerIndex >= 0 ? result.slice(markerIndex + marker.length) : result;
								result = header + adaptStatelessSkillContent(content);
							}
							break;
						case "write_file":
							result = await handler(
								args["filePath"] ?? "",
								args["content"] ?? "",
							);
							break;
						case "execute_command":
							result = await handler(
								args["command"] ?? "",
								args["timeout"] ?? "10000",
							);
							break;
						case "web_search":
							result = await handler(args["query"] ?? "");
							break;
						case "list_directory":
							result = await handler(args["dirPath"] ?? ".");
							break;
						case "search_files":
							result = await handler(args["keyword"] ?? "");
							break;
						case "list_causal_beliefs":
						case "list_scratchpads":
							result = await handler();
							break;
						case "open_causal_belief":
						case "open_scratchpad":
							result = await handler(args["id"] ?? "");
							_causalBelief = null;
							_scratchpad = null; // force re-load from disk
							break;
						case "create_causal_belief":
						case "create_scratchpad":
							result = await handler(
								args["title"] ?? "",
								args["objective"] ?? "",
								args["data_path"],
							);
							_causalBelief = null;
							_scratchpad = null; // force re-load from disk
							break;
						case "write_causal_belief":
						case "write_scratchpad":
							// Pass the structured update directly. The handler still accepts the
							// legacy `content` JSON string for older clients.
							result = await handler(args);
							_causalBelief = null;
							_scratchpad = null; // force re-load from disk
							break;
						case "causal_diagnose":
							result = await handler(
								args["data_path"] ?? "",
								args["treatment"] ?? undefined,
								args["outcome"] ?? undefined,
							);
							break;
						case "causal_discover":
							result = await handler(
								args["method"] ?? "",
								args["data_path"] ?? "",
								args["alpha"] ?? "",
								args["indep_test"] ?? "",
								args["lambda1"] ?? "",
								args["w_threshold"] ?? "",
								args["score_func"] ?? "",
								args["depth"] ?? "",
								args["timeout"] !== undefined ? String(args["timeout"]) : "",
								{ useCausalBelief: effectiveCapabilities.agentMode !== "wo_causal_state" },
							);
							break;
						case "causal_infer":
							// The shared mapper remains the single positional contract:
							// handler(...mapCausalInferToolArgs(args)) plus the optional runtime.
							result = await handler(
								...mapCausalInferToolArgs(args),
								{
									useCausalBelief: effectiveCapabilities.agentMode !== "wo_causal_state",
									graphEdges: parseExplicitGraphEdges(args["graph_edges"]),
								},
							);
							break;
						case "causal_scout":
							result = await handler(args);
							break;
						case "export_belief_adj_matrix":
						case "export_scratchpad_adj_matrix":
							result = await handler(args["path"] ?? "");
							break;
						case "search_causal_cognition":
						case "update_causal_cognition":
						case "write_cognition_note":
						case "instantiate_cognition_note":
							result = await handler(args);
							break;
						default:
							result =
								`Error: No argument mapping for tool '${name}'. ` +
								"Please update the orchestrator dispatch switch.";
					}
				} catch (e) {
					result = `Error executing ${name}: ${String(e)}`;
				}
			} else {
				result = `Error: Tool '${name}' not found in toolHandlers registry.`;
			}

			if (!capabilityError && isSuccessfulToolResult(result)) {
				if (CAUSAL_BELIEF_ENTRY_TOOLS.has(name)) {
					causalWorkflowEntered = true;
				}
				if (
					(name === "update_causal_cognition" || name === "write_cognition_note") &&
					isSuccessfulCognitionMemoryResult(result)
				) {
					successfulCognitionMemory = true;
				}
			}

			onUpdate(`EVENT:END:${name}`);
			onToolResult?.({
				tool: name,
				arguments: args,
				result,
				...(capabilityError ? { capabilityError } : {}),
			});

			// Send real-time causal belief update for tools that modify it
			if (
				stateEnabled &&
				!capabilityError &&
				(name === "write_causal_belief" ||
					name === "write_scratchpad" ||
					name === "instantiate_cognition_note" ||
					name === "create_causal_belief" ||
					name === "create_scratchpad" ||
					name === "open_causal_belief" ||
					name === "open_scratchpad" ||
					name === "causal_discover" ||
					name === "causal_infer")
			) {
				_causalBelief = loadActiveBelief();
				_scratchpad = _causalBelief;
				if (_causalBelief) {
					try {
						onUpdate(`CAUSAL_BELIEF:${JSON.stringify(_causalBelief)}`);
						onUpdate(`SCRATCHPAD:${JSON.stringify(_causalBelief)}`);
					} catch {
						/* non-fatal serialization error */
					}
				}
			}

			conversationHistory.push({
				role: "tool",
				tool_call_id: toolCall.id,
				content: result,
			});
		}
	}

	// A curator pass must never consume an otherwise usable user-facing answer,
	// even if it exhausts the ordinary agent iteration budget.
	if (emitAndFinalizePreservedCognitionAnswer()) return;

	if (iterations >= MAX_ITERATIONS) {
		onUpdate("WARN:Max iterations reached");
	}
}
