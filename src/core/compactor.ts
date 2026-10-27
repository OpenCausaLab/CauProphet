/**
 * compactor.ts — CausalClaw 上下文压缩引擎
 *
 * Two phases:
 *   Phase 1 — Microcompact (sync, zero LLM cost):
 *     Truncates oversized tool-call outputs in conversationHistory.
 *
 *   Phase 2 — Autocompact (async, uses main model):
 *     a. Silent Turn: lets agent flush important facts to AGENTS.md
 *     b. LLM summarisation: compresses turns[0..N-5] into a Markdown summary
 *
 * Internal constants (not exposed to users):
 *   COMPACT_THRESHOLD      = 0.80   trigger at 80% of contextWindow
 *   KEEP_LAST_M_TURNS      = 5      keep the newest 5 user-turns verbatim
 *   MICRO_MAX_TOOL_LINES   = 250    truncate tool output beyond 250 lines
 */

import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import { createModelCompletion, getConfiguredModel } from "@/core/llm";
import { GlobalState } from "@/core/state";
import type { CompactSummary } from "@/tools/session_tools";

// ─────────────────────────── Internal constants ───────────────────────────
const COMPACT_THRESHOLD    = 0.80;
const KEEP_LAST_M_TURNS    = 5;
const MICRO_MAX_TOOL_LINES = 250;

// ─────────────────────────────────────────────────────────────────────────
// Phase 1: Microcompact
// ─────────────────────────────────────────────────────────────────────────

/**
 * Truncates tool-call result messages that exceed MICRO_MAX_TOOL_LINES lines.
 * Runs synchronously in O(n) — zero LLM cost.
 */
export function microcompact(
	history: ChatCompletionMessageParam[],
): ChatCompletionMessageParam[] {
	return history.map((msg) => {
		if (msg.role !== "tool" || typeof msg.content !== "string") return msg;
		const lines = msg.content.split("\n");
		if (lines.length <= MICRO_MAX_TOOL_LINES) return msg;

		const stub =
			`[Tool output truncated by microcompact: ${lines.length} lines → showing first ${MICRO_MAX_TOOL_LINES}]\n` +
			lines.slice(0, MICRO_MAX_TOOL_LINES).join("\n") +
			"\n[...remaining output omitted...]";

		return { ...msg, content: stub };
	});
}

// ─────────────────────────────────────────────────────────────────────────
// Tool-pairing repair (defensive self-heal)
// ─────────────────────────────────────────────────────────────────────────

/**
 * Repairs the assistant-`tool_calls` ↔ `tool` pairing invariant that the
 * Chat Completions API enforces. Any breakage here poisons the whole
 * conversation (the API rejects every subsequent request with
 * "Messages with role 'tool' must be a response to a preceding message with
 * 'tool_calls'"), so we run this immediately before every API call.
 *
 * The pass is order-preserving and guarantees, for the returned array:
 *   - every assistant `tool_call` id is followed by exactly one matching
 *     `tool` message, in the same order as declared (missing ones are
 *     synthesised with a placeholder result);
 *   - `tool` messages that don't answer an immediately-preceding
 *     assistant `tool_calls` (orphans / duplicates) are dropped.
 *
 * This makes the loop resilient to partial tool execution (abort mid-chain),
 * skipped non-function tool calls, and JSON.parse crashes that leave a
 * dangling assistant turn — and it heals an already-corrupted in-memory
 * history on the very next request, without needing a process restart.
 */
export function sanitizeToolMessages(
	messages: ChatCompletionMessageParam[],
): ChatCompletionMessageParam[] {
	const out: ChatCompletionMessageParam[] = [];
	let i = 0;

	while (i < messages.length) {
		const msg = messages[i];
		if (!msg) { i++; continue; }

		const toolCalls =
			msg.role === "assistant"
				? (msg as { tool_calls?: Array<{ id: string; type?: string }> }).tool_calls
				: undefined;

		if (msg.role === "assistant" && toolCalls && toolCalls.length > 0) {
			out.push(msg);

			// Collect the run of tool messages that immediately follows.
			const responses = new Map<string, ChatCompletionMessageParam>();
			let j = i + 1;
			for (; j < messages.length; j++) {
				const t = messages[j];
				if (!t || t.role !== "tool") break;
				const id = (t as { tool_call_id?: string }).tool_call_id;
				if (id && !responses.has(id)) responses.set(id, t);
				// duplicates / unknown ids are simply skipped over and dropped
			}

			// Emit one response per declared tool_call, in declaration order.
			for (const tc of toolCalls) {
				const existing = responses.get(tc.id);
				if (existing) {
					out.push(existing);
				} else {
					out.push({
						role: "tool",
						tool_call_id: tc.id,
						content: "[No tool result was recorded for this call — it was interrupted or failed.]",
					} as ChatCompletionMessageParam);
				}
			}

			i = j;
			continue;
		}

		if (msg.role === "tool") {
			// Orphan tool message with no preceding assistant tool_calls — drop it.
			i++;
			continue;
		}

		out.push(msg);
		i++;
	}

	return out;
}

// ─────────────────────────────────────────────────────────────────────────
// Threshold check
// ─────────────────────────────────────────────────────────────────────────

/**
 * Returns true when accumulated token usage has crossed 80% of the
 * configured context window, indicating autocompact should run.
 */
export function shouldAutocompact(contextTokens: number): boolean {
	const config = GlobalState.getInstance().config;
	const limit  = config.contextWindow ?? 128_000;
	return contextTokens >= limit * COMPACT_THRESHOLD;
}

// ─────────────────────────────────────────────────────────────────────────
// Phase 2a: Silent Turn — pre-compact memory flush
// ─────────────────────────────────────────────────────────────────────────

/**
 * The content injected as a hidden user turn before compression.
 * The agent must either call write_file to persist key facts or reply
 * "NO_PERSIST_NEEDED" to skip at minimal cost.
 */
export const SILENT_TURN_CONTENT =
	"[SYSTEM — INTERNAL — NOT VISIBLE TO USER]\n\n" +
	"Your working context is about to be compressed. Before that happens:\n" +
	"1. Review the recent conversation for information worth long-term preservation:\n" +
	"   - User's explicit constraints or preferences stated during the session\n" +
	"   - Key architectural decisions that were made\n" +
	"   - Critical file paths, environment variables, or configuration values\n" +
	"2. If any such information exists, use the write_file tool to append it to\n" +
	"   AGENTS.md in the workspace root (create the file if it doesn't exist).\n" +
	"3. If nothing worth preserving was found, respond with exactly: NO_PERSIST_NEEDED\n\n" +
	"Be very selective — only persist facts that would be non-obvious to re-derive.";

// ─────────────────────────────────────────────────────────────────────────
// Phase 2b: LLM Autocompact
// ─────────────────────────────────────────────────────────────────────────

const COMPACT_SYSTEM_PROMPT = `You are a session summarizer for an AI research agent called CauProphet.
Compress the provided conversation history into a dense, structured summary.
Output ONLY the summary in this exact Markdown format, nothing else:

## Session Compact Summary
**Generated at**: {TIMESTAMP}
**Covers turns**: 1–{N}

### Current Objective
(1–2 sentences: what the user is trying to accomplish)

### Key Decisions & Findings
- (bullet point)

### Data & Environment
- Dataset: (file path and key characteristics, if mentioned)
- Methods tried: (list)

### Current State
(Free-form paragraph: where things stand, what is working, what is blocked)

### Open Questions / Next Steps
- (item)`;

export interface AutocompactResult {
	/** The Markdown summary covering the compressed portion. */
	summary: CompactSummary;
	/**
	 * New historyStartIndex to set on the session.
	 * turns[0..newHistoryStartIndex-1] are now covered by the summary.
	 */
	newHistoryStartIndex: number;
	/**
	 * Rebuilt conversationHistory to replace the in-memory array:
	 *   [summary assistant node, ...last-M-turns messages]
	 */
	newConversationHistory: ChatCompletionMessageParam[];
}

/**
 * Runs LLM-based autocompact on the in-memory conversationHistory.
 *
 * Split strategy: find the user-message boundary that leaves the last
 * KEEP_LAST_M_TURNS user turns untouched, compress everything before it.
 *
 * @param history         Current in-memory conversationHistory (may include tool messages)
 * @param persistedTurnCount  session.turns.length at call time (for index calculation)
 * @param onStatus        Status callback matching orchestrator protocol
 */
export async function autocompact(
	history: ChatCompletionMessageParam[],
	persistedTurnCount: number,
	onStatus: (s: string) => void,
	model?: string,
): Promise<AutocompactResult | null> {
	// Find the Nth-from-last user message boundary (excluding silent turns)
	const userIndices: number[] = [];
	for (let i = 0; i < history.length; i++) {
		const msg = history[i];
		if (msg && msg.role === "user" && msg.content !== SILENT_TURN_CONTENT) {
			userIndices.push(i);
		}
	}

	if (userIndices.length <= KEEP_LAST_M_TURNS) {
		onStatus("WARN:Autocompact skipped — fewer than " + (KEEP_LAST_M_TURNS + 1) + " user turns in history");
		return null;
	}

	// splitAt: first index of the "keep" region
	const keepFromIdx = userIndices[userIndices.length - KEEP_LAST_M_TURNS]!;
	const toCompress   = history.slice(0, keepFromIdx);
	const toKeep       = history.slice(keepFromIdx);

	onStatus("STATUS:Compressing context history…");

	try {
		const now   = new Date().toISOString();
		const turnN = userIndices.length - KEEP_LAST_M_TURNS; // number of user turns being compressed

		const prompt = COMPACT_SYSTEM_PROMPT
			.replace("{TIMESTAMP}", now)
			.replace("{N}",         String(turnN));

		const response = await createModelCompletion({
			model: model ?? getConfiguredModel(),
			messages: [
				{ role: "system", content: prompt },
				...sanitizeToolMessages(toCompress),
				{ role: "user", content: "Generate the compact summary for the above conversation." },
			],
		});

		const summaryContent =
			response.choices[0]?.message?.content?.trim() ??
			"[Summary generation failed]";

		// newHistoryStartIndex: index into session.turns[] where raw (uncompressed) turns begin.
		// We are compressing `turnN` user-assistant pairs, so the boundary should be at turn index
		// `turnN * 2` (2 persisted turns per pair: one user + one final assistant).
		//
		// Using `persistedTurnCount - KEEP_LAST_M_TURNS * 2` is incorrect when compaction
		// triggers at the start of a new agentic turn: at that point the current user message
		// is already persisted (making persistedTurnCount odd) but its assistant reply has not
		// yet been written. Subtracting from an odd total produces an off-by-one that places
		// the historyStartIndex inside a pair (pointing at an assistant turn instead of the
		// next user turn), causing buildWorkingHistory() to start context with a dangling
		// assistant message that has no preceding user message.
		//
		// Computing forward from `turnN` is always correct: `turnN` is the exact count of
		// complete user-turn pairs that went into `toCompress`.
		const newHistoryStartIndex = Math.min(persistedTurnCount, turnN * 2);

		const summary: CompactSummary = {
			generatedAt:     now,
			coversUpToIndex: newHistoryStartIndex,
			content:         summaryContent,
			turnCount:       newHistoryStartIndex,
		};

		// Rebuild conversation history with summary prepended
		const summaryNode: ChatCompletionMessageParam = {
			role:    "assistant",
			content: `[Context Summary — covers turns 1–${newHistoryStartIndex}]\n\n${summaryContent}`,
		};
		const newConversationHistory: ChatCompletionMessageParam[] = [summaryNode, ...toKeep];

		onStatus("STATUS:Context compression complete.");
		return { summary, newHistoryStartIndex, newConversationHistory };

	} catch (e) {
		onStatus(`WARN:Autocompact failed: ${String(e)}`);
		return null;
	}
}
