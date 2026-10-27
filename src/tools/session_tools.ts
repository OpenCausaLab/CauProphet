/**
 * session_tools.ts
 *
 * Manages **conversation sessions** — independent from causal scratchpads.
 * One session (conversation) can span multiple scratchpads; the agent decides
 * which scratchpad to use via its built-in tools.
 *
 * Storage: <cwd>/.causalclaw/sessions/{id}.json
 */

import { existsSync, mkdirSync, readdirSync, readFileSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import { atomicWriteFileSync } from "@/core/resource_lock";
import type { AgentMode } from "@/core/agent_capabilities";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ThinkBlock { kind: "think"; content: string; }
export interface ContentBlock { kind: "content"; content: string; }
export interface ToolBlock { kind: "tool"; tool: string; state: "running" | "done"; detail?: string; }
export interface WarnBlock { kind: "warn"; content: string; }
export type MessageBlock = ThinkBlock | ContentBlock | ToolBlock | WarnBlock;

/** A single tool invocation record stored for display purposes (no result). */
export interface PersistedToolCallRecord {
	tool: string;
	/** Human-readable argument summary (e.g. the command, file path, or search query). */
	detail?: string;
}

export interface PersistedChatTurn {
	role: "user" | "assistant";
	/** Plain text — stripped of <think> reasoning blocks. Null for tool-only turns. */
	content: string | null;
	timestamp: string;
	/** Accumulated <think>…</think> content from this turn (for display only). */
	think?: string;
	/** Tool calls made during this turn, stored for display restoration (no results). */
	toolCalls?: PersistedToolCallRecord[];
	/** Structured sequential blocks preserving the exact timeline of intermediate outputs, thoughts, and tool executions. */
	blocks?: MessageBlock[];
	/**
	 * Lifecycle status of this turn.
	 * - `"streaming"`: The agent loop is still in progress; the turn is being built incrementally.
	 * - `"done"`: The agent loop completed normally; the turn is final.
	 * When absent, the turn is assumed to be done (backwards compatibility).
	 */
	status?: "streaming" | "done";
}

/**
 * A compressed summary of session turns [0..coversUpToIndex-1].
 * The working context window is: [summary node] + turns[historyStartIndex..].
 */
export interface CompactSummary {
	generatedAt: string;
	/** turns[0..coversUpToIndex-1] are covered by this summary. */
	coversUpToIndex: number;
	/** Structured Markdown summary text. */
	content: string;
	/** Number of turns compressed (for display). */
	turnCount: number;
}

export interface Session {
	id: string;
	title: string;
	createdAt: string;
	updatedAt: string;
	/** Standard application mode for this conversation. */
	agentMode?: AgentMode;
	turns: PersistedChatTurn[];
	activeBeliefId?: string;
	/** Backward compatibility alias */
	activeScratchpadId?: string;
	/** Context window token occupancy from the last API call (used for auto-compact threshold). */
	contextTokens?: number;

	/**
	 * Latest recursive compact summary.
	 * When present, the LLM working context is:
	 *   [summary node]  +  turns[historyStartIndex..]
	 */
	compactSummary?: CompactSummary;

	/**
	 * Pointer into the turns array: turns before this index are represented
	 * by compactSummary, not sent raw to the LLM.
	 * Defaults to 0 (full history) when absent.
	 */
	historyStartIndex?: number;
}

export interface SessionSummary {
	id: string;
	title: string;
	createdAt: string;
	updatedAt: string;
	turnCount: number;
	isActive: boolean;
	agentMode?: AgentMode;
}

/** Worker processes pin all active-session lookups to their own session. */
function scopedSessionId(): string | null {
	const id = process.env["CAUPROPHET_SESSION_ID"]?.trim();
	return id || null;
}

// ---------------------------------------------------------------------------
// Storage helpers
// ---------------------------------------------------------------------------

export function getSessionDir(): string {
	const newDir = join(process.cwd(), ".cauprophet", "sessions");
	const legacyDir = join(process.cwd(), ".causalclaw", "sessions");
	if (existsSync(legacyDir) && !existsSync(newDir)) {
		return legacyDir;
	}
	return newDir;
}

export function getActiveSessionFile(): string {
	return join(getSessionDir(), "active_session.txt");
}

// ---------------------------------------------------------------------------
// CRUD
// ---------------------------------------------------------------------------

export function createSession(title = "New Conversation", agentMode: AgentMode = "full"): Session {
	return {
		id: randomUUID(),
		title,
		createdAt: new Date().toISOString(),
		updatedAt: new Date().toISOString(),
		agentMode,
		turns: [],
	};
}

export function persistSession(session: Session): void {
	try {
		const dir = getSessionDir();
		mkdirSync(dir, { recursive: true });
		atomicWriteFileSync(join(dir, `${session.id}.json`), `${JSON.stringify(session, null, 2)}\n`);
		// A worker must never move the UI's global active-session pointer while
		// another conversation is selected in the browser.
		if (!scopedSessionId()) atomicWriteFileSync(getActiveSessionFile(), session.id);
	} catch {
		// Non-fatal
	}
}

export function loadSessionById(id: string): Session | null {
	try {
		const path = join(getSessionDir(), `${id}.json`);
		if (!existsSync(path)) return null;
		return JSON.parse(readFileSync(path, "utf-8")) as Session;
	} catch {
		return null;
	}
}

export function loadActiveSession(): Session | null {
	const scopedId = scopedSessionId();
	if (scopedId) return loadSessionById(scopedId);
	const activeFile = getActiveSessionFile();
	if (!existsSync(activeFile)) return null;
	try {
		const activeId = readFileSync(activeFile, "utf-8").trim();
		return loadSessionById(activeId);
	} catch {
		return null;
	}
}

export function listSessions(): SessionSummary[] {
	const dir = getSessionDir();
	if (!existsSync(dir)) return [];

	let activeId = scopedSessionId() ?? "";
	if (!activeId) {
		try {
			activeId = readFileSync(getActiveSessionFile(), "utf-8").trim();
		} catch { /* no active yet */ }
	}

	const files = readdirSync(dir).filter((f) => f.endsWith(".json"));
	const summaries = files.flatMap((f) => {
		try {
			const s = JSON.parse(readFileSync(join(dir, f), "utf-8")) as Session;
			return [{
				id: s.id,
				title: s.title,
				createdAt: s.createdAt,
				updatedAt: s.updatedAt,
				turnCount: s.turns.length,
				isActive: s.id === activeId,
				agentMode: s.agentMode ?? "full",
			} satisfies SessionSummary];
		} catch { return []; }
	});
	summaries.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
	return summaries;
}

/**
 * Deletes a session by ID. If it was the active session, clears the pointer
 * so the next ensureSession() call creates a fresh one.
 */
export function deleteSession(id: string): boolean {
	const dir = getSessionDir();
	const path = join(dir, `${id}.json`);
	if (!existsSync(path)) return false;
	try {
		unlinkSync(path);
		// If the deleted session was the active one, remove the pointer
		const activeFile = getActiveSessionFile();
		if (existsSync(activeFile)) {
			try {
				const activeId = readFileSync(activeFile, "utf-8").trim();
				if (activeId === id) {
					unlinkSync(activeFile);
				}
			} catch { /* ignore */ }
		}
		return true;
	} catch {
		return false;
	}
}

/**
 * Appends a turn to the session and persists (returns updated session).
 */
export function appendTurnToSession(session: Session, turn: PersistedChatTurn, contextTokens?: number): Session {
	const updated: Session = {
		...session,
		turns: [...session.turns, turn],
		updatedAt: new Date().toISOString(),
	};
	if (contextTokens !== undefined) {
		updated.contextTokens = contextTokens;
	}
	persistSession(updated);
	return updated;
}

/**
 * Replaces the **last** turn in the session with `updatedTurn` and persists.
 * Used for incremental checkpoint-save during the agent loop: the in-flight
 * assistant turn is created early (status="streaming") and then updated
 * each time a new block (think / tool / content) is produced.
 *
 * Returns the updated Session object.
 */
export function updateLastTurnInSession(
	session: Session,
	updatedTurn: PersistedChatTurn,
	contextTokens?: number,
): Session {
	const turns = [...session.turns];
	if (turns.length === 0) {
		// Should not happen, but fall back to append
		turns.push(updatedTurn);
	} else {
		turns[turns.length - 1] = updatedTurn;
	}
	const updated: Session = {
		...session,
		turns,
		updatedAt: new Date().toISOString(),
	};
	if (contextTokens !== undefined) {
		updated.contextTokens = contextTokens;
	}
	persistSession(updated);
	return updated;
}

/**
 * Updates the session title and persists.
 */
export function updateSessionTitle(session: Session, title: string): Session {
	const updated: Session = { ...session, title, updatedAt: new Date().toISOString() };
	persistSession(updated);
	return updated;
}

/**
 * Reconstructs the **working** LLM context from a session.
 *
 * Working context = [compact summary node (if any)] + turns[historyStartIndex..]
 *
 * Historical turns before historyStartIndex are kept in turns[] for archival
 * purposes but are NOT sent to the LLM — they are represented by compactSummary.
 */
export function buildWorkingHistory(session: Session): ChatCompletionMessageParam[] {
	const startIdx = session.historyStartIndex ?? 0;

	const recentTurns: ChatCompletionMessageParam[] = session.turns
		.slice(startIdx)
		.filter((t) => t.content !== null)
		.map((t) => ({ role: t.role, content: t.content as string }));

	// Prepend the summary as an assistant context node when compression has occurred
	if (session.compactSummary && startIdx > 0) {
		const summaryMsg: ChatCompletionMessageParam = {
			role: "assistant",
			content:
				`[Context Summary — covers turns 1–${startIdx}]\n\n` +
				session.compactSummary.content,
		};
		return [summaryMsg, ...recentTurns];
	}

	return recentTurns;
}

/**
 * @deprecated Use buildWorkingHistory() instead, which respects the
 * historyStartIndex pointer and prepends compact summaries.
 * Kept for compatibility.
 */
export function sessionToLLMHistory(session: Session): ChatCompletionMessageParam[] {
	return buildWorkingHistory(session);
}
