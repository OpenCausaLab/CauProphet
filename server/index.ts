/**
 * CausalClaw Web Server
 *
 * Bridges the core orchestrator (src/core/orchestrator.ts) to a Web UI
 * via HTTP REST + WebSocket.
 *
 * Endpoints:
 *   GET  /api/config             → Returns current config (key masked)
 *   POST /api/config             → Saves updated config
 *   POST /api/test-connection    → Tests LLM API connectivity (server-side, no CORS)
 *   GET  /api/sessions           → Lists all conversation sessions
 *   POST /api/session/new        → Creates a new session
 *   POST /api/session/:id        → Switches to an existing session
 *   GET  /api/scratchpad         → Returns current scratchpad state
 *   GET  /api/causal-cognition/index → Returns the read-only cognition index
 *   GET  /api/scratchpads        → Lists all scratchpads (structured)
 *   POST /api/scratchpad/new     → Creates a new scratchpad
 *   POST /api/scratchpad/:id     → Switches to a scratchpad by id
 *   WS   /ws                     → Chat WebSocket channel
 */

import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { copyFileSync } from "node:fs";
import { GlobalState } from "@/core/state";
import {
	getCausalBelief,
	getScratchpad,
	getSession,
	resetSession,
	switchSession,
	triggerManualCompact,
	getContextTokens,
	getContextWindow,
	notifySessionRenamed,
} from "@/core/orchestrator";
import { resetOpenAIClient } from "@/core/llm";
import {
	loadActiveBelief,
	loadActiveScratchpad,
	loadBeliefById,
	loadScratchpadById,
	persistBelief,
	persistScratchpad,
	getBeliefDir,
	getScratchpadDir,
} from "@/tools/causal_belief_tools";
import {
	createSession,
	persistSession,
	listSessions,
	loadSessionById,
	loadActiveSession,
	updateSessionTitle,
	deleteSession,
	type Session,
} from "@/tools/session_tools";
import {
	createCausalBelief,
	createScratchpad,
	deserialiseBelief,
	deserialiseScratchpad,
	type CausalBelief,
	type CausalScratchpad,
} from "@/memory/causal_belief";
import {
	assessCausalRelation,
	CausalCognitionGraphStore,
	type CausalCognitionGraphData,
} from "@/memory/causal_cognition_graph";

const PORT        = Number(process.env["PORT"] ?? 3001);
const HOST        = process.env["HOST"] ?? "127.0.0.1";
const projectRoot = join(fileURLToPath(import.meta.url), "..", "..");
const webDistDir  = join(projectRoot, "web", "dist");
/** True when web/dist/index.html exists — production / npm-installed mode */
const hasStaticFrontend = existsSync(join(webDistDir, "index.html"));

// Supported MIME types for static file serving
const MIME: Record<string, string> = {
	".html": "text/html; charset=utf-8",
	".js":   "application/javascript",
	".mjs":  "application/javascript",
	".css":  "text/css",
	".svg":  "image/svg+xml",
	".png":  "image/png",
	".jpg":  "image/jpeg",
	".ico":  "image/x-icon",
	".json": "application/json",
	".woff2":"font/woff2",
	".woff": "font/woff",
	".ttf":  "font/ttf",
};

// ---------------------------------------------------------------------------
// Bootstrap: replicate the workspace init from cli/index.tsx
// ---------------------------------------------------------------------------
async function bootstrap() {
	const state = GlobalState.getInstance();
	const projectRoot2 = join(fileURLToPath(import.meta.url), "..", "..");

	// Change to workspace if configured
	if (state.config.workspace && existsSync(state.config.workspace)) {
		process.chdir(state.config.workspace);
	}

	// Copy AGENTS.md template if missing
	try {
		const templateAgentsMd = join(projectRoot2, "templates", "AGENTS.md");
		const workspaceAgentsMd = join(process.cwd(), "AGENTS.md");
		if (existsSync(templateAgentsMd) && !existsSync(workspaceAgentsMd)) {
			copyFileSync(templateAgentsMd, workspaceAgentsMd);
		}
	} catch {
		// Non-fatal
	}
}

// ---------------------------------------------------------------------------
// WebSocket message types (shared protocol with frontend)
// ---------------------------------------------------------------------------

// Client → Server
interface ClientMessage {
	type: "chat" | "reset" | "interrupt" | "watch";
	/** Session whose events this socket wants to receive. */
	sessionId?: string;
	/** For type=chat: the user's message text */
	text?: string;
	/** For type=chat: whether to stream tokens incrementally */
	streaming?: boolean;
	/** For type=reset: optional objective for the new session */
	objective?: string;
}

// Server → Client
type ServerMessage =
	| { type: "think"; content: string }
	| { type: "content"; content: string }
	| { type: "tool_start"; tool: string; detail?: string }
	| { type: "tool_end"; tool: string }
	| { type: "warn"; content: string }
	| { type: "scratchpad_update"; scratchpad: unknown }
	| { type: "done"; scratchpad: unknown | null; contextTokens: number; contextWindow: number; sessionId?: string }
	| { type: "job"; sessionId: string; state: "running" | "done" | "error" | "aborted"; startedAt?: string; error?: string }
	| { type: "error"; content: string };

// ---------------------------------------------------------------------------
// HTTP + WebSocket server
// ---------------------------------------------------------------------------
await bootstrap();

type JobSubscriber = { send: (data: string) => void };
type AgentJob = {
	sessionId: string;
	startedAt: string;
	process: ReturnType<typeof Bun.spawn>;
	subscribers: Set<JobSubscriber>;
	terminal: boolean;
};

/**
 * Owns detached agent workers.  There is at most one turn per conversation,
 * while different conversations run in parallel.  A socket is only a
 * subscriber; closing it never cancels the worker.
 */
class AgentJobManager {
	private readonly jobs = new Map<string, AgentJob>();

	start(sessionId: string, prompt: string, subscriber: JobSubscriber | undefined): boolean {
		if (this.jobs.has(sessionId)) return false;
		const startedAt = new Date().toISOString();
		const child = Bun.spawn([process.execPath, "run", join(projectRoot, "server", "agent_worker.ts")], {
			cwd: projectRoot,
			stdin: new Blob([JSON.stringify({ prompt })]),
			stdout: "pipe",
			stderr: "pipe",
			env: {
				...process.env,
				CAUPROPHET_SESSION_ID: sessionId,
				CAUPROPHET_AGENT_WORKSPACE: process.cwd(),
			},
		});
		const job: AgentJob = {
			sessionId,
			startedAt,
			process: child,
			subscribers: new Set(),
			terminal: false,
		};
		this.jobs.set(sessionId, job);
		if (subscriber) this.subscribe(sessionId, subscriber);
		void this.consume(job);
		return true;
	}

	isRunning(sessionId: string): boolean {
		return this.jobs.has(sessionId);
	}

	subscribe(sessionId: string, subscriber: JobSubscriber): void {
		this.unsubscribe(subscriber);
		const job = this.jobs.get(sessionId);
		if (!job) return;
		job.subscribers.add(subscriber);
		subscriber.send(JSON.stringify({
			type: "job",
			sessionId,
			state: "running",
			startedAt: job.startedAt,
		} satisfies ServerMessage));
	}

	unsubscribe(subscriber: JobSubscriber): void {
		for (const job of this.jobs.values()) job.subscribers.delete(subscriber);
	}

	interrupt(sessionId: string): boolean {
		const job = this.jobs.get(sessionId);
		if (!job) return false;
		job.process.kill();
		return true;
	}

	private broadcast(job: AgentJob, message: ServerMessage): void {
		const serialized = JSON.stringify(message);
		for (const subscriber of job.subscribers) {
			try {
				subscriber.send(serialized);
			} catch {
				job.subscribers.delete(subscriber);
			}
		}
	}

	private handleWorkerMessage(job: AgentJob, line: string): void {
		const prefix = "__CAUPROPHET_EVENT__";
		if (!line.startsWith(prefix)) return;
		let payload: Record<string, unknown>;
		try {
			payload = JSON.parse(line.slice(prefix.length)) as Record<string, unknown>;
		} catch {
			return;
		}
		if (payload["type"] === "update" && typeof payload["status"] === "string") {
			this.broadcast(job, parseUpdate(payload["status"]));
			return;
		}
		if (payload["type"] === "done") {
			job.terminal = true;
			const state = payload["aborted"] === true ? "aborted" : "done";
			this.broadcast(job, {
				type: "done",
				scratchpad: payload["scratchpad"] ?? null,
				contextTokens: typeof payload["contextTokens"] === "number" ? payload["contextTokens"] : 0,
				contextWindow: typeof payload["contextWindow"] === "number" ? payload["contextWindow"] : 128_000,
			});
			this.broadcast(job, { type: "job", sessionId: job.sessionId, state });
			return;
		}
		if (payload["type"] === "error") {
			job.terminal = true;
			const error = typeof payload["content"] === "string" ? payload["content"] : "Agent worker failed";
			this.broadcast(job, { type: "error", content: error });
			this.broadcast(job, { type: "job", sessionId: job.sessionId, state: "error", error });
		}
	}

	private async consume(job: AgentJob): Promise<void> {
		const stdout = job.process.stdout as ReadableStream<Uint8Array>;
		// Drain stderr concurrently; a verbose provider/tool must not block the
		// worker while the server is waiting for protocol events on stdout.
		const stderrDone = new Response(job.process.stderr as ReadableStream<Uint8Array>)
			.text()
			.catch(() => "");
		const reader = stdout.getReader();
		const decoder = new TextDecoder();
		let buffer = "";
		try {
			while (true) {
				const { value, done } = await reader.read();
				if (done) break;
				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split("\n");
				buffer = lines.pop() ?? "";
				for (const line of lines) this.handleWorkerMessage(job, line.trim());
			}
			buffer += decoder.decode();
			if (buffer.trim()) this.handleWorkerMessage(job, buffer.trim());
		} catch (error) {
			if (!job.terminal) {
				job.terminal = true;
				this.broadcast(job, { type: "error", content: String(error) });
			}
		} finally {
			await stderrDone;
			await job.process.exited;
			if (!job.terminal) {
				this.broadcast(job, { type: "job", sessionId: job.sessionId, state: "error", error: "Agent worker exited without a final result" });
			}
			if (this.jobs.get(job.sessionId) === job) this.jobs.delete(job.sessionId);
		}
	}
}

const agentJobs = new AgentJobManager();

const server = Bun.serve({
	port: PORT,
	hostname: HOST,

	// ---- HTTP routes ----
	async fetch(req, server) {
		const url = new URL(req.url);
		if (!requestOriginAllowed(req)) {
			return new Response("Forbidden origin", { status: 403 });
		}

		// CORS preflight
		if (req.method === "OPTIONS") {
			return corsResponse(new Response(null, { status: 204 }));
		}

		// WebSocket upgrade
		if (url.pathname === "/ws") {
			const upgraded = server.upgrade(req);
			if (!upgraded) {
				return corsResponse(new Response("WebSocket upgrade failed", { status: 400 }));
			}
			return undefined as unknown as Response;
		}

		// --- REST API ---
		if (url.pathname === "/api/config") {
			if (req.method === "GET") return handleGetConfig();
			if (req.method === "POST") return handlePostConfig(req);
		}

		if (url.pathname === "/api/test-connection" && req.method === "POST") {
			return handleTestConnection(req);
		}

		if (url.pathname === "/api/causal-cognition/index" && req.method === "GET") {
			return handleGetCausalCognitionIndex();
		}

		// ---- Upload endpoint ----
		if (url.pathname === "/api/upload" && req.method === "POST") {
			return handleUploadFile(req);
		}

		// ---- Session endpoints ----
		if (url.pathname === "/api/session/active" && req.method === "GET") {
			const session = getSession();
			const recentTurns = session.turns.slice(-20);
			const scratchpad = session.activeScratchpadId
				? loadScratchpadById(session.activeScratchpadId)
				: null;
			return jsonResponse({
				...session,
				recentTurns,
				scratchpad,
				contextTokens: getContextTokens(),
				contextWindow: getContextWindow(),
				isRunning: agentJobs.isRunning(session.id),
			});
		}

		if (url.pathname === "/api/sessions" && req.method === "GET") {
			return jsonResponse(listSessions().map((session) => ({
				...session,
				isRunning: agentJobs.isRunning(session.id),
			})));
		}

		if (url.pathname === "/api/session/new" && req.method === "POST") {
			return handleNewSession(req);
		}

		if (url.pathname === "/api/session/compact" && req.method === "POST") {
			return handleCompactSession();
		}

		const sessionSwitchMatch = url.pathname.match(/^\/api\/session\/(.+)$/);
		if (sessionSwitchMatch) {
			if (req.method === "POST") return handleSwitchSession(sessionSwitchMatch[1] ?? "");
			if (req.method === "PATCH") return handleRenameSession(sessionSwitchMatch[1] ?? "", req);
			if (req.method === "DELETE") return handleDeleteSession(sessionSwitchMatch[1] ?? "");
		}

		// ---- Causal Belief / Scratchpad endpoints ----
		if ((url.pathname === "/api/causal-belief" || url.pathname === "/api/scratchpad") && req.method === "GET") {
			return handleGetScratchpad();
		}

		if ((url.pathname === "/api/causal-beliefs" || url.pathname === "/api/scratchpads") && req.method === "GET") {
			return handleListScratchpads();
		}

		if ((url.pathname === "/api/causal-belief/new" || url.pathname === "/api/scratchpad/new") && req.method === "POST") {
			return handleNewScratchpad(req);
		}

		const switchBeliefMatch = url.pathname.match(/^\/api\/(?:causal-belief|scratchpad)\/(.+)$/);
		if (switchBeliefMatch && req.method === "POST") {
			return handleSwitchScratchpad(switchBeliefMatch[1] ?? "");
		}

		// ── Static file serving (production / npm-installed mode) ────────────
		// Serves web/dist/ when it exists. Falls back to index.html for SPA routing.
		if (hasStaticFrontend) {
			const pathname  = url.pathname === "/" ? "/index.html" : url.pathname;
			const filePath  = join(webDistDir, pathname);
			const ext       = pathname.slice(pathname.lastIndexOf("."));
			const mimeType  = MIME[ext] ?? "application/octet-stream";

			if (existsSync(filePath)) {
				return corsResponse(
					new Response(Bun.file(filePath), {
						headers: { "Content-Type": mimeType },
					}),
				);
			}

			// SPA fallback: unknown path → serve index.html (client handles routing)
			return corsResponse(
				new Response(Bun.file(join(webDistDir, "index.html")), {
					headers: { "Content-Type": "text/html; charset=utf-8" },
				}),
			);
		}

		return corsResponse(new Response("Not found", { status: 404 }));
	},

	// ---- WebSocket handlers ----
	websocket: {
		open(ws) {
			console.log("[WS] Client connected");
		},

		async message(ws, raw) {
			let msg: ClientMessage;
			try {
				msg = JSON.parse(typeof raw === "string" ? raw : raw.toString()) as ClientMessage;
			} catch {
				ws.send(JSON.stringify({ type: "error", content: "Invalid JSON" } satisfies ServerMessage));
				return;
			}

			if (msg.type === "interrupt") {
				const sessionId = msg.sessionId?.trim() || getSession().id;
				if (!agentJobs.interrupt(sessionId)) {
					ws.send(JSON.stringify({ type: "warn", content: "No running agent turn for this conversation." } satisfies ServerMessage));
				}
				return;
			}

			if (msg.type === "watch") {
				const sessionId = msg.sessionId?.trim();
				if (sessionId) agentJobs.subscribe(sessionId, ws);
				return;
			}

			if (msg.type === "reset") {
				const selectedId = msg.sessionId?.trim();
				if (selectedId && agentJobs.isRunning(selectedId)) {
					ws.send(JSON.stringify({ type: "error", content: "This conversation is still running; wait for it to finish before resetting." } satisfies ServerMessage));
					return;
				}
				if (selectedId) switchSession(selectedId);
				// Create a new session (clear history); leave scratchpad untouched.
				resetSession();
				ws.send(JSON.stringify({
					type: "done",
					scratchpad: getScratchpad(),
					contextTokens: getContextTokens(),
					contextWindow: getContextWindow(),
					sessionId: getSession().id,
				} satisfies ServerMessage));
				return;
			}

			if (msg.type === "chat" && msg.text) {
				const sessionId = msg.sessionId?.trim() || getSession().id;
				const session = loadSessionById(sessionId);
				if (!session) {
					ws.send(JSON.stringify({ type: "error", content: `No session with id ${sessionId}` } satisfies ServerMessage));
					return;
				}
				if (!agentJobs.start(sessionId, msg.text, ws)) {
					ws.send(JSON.stringify({ type: "error", content: "This conversation already has a running agent turn." } satisfies ServerMessage));
				}
			}
		},

		close(ws) {
			agentJobs.unsubscribe(ws);
			console.log("[WS] Client disconnected");
		},
	},
});

console.log(`[CauProphet Server] Listening on http://${HOST}:${server.port}`);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseUpdate(status: string): ServerMessage {
	if (status.startsWith("THINK:")) {
		return { type: "think", content: status.slice(6) };
	}
	if (status.startsWith("CONTENT:")) {
		return { type: "content", content: status.slice(8) };
	}
	if (status.startsWith("CAUSAL_BELIEF:")) {
		try {
			const scratchpad = JSON.parse(status.slice(14));
			return { type: "scratchpad_update", scratchpad };
		} catch {
			return { type: "content", content: "" };
		}
	}
	if (status.startsWith("SCRATCHPAD:")) {
		try {
			const scratchpad = JSON.parse(status.slice(11));
			return { type: "scratchpad_update", scratchpad };
		} catch {
			// Malformed JSON — ignore
			return { type: "content", content: "" };
		}
	}
	if (status.startsWith("EVENT:START:")) {
		const payload = status.slice(12);
		const colonIdx = payload.indexOf(":");
		if (colonIdx !== -1) {
			const tool = payload.slice(0, colonIdx);
			const detail = payload.slice(colonIdx + 1);
			return { type: "tool_start", tool, detail };
		}
		return { type: "tool_start", tool: payload };
	}
	if (status.startsWith("EVENT:END:")) {
		return { type: "tool_end", tool: status.slice(10) };
	}
	if (status.startsWith("WARN:")) {
		return { type: "warn", content: status.slice(5) };
	}
	return { type: "content", content: status };
}

function corsResponse(res: Response): Response {
	res.headers.set("Access-Control-Allow-Origin", "*");
	res.headers.set("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS");
	res.headers.set("Access-Control-Allow-Headers", "Content-Type");
	return res;
}

function requestOriginAllowed(req: Request): boolean {
	const origin = req.headers.get("Origin");
	if (!origin) return true;
	if (
		process.env["CAUPROPHET_ALLOWED_ORIGIN"] &&
		origin === process.env["CAUPROPHET_ALLOWED_ORIGIN"]
	) {
		return true;
	}
	try {
		const hostname = new URL(origin).hostname;
		return (
			hostname === "localhost" ||
			hostname === "127.0.0.1" ||
			hostname === "::1" ||
			hostname === "[::1]"
		);
	} catch {
		return false;
	}
}

function jsonResponse(data: unknown, status = 200): Response {
	return corsResponse(
		new Response(JSON.stringify(data), {
			status,
			headers: { "Content-Type": "application/json" },
		}),
	);
}

function handleGetConfig(): Response {
	const cfg = GlobalState.getInstance().config;
	return jsonResponse({
		openaiBaseUrl: cfg.openaiBaseUrl,
		// Mask the API key — only show if set
		openaiApiKey: cfg.openaiApiKey ? "••••••••" + cfg.openaiApiKey.slice(-4) : "",
		model: cfg.model,
		workspace: cfg.workspace ?? "",
		contextWindow: cfg.contextWindow ?? 128_000,
	searchProvider: cfg.searchProvider ?? "ddg",
	braveApiKey: cfg.braveApiKey ? "••••••••" + cfg.braveApiKey.slice(-4) : "",
	bochaApiKey: cfg.bochaApiKey ? "••••••••" + cfg.bochaApiKey.slice(-4) : "",
	});
}

async function handlePostConfig(req: Request): Promise<Response> {
	let body: Record<string, unknown>;
	try {
		body = (await req.json()) as Record<string, string>;
	} catch {
		return jsonResponse({ error: "Invalid JSON" }, 400);
	}

	const state = GlobalState.getInstance();

	if (typeof body["openaiBaseUrl"] === "string") state.config.openaiBaseUrl = body["openaiBaseUrl"];
	if (typeof body["model"] === "string") state.config.model = body["model"];
	if (typeof body["workspace"] === "string") state.config.workspace = body["workspace"];
	if (body["contextWindow"] !== undefined) {
		const cw = Number(body["contextWindow"]);
		if (Number.isInteger(cw) && cw > 0) state.config.contextWindow = cw;
	}

	// Web search provider
	if (body["searchProvider"] !== undefined) {
		const sp = body["searchProvider"];
		if (sp === "ddg" || sp === "brave" || sp === "bocha") state.config.searchProvider = sp;
	}

	// Only update keys if a real key is sent (not the masked placeholder)
	if (typeof body["openaiApiKey"] === "string" && body["openaiApiKey"] && !body["openaiApiKey"].startsWith("••••")) {
		state.config.openaiApiKey = body["openaiApiKey"];
	}
	if (typeof body["braveApiKey"] === "string" && body["braveApiKey"] && !body["braveApiKey"].startsWith("••••")) {
		state.config.braveApiKey = body["braveApiKey"];
	}
	if (typeof body["bochaApiKey"] === "string" && body["bochaApiKey"] && !body["bochaApiKey"].startsWith("••••")) {
		state.config.bochaApiKey = body["bochaApiKey"];
	}

	state.saveConfig();
	// Invalidate the cached OpenAI client so next request picks up the new config
	resetOpenAIClient();

	// If workspace changed, chdir
	if (typeof body["workspace"] === "string" && body["workspace"] && existsSync(body["workspace"])) {
		process.chdir(body["workspace"]);
	}

	return jsonResponse({ ok: true });
}

function handleGetScratchpad(): Response {
	const pad = loadActiveScratchpad();
	return jsonResponse(pad);
}

function cognitionIndexPayload(
	graph: CausalCognitionGraphData,
	store: CausalCognitionGraphStore,
) {
	const degree = new Map<string, { incoming: number; outgoing: number }>();
	for (const concept of graph.concepts) {
		degree.set(concept.id, { incoming: 0, outgoing: 0 });
	}
	let activeRelations = 0;
	for (const relation of graph.relations) {
		if (!assessCausalRelation(relation).active) continue;
		activeRelations += 1;
		const cause = degree.get(relation.causeId);
		const effect = degree.get(relation.effectId);
		if (cause) cause.outgoing += 1;
		if (effect) effect.incoming += 1;
	}

	return {
		source: "active",
		sourceLabel: "Workspace",
		storageRelativePath: ".cauprophet/causal_cognition.json",
		updatedAt: existsSync(store.graphPath)
			? statSync(store.graphPath).mtime.toISOString()
			: null,
		schemaVersion: graph.schemaVersion,
		counts: {
			concepts: graph.concepts.length,
			relationVariants: graph.relations.length,
			activeRelations,
			relationClaims: graph.relations.reduce(
				(total, relation) => total + relation.claims.length,
				0,
			),
			episodes: graph.episodes.length,
		},
		concepts: [...graph.concepts]
			.sort((left, right) => left.label.localeCompare(right.label))
			.map((concept) => ({
				id: concept.id,
				label: concept.label,
				aliases: concept.aliases,
				keywords: concept.keywords ?? [],
				description: concept.description,
				activeIncoming: degree.get(concept.id)?.incoming ?? 0,
				activeOutgoing: degree.get(concept.id)?.outgoing ?? 0,
			})),
		relations: [...graph.relations]
			.sort((left, right) => left.id.localeCompare(right.id))
			.map((relation) => ({
				id: relation.id,
				causeId: relation.causeId,
				effectId: relation.effectId,
				mechanism: relation.mechanism,
				contexts: relation.contexts,
				lag: relation.lag,
				claims: relation.claims,
				assessment: assessCausalRelation(relation),
			})),
	};
}

function handleGetCausalCognitionIndex(): Response {
	const store = new CausalCognitionGraphStore();
	return jsonResponse(cognitionIndexPayload(store.load(), store));
}

/** Returns structured scratchpad list. */
function handleListScratchpads(): Response {
	try {
		const dir = getScratchpadDir();
		if (!existsSync(dir)) return jsonResponse([]);

		const session = loadActiveSession();
		const activeId = session?.activeScratchpadId ?? "";

		const files = readdirSync(dir).filter((f) => /^[0-9a-f-]{36}\.json$/.test(f));
		const pads = files.flatMap((f) => {
			try {
				const pad = deserialiseScratchpad(readFileSync(join(dir, f), "utf-8"));
				return [{ id: pad.id, title: pad.title, objective: pad.objective, lastUpdated: pad.lastUpdated, isActive: pad.id === activeId }];
			} catch { return []; }
		});
		pads.sort((a, b) => b.lastUpdated.localeCompare(a.lastUpdated));
		return jsonResponse(pads);
	} catch {
		return jsonResponse([]);
	}
}

async function handleNewScratchpad(req: Request): Promise<Response> {
	let body: { title?: string; objective?: string } = {};
	try {
		body = (await req.json()) as { title?: string; objective?: string };
	} catch {
		// Defaults are fine
	}
	const pad = createScratchpad(body.objective ?? "");
	if (body.title) pad.title = body.title;
	persistScratchpad(pad);
	return jsonResponse(pad);
}

// ---- Session handlers ----

async function handleNewSession(req: Request): Promise<Response> {
	let body: { title?: string } = {};
	try {
		body = (await req.json()) as { title?: string };
	} catch { /* use defaults */ }
	const session = createSession(body.title ?? "New Conversation");
	persistSession(session);
	// Switch orchestrator to this session (resets _contextTokens to 0)
	switchSession(session.id);
	// Return same shape as handleSwitchSession so the frontend gets explicit
	// contextTokens: 0 rather than undefined (which leaks the previous session's value).
	return jsonResponse({
		...session,
		recentTurns: [],
		scratchpad: null,
		contextTokens: getContextTokens(), // always 0 after switchSession on a fresh session
		contextWindow: getContextWindow(),
		isRunning: false,
	});
}

function handleSwitchSession(id: string): Response {
	const session = switchSession(id);
	if (!session) return jsonResponse({ error: `No session with id ${id}` }, 404);
	const recentTurns = session.turns.slice(-20);
	// Also return the scratchpad linked to this session
	const scratchpad = session.activeScratchpadId
		? loadScratchpadById(session.activeScratchpadId)
		: null;
	return jsonResponse({
		...session,
		recentTurns,
		scratchpad,
		contextTokens: getContextTokens(),
		contextWindow: getContextWindow(),
		isRunning: agentJobs.isRunning(session.id),
	});
}

async function handleRenameSession(id: string, req: Request): Promise<Response> {
	if (agentJobs.isRunning(id)) {
		return jsonResponse({ error: "Cannot rename a conversation while its agent turn is running" }, 409);
	}
	let body: { title?: string } = {};
	try {
		body = (await req.json()) as { title?: string };
	} catch {
		return jsonResponse({ error: "Invalid request body" }, 400);
	}
	const title = (body.title ?? "").trim();
	if (!title) return jsonResponse({ error: "Title cannot be empty" }, 400);

	const session = loadSessionById(id);
	if (!session) return jsonResponse({ error: `No session with id ${id}` }, 404);

	const updated = updateSessionTitle(session, title);
	notifySessionRenamed(id, title);
	return jsonResponse(updated);
}

function handleDeleteSession(id: string): Response {
	if (agentJobs.isRunning(id)) {
		return jsonResponse({ error: "Cannot delete a conversation while its agent turn is running" }, 409);
	}
	const deleted = deleteSession(id);
	if (!deleted) return jsonResponse({ error: `No session with id ${id}` }, 404);

	// If the deleted session was active, auto-create a fresh one
	const active = loadActiveSession();
	if (!active) {
		const fresh = createSession();
		persistSession(fresh);
		switchSession(fresh.id);
	}

	return jsonResponse({ ok: true });
}

// ---- Compact handler ----

async function handleCompactSession(): Promise<Response> {
	try {
		if (agentJobs.isRunning(getSession().id)) {
			return jsonResponse({ ok: false, error: "Cannot compact a conversation while its agent turn is running" }, 409);
		}
		const { ran } = await triggerManualCompact(() => {});
		return jsonResponse({
			ok: true,
			ran,
			contextTokens: getContextTokens(),
			contextWindow: getContextWindow(),
		});
	} catch (e) {
		return jsonResponse({ ok: false, error: String(e) }, 500);
	}
}

// ---- Scratchpad handlers ----

function handleSwitchScratchpad(id: string): Response {
	try {
		const pad = loadScratchpadById(id);
		if (!pad) return jsonResponse({ error: `No scratchpad with id ${id}` }, 404);
		persistScratchpad(pad); // updates session pointer
		return jsonResponse(pad);
	} catch (e) {
		return jsonResponse({ error: String(e) }, 404);
	}
}
async function handleTestConnection(req: Request): Promise<Response> {
	let body: { baseUrl?: string; apiKey?: string } = {};
	try {
		body = (await req.json()) as { baseUrl?: string; apiKey?: string };
	} catch {
		return jsonResponse({ ok: false, message: "Invalid request body" }, 400);
	}

	const baseUrl = (body.baseUrl ?? "").replace(/\/$/, "");
	// If no key sent from frontend (user hasn't changed the masked placeholder), use stored config key
	const storedKey = GlobalState.getInstance().config.openaiApiKey ?? "";
	const apiKey = body.apiKey ?? storedKey;

	if (!baseUrl) {
		return jsonResponse({ ok: false, message: "Base URL is required" });
	}

	try {
		const res = await fetch(`${baseUrl}/models`, {
			headers: { Authorization: `Bearer ${apiKey}` },
			signal: AbortSignal.timeout(8000),
		});

		const contentType = res.headers.get("content-type") ?? "";
		if (!contentType.includes("application/json")) {
			// Server responded but not with JSON — likely wrong URL or non-API server
			const text = await res.text();
			return jsonResponse({
				ok: false,
				message: `Server responded with non-JSON (HTTP ${res.status}). Is the Base URL correct? Preview: ${text.slice(0, 120)}`,
			});
		}

		if (res.ok) {
			const data = await res.json() as { data?: unknown[] };
			const count = Array.isArray(data.data) ? data.data.length : "?";
			return jsonResponse({ ok: true, message: `Connected — ${count} models available` });
		}

		// Non-200 JSON response (e.g. 401 Unauthorized)
		let errMsg = `HTTP ${res.status}`;
		try {
			const errBody = await res.json() as { error?: { message?: string } | string };
			if (typeof errBody.error === "string") errMsg += `: ${errBody.error}`;
			else if (errBody.error?.message) errMsg += `: ${errBody.error.message}`;
		} catch { /* ignore */ }
		return jsonResponse({ ok: false, message: errMsg });

	} catch (e) {
		const msg = e instanceof Error ? e.message : String(e);
		return jsonResponse({ ok: false, message: `Connection failed: ${msg}` });
	}
}

async function handleUploadFile(req: Request): Promise<Response> {
	try {
		const formData = await req.formData();
		const file = formData.get("file") as File | null;
		if (!file) {
			return jsonResponse({ error: "No file uploaded" }, 400);
		}

		const workspace = GlobalState.getInstance().config.workspace ?? process.cwd();
		const targetPath = join(workspace, file.name || "uploaded_file");
		const bytes = await file.arrayBuffer();
		await Bun.write(targetPath, bytes);

		return jsonResponse({ ok: true, name: file.name || "uploaded_file", path: targetPath });
	} catch (e) {
		return jsonResponse({ error: String(e) }, 500);
	}
}
