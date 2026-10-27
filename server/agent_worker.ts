/**
 * One detached agent turn.  The web server starts one process per session so
 * the orchestrator's legacy module-level state cannot leak across chats.
 * Communication is newline-delimited and prefixed so incidental logs from a
 * tool or provider never get mistaken for protocol messages.
 */

const EVENT_PREFIX = "__CAUPROPHET_EVENT__";

function emit(payload: Record<string, unknown>): void {
	process.stdout.write(`${EVENT_PREFIX}${JSON.stringify(payload)}\n`);
}

let input = "";
try {
	input = await new Response(Bun.stdin).text();
	const request = JSON.parse(input.trim()) as {
		prompt?: string;
	};
	if (!request.prompt?.trim()) throw new Error("Worker received an empty prompt");

	const workspace = process.env["CAUPROPHET_AGENT_WORKSPACE"]?.trim();
	if (workspace) process.chdir(workspace);

	// Dynamic imports happen after chdir so all workspace-relative stores point
	// at this run's configured workspace while the session id remains scoped by
	// the process environment.
	const {
		runAgentLoop,
		getScratchpad,
		getContextTokens,
		getContextWindow,
	} = await import("@/core/orchestrator");
	const abortController = new AbortController();
	process.on("SIGTERM", () => abortController.abort());
	process.on("SIGINT", () => abortController.abort());

	await runAgentLoop(request.prompt, {
		onUpdate: (status) => emit({ type: "update", status }),
		signal: abortController.signal,
	});

	emit({
		type: "done",
		scratchpad: getScratchpad(),
		contextTokens: getContextTokens(),
		contextWindow: getContextWindow(),
		aborted: abortController.signal.aborted,
	});
} catch (error) {
	emit({ type: "error", content: String(error) });
	process.exitCode = 1;
}
