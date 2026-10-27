/** Tools for retaining, revising and applying written Causal Cognition. */
import { CognitionNoteStore, type CognitionNoteInput } from "@/memory/cognition_notes";
import { loadActiveBelief, writeCausalBelief } from "@/tools/causal_belief_tools";

export const writeCognitionNoteTool = {
	type: "function" as const,
	function: {
		name: "write_cognition_note",
		description: "Save or revise useful observations, reusable mechanisms, strategies, or unresolved hypotheses in natural language. Prefer keeping useful knowledge here rather than only in a task-local Belief. Tentative knowledge is allowed if uncertainty and evidence are clear. No graph or existing Belief is required. Omit id to derive it from title; omit expected_version for a normal serial update.",
		parameters: {
			type: "object",
			properties: {
				id: {type: "string"}, title: {type: "string"}, note: {type: "string"},
				expected_version: {type: "integer", minimum: 0}, revision_reason: {type: "string"},
			},
			required: ["title", "note"], additionalProperties: false,
		},
	},
};

export const instantiateCognitionNoteTool = {
	type: "function" as const,
	function: {
		name: "instantiate_cognition_note",
		description: "Apply a retrieved mechanism to the active task-local Belief using your natural-language application. Omit version to use the current visible version. Does not modify persistent knowledge.",
		parameters: {
			type: "object",
			properties: {
				id: {type: "string"}, version: {type: "integer", minimum: 1}, application: {type: "string"},
			},
			required: ["id", "application"], additionalProperties: false,
		},
	},
};

function objectInput(input: unknown): Record<string, unknown> {
	const value = typeof input === "string" ? JSON.parse(input) : input;
	if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Tool input must be an object");
	return value as Record<string, unknown>;
}

export async function writeCognitionNote(input: unknown): Promise<string> {
	try {
		const raw = objectInput(input);
		for (const key of ["id", "revision_reason"] as const) {
			if (raw[key] !== undefined && typeof raw[key] !== "string") throw new Error(`${key} must be a string`);
		}
		const record = new CognitionNoteStore().write(raw as unknown as CognitionNoteInput);
		return ["# Written Causal Cognition", `- Cognition note saved: \`${record.id}\` (version ${record.version})`,
			`- Title: ${record.title}`, `- Available at: ${record.available_at}`].join("\n");
	} catch (error) {
		return `Error: Written Cognition update failed: ${String(error)}`;
	}
}

export async function instantiateCognitionNote(input: unknown): Promise<string> {
	try {
		const raw = objectInput(input);
		if (typeof raw.id !== "string" || typeof raw.application !== "string" || !raw.application.trim()) {
			throw new Error("A note ID and non-empty task application are required");
		}
		if (raw.version !== undefined && (!Number.isInteger(raw.version) || Number(raw.version) < 1)) {
			throw new Error("version must be a positive integer");
		}
		if (!loadActiveBelief()) throw new Error("Create or open a task-local Belief before applying knowledge");
		const record = new CognitionNoteStore().current(raw.id, raw.version as number | undefined);
		return await writeCausalBelief({append_notes:
			`Mechanism ${record.id} v${record.version}: ${record.note}\nTask application: ${raw.application}`});
	} catch (error) {
		return `Error: Cognition application failed: ${String(error)}`;
	}
}
