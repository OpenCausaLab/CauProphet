/** Revisable written knowledge alongside the persistent causal graph. */
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { atomicWriteFileSync, withFileLockSync } from "@/core/resource_lock";

export interface CognitionNoteRevision {
	id: string;
	version: number;
	title: string;
	note: string;
	revision_reason: string;
	available_at: string;
}

export interface CognitionNoteInput {
	id?: string;
	title: string;
	note: string;
	expected_version?: number;
	revision_reason?: string;
}

function timestamp(value: string): number {
	if (!/(Z|[+-]\d\d:\d\d)$/.test(value) || !Number.isFinite(Date.parse(value))) {
		throw new Error("Expected an ISO timestamp with a timezone");
	}
	return Date.parse(value);
}

function terms(text: string): string[] {
	return text.toLowerCase().match(/[\p{L}\p{N}]+/gu)?.filter(term => term.length > 1) ?? [];
}

export class CognitionNoteStore {
	readonly filePath: string;

	constructor(workspaceDir = process.cwd()) {
		this.filePath = join(workspaceDir, ".cauprophet", "cognition_notes.json");
	}

	history(): CognitionNoteRevision[] {
		if (!existsSync(this.filePath)) return [];
		const data = JSON.parse(readFileSync(this.filePath, "utf8"));
		if (data.schemaVersion !== 1 || !Array.isArray(data.revisions)) {
			throw new Error("Unsupported written Cognition format");
		}
		for (const record of data.revisions) {
			if (typeof record.id !== "string" || !/^[a-zA-Z0-9_-]{1,80}$/.test(record.id) ||
				!Number.isInteger(record.version) || record.version < 1 ||
				typeof record.title !== "string" || typeof record.note !== "string" ||
				typeof record.revision_reason !== "string" || typeof record.available_at !== "string") {
				throw new Error("Invalid written Cognition revision");
			}
			timestamp(record.available_at);
		}
		return data.revisions;
	}

	visible(asOf = new Date().toISOString()): CognitionNoteRevision[] {
		const cutoff = timestamp(asOf);
		const selected = new Map<string, CognitionNoteRevision>();
		for (const record of this.history()) {
			if (timestamp(record.available_at) > cutoff) continue;
			const previous = selected.get(record.id);
			if (!previous || record.version > previous.version) selected.set(record.id, record);
		}
		return [...selected.values()];
	}

	write(input: CognitionNoteInput, availableAt = new Date().toISOString()): CognitionNoteRevision {
		if (typeof input.title !== "string" || !input.title.trim() ||
			typeof input.note !== "string" || !input.note.trim()) {
			throw new Error("A non-empty title and note are required");
		}
		const id = input.id ?? "note_" + createHash("sha256")
			.update(input.title.trim().toLowerCase()).digest("hex").slice(0, 16);
		if (!/^[a-zA-Z0-9_-]{1,80}$/.test(id)) throw new Error("Invalid note ID");
		if (input.expected_version !== undefined &&
			(!Number.isInteger(input.expected_version) || input.expected_version < 0)) {
			throw new Error("expected_version must be a non-negative integer");
		}
		const available = timestamp(availableAt);
		return withFileLockSync(this.filePath, () => {
			const history = this.history();
			const previous = history.filter(record => record.id === id).at(-1);
			if (previous && timestamp(previous.available_at) > available) {
				throw new Error("Cannot revise knowledge from an earlier time");
			}
			if (input.expected_version !== undefined && input.expected_version !== (previous?.version ?? 0)) {
				throw new Error("Version changed; retrieve the current note before revising");
			}
			const record: CognitionNoteRevision = {
				id, version: (previous?.version ?? 0) + 1, title: input.title.trim(),
				note: input.note, revision_reason: input.revision_reason ?? "", available_at: availableAt,
			};
			atomicWriteFileSync(this.filePath, JSON.stringify({schemaVersion: 1, revisions: [...history, record]}, null, 2) + "\n");
			return record;
		});
	}

	current(id: string, version?: number): CognitionNoteRevision {
		const record = this.visible().find(record => record.id === id);
		if (!record || (version !== undefined && record.version !== version)) {
			throw new Error("Mechanism version unavailable or superseded; retrieve the current note");
		}
		return record;
	}

	search(keywords: string[] = [], context = "", asOf?: string): {record: CognitionNoteRevision; score: number}[] {
		const query = [...new Set(terms(keywords.join(" ") + " " + context))];
		const records = this.visible(asOf);
		const contextTerms = terms(context);
		return records.map(record => {
			const title = terms(record.title), body = terms(record.note);
			let score = 0;
			for (const word of query) {
				const frequency = records.filter(item => terms(item.title + " " + item.note).includes(word)).length;
				const weight = Math.log(1 + (records.length - frequency + 0.5) / (frequency + 0.5));
				score += weight * (3 * Number(title.includes(word)) + Number(body.includes(word)));
			}
			for (let index = 1; index < contextTerms.length; index++) {
				if (record.note.toLowerCase().includes(contextTerms[index - 1] + " " + contextTerms[index])) score++;
			}
			return {record, score};
		}).filter(match => query.length === 0 || match.score > 0)
			.sort((left, right) => right.score - left.score || left.record.id.localeCompare(right.record.id)).slice(0, 4);
	}
}
