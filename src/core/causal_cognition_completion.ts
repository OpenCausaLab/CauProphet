export const COGNITION_SKIP_CATEGORIES = [
	"episode-specific",
	"direction-unresolved",
	"already-represented",
	"grounding-unresolved",
	"temporal-only",
] as const;

export type CognitionSkipCategory = (typeof COGNITION_SKIP_CATEGORIES)[number];

export interface CognitionSkipDecision {
	category: CognitionSkipCategory;
	reason: string;
	marker: string;
}

const SKIP_PATTERN =
	/Cognition\s+Lift[：:]\s*skipped\s*[—–:-]\s*(?:\[(episode-specific|direction-unresolved|already-represented|grounding-unresolved|temporal-only)\]|(episode-specific|direction-unresolved|already-represented|grounding-unresolved|temporal-only))\s*(?:[;；:：]\s*)?([^\n]+)/gi;
const GENERIC_SKIP_REASON =
	/^(?:<[^>]+>|n\/?a|none|unknown|tbd|specific reason|no eligible relation|nothing to lift|insufficient evidence|lack(?:s|ing)? (?:sufficient )?evidence|not identifiable|unidentifiable|证据不足|缺乏(?:充分)?证据|不可识别|无法识别|无可提升关系|没有可提升关系)[.!。！\s]*$/i;

/** Return the last structured no-Lift decision in a Belief note. */
export function parseCognitionSkipDecision(
	notes: string | null | undefined,
): CognitionSkipDecision | null {
	if (!notes) return null;
	let decision: CognitionSkipDecision | null = null;
	for (const match of notes.matchAll(SKIP_PATTERN)) {
		const category = (match[1] ?? match[2]) as
			| CognitionSkipCategory
			| undefined;
		const reason = match[3]?.trim() ?? "";
		if (
			!category ||
			reason.replace(/\s/g, "").length < 8 ||
			GENERIC_SKIP_REASON.test(reason)
		) {
			continue;
		}
		decision = {
			category,
			reason,
			marker: `${category}:${reason}`,
		};
	}
	return decision;
}

export function hasFreshCognitionSkipDecision(input: {
	initialBeliefId: string | null;
	initialDecision: CognitionSkipDecision | null;
	currentBeliefId: string | null;
	currentNotes: string | null | undefined;
}): boolean {
	const current = parseCognitionSkipDecision(input.currentNotes);
	if (!current) return false;
	return (
		input.currentBeliefId !== input.initialBeliefId ||
		current.marker !== input.initialDecision?.marker
	);
}

export function isSuccessfulToolResult(result: string): boolean {
	return !/^\s*Error(?:\s|:)/i.test(result);
}

/** Distinguish a real episode Lift from maintenance-only cognition updates. */
export function isSuccessfulCognitionLiftResult(result: string): boolean {
	return (
		isSuccessfulToolResult(result) &&
		/^\s*- Episode:\s*`[^`]+`\s*$/m.test(result) &&
		/^\s*- Lift created \d+ concepts and \d+ relation variants; updated \d+;/m.test(
			result,
		)
	);
}

/** Experience or written knowledge can complete a memory review without causal Lift. */
export function isSuccessfulCognitionMemoryResult(result: string): boolean {
	return isSuccessfulCognitionLiftResult(result) || (
		isSuccessfulToolResult(result) && (/^\s*- Experience saved: `[^`]+`\s*$/m.test(result) ||
			/^\s*- Cognition note saved: `[^`]+` \(version \d+\)\s*$/m.test(result))
	);
}

export type CognitionReflectionDecision = "none" | "offer" | "accept";

/** Offer one optional reflection, then always let the agent's answer proceed. */
export function cognitionReflectionDecision(input: {
	required: boolean;
	workflowEntered: boolean;
	lifted: boolean;
	advisoryOffered: boolean;
}): CognitionReflectionDecision {
	if (!input.required || !input.workflowEntered || input.lifted) {
		return "none";
	}
	return input.advisoryOffered ? "accept" : "offer";
}

export const COGNITION_COMPLETION_ADVISORY_PROMPT = [
	"[INTERNAL CAUSAL COGNITION CURATOR PASS]",
	"The user-facing answer is complete and preserved for delivery. Do not repeat it, change it, restart the task, or perform more web research.",
	"Inspect the task actions, tool results, execution observations, and active Belief for 1-3 useful long-term lessons. Cognition stores practical experience as well as causal knowledge: API behavior, preconditions, side effects, successful strategies, and error recovery can all help future tasks.",
	"Search for relevant prior memory. Use write_cognition_note when available to retain or revise a useful written explanation, or update_causal_cognition with experience for a useful lesson even when the Belief has no directed edges. In a written note, state the mechanism, conditions, evidence, and uncertainty. For an experience object, include title, summary, keywords, context, and concrete evidence already obtained. A single observed episode or API contract is sufficient if applicability and uncertainty are clear. For reusable causal edges, retain the existing grounded Lift and evidence requirements; use hypothesized for mechanisms not outcome-validated.",
	"Generalize concrete entities into the operation, precondition, result, and lesson where useful. Keep unresolved causal direction as uncertainty in experience. Leave memory unchanged when no useful new lesson or evidence remains; a note in Belief is task-local and does not persist an experience in Cognition.",
	"Never create a relation merely to satisfy this pass. End with a terse internal acknowledgement; it will not replace the preserved user-facing answer.",
].join("\n");
