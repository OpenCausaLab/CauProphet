/**
 * Causal Scout
 *
 * A small, assistive hypothesis-search layer for the active Causal Belief.
 * The LLM supplies a typed Scout Program; this module deterministically
 * serializes it to ASP, optionally asks Clingo for several stable models, and
 * interprets the resulting local structures as hypothesis cards.  Scout never
 * writes the active belief and never changes causal_infer behaviour.
 */

import { spawn } from "node:child_process";
import { join } from "node:path";
import type { CausalBelief } from "@/memory/causal_belief";
import { loadActiveBelief } from "@/tools/causal_belief_tools";
import { getEstimandRelevantGraphBlockers } from "@/tools/causal_tools";

// ---------------------------------------------------------------------------
// Typed Scout Program contract
// ---------------------------------------------------------------------------

export type ScoutEdgeType =
	| "directed"
	| "undirected"
	| "bidirected"
	| "partially_directed";

export type ScoutEdgeTuple = readonly [string, string];
export type ScoutTriple = readonly [string, string, string];
export type ScoutGraphSemantics = "candidate_dag" | "pdag" | "cpdag";

export interface ScoutQuery {
	treatment: string;
	outcome: string;
	estimand?: string;
	claim?: string;
}

export interface ScoutTemporalRole {
	node: string;
	role:
		| "pre_treatment"
		| "post_treatment"
		| "treatment"
		| "outcome"
		| "unknown"
		| (string & {});
}

export interface ScoutAssociation {
	from?: string;
	to?: string;
	nodes?: string[];
	type?: string;
	description?: string;
	source?: string;
}

export interface ScoutIndependenceFact {
	id?: string;
	from: string;
	to: string;
	conditioned_on?: string[];
	independent: boolean;
	/** Evidence credibility, not a CI p-value or a probability of graph truth. */
	confidence?: number;
	source?: string;
}

export interface NormalizedScoutIndependenceFact extends ScoutIndependenceFact {
	id: string;
	conditioned_on: string[];
	confidence: number;
	source: string;
}

export interface ScoutCandidateHypothesis {
	id: string;
	candidate_edges: ScoutEdgeTuple[];
	rationale?: string;
	support?: string[];
	counterevidence?: string[];
	/** Optional soft preference in [0, 1]; it is only used for ranking. */
	confidence?: number;
}

export interface ScoutFacts {
	nodes?: string[];
	node_descriptions?:
		| Record<string, string>
		| Array<{ node: string; description: string }>;
	known_edges?: ScoutEdgeTuple[] | ScoutKnownEdge[];
	temporal_roles?: ScoutTemporalRole[];
	associations?: ScoutAssociation[];
	/** Hard structural constraints supplied by data or domain knowledge. */
	required_edges?: ScoutEdgeTuple[];
	forbidden_edges?: ScoutEdgeTuple[];
	incompatible_hypotheses?: Array<readonly [string, string]>;
	/** Tested CI/dependence evidence, compared with complete candidate DAGs. */
	conditional_independencies?: ScoutIndependenceFact[];
	/** Optional search skeleton: candidate arrows may only orient these pairs. */
	skeleton?: ScoutEdgeTuple[];
	/** Explicit earlier/later pairs; unknown timing is not a restriction. */
	temporal_order?: ScoutEdgeTuple[];
	/** Whether undirected known edges are a valid PDAG/CPDAG, not arbitrary uncertainty. */
	graph_semantics?: ScoutGraphSemantics;
	/** Definite non-collider triples from PC/FCI separating-set information. */
	definite_noncolliders?: ScoutTriple[];
}

export interface ScoutKnownEdge {
	from: string;
	to: string;
	edgeType?: ScoutEdgeType;
}

export interface CausalScoutInput {
	query: ScoutQuery;
	facts?: ScoutFacts;
	candidate_hypotheses?: ScoutCandidateHypothesis[];
	top_k?: number;
}

export interface NormalizedScoutCandidate {
	id: string;
	candidate_edges: ScoutEdgeTuple[];
	rationale: string;
	support: string[];
	counterevidence: string[];
	confidence: number;
}

export interface ScoutEffectEvidence {
	treatment: string;
	outcome: string;
	ate: number | null;
	adjustment_set: string[];
	refuted: boolean;
	validity?: string;
}

export interface NormalizedScoutProgram {
	query: Required<Pick<ScoutQuery, "treatment" | "outcome">> & {
		estimand: string;
		claim: string;
	};
	facts: {
		nodes: string[];
		node_descriptions: Record<string, string>;
		node_sources: Record<string, string>;
		known_edges: ScoutKnownEdge[];
		temporal_roles: ScoutTemporalRole[];
		associations: ScoutAssociation[];
		required_edges: ScoutEdgeTuple[];
		forbidden_edges: ScoutEdgeTuple[];
		incompatible_hypotheses: Array<readonly [string, string]>;
		conditional_independencies: NormalizedScoutIndependenceFact[];
		skeleton?: ScoutEdgeTuple[];
		temporal_order: ScoutEdgeTuple[];
		graph_semantics: ScoutGraphSemantics;
		definite_noncolliders: ScoutTriple[];
	};
	belief_context: {
		objective: string;
		notes: string;
		effects: ScoutEffectEvidence[];
	};
	candidate_hypotheses: NormalizedScoutCandidate[];
	/** Human-facing labels retained after ASP-safe identifiers are generated. */
	display_names: Record<string, string>;
}

export interface ScoutModel {
	answer_set: number;
	hypotheses: string[];
	derived_motifs: string[];
	selected_edges: ScoutEdgeTuple[];
	ranking: { soft_cost: number };
	evidence?: { matched: string[]; conflicted: string[] };
	meek_orientations?: ScoutEdgeTuple[];
}

export interface ScoutSearch {
	status: "satisfiable" | "relaxed" | "unsatisfiable" | "incomplete";
	exhaustive: boolean;
	structurally_valid_count: number;
	compatible_count: number;
	returned_count: number;
	retained_evidence: string[];
	relaxed_evidence: Array<{
		id: string;
		confidence: number;
		source: string;
		reason: string;
	}>;
	rejected_models: Array<{ hypotheses: string[]; reasons: string[] }>;
	diagnostics_exhaustive: boolean;
	discriminating_tests: Array<{
		from: string;
		to: string;
		conditioned_on: string[];
		independent_in: number[];
		dependent_in: number[];
		reason: string;
	}>;
	assumptions: string[];
	warnings: string[];
	compelled_edges: ScoutEdgeTuple[];
	reversible_pairs: ScoutEdgeTuple[];
}

export interface ScoutCard {
	title: string;
	hypothesis: string;
	/** Compact aliases retained for callers using the initial Scout sketch. */
	mechanism: string;
	why_considered: string[];
	graph_consequence: string[];
	structural_consequences: string[];
	what_would_change: string[];
	supporting_evidence: string[];
	support: string[];
	counterevidence: string[];
	discriminating_probes: string[];
	probes: string[];
	status_for_llm: string;
}

export interface CausalScoutResult {
	backend: "clingo" | "unavailable";
	clingo_available: boolean;
	program: NormalizedScoutProgram;
	asp: string;
	models: ScoutModel[];
	cards: ScoutCard[];
	message: string;
	search?: ScoutSearch;
}

export interface ScoutAnswerSet {
	atoms: string[];
	cost?: number;
	evidence?: { matched: string[]; conflicted: string[] };
}

// ---------------------------------------------------------------------------
// Normalization and program construction
// ---------------------------------------------------------------------------

/** Make a stable, ASP-safe identifier while retaining the original label. */
export function normalizeScoutIdentifier(value: unknown): string {
	const source = String(value ?? "")
		.normalize("NFKC")
		.trim();
	const text = Array.from(source, (character) => {
		if (/[A-Za-z0-9_]/.test(character)) return character;
		if (/\p{L}|\p{N}/u.test(character))
			return `u${character.codePointAt(0)?.toString(16) ?? "0"}`;
		return "_";
	})
		.join("")
		.replace(/^_+|_+$/g, "")
		.replace(/_+/g, "_");
	if (!text) return "unnamed";
	return /^[a-z]/.test(text) ? text : `v_${text}`;
}

/** Remove control characters and bound free text before it enters a result. */
export function normalizeScoutDescription(
	value: unknown,
	maxLength = 500,
): string {
	const text = Array.from(String(value ?? ""), (character) => {
		const code = character.charCodeAt(0);
		return code < 0x20 || code === 0x7f ? " " : character;
	})
		.join("")
		.replace(/\s+/g, " ")
		.trim();
	return text.length <= maxLength ? text : `${text.slice(0, maxLength - 1)}…`;
}

function asStringArray(value: unknown): string[] {
	if (!Array.isArray(value)) return [];
	return value.map((item) => normalizeScoutDescription(item)).filter(Boolean);
}

function asEdgeTuple(value: unknown): ScoutEdgeTuple | null {
	if (!Array.isArray(value) || value.length < 2) return null;
	const from = normalizeScoutIdentifier(value[0]);
	const to = normalizeScoutIdentifier(value[1]);
	if (from === "unnamed" || to === "unnamed") return null;
	return [from, to];
}

function normalizeKnownEdges(value: unknown): ScoutKnownEdge[] {
	if (!Array.isArray(value)) return [];
	const result: ScoutKnownEdge[] = [];
	for (const item of value) {
		if (Array.isArray(item)) {
			const edge = asEdgeTuple(item);
			if (edge)
				result.push({ from: edge[0], to: edge[1], edgeType: "directed" });
			continue;
		}
		if (!item || typeof item !== "object") continue;
		const record = item as Record<string, unknown>;
		const edge = asEdgeTuple([record.from, record.to]);
		if (edge) {
			const edgeType = [
				"directed",
				"undirected",
				"bidirected",
				"partially_directed",
			].includes(String(record.edgeType))
				? (String(record.edgeType) as ScoutEdgeType)
				: "directed";
			result.push({ from: edge[0], to: edge[1], edgeType });
		}
	}
	return result;
}

function normalizeEdgeTuples(value: unknown): ScoutEdgeTuple[] {
	if (!Array.isArray(value)) return [];
	return [
		...new Map(
			value
				.map(asEdgeTuple)
				.filter((edge): edge is ScoutEdgeTuple => edge !== null)
				.map((edge) => [candidateKey(edge), edge]),
		).values(),
	].sort((left, right) =>
		candidateKey(left).localeCompare(candidateKey(right)),
	);
}

function normalizeHypothesisPairs(
	value: unknown,
): Array<readonly [string, string]> {
	if (!Array.isArray(value)) return [];
	return [
		...new Map(
			value
				.filter((pair) => Array.isArray(pair) && pair.length >= 2)
				.map(
					(pair) =>
						[
							normalizeScoutIdentifier(pair[0]),
							normalizeScoutIdentifier(pair[1]),
						] as const,
				)
				.filter(([left, right]) => left !== "unnamed" && right !== "unnamed")
				.map((pair) => [`${pair[0]}\0${pair[1]}`, pair] as const),
		).values(),
	].sort((left, right) =>
		`${left[0]}\0${left[1]}`.localeCompare(`${right[0]}\0${right[1]}`),
	);
}

function normalizeTriples(value: unknown): ScoutTriple[] {
	if (!Array.isArray(value)) return [];
	return [
		...new Map(
			value
				.filter((triple) => Array.isArray(triple) && triple.length >= 3)
				.map((triple) => {
					const first = normalizeScoutIdentifier(triple[0]);
					const middle = normalizeScoutIdentifier(triple[1]);
					const third = normalizeScoutIdentifier(triple[2]);
					const outer = [first, third].sort((left, right) =>
						left.localeCompare(right),
					);
					return [
						outer[0] ?? "unnamed",
						middle,
						outer[1] ?? "unnamed",
					] as const;
				})
				.filter(
					([first, middle, third]) =>
						first !== "unnamed" &&
						middle !== "unnamed" &&
						third !== "unnamed" &&
						first !== middle &&
						middle !== third &&
						first !== third,
				)
				.map((triple) => [triple.join("\0"), triple] as const),
		).values(),
	].sort((left, right) => left.join("\0").localeCompare(right.join("\0")));
}

function normalizeConditionalIndependencies(
	value: unknown,
): NormalizedScoutIndependenceFact[] {
	if (!Array.isArray(value)) return [];
	const ids = new Set<string>();
	return value
		.map((item, index) => {
			if (!item || typeof item !== "object")
				throw new Error("CI evidence must be an object.");
			const record = item as Record<string, unknown>;
			const from = normalizeScoutIdentifier(record.from);
			const to = normalizeScoutIdentifier(record.to);
			if (
				from === "unnamed" ||
				to === "unnamed" ||
				from === to ||
				typeof record.independent !== "boolean"
			)
				throw new Error(
					"CI evidence needs distinct endpoints and an explicit independent boolean.",
				);
			const conditionedOn = Array.isArray(record.conditioned_on)
				? record.conditioned_on.map(normalizeScoutIdentifier)
				: [];
			if (
				conditionedOn.includes(from) ||
				conditionedOn.includes(to) ||
				conditionedOn.includes("unnamed")
			)
				throw new Error(
					"CI conditioning nodes must be disjoint from the endpoints.",
				);
			const id = normalizeScoutIdentifier(record.id || `ci_${index + 1}`);
			if (ids.has(id)) throw new Error(`Duplicate CI evidence id: ${id}`);
			ids.add(id);
			const confidence =
				record.confidence == null ? 0.5 : Number(record.confidence);
			if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1)
				throw new Error("CI confidence must be in [0,1].");
			return {
				id,
				from,
				to,
				conditioned_on: [...new Set(conditionedOn)].sort(),
				independent: record.independent,
				confidence,
				source: normalizeScoutDescription(record.source),
			};
		})
		.sort((left, right) =>
			JSON.stringify(left).localeCompare(JSON.stringify(right)),
		);
}

function edgeKey(edge: ScoutKnownEdge | ScoutEdgeTuple): string {
	if ("from" in edge)
		return `${edge.from}\0${edge.to}\0${edge.edgeType ?? "directed"}`;
	return `${edge[0]}\0${edge[1]}\0directed`;
}

function candidateKey(edge: ScoutEdgeTuple): string {
	return `${edge[0]}\0${edge[1]}`;
}

function inferTemporalRole(
	role: string | undefined,
	source: string,
	description: string,
): ScoutTemporalRole["role"] {
	const normalizedRole = role ? normalizeScoutIdentifier(role) : "";
	if (
		normalizedRole === "treatment" ||
		normalizedRole === "outcome" ||
		normalizedRole === "pre_treatment" ||
		normalizedRole === "post_treatment" ||
		normalizedRole === "selection"
	) {
		return normalizedRole;
	}
	if (
		normalizedRole === "baseline" ||
		/pre[-_ ]?treatment|baseline|prior to|before treatment/i.test(
			`${source} ${description}`,
		)
	) {
		return "pre_treatment";
	}
	return "unknown";
}

/** Merge user facts with the active Belief without mutating or persisting it. */
export function buildScoutProgram(
	input: CausalScoutInput,
	belief: CausalBelief | null = null,
): NormalizedScoutProgram {
	const displayNames: Record<string, string> = {};
	const descriptions: Record<string, string> = {};
	const sources: Record<string, string> = {};
	const remember = (value: unknown): string => {
		const normalized = normalizeScoutIdentifier(value);
		const label = normalizeScoutDescription(value) || normalized;
		if (!displayNames[normalized]) displayNames[normalized] = label;
		return normalized;
	};

	const queryRecord = input.query ?? ({} as ScoutQuery);
	const treatment = remember(queryRecord.treatment);
	const outcome = remember(queryRecord.outcome);
	const beliefNodes = (belief?.variables ?? []).map((variable) => {
		const node = remember(variable.name);
		const description = normalizeScoutDescription(variable.description);
		const source = normalizeScoutDescription(variable.source);
		if (description) descriptions[node] = description;
		if (source) sources[node] = source;
		return {
			node,
			role: inferTemporalRole(variable.role, source, description),
		};
	});
	const rawFacts = input.facts ?? {};
	const nodes = new Set<string>([treatment, outcome]);
	for (const node of beliefNodes) nodes.add(node.node);
	for (const node of rawFacts.nodes ?? []) nodes.add(remember(node));
	const rawDescriptions = Array.isArray(rawFacts.node_descriptions)
		? Object.fromEntries(
				rawFacts.node_descriptions.map((item) => [item.node, item.description]),
			)
		: (rawFacts.node_descriptions ?? {});
	for (const [rawNode, rawDescription] of Object.entries(rawDescriptions)) {
		const node = remember(rawNode);
		descriptions[node] = normalizeScoutDescription(rawDescription);
		nodes.add(node);
	}

	const knownEdges: ScoutKnownEdge[] = [];
	for (const edge of belief?.edges ?? []) {
		const from = remember(edge.from);
		const to = remember(edge.to);
		knownEdges.push({ from, to, edgeType: edge.edgeType });
		nodes.add(from);
		nodes.add(to);
	}
	for (const edge of normalizeKnownEdges(rawFacts.known_edges)) {
		const from = remember(edge.from);
		const to = remember(edge.to);
		knownEdges.push({ from, to, edgeType: edge.edgeType });
		nodes.add(from);
		nodes.add(to);
	}

	const uniqueKnownEdges = [
		...new Map(knownEdges.map((edge) => [edgeKey(edge), edge])).values(),
	].sort((left, right) => edgeKey(left).localeCompare(edgeKey(right)));

	const temporalByNode = new Map<string, ScoutTemporalRole>();
	for (const role of beliefNodes)
		temporalByNode.set(role.node, { node: role.node, role: role.role });
	for (const rawRole of rawFacts.temporal_roles ?? []) {
		const node = remember(rawRole.node);
		temporalByNode.set(node, {
			node,
			role: inferTemporalRole(rawRole.role, "", ""),
		});
		nodes.add(node);
	}
	const temporalRoles = [...temporalByNode.values()].sort((left, right) =>
		left.node.localeCompare(right.node),
	);

	const associations: ScoutAssociation[] = (rawFacts.associations ?? [])
		.map((association) => ({
			...(association.from ? { from: remember(association.from) } : {}),
			...(association.to ? { to: remember(association.to) } : {}),
			...(association.nodes ? { nodes: association.nodes.map(remember) } : {}),
			...(association.type
				? { type: normalizeScoutDescription(association.type, 80) }
				: {}),
			...(association.description
				? { description: normalizeScoutDescription(association.description) }
				: {}),
			...(association.source
				? { source: normalizeScoutDescription(association.source) }
				: {}),
		}))
		.sort((left, right) =>
			JSON.stringify(left).localeCompare(JSON.stringify(right)),
		);
	const requiredEdges = normalizeEdgeTuples(rawFacts.required_edges);
	const forbiddenEdges = normalizeEdgeTuples(rawFacts.forbidden_edges);
	const incompatibleHypotheses = normalizeHypothesisPairs(
		rawFacts.incompatible_hypotheses,
	);
	const conditionalIndependencies = normalizeConditionalIndependencies(
		rawFacts.conditional_independencies,
	);
	const graphSemantics: ScoutGraphSemantics =
		rawFacts.graph_semantics === "cpdag" || rawFacts.graph_semantics === "pdag"
			? rawFacts.graph_semantics
			: "candidate_dag";
	const definiteNoncolliders = normalizeTriples(rawFacts.definite_noncolliders);
	const skeleton =
		rawFacts.skeleton === undefined
			? undefined
			: normalizeEdgeTuples(rawFacts.skeleton);
	const temporalOrder = normalizeEdgeTuples(rawFacts.temporal_order);
	// Only explicit timing creates hard constraints; prose-derived roles remain hints.
	for (const item of rawFacts.temporal_roles ?? []) {
		const node = remember(item.node);
		if (item.role === "pre_treatment" && node !== treatment)
			temporalOrder.push([node, treatment]);
		if (item.role === "post_treatment" && node !== treatment)
			temporalOrder.push([treatment, node]);
	}
	for (const [from, to] of [
		...requiredEdges,
		...forbiddenEdges,
		...(skeleton ?? []),
		...temporalOrder,
	]) {
		nodes.add(from);
		nodes.add(to);
	}
	for (const fact of conditionalIndependencies) {
		nodes.add(fact.from);
		nodes.add(fact.to);
		for (const node of fact.conditioned_on) nodes.add(node);
	}

	const candidates: NormalizedScoutCandidate[] = [];
	for (const rawCandidate of input.candidate_hypotheses ?? []) {
		if (!rawCandidate?.id) continue;
		const id = remember(rawCandidate.id);
		const candidateEdges = [
			...new Map(
				(rawCandidate.candidate_edges ?? [])
					.map(asEdgeTuple)
					.filter((edge): edge is ScoutEdgeTuple => edge !== null)
					.map((edge) => [candidateKey(edge), edge]),
			).values(),
		].sort((left, right) =>
			candidateKey(left).localeCompare(candidateKey(right)),
		);
		for (const [from, to] of candidateEdges) {
			nodes.add(from);
			nodes.add(to);
		}
		const confidence = Number(rawCandidate.confidence);
		candidates.push({
			id,
			candidate_edges: candidateEdges,
			rationale: normalizeScoutDescription(rawCandidate.rationale),
			support: asStringArray(rawCandidate.support),
			counterevidence: asStringArray(rawCandidate.counterevidence),
			confidence: Number.isFinite(confidence)
				? Math.min(1, Math.max(0, confidence))
				: 0.5,
		});
	}
	const uniqueCandidates = [
		...new Map(
			candidates.map((candidate) => [candidate.id, candidate]),
		).values(),
	].sort((left, right) => left.id.localeCompare(right.id));
	const effects: ScoutEffectEvidence[] = (belief?.effects ?? []).map(
		(effect) => ({
			treatment: remember(effect.treatment),
			outcome: remember(effect.outcome),
			ate:
				typeof effect.ate === "number" && Number.isFinite(effect.ate)
					? effect.ate
					: null,
			adjustment_set: (effect.adjustmentSet ?? [])
				.map(remember)
				.sort((left, right) => left.localeCompare(right)),
			refuted: effect.refuted === true,
			...(effect.validity?.status ? { validity: effect.validity.status } : {}),
		}),
	);

	return {
		query: {
			treatment,
			outcome,
			estimand: normalizeScoutDescription(
				queryRecord.estimand || "total_effect",
				120,
			),
			claim: normalizeScoutDescription(queryRecord.claim),
		},
		facts: {
			nodes: [...nodes].sort((left, right) => left.localeCompare(right)),
			node_descriptions: Object.fromEntries(
				Object.entries(descriptions).sort(([left], [right]) =>
					left.localeCompare(right),
				),
			),
			node_sources: Object.fromEntries(
				Object.entries(sources).sort(([left], [right]) =>
					left.localeCompare(right),
				),
			),
			known_edges: uniqueKnownEdges,
			temporal_roles: temporalRoles,
			associations,
			required_edges: requiredEdges,
			forbidden_edges: forbiddenEdges,
			incompatible_hypotheses: incompatibleHypotheses,
			conditional_independencies: conditionalIndependencies,
			...(skeleton === undefined ? {} : { skeleton }),
			temporal_order: normalizeEdgeTuples(temporalOrder),
			graph_semantics: graphSemantics,
			definite_noncolliders: definiteNoncolliders,
		},
		belief_context: {
			objective: normalizeScoutDescription(belief?.objective),
			notes: normalizeScoutDescription(belief?.notes, 2_000),
			effects,
		},
		candidate_hypotheses: uniqueCandidates,
		display_names: Object.fromEntries(
			Object.entries(displayNames).sort(([left], [right]) =>
				left.localeCompare(right),
			),
		),
	};
}

function aspAtom(value: string): string {
	return normalizeScoutIdentifier(value);
}

/** Deterministically compile a typed Scout Program into a finite ASP program. */
export function serializeScoutProgram(
	program: NormalizedScoutProgram | CausalScoutInput,
): string {
	const scoutProgram: NormalizedScoutProgram =
		"display_names" in program ? program : buildScoutProgram(program);
	const lines: string[] = [
		"% Causal Scout: generated from a typed program; candidate hypotheses remain optional.",
		`query_treatment(${aspAtom(scoutProgram.query.treatment)}).`,
		`query_outcome(${aspAtom(scoutProgram.query.outcome)}).`,
	];
	if (isCausalClaim(scoutProgram.query.claim)) lines.push("causal_claim.");
	for (const node of scoutProgram.facts.nodes)
		lines.push(`node(${aspAtom(node)}).`);
	for (const edge of scoutProgram.facts.known_edges) {
		lines.push(
			`known_edge(${aspAtom(edge.from)},${aspAtom(edge.to)},${edge.edgeType}).`,
		);
	}
	for (const role of scoutProgram.facts.temporal_roles) {
		lines.push(
			`temporal_role(${aspAtom(role.node)},${normalizeScoutIdentifier(role.role)}).`,
		);
	}
	for (const association of scoutProgram.facts.associations) {
		if (association.from && association.to) {
			lines.push(
				`association(${aspAtom(association.from)},${aspAtom(association.to)},${normalizeScoutIdentifier(association.type || "observational")}).`,
			);
		}
	}
	for (const [from, to] of scoutProgram.facts.required_edges)
		lines.push(`required_edge(${aspAtom(from)},${aspAtom(to)}).`);
	for (const [from, to] of scoutProgram.facts.forbidden_edges)
		lines.push(`forbidden_edge(${aspAtom(from)},${aspAtom(to)}).`);
	for (const [left, right] of scoutProgram.facts.incompatible_hypotheses)
		lines.push(`incompatible_hypothesis(${aspAtom(left)},${aspAtom(right)}).`);
	if (scoutProgram.facts.skeleton !== undefined) {
		lines.push("restrict_skeleton.");
		for (const [from, to] of scoutProgram.facts.skeleton)
			lines.push(`skeleton(${aspAtom(from)},${aspAtom(to)}).`);
	}
	for (const [earlier, later] of scoutProgram.facts.temporal_order)
		lines.push(`before(${aspAtom(earlier)},${aspAtom(later)}).`);
	if (scoutProgram.facts.graph_semantics !== "candidate_dag") {
		lines.push("orientation_closure.");
		if (scoutProgram.facts.graph_semantics === "cpdag")
			lines.push("cpdag_input.");
		for (const [left, middle, right] of scoutProgram.facts
			.definite_noncolliders)
			lines.push(
				`definite_noncollider(${aspAtom(left)},${aspAtom(middle)},${aspAtom(right)}).`,
			);
	}
	for (const fact of scoutProgram.facts.conditional_independencies) {
		lines.push(
			`ci_evidence(${aspAtom(fact.id)},${aspAtom(fact.from)},${aspAtom(fact.to)},${fact.independent ? "independent" : "dependent"},${Math.round(fact.confidence * 1000)}).`,
		);
		for (const node of fact.conditioned_on)
			lines.push(`ci_condition(${aspAtom(fact.id)},${aspAtom(node)}).`);
	}
	for (const candidate of scoutProgram.candidate_hypotheses) {
		lines.push(`candidate_hypothesis(${aspAtom(candidate.id)}).`);
		for (const [from, to] of candidate.candidate_edges) {
			lines.push(
				`hypothesis_edge(${aspAtom(candidate.id)},${aspAtom(from)},${aspAtom(to)}).`,
			);
		}
	}

	lines.push(
		"",
		"% Enumerate competing local explanations; no candidate is mandatory.",
		"{ selected_hypothesis(H) } :- candidate_hypothesis(H).",
		"selected_edge(U,V) :- known_edge(U,V,directed).",
		"1 { selected_edge(U,V); selected_edge(V,U) } 1 :- known_edge(U,V,undirected).",
		"selected_edge(U,V) :- selected_hypothesis(H), hypothesis_edge(H,U,V).",
		"% Hard structural constraints from data or domain knowledge.",
		"violation(required_edge,U,V) :- required_edge(U,V), not selected_edge(U,V).",
		"violation(forbidden_edge,U,V) :- selected_edge(U,V), forbidden_edge(U,V).",
		"violation(incompatible,H1,H2) :- selected_hypothesis(H1), selected_hypothesis(H2), incompatible_hypothesis(H1,H2).",
		"path(X,Y) :- selected_edge(X,Y).",
		"path(X,Y) :- selected_edge(X,Z), path(Z,Y).",
		"violation(cycle,X,X) :- path(X,X).",
		"earlier(X,Y) :- before(X,Y).",
		"earlier(X,Y) :- before(X,Z), earlier(Z,Y).",
		"violation(temporal_order,U,V) :- path(U,V), earlier(V,U).",
		"violation(temporal_cycle,X,X) :- earlier(X,X).",
		"allowed_pair(U,V) :- skeleton(U,V).",
		"allowed_pair(U,V) :- skeleton(V,U).",
		"violation(skeleton,U,V) :- restrict_skeleton, selected_edge(U,V), not allowed_pair(U,V).",
		"adjacent(U,V) :- selected_edge(U,V).",
		"adjacent(U,V) :- selected_edge(V,U).",
		"undirected_pair(U,V) :- known_edge(U,V,undirected).",
		"undirected_pair(U,V) :- known_edge(V,U,undirected).",
		"oriented_edge(U,V) :- selected_edge(U,V).",
		"definite_noncollider(C,B,A) :- definite_noncollider(A,B,C).",
		"meek_r1_triple(A,B,C) :- definite_noncollider(A,B,C).",
		"meek_r1_triple(A,B,C) :- cpdag_input, node(A), node(B), node(C), A != B, B != C, A != C.",
		"meek_force(B,C,r1) :- orientation_closure, meek_r1_triple(A,B,C), oriented_edge(A,B), undirected_pair(B,C), node(A), node(B), node(C), A != B, B != C, A != C, not adjacent(A,C).",
		"meek_force(A,C,r2) :- orientation_closure, oriented_edge(A,B), oriented_edge(B,C), undirected_pair(A,C), node(A), node(B), node(C), A != B, B != C, A != C.",
		"meek_force(A,B,r3) :- orientation_closure, undirected_pair(A,B), adjacent(A,C), adjacent(A,D), oriented_edge(C,B), oriented_edge(D,B), node(A), node(B), node(C), node(D), A != B, A != C, A != D, B != C, B != D, C != D, not adjacent(C,D).",
		"meek_force(A,B,r4) :- orientation_closure, undirected_pair(A,B), adjacent(A,D), adjacent(A,C), oriented_edge(D,C), oriented_edge(C,B), node(A), node(B), node(C), node(D), A != B, A != C, A != D, B != C, B != D, C != D, not adjacent(B,D).",
		"oriented_edge(U,V) :- meek_force(U,V,_).",
		"violation(meek_conflict,U,V) :- meek_force(U,V,_), selected_edge(V,U).",
		"violation(meek_missing,U,V) :- meek_force(U,V,_), not selected_edge(U,V).",
		"#external scout_diagnostics.",
		":- violation(K,U,V), not scout_diagnostics.",
		"% CI implications are checked by the existing NetworkX d-separation implementation in the bridge.",
		"pre_treatment(Z) :- temporal_role(Z,pre_treatment).",
		"pre_treatment(Z) :- temporal_role(Z,unknown).",
		"potential_common_cause(Z,X,Y) :- selected_edge(Z,X), selected_edge(Z,Y), pre_treatment(Z), query_treatment(X), query_outcome(Y).",
		"candidate_confounder(Z) :- potential_common_cause(Z,X,Y).",
		"possible_mediator(X,M,Y) :- selected_edge(X,M), selected_edge(M,Y), query_treatment(X), query_outcome(Y).",
		"possible_collider(X,C,Y) :- selected_edge(X,C), selected_edge(Y,C), query_treatment(X), query_outcome(Y).",
		"selection_node(S) :- temporal_role(S,selection).",
		"possible_selection_collider(X,S,U) :- selected_edge(X,S), selected_edge(U,S), selection_node(S), query_treatment(X), U != X.",
		"association_intervention_gap :- causal_claim, association(X,Y,_), query_treatment(X), query_outcome(Y), not observed_intervention(X,Y).",
		"",
		"% The bridge relaxes CI evidence before applying semantic/parsimony tie-breaking and top-k.",
		"#show selected_hypothesis/1.",
		"#show selected_edge/2.",
		"#show meek_force/3.",
		"#show violation/3.",
		"#show potential_common_cause/3.",
		"#show candidate_confounder/1.",
		"#show possible_mediator/3.",
		"#show possible_collider/3.",
		"#show possible_selection_collider/3.",
		"#show association_intervention_gap/0.",
	);
	return `${lines.join("\n")}\n`;
}

// ---------------------------------------------------------------------------
// Answer-set interpretation and hypothesis cards
// ---------------------------------------------------------------------------

function parseAtom(atom: string): { name: string; args: string[] } | null {
	const match = atom.trim().match(/^([a-z_][a-z0-9_]*)(?:\(([^)]*)\))?$/i);
	if (!match) return null;
	const args = match[2] ? match[2].split(",").map((part) => part.trim()) : [];
	return { name: (match[1] ?? "").toLowerCase(), args };
}

function directedEdges(
	model: ScoutModel,
	program: NormalizedScoutProgram,
): Set<string> {
	const edges = new Set<string>();
	for (const edge of model.selected_edges) edges.add(candidateKey(edge));
	for (const edge of program.facts.known_edges) {
		if (edge.edgeType === "directed")
			edges.add(candidateKey([edge.from, edge.to]));
	}
	return edges;
}

function motifLabel(
	motif: string,
	args: string[],
	program: NormalizedScoutProgram,
): string {
	const label = (name: string): string => program.display_names[name] || name;
	if (motif === "potential_common_cause")
		return `${label(args[0] || "")}: ${label(args[0] || "")} -> ${label(args[1] || "")} and ${label(args[2] || "")}`;
	if (motif === "possible_mediator")
		return `${label(args[0] || "")} -> ${label(args[1] || "")} -> ${label(args[2] || "")}`;
	if (motif === "possible_selection_collider")
		return `${label(args[0] || "")} -> ${label(args[1] || "")} <- ${label(args[2] || "")} (selection)`;
	if (motif === "possible_collider")
		return `${label(args[0] || "")} -> ${label(args[1] || "")} <- ${label(args[2] || "")}`;
	return motif;
}

function currentGraphCaveats(
	program: NormalizedScoutProgram,
	belief: CausalBelief | null,
): string[] {
	if (!belief?.edges?.length) return [];
	const blockers = getEstimandRelevantGraphBlockers(
		belief.edges,
		program.query.treatment,
		program.query.outcome,
	);
	if (blockers.length === 0) return [];
	return [
		"The current treatment/outcome graph still contains " +
			blockers.length +
			" edges with uncertain direction or potential latent confounding. Scout highlights this consequence; the existing inference tools determine the next step.",
	];
}

function isCausalClaim(claim: string): boolean {
	return /cause|causal|effect|interven|impact|do\s*\(|导致|因果|干预|影响|效果/i.test(
		claim,
	);
}

function isAssociational(association: ScoutAssociation): boolean {
	return (
		!association.type ||
		/association|observational|correlation|descriptive|survey|相关|观察|相关性/i.test(
			association.type,
		)
	);
}

function hasAssociationInterventionGap(
	program: NormalizedScoutProgram,
	belief: CausalBelief | null,
): boolean {
	if (!isCausalClaim(program.query.claim)) return false;
	const relevant = program.facts.associations.filter((association) => {
		const values = new Set([
			...(association.from ? [association.from] : []),
			...(association.to ? [association.to] : []),
			...(association.nodes ?? []),
		]);
		return (
			values.has(program.query.treatment) || values.has(program.query.outcome)
		);
	});
	if (relevant.length === 0) return false;
	const hasMatchingEffect = (belief?.effects ?? []).some(
		(effect) =>
			normalizeScoutIdentifier(effect.treatment) === program.query.treatment &&
			normalizeScoutIdentifier(effect.outcome) === program.query.outcome &&
			typeof effect.ate === "number" &&
			Number.isFinite(effect.ate),
	);
	return !hasMatchingEffect && relevant.every(isAssociational);
}

function modelMotifs(
	model: ScoutModel,
	program: NormalizedScoutProgram,
): { motifs: string[]; motifArgs: Array<{ name: string; args: string[] }> } {
	const edges = directedEdges(model, program);
	const treatment = program.query.treatment;
	const outcome = program.query.outcome;
	const motifs: string[] = [];
	const motifArgs: Array<{ name: string; args: string[] }> = [];
	const nodes = program.facts.nodes;
	const role = new Map(
		program.facts.temporal_roles.map((item) => [item.node, item.role]),
	);
	for (const node of nodes) {
		if (node === treatment || node === outcome) continue;
		const common =
			edges.has(candidateKey([node, treatment])) &&
			edges.has(candidateKey([node, outcome]));
		if (
			common &&
			(role.get(node) === "pre_treatment" ||
				role.get(node) === "unknown" ||
				!role.has(node))
		) {
			motifs.push("potential_common_cause");
			motifArgs.push({
				name: "potential_common_cause",
				args: [node, treatment, outcome],
			});
		}
		for (const edgeValue of edges) {
			const [u, v] = edgeValue.split("\0");
			if (!u || !v) continue;
			if (
				u === treatment &&
				v === node &&
				edges.has(candidateKey([node, outcome]))
			) {
				motifs.push("possible_mediator");
				motifArgs.push({
					name: "possible_mediator",
					args: [treatment, node, outcome],
				});
			}
		}
		if (edges.has(candidateKey([treatment, node]))) {
			for (const other of nodes) {
				if (other === outcome && edges.has(candidateKey([other, node]))) {
					// Keep the collider local to X -> C <- Y; no claim about an
					// adjustment variable is made here.
					motifs.push("possible_collider");
					motifArgs.push({
						name: "possible_collider",
						args: [treatment, node, other],
					});
				}
				if (
					other !== treatment &&
					edges.has(candidateKey([other, node])) &&
					role.get(node) === "selection"
				) {
					motifs.push("possible_selection_collider");
					motifArgs.push({
						name: "possible_selection_collider",
						args: [treatment, node, other],
					});
				}
			}
		}
	}
	return { motifs: [...new Set(motifs)], motifArgs };
}

function consequenceText(
	motif: string,
	args: string[],
	program: NormalizedScoutProgram,
	belief: CausalBelief | null,
): string {
	const label = (name: string): string => program.display_names[name] || name;
	if (motif === "potential_common_cause") {
		if (!args[0] || !args[1] || !args[2])
			return "A candidate pre-treatment common cause could open a backdoor path; verify timing and adjustment with existing tools.";
		const effect = (belief?.effects ?? []).find(
			(item) =>
				normalizeScoutIdentifier(item.treatment) === program.query.treatment &&
				normalizeScoutIdentifier(item.outcome) === program.query.outcome,
		);
		if (
			effect?.adjustmentSet?.some(
				(item) => normalizeScoutIdentifier(item) === args[0],
			)
		)
			return `The existing estimate records adjustment for ${label(args[0] || "")}; its measurement and temporal ordering still need verification.`;
		return `If ${label(args[0] || "")} precedes treatment, ${label(args[0] || "")} -> ${label(args[1] || "")} and ${label(args[0] || "")} -> ${label(args[2] || "")} form a potential backdoor path; adjustment choices and ATE identification may change.`;
	}
	if (motif === "possible_mediator")
		return args[1]
			? `Adjusting for ${label(args[1])} may block a mediated treatment -> outcome path and change the estimand from a total effect to a controlled direct effect.`
			: "A candidate mediated path may cause adjustment to change the total-effect estimand; verify the estimand.";
	if (motif === "possible_collider")
		return args[1]
			? `Conditioning on ${label(args[1])} may open a noncausal path; compare identification and estimation with and without adjustment for this node.`
			: "Conditioning on a candidate collider may open a noncausal path; compare adjustment choices.";
	if (motif === "possible_selection_collider")
		return args[1]
			? "Selection node " +
					label(args[1]) +
					" may be jointly determined by treatment and another variable; conditioning on it may introduce selection bias."
			: "A candidate selection node may introduce selection bias; check inclusion criteria and the measurement process.";
	return motif;
}

function cardForCandidate(
	candidate: NormalizedScoutCandidate,
	program: NormalizedScoutProgram,
	belief: CausalBelief | null,
	model?: ScoutModel,
): ScoutCard {
	const label = (name: string): string => program.display_names[name] || name;
	const mechanism =
		candidate.candidate_edges
			.map(([from, to]) => `${label(from)} -> ${label(to)}`)
			.join(" and ") || "no directed edge supplied";
	const structural = model?.derived_motifs ?? [];
	const consequences = structural.map((motif) => {
		return consequenceText(motif, [], program, belief);
	});
	consequences.push(...currentGraphCaveats(program, belief));
	const probes: string[] = [];
	if (
		candidate.candidate_edges.some(
			([from, to]) =>
				from === program.query.treatment || to === program.query.treatment,
		)
	) {
		probes.push(
			`Check temporal ordering, baseline definitions, and missingness patterns for variables involving ${label(candidate.id)}.`,
		);
	}
	probes.push(
		`Compare adjustment sets with and without ${candidate.candidate_edges.map(([from]) => label(from)).join(", ")}, and run the existing causal_infer/refutation tools.`,
	);
	if (candidate.candidate_edges.length === 0)
		probes.push("Add at least one candidate edge supported by data or semantics before requesting Scout enumeration.");
	return {
		title: `${mechanism} — candidate mechanism`,
		hypothesis: mechanism,
		mechanism,
		why_considered: candidate.rationale
			? [candidate.rationale]
			: ["Structured candidate supplied by the caller; not yet established."],
		graph_consequence:
			structural.length > 0
				? structural
				: ["Candidate edges are retained; no initial motif has been derived from the current local structure."],
		structural_consequences:
			structural.length > 0
				? structural
				: ["Candidate edges are retained; no initial motif has been derived from the current local structure."],
		what_would_change:
			consequences.length > 0
				? consequences
				: [
						"Temporal or measurement evidence may change adjustment, identification, or intervention interpretation.",
					],
		supporting_evidence: candidate.support,
		support: candidate.support,
		counterevidence: candidate.counterevidence,
		discriminating_probes: probes,
		probes,
		status_for_llm: model
			? "candidate to inspect; this card is not proof of a hidden cause"
			: "candidate to inspect; symbolic enumeration unavailable",
	};
}

function cardForMotif(
	motif: string,
	args: string[],
	program: NormalizedScoutProgram,
	belief: CausalBelief | null,
): ScoutCard {
	const label = (name: string): string => program.display_names[name] || name;
	const motifText = motifLabel(motif, args, program);
	const probes =
		motif === "potential_common_cause"
			? [
					`Retrieve baseline proxies or historical records for ${label(args[0] || "")}, then compare stratified or sensitivity analyses.`,
					"Compare d-separation, ATE, and refutation results across adjustment sets.",
				]
			: motif === "possible_mediator"
				? [
						`Clarify whether the target is the total effect or a controlled direct effect; compare estimates with and without adjustment for ${label(args[1] || "")}.`,
					]
				: motif === "possible_selection_collider"
					? [
							"Check the inclusion criteria and missingness mechanism of the selection node, and compare paths before and after conditioning.",
						]
					: [
							`Check whether ${label(args[1] || "")} is a selection or measurement node; compare paths and estimates with and without conditioning.`,
						];
	const consequence = consequenceText(motif, args, program, belief);
	return {
		title: `${motif}: ${motifText}`,
		hypothesis: motifText,
		mechanism: motifText,
		why_considered: [
			"Derived locally from the current Belief, query, and structured candidate edges; this does not establish the true mechanism.",
		],
		graph_consequence: [consequence],
		structural_consequences: [consequence],
		what_would_change: [consequence, ...currentGraphCaveats(program, belief)],
		supporting_evidence: [],
		support: [],
		counterevidence: [],
		discriminating_probes: probes,
		probes,
		status_for_llm:
			"candidate to inspect; retain uncertainty and validate with existing causal tools",
	};
}

function associationGapCard(program: NormalizedScoutProgram): ScoutCard {
	return {
		title:
			"Association evidence does not by itself establish the intervention claim",
		hypothesis: `${program.query.claim || "causal claim"} may exceed the available associational evidence`,
		mechanism: `${program.query.claim || "causal claim"} may exceed the available associational evidence`,
		why_considered: [
			"The current evidence is marked observational/association, while the query uses causal/intervention language.",
		],
		graph_consequence: ["P(Y|X) and P(Y|do(X)) have not yet been connected by a study design or assumptions."],
		structural_consequences: ["P(Y|X) and P(Y|do(X)) have not yet been connected by a study design or assumptions."],
		what_would_change: [
			"Clarify the estimand, temporal ordering, and exchangeability assumptions, or introduce an experimental or quasi-experimental design.",
		],
		supporting_evidence: [],
		support: [],
		counterevidence: [],
		discriminating_probes: [
			"Verify the estimand and whether an effect estimate, randomized or natural experiment, or verifiable adjustment set exists for this treatment/outcome.",
			"If only correlation is available, retain an association interpretation rather than treating it as an intervention effect.",
		],
		probes: [
			"Verify the estimand and whether an effect estimate, randomized or natural experiment, or verifiable adjustment set exists for this treatment/outcome.",
			"If only correlation is available, retain an association interpretation rather than treating it as an intervention effect.",
		],
		status_for_llm:
			"currently unsupported intervention interpretation; clarify estimand and design",
	};
}

/** Rank without pruning: every answer set remains available to the LLM. */
export function rankScoutModels(
	models: ScoutModel[],
	program: NormalizedScoutProgram,
): ScoutModel[] {
	return models
		.map((model, index) => {
			const selected = new Set(model.hypotheses);
			let softCost = 0;
			for (const candidate of program.candidate_hypotheses) {
				if (selected.has(candidate.id)) {
					softCost +=
						1 - candidate.confidence + candidate.candidate_edges.length * 0.1;
				} else {
					softCost +=
						candidate.confidence * 0.25 + candidate.support.length * 0.05;
				}
			}
			// Keep a model's parsed cost only as a lower-bound hint; never let it
			// remove or collapse a competing answer set.
			if (Number.isFinite(model.ranking.soft_cost))
				softCost += Math.max(0, model.ranking.soft_cost) * 0.001;
			return {
				...model,
				ranking: { soft_cost: Number(softCost.toFixed(6)) },
				answer_set: model.answer_set || index + 1,
			};
		})
		.sort(
			(left, right) =>
				left.ranking.soft_cost - right.ranking.soft_cost ||
				left.answer_set - right.answer_set,
		);
}

/** Convert parsed Clingo atoms (or fixtures) into multiplicity-preserving models. */
export function interpretScoutAnswerSets(
	answerSets: ScoutAnswerSet[],
	program: NormalizedScoutProgram,
	belief: CausalBelief | null = null,
): { models: ScoutModel[]; cards: ScoutCard[] } {
	const motifArgsByAnswerSet = new Map<
		number,
		Array<{ name: string; args: string[] }>
	>();
	const models: ScoutModel[] = answerSets.map((answerSet, index) => {
		const selectedHypotheses: string[] = [];
		const selectedEdges: ScoutEdgeTuple[] = [];
		const meekOrientations: ScoutEdgeTuple[] = [];
		const derivedMotifs: string[] = [];
		const motifArgs: Array<{ name: string; args: string[] }> = [];
		for (const atom of answerSet.atoms) {
			const parsed = parseAtom(atom);
			if (!parsed) continue;
			if (parsed.name === "selected_hypothesis" && parsed.args[0])
				selectedHypotheses.push(parsed.args[0]);
			if (parsed.name === "selected_edge" && parsed.args.length >= 2) {
				const from = parsed.args[0];
				const to = parsed.args[1];
				if (from && to) selectedEdges.push([from, to]);
			}
			if (parsed.name === "meek_force" && parsed.args.length >= 2) {
				const from = parsed.args[0];
				const to = parsed.args[1];
				if (from && to) meekOrientations.push([from, to]);
			}
			if (
				parsed.name === "potential_common_cause" ||
				parsed.name === "possible_mediator" ||
				parsed.name === "possible_collider" ||
				parsed.name === "possible_selection_collider"
			) {
				derivedMotifs.push(parsed.name);
				motifArgs.push({ name: parsed.name, args: parsed.args });
			}
			if (parsed.name === "association_intervention_gap")
				derivedMotifs.push(parsed.name);
		}
		const model: ScoutModel = {
			answer_set: index + 1,
			hypotheses: [...new Set(selectedHypotheses)].sort(),
			derived_motifs: [...new Set(derivedMotifs)].sort(),
			selected_edges: [
				...new Map(
					selectedEdges.map((edge) => [candidateKey(edge), edge]),
				).values(),
			].sort((left, right) =>
				candidateKey(left).localeCompare(candidateKey(right)),
			),
			ranking: { soft_cost: answerSet.cost ?? 0 },
			...(answerSet.evidence ? { evidence: answerSet.evidence } : {}),
			...(meekOrientations.length > 0
				? {
						meek_orientations: [
							...new Map(
								meekOrientations.map((edge) => [candidateKey(edge), edge]),
							).values(),
						],
					}
				: {}),
		};
		const inferred = modelMotifs(model, program);
		if (inferred.motifs.length > 0) {
			model.derived_motifs = [
				...new Set([...model.derived_motifs, ...inferred.motifs]),
			].sort();
			for (const inferredArg of inferred.motifArgs) motifArgs.push(inferredArg);
		}
		motifArgsByAnswerSet.set(model.answer_set, motifArgs);
		return model;
	});
	const rankedModels = rankScoutModels(models, program);
	const cards: ScoutCard[] = [];
	const seenCards = new Set<string>();
	for (const model of rankedModels) {
		const motifArgs = motifArgsByAnswerSet.get(model.answer_set) ?? [];
		for (const motif of motifArgs) {
			if (motif.name === "association_intervention_gap") continue;
			const key = `${motif.name}:${motif.args.join("\0")}`;
			if (seenCards.has(key)) continue;
			seenCards.add(key);
			cards.push(cardForMotif(motif.name, motif.args, program, belief));
		}
		for (const hypothesis of model.hypotheses) {
			const candidate = program.candidate_hypotheses.find(
				(item) => item.id === hypothesis,
			);
			if (!candidate || seenCards.has(`candidate:${candidate.id}`)) continue;
			seenCards.add(`candidate:${candidate.id}`);
			cards.push(cardForCandidate(candidate, program, belief, model));
		}
	}
	for (const candidate of program.candidate_hypotheses) {
		if (seenCards.has(`candidate:${candidate.id}`)) continue;
		seenCards.add(`candidate:${candidate.id}`);
		cards.push(cardForCandidate(candidate, program, belief));
	}
	if (hasAssociationInterventionGap(program, belief))
		cards.push(associationGapCard(program));
	return { models: rankedModels, cards };
}

function unenumeratedCards(
	program: NormalizedScoutProgram,
	belief: CausalBelief | null,
): ScoutCard[] {
	const cards: ScoutCard[] = [];
	const knownModel: ScoutModel = {
		answer_set: 0,
		hypotheses: [],
		derived_motifs: [],
		selected_edges: program.facts.known_edges
			.filter((edge) => edge.edgeType === "directed")
			.map((edge) => [edge.from, edge.to]),
		ranking: { soft_cost: 0 },
	};
	const knownMotifs = modelMotifs(knownModel, program);
	for (const motif of knownMotifs.motifArgs)
		cards.push(cardForMotif(motif.name, motif.args, program, belief));
	for (const candidate of program.candidate_hypotheses)
		cards.push(cardForCandidate(candidate, program, belief));
	if (hasAssociationInterventionGap(program, belief))
		cards.push(associationGapCard(program));
	return cards;
}

// ---------------------------------------------------------------------------
// Optional Clingo worker
// ---------------------------------------------------------------------------

interface ClingoRunResult {
	available: boolean;
	answerSets: ScoutAnswerSet[];
	search?: ScoutSearch;
	error?: string;
}

function parseClingoOutput(stdout: string): {
	answerSets: ScoutAnswerSet[];
	search?: ScoutSearch;
} {
	const trimmed = stdout.trim();
	if (trimmed.startsWith("{")) {
		try {
			const parsed = JSON.parse(trimmed) as {
				answer_sets?: Array<{
					atoms?: unknown;
					cost?: unknown;
					matched?: string[];
					conflicted?: string[];
				}>;
				search?: ScoutSearch;
			};
			if (Array.isArray(parsed.answer_sets)) {
				return {
					answerSets: parsed.answer_sets.map((answerSet) => ({
						atoms: Array.isArray(answerSet.atoms)
							? answerSet.atoms.map(String).filter(Boolean)
							: [],
						cost: Number.isFinite(Number(answerSet.cost))
							? Number(answerSet.cost)
							: undefined,
						evidence: {
							matched: answerSet.matched ?? [],
							conflicted: answerSet.conflicted ?? [],
						},
					})),
					search: parsed.search,
				};
			}
		} catch {
			// Fall through to the text parser for a native Clingo invocation.
		}
	}
	const answerSets: ScoutAnswerSet[] = [];
	const regex =
		/Answer:\s*(\d+)\s*\n([\s\S]*?)(?=\nAnswer:\s*\d+|\n(?:OPTIMUM FOUND|SATISFIABLE|UNSATISFIABLE)|$)/g;
	for (const match of stdout.matchAll(regex)) {
		const atoms = (match[2] ?? "").trim().split(/\s+/).filter(Boolean);
		answerSets.push({ atoms });
	}
	return { answerSets };
}

function runClingo(
	asp: string,
	topK: number,
	evidence: NormalizedScoutIndependenceFact[],
	nodes: string[],
	candidates: NormalizedScoutCandidate[],
): Promise<ClingoRunResult> {
	return new Promise((resolve) => {
		let settled = false;
		const bridgePath = join(
			process.cwd(),
			"python",
			"scripts",
			"causal_scout_clingo.py",
		);
		const child = spawn(
			"uv",
			["run", "--project", "python", "python", bridgePath, String(topK)],
			{
				cwd: process.cwd(),
				stdio: ["pipe", "pipe", "pipe"],
			},
		);
		let stdout = "";
		let stderr = "";
		const finish = (result: ClingoRunResult): void => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			resolve(result);
		};
		const timeout = setTimeout(() => {
			child.kill();
			finish({
				available: true,
				answerSets: [],
				error: "Clingo timed out after 5 seconds.",
			});
		}, 5_000);
		child.stdout.on("data", (chunk: Buffer) => {
			stdout += chunk.toString();
		});
		child.stderr.on("data", (chunk: Buffer) => {
			stderr += chunk.toString();
		});
		child.on("error", (error: NodeJS.ErrnoException) => {
			finish({
				available: false,
				answerSets: [],
				error:
					error.code === "ENOENT"
						? "uv or the required Clingo dependency is unavailable; symbolic backend skipped."
						: String(error),
			});
		});
		child.on("close", (code) => {
			if (code === 0 || code === 10 || code === 20) {
				const parsed = parseClingoOutput(stdout);
				finish({
					available: true,
					answerSets: parsed.answerSets,
					search: parsed.search,
				});
			} else {
				finish({
					available: false,
					answerSets: parseClingoOutput(stdout).answerSets,
					error: stderr.trim() || `Clingo exited with code ${code}.`,
				});
			}
		});
		child.stdin.on("error", (error) => {
			finish({
				available: true,
				answerSets: parseClingoOutput(stdout).answerSets,
				error: "Clingo stdin failed: " + String(error),
			});
		});
		child.stdin.write(
			JSON.stringify({ program: asp, evidence, nodes, candidates }),
		);
		child.stdin.end();
	});
}

/** Execute Scout with the project's Clingo backend and return a JSON-ready result. */
export async function runCausalScout(
	input: CausalScoutInput,
	belief: CausalBelief | null = loadActiveBelief(),
): Promise<CausalScoutResult> {
	const program = buildScoutProgram(input, belief);
	const asp = serializeScoutProgram(program);
	const topK = Math.min(32, Math.max(1, Math.floor(Number(input.top_k) || 8)));
	const clingo = await runClingo(
		asp,
		topK,
		program.facts.conditional_independencies,
		program.facts.nodes,
		program.candidate_hypotheses,
	);
	if (!clingo.available) {
		return {
			backend: "unavailable",
			clingo_available: false,
			program,
			asp,
			models: [],
			cards: unenumeratedCards(program, belief),
			search: clingo.search,
			message: clingo.error
				? `Clingo backend unavailable: ${clingo.error}`
				: "Clingo unavailable; candidate cards are returned without fabricated answer sets.",
		};
	}
	const interpreted =
		clingo.answerSets.length > 0
			? interpretScoutAnswerSets(clingo.answerSets, program, belief)
			: { models: [], cards: unenumeratedCards(program, belief) };
	const searchSummary = clingo.search
		? ` Structural search kept ${clingo.search.compatible_count}/${clingo.search.structurally_valid_count} compatible graph(s)` +
			(clingo.search.relaxed_evidence.length > 0
				? ` after relaxing ${clingo.search.relaxed_evidence.length} lower-confidence CI fact(s).`
				: ".")
		: "";
	return {
		backend: "clingo",
		clingo_available: true,
		program,
		asp,
		models: interpreted.models,
		cards: interpreted.cards,
		search: clingo.search,
		message:
			clingo.error ||
			`Clingo enumerated ${interpreted.models.length} local answer set(s); models remain competing candidates.${searchSummary}`,
	};
}

// ---------------------------------------------------------------------------
// Built-in tool definition and handler
// ---------------------------------------------------------------------------

export const causalScoutTool = {
	type: "function" as const,
	function: {
		name: "causal_scout",
		description:
			"Assistive causal hypothesis search. Pass a typed query plus candidate_hypotheses and, when available, required/forbidden edges, a skeleton, temporal order, and tested conditional-independence facts. For a valid CPDAG/PDAG input, set graph_semantics and provide definite non-collider triples so the ASP program applies a sound Meek orientation closure. The tool checks candidate DAGs with d-separation, progressively relaxes lower-confidence conflicting CI facts, and returns competing graphs, compelled/reversible directions, rejection reasons, and discriminating probes. It never proves hidden confounders, rewrites the active Causal Belief, or changes causal_infer.",
		parameters: {
			type: "object",
			properties: {
				query: {
					type: "object",
					properties: {
						treatment: { type: "string" },
						outcome: { type: "string" },
						estimand: {
							type: "string",
							description: "Requested estimand, e.g. total_effect.",
						},
						claim: {
							type: "string",
							description: "Current causal or intervention claim.",
						},
					},
					required: ["treatment", "outcome"],
				},
				facts: {
					type: "object",
					properties: {
						nodes: { type: "array", items: { type: "string" } },
						node_descriptions: {
							type: "array",
							items: {
								type: "object",
								properties: {
									node: { type: "string" },
									description: { type: "string" },
								},
								required: ["node", "description"],
							},
							description: "Optional measurement descriptions.",
						},
						known_edges: {
							type: "array",
							items: {
								type: "array",
								items: { type: "string" },
								minItems: 2,
								maxItems: 2,
							},
						},
						temporal_roles: {
							type: "array",
							items: {
								type: "object",
								properties: {
									node: { type: "string" },
									role: { type: "string" },
								},
								required: ["node", "role"],
							},
						},
						associations: {
							type: "array",
							items: {
								type: "object",
								properties: {
									from: { type: "string" },
									to: { type: "string" },
									type: { type: "string" },
									description: { type: "string" },
									source: { type: "string" },
								},
							},
						},
						required_edges: {
							type: "array",
							items: {
								type: "array",
								items: { type: "string" },
								minItems: 2,
								maxItems: 2,
							},
						},
						forbidden_edges: {
							type: "array",
							items: {
								type: "array",
								items: { type: "string" },
								minItems: 2,
								maxItems: 2,
							},
						},
						incompatible_hypotheses: {
							type: "array",
							items: {
								type: "array",
								items: { type: "string" },
								minItems: 2,
								maxItems: 2,
							},
						},
						conditional_independencies: {
							type: "array",
							items: {
								type: "object",
								properties: {
									id: { type: "string" },
									from: { type: "string" },
									to: { type: "string" },
									conditioned_on: { type: "array", items: { type: "string" } },
									independent: { type: "boolean" },
									confidence: { type: "number", minimum: 0, maximum: 1 },
									source: { type: "string" },
								},
								required: ["from", "to", "independent"],
							},
						},
						graph_semantics: {
							type: "string",
							enum: ["candidate_dag", "pdag", "cpdag"],
							description:
								"Use cpdag/pdag only when undirected edges come from a valid partially oriented graph and collider/non-collider information is available.",
						},
						definite_noncolliders: {
							type: "array",
							items: {
								type: "array",
								items: { type: "string" },
								minItems: 3,
								maxItems: 3,
							},
							description:
								"Unshielded triples known not to be colliders, typically derived from PC separating sets.",
						},
						skeleton: {
							type: "array",
							items: {
								type: "array",
								items: { type: "string" },
								minItems: 2,
								maxItems: 2,
							},
						},
						temporal_order: {
							type: "array",
							items: {
								type: "array",
								items: { type: "string" },
								minItems: 2,
								maxItems: 2,
							},
						},
					},
				},
				candidate_hypotheses: {
					type: "array",
					items: {
						type: "object",
						properties: {
							id: { type: "string" },
							candidate_edges: {
								type: "array",
								items: {
									type: "array",
									items: { type: "string" },
									minItems: 2,
									maxItems: 2,
								},
							},
							rationale: { type: "string" },
							support: { type: "array", items: { type: "string" } },
							counterevidence: { type: "array", items: { type: "string" } },
							confidence: { type: "number", minimum: 0, maximum: 1 },
						},
						required: ["id", "candidate_edges"],
					},
				},
				top_k: { type: "integer", minimum: 1, maximum: 32, default: 8 },
			},
			required: ["query"],
		},
	},
};

export async function causalScout(
	input: Record<string, unknown>,
): Promise<string> {
	try {
		const result = await runCausalScout(input as unknown as CausalScoutInput);
		return JSON.stringify(result, null, 2);
	} catch (error) {
		return `Error running causal scout: ${String(error)}`;
	}
}
