/**
 * Causal Cognition research prototype.
 *
 * A persistent causal concept graph and task-local Causal Beliefs co-grow
 * through Index search -> graph exploration -> task validation -> Lift.
 * Retrieval is deliberately LLM-directed: the model chooses keywords and may
 * follow returned frontier concepts for as many bounded calls as the task needs.
 */

import { randomUUID } from "node:crypto";
import { mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { atomicWriteFileSync, withFileLockSync } from "@/core/resource_lock";

import type {
	CausalBelief,
	CausalEdge,
	CausalVariable,
} from "@/memory/causal_belief";
import {
	assessCausalRelationBelief,
	type CausalRelationBeliefAssessment,
	type CausalRelationClaimConfidence,
	type CausalRelationVerdict,
} from "@/memory/causal_relation_belief";

export const CAUSAL_COGNITION_GRAPH_SCHEMA_VERSION = 4;

export interface CausalConcept {
	id: string;
	label: string;
	aliases: string[];
	/** Search cues are topical; unlike aliases, they do not assert identity. */
	keywords?: string[];
	description?: string;
}

export type RelationVerdict = CausalRelationVerdict;
export type RelationConfidence = CausalRelationClaimConfidence;

/** One episode-level claim about one contextual relation variant. */
export interface RelationClaim {
	episodeId: string;
	verdict: RelationVerdict;
	confidence?: RelationConfidence;
	note?: string;
}

export const CAUSAL_RELATION_LAGS = [
	"unknown",
	"immediate",
	"days",
	"weeks",
	"months",
	"quarters",
	"years",
] as const;
export type CausalRelationLag = (typeof CAUSAL_RELATION_LAGS)[number];

/**
 * A relation variant is identified by directed endpoints + lag + exact context.
 * Its claims array contains multiple episode-level assertions; the relation is not a
 * single unqualified truth assertion.
 */
export interface CausalRelation {
	id: string;
	causeId: string;
	effectId: string;
	mechanism?: string;
	contexts: string[];
	lag?: string;
	claims: RelationClaim[];
}

export type GroundingKind =
	| "exact"
	| "alias"
	| "proxy"
	| "hypothesis"
	| "ambiguous";

export interface CausalGrounding {
	variable: string;
	conceptId: string;
	kind: GroundingKind;
	confidence: number;
	rationale?: string;
}

export interface CausalCognitionEpisode {
	id: string;
	beliefId: string;
	objective: string;
	context: string[];
	groundings: CausalGrounding[];
	localGraph: {
		variables: string[];
		edges: CausalEdge[];
	};
	summary?: string;
	/** Searchable execution experience; its evidence is not a causal relation claim. */
	experience?: { title: string; keywords: string[]; evidence: string };
	createdAt: string;
}

export interface CognitionExperienceInput {
	title: string;
	summary: string;
	keywords: string[];
	context: string[];
	evidence: string;
}

/** Persist an experience using the same episode store, including when no graph exists. */
export function rememberCognitionExperience(
	inputGraph: CausalCognitionGraphData,
	input: CognitionExperienceInput,
	belief?: CausalBelief | null,
): { graph: CausalCognitionGraphData; episode: CausalCognitionEpisode } {
	if (!input.title.trim() || !input.summary.trim() || !input.evidence.trim()) {
		throw new Error("Experience requires a title, summary, and evidence from the task or tool documentation.");
	}
	const graph = normalizeGraph(structuredClone(inputGraph));
	const episode: CausalCognitionEpisode = {
		id: randomUUID(),
		beliefId: belief?.id ?? "",
		objective: belief?.objective || input.title.trim(),
		context: uniqueStrings(input.context),
		groundings: [],
		localGraph: { variables: [], edges: [] },
		summary: input.summary.trim(),
		experience: {
			title: input.title.trim(),
			keywords: uniqueStrings(input.keywords),
			evidence: input.evidence.trim(),
		},
		createdAt: new Date().toISOString(),
	};
	graph.episodes.push(episode);
	return { graph, episode };
}

export interface CausalCognitionGraphData {
	schemaVersion: number;
	concepts: CausalConcept[];
	relations: CausalRelation[];
	episodes: CausalCognitionEpisode[];
}

export type CausalSearchDirection = "upstream" | "downstream" | "both";

export interface CausalConceptIndexMatch {
	conceptId: string;
	label: string;
	aliases: string[];
	keywords: string[];
	description?: string;
	score: number;
	matchedBy: string[];
	activeIncoming: number;
	activeOutgoing: number;
}

export interface CausalRelationSearchMatch {
	relation: CausalRelation;
	assessment: CausalRelationBeliefAssessment;
	contextSimilarity: number;
	score: number;
	seedConceptIds: string[];
	neighborConceptIds: string[];
}

export interface CausalFrontierConcept {
	conceptId: string;
	label: string;
	aliases: string[];
	keywords: string[];
	viaRelationIds: string[];
	activeIncoming: number;
	activeOutgoing: number;
}

export interface CausalCognitionSearchOptions {
	/** Free-form LLM-chosen index terms. */
	keywords?: string[];
	/** Exact concept IDs from an earlier search/frontier result. */
	conceptIds?: string[];
	/** Concepts already inspected by the caller; suppresses repeated frontier. */
	knownConceptIds?: string[];
	direction?: CausalSearchDirection;
	context?: string[];
	conceptLimit?: number;
	relationLimit?: number;
}

export interface CausalCognitionSearchResult {
	queryKeywords: string[];
	indexMatches: CausalConceptIndexMatch[];
	seedConceptIds: string[];
	relations: CausalRelationSearchMatch[];
	frontier: CausalFrontierConcept[];
	experienceMatches: Array<{ episode: CausalCognitionEpisode; score: number }>;
	warnings: string[];
	visitedRelationCount: number;
	truncated: boolean;
}

export interface LiftRelationInput {
	fromVariable: string;
	toVariable: string;
	verdict: RelationVerdict;
	confidence?: RelationConfidence;
	mechanism?: string;
	lag?: string;
	note?: string;
}

export interface LiftCausalBeliefInput {
	episodeId?: string;
	groundings?: CausalGrounding[];
	variableAliases?: Record<string, string>;
	context?: string[];
	relations?: LiftRelationInput[];
	summary?: string;
	createdAt?: string;
}

export interface LiftCausalBeliefResult {
	graph: CausalCognitionGraphData;
	episode: CausalCognitionEpisode;
	createdConceptIds: string[];
	createdRelationIds: string[];
	updatedRelationIds: string[];
	mergedConcepts: Array<{ sourceConceptId: string; targetConceptId: string }>;
	warnings: string[];
}

export interface CausalConceptUpdateInput {
	conceptId: string;
	label?: string;
	description?: string;
	addAliases?: string[];
	removeAliases?: string[];
	addKeywords?: string[];
	removeKeywords?: string[];
}

export interface CausalConceptMergeInput {
	sourceConceptId: string;
	targetConceptId: string;
}

export interface RelationClaimRetractionInput {
	relationId: string;
	episodeId: string;
}

export interface MaintainCausalCognitionInput {
	conceptUpdates?: CausalConceptUpdateInput[];
	merges?: CausalConceptMergeInput[];
	retractClaims?: RelationClaimRetractionInput[];
	removeConceptIds?: string[];
	autoMerge?: boolean;
}

export interface MaintainCausalCognitionResult {
	graph: CausalCognitionGraphData;
	updatedConceptIds: string[];
	mergedConcepts: Array<{ sourceConceptId: string; targetConceptId: string }>;
	retractedClaimCount: number;
	removedRelationIds: string[];
	removedConceptIds: string[];
	warnings: string[];
}

const EMPTY_GRAPH = (): CausalCognitionGraphData => ({
	schemaVersion: CAUSAL_COGNITION_GRAPH_SCHEMA_VERSION,
	concepts: [],
	relations: [],
	episodes: [],
});

const KEYWORD_STOPWORDS = new Set([
	"about",
	"after",
	"before",
	"data",
	"episode",
	"from",
	"into",
	"other",
	"task",
	"that",
	"the",
	"this",
	"unknown",
	"variable",
	"with",
]);

const CANONICAL_RELATION_LAGS = new Set<string>(CAUSAL_RELATION_LAGS);
const STABLE_CONTEXT_TAG = /^(?:domain|scope):\S(?:.*\S)?$/iu;
const EPHEMERAL_CONTEXT_MARKER =
	/(?:\b(?:run|task|cutoff|resolution|resolved|deadline|date|year|month|as[- ]of)\b|(?:19|20)\d{2}|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b|\d{1,2}\s*(?:月|日|号)|(?:任务|轮次|截止|揭晓|日期|年份|月份))/iu;

function normalize(value: string): string {
	return value
		.normalize("NFKC")
		.toLocaleLowerCase()
		.replace(/[^\p{L}\p{N}]+/gu, " ")
		.trim();
}

function tokens(value: string): Set<string> {
	return new Set(normalize(value).split(/\s+/u).filter(Boolean));
}

function jaccard(left: Set<string>, right: Set<string>): number {
	if (left.size === 0 || right.size === 0) return 0;
	let intersection = 0;
	for (const item of left) if (right.has(item)) intersection += 1;
	return intersection / (left.size + right.size - intersection);
}

function clamp01(value: number): number {
	return Math.max(0, Math.min(1, value));
}

function boundedInteger(
	value: number | undefined,
	fallback: number,
	minimum: number,
	maximum: number,
): number {
	if (!Number.isFinite(value)) return fallback;
	return Math.max(minimum, Math.min(maximum, Math.floor(value ?? fallback)));
}

function uniqueStrings(values: readonly string[]): string[] {
	const seen = new Set<string>();
	const result: string[] = [];
	for (const raw of values) {
		const value = raw.trim();
		const key = normalize(value);
		if (!value || !key || seen.has(key)) continue;
		seen.add(key);
		result.push(value);
	}
	return result;
}

function extractKeywords(value: string, limit = 12): string[] {
	return [...tokens(value)]
		.filter((item) => item.length >= 3 && !KEYWORD_STOPWORDS.has(item))
		.slice(0, limit);
}

function isCanonicalRelationLag(value: string | undefined): boolean {
	return Boolean(value && CANONICAL_RELATION_LAGS.has(value.trim()));
}

function invalidLiftContext(value: string): boolean {
	return (
		!STABLE_CONTEXT_TAG.test(value) || EPHEMERAL_CONTEXT_MARKER.test(value)
	);
}

function hasText(value: string | undefined): boolean {
	return Boolean(value?.trim());
}

function contextSignature(values: readonly string[]): string {
	return uniqueStrings(values.map(normalize)).sort().join("|");
}

function relationKey(
	relation: Pick<CausalRelation, "causeId" | "effectId" | "contexts" | "lag">,
): string {
	return [
		relation.causeId,
		relation.effectId,
		relation.lag ?? "",
		contextSignature(relation.contexts),
	].join("::");
}

function confidenceRank(value: RelationConfidence | undefined): number {
	if (value === "low") return 1;
	if (value === "medium") return 2;
	return 3;
}

function normalizeGraph(
	input: CausalCognitionGraphData,
): CausalCognitionGraphData {
	const usedRelationIds = new Set<string>();
	const normalized: CausalCognitionGraphData = {
		schemaVersion: CAUSAL_COGNITION_GRAPH_SCHEMA_VERSION,
		concepts: input.concepts.map((concept) => ({
			...concept,
			aliases: uniqueStrings(concept.aliases ?? []),
			keywords: uniqueStrings(concept.keywords ?? []),
		})),
		relations: input.relations.map((relation) => {
			const serialized = relation as CausalRelation & {
				evidence?: RelationClaim[];
				kind?: unknown;
			};
			const baseId =
				serialized.id.replace(/--(?:causes|inhibits)--/gu, "--to--") ||
				`${serialized.causeId}--to--${serialized.effectId}`;
			const id = uniqueId(baseId, usedRelationIds);
			usedRelationIds.add(id);
			return {
				id,
				causeId: serialized.causeId,
				effectId: serialized.effectId,
				...(serialized.mechanism ? { mechanism: serialized.mechanism } : {}),
				contexts: uniqueStrings(serialized.contexts ?? []),
				...(serialized.lag ? { lag: serialized.lag } : {}),
				claims: (serialized.claims ?? serialized.evidence ?? []).map(
					(claim) => ({
						...claim,
					}),
				),
			};
		}),
		episodes: input.episodes.map((episode) => structuredClone(episode)),
	};
	deduplicateRelations(normalized);
	return normalized;
}

export function assessCausalRelation(
	relation: CausalRelation,
): CausalRelationBeliefAssessment {
	return assessCausalRelationBelief(relation.claims);
}

function activeDegree(
	graph: CausalCognitionGraphData,
	conceptId: string,
): { incoming: number; outgoing: number } {
	let incoming = 0;
	let outgoing = 0;
	for (const relation of graph.relations) {
		if (!assessCausalRelation(relation).active) continue;
		if (relation.effectId === conceptId) incoming += 1;
		if (relation.causeId === conceptId) outgoing += 1;
	}
	return { incoming, outgoing };
}

export function causalContextSimilarity(
	relationContext: readonly string[],
	queryContext: readonly string[],
): number {
	if (relationContext.length === 0 || queryContext.length === 0) return 0.6;
	return jaccard(
		tokens(relationContext.join(" ")),
		tokens(queryContext.join(" ")),
	);
}

function conceptIndexScore(
	concept: CausalConcept,
	query: string,
): { score: number; matchedBy: string[] } {
	const normalizedQuery = normalize(query);
	const queryTokens = tokens(query);
	let score = 0;
	const matchedBy: string[] = [];
	if (
		normalize(concept.id) === normalizedQuery ||
		normalize(concept.label) === normalizedQuery
	) {
		score = 1;
		matchedBy.push("id/label exact");
	}
	for (const alias of concept.aliases) {
		if (normalize(alias) === normalizedQuery) {
			score = Math.max(score, 0.95);
			matchedBy.push(`alias:${alias}`);
		}
	}
	for (const keyword of concept.keywords ?? []) {
		if (normalize(keyword) === normalizedQuery) {
			score = Math.max(score, 0.85);
			matchedBy.push(`keyword:${keyword}`);
		}
	}
	const fields: Array<[string, string, number]> = [
		["label", concept.label, 0.8],
		["aliases", concept.aliases.join(" "), 0.75],
		["keywords", (concept.keywords ?? []).join(" "), 0.7],
		["description", concept.description ?? "", 0.45],
	];
	for (const [name, value, weight] of fields) {
		const overlap = jaccard(queryTokens, tokens(value));
		if (overlap <= 0) continue;
		const candidate = weight * overlap;
		if (candidate > score) score = candidate;
		matchedBy.push(`${name}:token-overlap`);
	}
	return { score: clamp01(score), matchedBy: uniqueStrings(matchedBy) };
}

function relationTextScore(relation: CausalRelation, query: string): number {
	const text = [
		relation.mechanism ?? "",
		relation.contexts.join(" "),
		relation.lag ?? "",
	].join(" ");
	return jaccard(tokens(query), tokens(text));
}

export function searchCausalCognition(
	inputGraph: CausalCognitionGraphData,
	options: CausalCognitionSearchOptions = {},
): CausalCognitionSearchResult {
	const graph = normalizeGraph(inputGraph);
	const warnings: string[] = [];
	const queryKeywords = uniqueStrings(options.keywords ?? []).slice(0, 8);
	const conceptLimit = boundedInteger(options.conceptLimit, 6, 1, 12);
	const relationLimit = boundedInteger(options.relationLimit, 8, 1, 20);
	const rankedExperiences = graph.episodes
		.filter((episode) => episode.experience)
		.map((episode) => ({
			episode,
			score: Math.max(0, ...queryKeywords.map((query) => conceptIndexScore({
				id: episode.id,
				label: episode.experience!.title,
				aliases: [],
				keywords: episode.experience!.keywords,
				description: [episode.summary, episode.experience!.evidence, ...episode.context].join(" "),
			}, query).score)),
		}))
		.filter((match) => match.score > 0)
		.sort((a, b) => b.score - a.score || a.episode.id.localeCompare(b.episode.id));
	const experienceMatches = rankedExperiences.slice(0, conceptLimit);
	const conceptById = new Map(
		graph.concepts.map((concept) => [concept.id, concept]),
	);
	const hitMap = new Map<string, { score: number; matchedBy: string[] }>();

	for (const concept of graph.concepts) {
		for (const query of queryKeywords) {
			const match = conceptIndexScore(concept, query);
			if (match.score <= 0) continue;
			const previous = hitMap.get(concept.id);
			hitMap.set(concept.id, {
				score: Math.max(previous?.score ?? 0, match.score),
				matchedBy: uniqueStrings([
					...(previous?.matchedBy ?? []),
					...match.matchedBy,
				]),
			});
		}
	}

	// Mechanism/context terms are searchable, but never imply concept identity.
	for (const relation of graph.relations) {
		for (const query of queryKeywords) {
			const overlap = relationTextScore(relation, query);
			if (overlap <= 0) continue;
			for (const conceptId of [relation.causeId, relation.effectId]) {
				if (!conceptById.has(conceptId)) continue;
				const previous = hitMap.get(conceptId);
				hitMap.set(conceptId, {
					score: Math.max(previous?.score ?? 0, 0.55 * overlap),
					matchedBy: uniqueStrings([
						...(previous?.matchedBy ?? []),
						`relation:${relation.id}`,
					]),
				});
			}
		}
	}

	const indexMatches = [...hitMap.entries()]
		.map(([conceptId, match]): CausalConceptIndexMatch | undefined => {
			const concept = conceptById.get(conceptId);
			if (!concept) return undefined;
			const degree = activeDegree(graph, conceptId);
			return {
				conceptId,
				label: concept.label,
				aliases: [...concept.aliases],
				keywords: [...(concept.keywords ?? [])],
				...(concept.description ? { description: concept.description } : {}),
				score: match.score,
				matchedBy: match.matchedBy,
				activeIncoming: degree.incoming,
				activeOutgoing: degree.outgoing,
			};
		})
		.filter((item): item is CausalConceptIndexMatch => item !== undefined)
		.sort(
			(left, right) =>
				right.score - left.score ||
				left.conceptId.localeCompare(right.conceptId),
		)
		.slice(0, conceptLimit);

	const requestedSeeds = uniqueStrings(options.conceptIds ?? []);
	const seedConceptIds: string[] = [];
	for (const conceptId of requestedSeeds) {
		if (conceptById.has(conceptId)) seedConceptIds.push(conceptId);
		else warnings.push(`Unknown concept ID: ${conceptId}`);
	}
	if (queryKeywords.length === 0 && requestedSeeds.length === 0) {
		warnings.push(
			"Provide index keywords or exact concept IDs from a prior result.",
		);
	}
	if (queryKeywords.length > 0 && indexMatches.length === 0 && experienceMatches.length === 0) {
		warnings.push(
			"No concept, relation, or experience matched the supplied keywords.",
		);
	}

	const known = new Set(uniqueStrings(options.knownConceptIds ?? []));
	const seedSet = new Set(seedConceptIds);
	const direction = options.direction ?? "both";
	const context = uniqueStrings(options.context ?? []);
	const candidates = new Map<string, CausalRelationSearchMatch>();
	let visitedRelationCount = 0;
	for (const relation of graph.relations) {
		const assessment = assessCausalRelation(relation);
		if (!assessment.active) continue;
		const upstreamSeeds = seedConceptIds.filter(
			(id) => relation.effectId === id,
		);
		const downstreamSeeds = seedConceptIds.filter(
			(id) => relation.causeId === id,
		);
		const allowedSeeds = [
			...(direction === "downstream" ? [] : upstreamSeeds),
			...(direction === "upstream" ? [] : downstreamSeeds),
		];
		if (allowedSeeds.length === 0) continue;
		visitedRelationCount += 1;
		const neighbors = uniqueStrings(
			allowedSeeds.map((seed) =>
				relation.causeId === seed ? relation.effectId : relation.causeId,
			),
		).filter((id) => !known.has(id) && !seedSet.has(id));
		if (neighbors.length === 0) continue;
		const contextSimilarity = causalContextSimilarity(
			relation.contexts,
			context,
		);
		candidates.set(relation.id, {
			relation: structuredClone(relation),
			assessment,
			contextSimilarity,
			score: clamp01(0.7 * assessment.posterior + 0.3 * contextSimilarity),
			seedConceptIds: uniqueStrings(allowedSeeds),
			neighborConceptIds: neighbors,
		});
	}
	const rankedRelations = [...candidates.values()].sort(
		(left, right) =>
			right.score - left.score ||
			left.relation.id.localeCompare(right.relation.id),
	);
	const relations = rankedRelations.slice(0, relationLimit);
	const frontierMap = new Map<string, CausalFrontierConcept>();
	for (const match of relations) {
		for (const conceptId of match.neighborConceptIds) {
			const concept = conceptById.get(conceptId);
			if (!concept) continue;
			const degree = activeDegree(graph, conceptId);
			const previous = frontierMap.get(conceptId);
			frontierMap.set(conceptId, {
				conceptId,
				label: concept.label,
				aliases: [...concept.aliases],
				keywords: [...(concept.keywords ?? [])],
				viaRelationIds: uniqueStrings([
					...(previous?.viaRelationIds ?? []),
					match.relation.id,
				]),
				activeIncoming: degree.incoming,
				activeOutgoing: degree.outgoing,
			});
		}
	}
	if (seedConceptIds.length > 0 && relations.length === 0) {
		warnings.push(
			"No new active relation frontier was found for these concepts and direction.",
		);
	}
	return {
		queryKeywords,
		indexMatches,
		experienceMatches,
		seedConceptIds,
		relations,
		frontier: [...frontierMap.values()].sort((left, right) =>
			left.conceptId.localeCompare(right.conceptId),
		),
		warnings,
		visitedRelationCount,
		truncated: rankedRelations.length > relations.length || rankedExperiences.length > experienceMatches.length,
	};
}

function slug(value: string): string {
	return normalize(value).replace(/\s+/gu, "-") || randomUUID();
}

function uniqueId(base: string, used: Set<string>): string {
	if (!used.has(base)) return base;
	let index = 2;
	while (used.has(`${base}-${index}`)) index += 1;
	return `${base}-${index}`;
}

function automaticConceptCandidate(
	graph: CausalCognitionGraphData,
	variable: CausalVariable,
): CausalGrounding | undefined {
	const variableName = normalize(variable.name);
	const exact = graph.concepts.filter(
		(concept) =>
			normalize(concept.label) === variableName ||
			concept.aliases.some((alias) => normalize(alias) === variableName),
	);
	if (exact.length === 1) {
		const concept = exact[0];
		if (!concept) return undefined;
		return {
			variable: variable.name,
			conceptId: concept.id,
			kind: normalize(concept.label) === variableName ? "exact" : "alias",
			confidence: normalize(concept.label) === variableName ? 1 : 0.95,
			rationale: "Conservative exact label/alias match",
		};
	}
	if (exact.length > 1) {
		return {
			variable: variable.name,
			conceptId: exact.map((concept) => concept.id).sort()[0] ?? "",
			kind: "ambiguous",
			confidence: 0,
			rationale: `Multiple exact matches: ${exact
				.map((concept) => concept.id)
				.join(", ")}`,
		};
	}
	const variableTokens = tokens(`${variable.name} ${variable.description}`);
	const ranked = graph.concepts
		.map((concept) => ({
			concept,
			score: jaccard(
				variableTokens,
				tokens(
					`${concept.label} ${concept.aliases.join(" ")} ${concept.description ?? ""}`,
				),
			),
		}))
		.filter((candidate) => candidate.score >= 0.85)
		.sort(
			(left, right) =>
				right.score - left.score ||
				left.concept.id.localeCompare(right.concept.id),
		);
	const best = ranked[0];
	if (!best) return undefined;
	if (ranked[1] && best.score - ranked[1].score < 0.1) {
		return {
			variable: variable.name,
			conceptId: best.concept.id,
			kind: "ambiguous",
			confidence: best.score,
			rationale: `Near-tie with ${ranked[1].concept.id}`,
		};
	}
	return {
		variable: variable.name,
		conceptId: best.concept.id,
		kind: "hypothesis",
		confidence: best.score,
		rationale: "High-overlap conservative automatic match",
	};
}

function addVariableVocabulary(
	concept: CausalConcept,
	variable: CausalVariable,
): boolean {
	const before = JSON.stringify([concept.aliases, concept.keywords]);
	if (normalize(variable.name) !== normalize(concept.label)) {
		concept.aliases = uniqueStrings([...concept.aliases, variable.name]);
	}
	concept.keywords = uniqueStrings([
		...(concept.keywords ?? []),
		...extractKeywords(`${variable.description} ${variable.source}`),
	]);
	return before !== JSON.stringify([concept.aliases, concept.keywords]);
}

function claimForEpisode(
	claims: RelationClaim[],
	episodeId: string,
): RelationClaim | undefined {
	return claims.find((claim) => claim.episodeId === episodeId);
}

function relationInputConfidence(
	input: LiftRelationInput,
	edge: CausalEdge | undefined,
	warnings: string[],
): RelationConfidence {
	const requested = input.confidence ?? edge?.confidence ?? "medium";
	if (
		input.verdict !== "contradicted" &&
		edge &&
		confidenceRank(requested) > confidenceRank(edge.confidence)
	) {
		warnings.push(
			`Claim confidence clamped to final directed Belief confidence for ${input.fromVariable} -> ${input.toVariable}: ${requested} -> ${edge.confidence}.`,
		);
		return edge.confidence;
	}
	return requested;
}

function duplicateConceptPair(
	graph: CausalCognitionGraphData,
): { sourceConceptId: string; targetConceptId: string } | undefined {
	const degree = (conceptId: string) =>
		graph.relations.filter(
			(relation) =>
				relation.causeId === conceptId || relation.effectId === conceptId,
		).length;
	for (let leftIndex = 0; leftIndex < graph.concepts.length; leftIndex += 1) {
		const left = graph.concepts[leftIndex];
		if (!left) continue;
		for (
			let rightIndex = leftIndex + 1;
			rightIndex < graph.concepts.length;
			rightIndex += 1
		) {
			const right = graph.concepts[rightIndex];
			if (!right) continue;
			const sameLabel = normalize(left.label) === normalize(right.label);
			const labelAlias =
				left.aliases.some(
					(alias) => normalize(alias) === normalize(right.label),
				) ||
				right.aliases.some(
					(alias) => normalize(alias) === normalize(left.label),
				);
			if (!sameLabel && !labelAlias) continue;
			const ranked = [left, right].sort(
				(a, b) =>
					degree(b.id) - degree(a.id) ||
					b.aliases.length +
						(b.keywords?.length ?? 0) -
						(a.aliases.length + (a.keywords?.length ?? 0)) ||
					a.id.localeCompare(b.id),
			);
			const target = ranked[0];
			const source = ranked[1];
			if (target && source) {
				return {
					sourceConceptId: source.id,
					targetConceptId: target.id,
				};
			}
		}
	}
	return undefined;
}

function mergeEpisodeClaims(
	left: RelationClaim,
	right: RelationClaim,
): RelationClaim {
	if (left.verdict === right.verdict) {
		return confidenceRank(right.confidence) > confidenceRank(left.confidence)
			? { ...right }
			: { ...left };
	}
	return {
		episodeId: left.episodeId,
		verdict: "hypothesized",
		confidence: "low",
		note: "Conflicting duplicate claims were merged as unresolved.",
	};
}

function deduplicateRelations(graph: CausalCognitionGraphData): void {
	const byKey = new Map<string, CausalRelation>();
	const deduplicated: CausalRelation[] = [];
	for (const relation of graph.relations) {
		if (relation.causeId === relation.effectId) continue;
		const key = relationKey(relation);
		const existing = byKey.get(key);
		if (!existing) {
			const clone = structuredClone(relation);
			byKey.set(key, clone);
			deduplicated.push(clone);
			continue;
		}
		if (!existing.mechanism && relation.mechanism) {
			existing.mechanism = relation.mechanism;
		}
		for (const claim of relation.claims) {
			const prior = claimForEpisode(existing.claims, claim.episodeId);
			if (!prior) existing.claims.push(structuredClone(claim));
			else Object.assign(prior, mergeEpisodeClaims(prior, claim));
		}
	}
	graph.relations = deduplicated;
}

function mergeConcept(
	graph: CausalCognitionGraphData,
	merge: CausalConceptMergeInput,
): boolean {
	if (merge.sourceConceptId === merge.targetConceptId) return false;
	const source = graph.concepts.find(
		(concept) => concept.id === merge.sourceConceptId,
	);
	const target = graph.concepts.find(
		(concept) => concept.id === merge.targetConceptId,
	);
	if (!source || !target) return false;
	target.aliases = uniqueStrings([
		...target.aliases,
		source.label,
		...source.aliases,
	]).filter((alias) => normalize(alias) !== normalize(target.label));
	target.keywords = uniqueStrings([
		...(target.keywords ?? []),
		...(source.keywords ?? []),
	]);
	if (!target.description && source.description) {
		target.description = source.description;
	}
	for (const relation of graph.relations) {
		if (relation.causeId === source.id) relation.causeId = target.id;
		if (relation.effectId === source.id) relation.effectId = target.id;
	}
	for (const episode of graph.episodes) {
		for (const grounding of episode.groundings) {
			if (grounding.conceptId === source.id) grounding.conceptId = target.id;
		}
		episode.groundings = [
			...new Map(
				episode.groundings.map((grounding) => [grounding.variable, grounding]),
			).values(),
		];
	}
	graph.concepts = graph.concepts.filter((concept) => concept.id !== source.id);
	deduplicateRelations(graph);
	return true;
}

function autoMergeConcepts(
	graph: CausalCognitionGraphData,
): Array<{ sourceConceptId: string; targetConceptId: string }> {
	const merged: Array<{
		sourceConceptId: string;
		targetConceptId: string;
	}> = [];
	let candidate = duplicateConceptPair(graph);
	while (candidate) {
		if (!mergeConcept(graph, candidate)) break;
		merged.push(candidate);
		candidate = duplicateConceptPair(graph);
	}
	return merged;
}

export function liftCausalBelief(
	inputGraph: CausalCognitionGraphData,
	belief: CausalBelief,
	input: LiftCausalBeliefInput = {},
): LiftCausalBeliefResult {
	const graph = normalizeGraph(structuredClone(inputGraph));
	const episodeId = input.episodeId ?? randomUUID();
	const createdConceptIds: string[] = [];
	const createdRelationIds: string[] = [];
	const updatedRelationIds: string[] = [];
	const warnings: string[] = [];
	const usedConceptIds = new Set(graph.concepts.map((concept) => concept.id));
	const beliefVariables = new Map(
		belief.variables.map((variable) => [variable.name, variable]),
	);
	// Resolve surface-form differences before checking the task-local relation.
	const resolveVariable = (name: string): string => {
		if (beliefVariables.has(name)) return name;
		const key = normalize(name);
		const names = [...beliefVariables.keys()];
		const exact = names.filter((candidate) => normalize(candidate) === key);
		if (exact.length === 1) return exact[0]!;
		const aliases = Object.entries(input.variableAliases ?? {})
			.filter(([alias]) => normalize(alias) === key)
			.map(([, target]) => target);
		if (aliases.length > 0) {
			const targets = names.filter((candidate) =>
				aliases.some((target) => normalize(target) === normalize(candidate)),
			);
			if (targets.length === 1) return targets[0]!;
			warnings.push(`Unresolved variable alias ${name}; choose one Belief variable: ${names.join(", ")}.`);
			return name;
		}
		const candidates = exact.length > 1 ? exact : names.filter((candidate) => {
			const other = normalize(candidate);
			const [short, long] = key.length <= other.length ? [key, other] : [other, key];
			if (short.length < 4 || short.length / long.length < 0.5) return false;
			// CJK has no word separators; Latin matches must respect word boundaries.
			return /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]/u.test(short)
				? long.includes(short)
				: ` ${long} `.includes(` ${short} `);
		});
		if (candidates.length === 1) return candidates[0]!;
		warnings.push(`Unresolved Lift variable ${name}; use variable_aliases to map it to one Belief variable: ${(candidates.length ? candidates : names).join(", ")}.`);
		return name;
	};
	const directedEdges = new Map(
		belief.edges
			.filter((edge) => edge.edgeType === "directed")
			.map((edge) => [`${edge.from}->${edge.to}`, edge] as const),
	);
	const relationContext = uniqueStrings(input.context ?? []);
	const invalidContexts = relationContext.filter(invalidLiftContext);
	if (invalidContexts.length > 0) {
		warnings.push(
			`Lift context must use stable domain:/scope: taxonomy tags without dates or task/run/cutoff/resolution markers; rejected: ${invalidContexts.join(", ")}.`,
		);
	}
	const requestedRelations: LiftRelationInput[] = input.relations
		? [...input.relations]
		: belief.edges
				.filter((edge) => edge.edgeType === "directed")
				.map((edge) => ({
					fromVariable: edge.from,
					toVariable: edge.to,
					verdict: "hypothesized" as const,
					confidence: edge.confidence,
					lag: "unknown",
					...(edge.rationale ? { mechanism: edge.rationale } : {}),
				}));
	const relationInputs = requestedRelations.map((relation) => ({
		...relation,
		fromVariable: resolveVariable(relation.fromVariable),
		toVariable: resolveVariable(relation.toVariable),
	})).filter((relation) => {
		if (
			!beliefVariables.has(relation.fromVariable) ||
			!beliefVariables.has(relation.toVariable) ||
			relation.fromVariable === relation.toVariable
		) {
			warnings.push(
				`Ignored invalid relation ${relation.fromVariable} -> ${relation.toVariable}.`,
			);
			return false;
		}
		if (invalidContexts.length > 0) return false;
		if (!isCanonicalRelationLag(relation.lag)) {
			warnings.push(
				`Ignored relation with missing or non-canonical lag (${relation.lag ?? "missing"}); use unknown|immediate|days|weeks|months|quarters|years: ${relation.fromVariable} -> ${relation.toVariable}.`,
			);
			return false;
		}
		const matchingEdge = directedEdges.get(
			`${relation.fromVariable}->${relation.toVariable}`,
		);
		if (relation.verdict !== "contradicted" && !matchingEdge) {
			warnings.push(
				`Ignored ${relation.verdict} relation absent from the final directed Belief: ${relation.fromVariable} -> ${relation.toVariable}.`,
			);
			return false;
		}
		if (
			(relation.verdict === "supported" ||
				relation.verdict === "contradicted") &&
			(!hasText(relation.mechanism) || !hasText(relation.note))
		) {
			warnings.push(
				`Ignored ${relation.verdict} relation without a non-empty mechanism and claim note: ${relation.fromVariable} -> ${relation.toVariable}.`,
			);
			return false;
		}
		if (
			relation.verdict === "supported" &&
			matchingEdge &&
			(!hasText(matchingEdge.rationale) || !hasText(matchingEdge.method))
		) {
			warnings.push(
				`Ignored supported relation because the matching final directed Belief edge lacks rationale or method: ${relation.fromVariable} -> ${relation.toVariable}.`,
			);
			return false;
		}
		return true;
	});
	const selectedVariableNames = new Set(
		relationInputs.flatMap((relation) => [
			relation.fromVariable,
			relation.toVariable,
		]),
	);
	const creatableVariableNames = new Set(
		relationInputs
			.filter((relation) => relation.verdict !== "contradicted")
			.flatMap((relation) => [relation.fromVariable, relation.toVariable]),
	);
	const suppliedByVariable = new Map<string, CausalGrounding[]>();
	const invalidSuppliedVariables = new Set<string>();
	for (const suppliedGrounding of input.groundings ?? []) {
		const grounding = { ...suppliedGrounding, variable: resolveVariable(suppliedGrounding.variable) };
		if (!selectedVariableNames.has(grounding.variable)) continue;
		if (!usedConceptIds.has(grounding.conceptId)) {
			invalidSuppliedVariables.add(grounding.variable);
			warnings.push(
				`Ignored grounding to unknown concept ${grounding.conceptId}.`,
			);
			continue;
		}
		const entries = suppliedByVariable.get(grounding.variable) ?? [];
		entries.push({
			...grounding,
			confidence: clamp01(grounding.confidence),
		});
		suppliedByVariable.set(grounding.variable, entries);
	}
	const groundings: CausalGrounding[] = [];
	const groundingByVariable = new Map<string, CausalGrounding>();
	for (const variableName of selectedVariableNames) {
		const variable = beliefVariables.get(variableName);
		if (!variable) continue;
		const supplied = [...(suppliedByVariable.get(variableName) ?? [])].sort(
			(left, right) =>
				right.confidence - left.confidence ||
				left.conceptId.localeCompare(right.conceptId),
		);
		let grounding: CausalGrounding | undefined;
		let groundingBlocked = false;
		if (supplied[0]) {
			const nearTie =
				supplied[1] && supplied[0].confidence - supplied[1].confidence < 0.05;
			if (
				nearTie ||
				supplied[0].kind === "ambiguous" ||
				supplied[0].confidence < 0.5
			) {
				groundingBlocked = true;
				warnings.push(
					`Ambiguous or low-confidence Lift grounding blocked for ${variableName}.`,
				);
			} else grounding = supplied[0];
		} else if (invalidSuppliedVariables.has(variableName)) {
			groundingBlocked = true;
		} else {
			const automatic = automaticConceptCandidate(graph, variable);
			if (automatic?.kind === "ambiguous") {
				groundingBlocked = true;
				warnings.push(
					`Ambiguous automatic Lift grounding blocked for ${variableName}: ${automatic.rationale ?? "near tie"}.`,
				);
			} else grounding = automatic;
		}
		if (
			!grounding &&
			!groundingBlocked &&
			creatableVariableNames.has(variableName)
		) {
			const conceptId = uniqueId(slug(variable.name), usedConceptIds);
			usedConceptIds.add(conceptId);
			graph.concepts.push({
				id: conceptId,
				label: variable.name,
				aliases: [],
				keywords: extractKeywords(`${variable.description} ${variable.source}`),
				description: variable.description,
			});
			grounding = {
				variable: variable.name,
				conceptId,
				kind: "hypothesis",
				confidence: 0.5,
				rationale: "Created from a selected task-local relation during Lift",
			};
			createdConceptIds.push(conceptId);
		}
		if (!grounding) continue;
		const concept = graph.concepts.find(
			(item) => item.id === grounding?.conceptId,
		);
		if (!concept) continue;
		groundings.push(grounding);
		groundingByVariable.set(variableName, grounding);
	}

	const successfulVariableNames = new Set<string>();
	for (const relationInput of relationInputs) {
		const causeId = groundingByVariable.get(
			relationInput.fromVariable,
		)?.conceptId;
		const effectId = groundingByVariable.get(
			relationInput.toVariable,
		)?.conceptId;
		if (!causeId || !effectId || causeId === effectId) {
			warnings.push(
				`Relation skipped because its endpoints were not grounded: ${relationInput.fromVariable} -> ${relationInput.toVariable}.`,
			);
			continue;
		}
		const lag = relationInput.lag?.trim();
		const mechanism = relationInput.mechanism?.trim();
		const note = relationInput.note?.trim();
		const key = relationKey({
			causeId,
			effectId,
			contexts: relationContext,
			lag,
		});
		const existing = graph.relations.find(
			(relation) => relationKey(relation) === key,
		);
		const claim: RelationClaim = {
			episodeId,
			verdict: relationInput.verdict,
			confidence: relationInputConfidence(
				relationInput,
				directedEdges.get(
					`${relationInput.fromVariable}->${relationInput.toVariable}`,
				),
				warnings,
			),
			...(note ? { note } : {}),
		};
		if (existing) {
			const prior = claimForEpisode(existing.claims, episodeId);
			if (prior) Object.assign(prior, claim);
			else existing.claims.push(claim);
			if (mechanism) {
				existing.mechanism = mechanism;
			}
			updatedRelationIds.push(existing.id);
			successfulVariableNames.add(relationInput.fromVariable);
			successfulVariableNames.add(relationInput.toVariable);
			continue;
		}
		if (relationInput.verdict === "contradicted") {
			warnings.push(
				`Contradiction did not match an existing contextual relation: ${causeId} -> ${effectId}.`,
			);
			continue;
		}
		const relationId = uniqueId(
			`${causeId}--to--${effectId}`,
			new Set(graph.relations.map((relation) => relation.id)),
		);
		graph.relations.push({
			id: relationId,
			causeId,
			effectId,
			...(mechanism ? { mechanism } : {}),
			contexts: relationContext,
			...(lag ? { lag } : {}),
			claims: [claim],
		});
		createdRelationIds.push(relationId);
		successfulVariableNames.add(relationInput.fromVariable);
		successfulVariableNames.add(relationInput.toVariable);
	}

	const unconnectedCreated = new Set(
		createdConceptIds.filter(
			(conceptId) =>
				!graph.relations.some(
					(relation) =>
						relation.causeId === conceptId || relation.effectId === conceptId,
				),
		),
	);
	if (unconnectedCreated.size > 0) {
		graph.concepts = graph.concepts.filter(
			(concept) => !unconnectedCreated.has(concept.id),
		);
		for (let index = groundings.length - 1; index >= 0; index -= 1) {
			if (unconnectedCreated.has(groundings[index]?.conceptId ?? "")) {
				groundings.splice(index, 1);
			}
		}
		warnings.push(
			`Discarded ${unconnectedCreated.size} unconnected concept candidate(s) after blocked Lift relations.`,
		);
	}
	for (const variableName of successfulVariableNames) {
		const variable = beliefVariables.get(variableName);
		const conceptId = groundingByVariable.get(variableName)?.conceptId;
		const concept = graph.concepts.find((item) => item.id === conceptId);
		if (variable && concept) addVariableVocabulary(concept, variable);
	}

	const episode: CausalCognitionEpisode = {
		id: episodeId,
		beliefId: belief.id,
		objective: belief.objective,
		context: relationContext,
		groundings: groundings.filter((grounding) =>
			successfulVariableNames.has(grounding.variable),
		),
		localGraph: {
			variables: belief.variables.map((variable) => variable.name),
			edges: structuredClone(belief.edges),
		},
		...(input.summary ? { summary: input.summary } : {}),
		createdAt: input.createdAt ?? new Date().toISOString(),
	};
	const hasSuccessfulClaim =
		createdRelationIds.length > 0 || updatedRelationIds.length > 0;
	if (hasSuccessfulClaim) {
		graph.episodes = graph.episodes.filter((item) => item.id !== episodeId);
		graph.episodes.push(episode);
	}
	const mergedConcepts = hasSuccessfulClaim ? autoMergeConcepts(graph) : [];

	return {
		graph,
		episode,
		createdConceptIds: createdConceptIds.filter((id) =>
			graph.concepts.some((concept) => concept.id === id),
		),
		createdRelationIds: uniqueStrings(createdRelationIds),
		updatedRelationIds: uniqueStrings(updatedRelationIds),
		mergedConcepts,
		warnings,
	};
}

export function maintainCausalCognition(
	inputGraph: CausalCognitionGraphData,
	input: MaintainCausalCognitionInput = {},
): MaintainCausalCognitionResult {
	const graph = normalizeGraph(structuredClone(inputGraph));
	const updatedConceptIds: string[] = [];
	const mergedConcepts: Array<{
		sourceConceptId: string;
		targetConceptId: string;
	}> = [];
	const removedRelationIds: string[] = [];
	const removedConceptIds: string[] = [];
	const warnings: string[] = [];
	let retractedClaimCount = 0;

	for (const update of input.conceptUpdates ?? []) {
		const concept = graph.concepts.find((item) => item.id === update.conceptId);
		if (!concept) {
			warnings.push(`Unknown concept update target: ${update.conceptId}`);
			continue;
		}
		if (update.label?.trim()) concept.label = update.label.trim();
		if (update.description !== undefined) {
			concept.description = update.description.trim() || undefined;
		}
		const removeAliases = new Set((update.removeAliases ?? []).map(normalize));
		const removeKeywords = new Set(
			(update.removeKeywords ?? []).map(normalize),
		);
		concept.aliases = uniqueStrings([
			...concept.aliases.filter((item) => !removeAliases.has(normalize(item))),
			...(update.addAliases ?? []),
		]).filter((alias) => normalize(alias) !== normalize(concept.label));
		concept.keywords = uniqueStrings([
			...(concept.keywords ?? []).filter(
				(item) => !removeKeywords.has(normalize(item)),
			),
			...(update.addKeywords ?? []),
		]);
		updatedConceptIds.push(concept.id);
	}

	for (const retraction of input.retractClaims ?? []) {
		const relation = graph.relations.find(
			(item) => item.id === retraction.relationId,
		);
		if (!relation) {
			warnings.push(
				`Unknown relation retraction target: ${retraction.relationId}`,
			);
			continue;
		}
		const before = relation.claims.length;
		relation.claims = relation.claims.filter(
			(claim) => claim.episodeId !== retraction.episodeId,
		);
		retractedClaimCount += before - relation.claims.length;
		if (before === relation.claims.length) {
			warnings.push(
				`No claim from episode ${retraction.episodeId} on relation ${relation.id}.`,
			);
		}
	}
	for (const relation of [...graph.relations]) {
		if (relation.claims.length > 0) continue;
		removedRelationIds.push(relation.id);
		graph.relations = graph.relations.filter((item) => item.id !== relation.id);
	}

	for (const merge of input.merges ?? []) {
		if (mergeConcept(graph, merge)) mergedConcepts.push(merge);
		else {
			warnings.push(
				`Invalid concept merge: ${merge.sourceConceptId} -> ${merge.targetConceptId}`,
			);
		}
	}
	if (input.autoMerge !== false) {
		mergedConcepts.push(...autoMergeConcepts(graph));
	}

	for (const conceptId of uniqueStrings(input.removeConceptIds ?? [])) {
		const concept = graph.concepts.find((item) => item.id === conceptId);
		if (!concept) {
			warnings.push(`Unknown concept removal target: ${conceptId}`);
			continue;
		}
		const incident = graph.relations.some(
			(relation) =>
				relation.causeId === conceptId || relation.effectId === conceptId,
		);
		if (incident) {
			warnings.push(
				`Concept ${conceptId} still has incident relations; merge or retract them first.`,
			);
			continue;
		}
		graph.concepts = graph.concepts.filter((item) => item.id !== conceptId);
		removedConceptIds.push(conceptId);
	}

	return {
		graph,
		updatedConceptIds: uniqueStrings(updatedConceptIds),
		mergedConcepts,
		retractedClaimCount,
		removedRelationIds: uniqueStrings(removedRelationIds),
		removedConceptIds,
		warnings,
	};
}

export function renderCausalCognitionIndex(
	inputGraph: CausalCognitionGraphData,
): string {
	const graph = normalizeGraph(inputGraph);
	const lines = [
		"# Causal Cognition Index",
		"",
		"> Search vocabulary for the causal concept graph. Runtime search uses the JSON graph; this Markdown file is a human-readable map.",
		"",
		"| Concept | Aliases | Keywords | Active incoming | Active outgoing |",
		"|---|---|---|---:|---:|",
	];
	for (const concept of [...graph.concepts].sort((a, b) =>
		a.label.localeCompare(b.label),
	)) {
		const degree = activeDegree(graph, concept.id);
		lines.push(
			`| ${concept.label} (\`${concept.id}\`) | ${concept.aliases.join(", ") || "—"} | ${(concept.keywords ?? []).join(", ") || "—"} | ${degree.incoming} | ${degree.outgoing} |`,
		);
	}
	lines.push(
		"",
		`Episodes: ${graph.episodes.length}`,
		`Relation variants: ${graph.relations.length}`,
	);
	const experiences = graph.episodes.filter((episode) => episode.experience);
	if (experiences.length) lines.push("", "## Experience memory", "");
	for (const episode of experiences) {
		lines.push(
			`### ${episode.experience!.title} (\`${episode.id}\`)`,
			`Keywords: ${episode.experience!.keywords.join(", ")}; context: ${episode.context.join(", ")}`,
			episode.summary ?? "",
			`Evidence: ${episode.experience!.evidence}`,
			"",
		);
	}
	return lines.join("\n");
}

export class CausalCognitionGraphStore {
	readonly graphPath: string;
	readonly indexPath: string;

	constructor(workspaceDir = process.cwd()) {
		this.graphPath = join(workspaceDir, ".cauprophet", "causal_cognition.json");
		this.indexPath = join(
			workspaceDir,
			".cauprophet",
			"causal_cognition_index.md",
		);
	}

	load(): CausalCognitionGraphData {
		try {
			const parsed = JSON.parse(
				readFileSync(this.graphPath, "utf8"),
			) as CausalCognitionGraphData;
			if (
				![1, 2, 3, CAUSAL_COGNITION_GRAPH_SCHEMA_VERSION].includes(
					parsed.schemaVersion,
				) ||
				!Array.isArray(parsed.concepts) ||
				!Array.isArray(parsed.relations) ||
				!Array.isArray(parsed.episodes)
			) {
				throw new Error("Unsupported Causal Cognition graph schema");
			}
			return normalizeGraph(parsed);
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code === "ENOENT") {
				return EMPTY_GRAPH();
			}
			throw error;
		}
	}

	save(graph: CausalCognitionGraphData): void {
		withFileLockSync(this.graphPath, () => this.saveUnlocked(graph));
	}

	private saveUnlocked(graph: CausalCognitionGraphData): void {
		const normalized = normalizeGraph(graph);
		mkdirSync(dirname(this.graphPath), { recursive: true });
		atomicWriteFileSync(this.graphPath, `${JSON.stringify(normalized, null, 2)}\n`);
		atomicWriteFileSync(this.indexPath, `${renderCausalCognitionIndex(normalized)}\n`);
	}

	search(
		options: CausalCognitionSearchOptions = {},
	): CausalCognitionSearchResult {
		return searchCausalCognition(this.load(), options);
	}

	lift(
		belief: CausalBelief,
		input: LiftCausalBeliefInput = {},
	): LiftCausalBeliefResult {
		return withFileLockSync(this.graphPath, () => {
			const result = liftCausalBelief(this.load(), belief, input);
			this.saveUnlocked(result.graph);
			return result;
		});
	}

	maintain(
		input: MaintainCausalCognitionInput = {},
	): MaintainCausalCognitionResult {
		return withFileLockSync(this.graphPath, () => {
			const result = maintainCausalCognition(this.load(), input);
			this.saveUnlocked(result.graph);
			return result;
		});
	}
}
