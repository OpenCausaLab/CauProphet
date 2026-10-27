/** LLM-facing search/update bridge for the persistent causal concept graph. */

import { Buffer } from "node:buffer";
import { CognitionNoteStore } from "@/memory/cognition_notes";

import {
	CAUSAL_RELATION_LAGS,
	CausalCognitionGraphStore,
	type CausalConceptMergeInput,
	type CausalConceptUpdateInput,
	type CausalCognitionGraphData,
	type CausalCognitionSearchResult,
	type CausalGrounding,
	type CausalRelation,
	type GroundingKind,
	type LiftRelationInput,
	type RelationClaimRetractionInput,
	liftCausalBelief,
	rememberCognitionExperience,
	maintainCausalCognition,
} from "@/memory/causal_cognition_graph";
import { withFileLockSync } from "@/core/resource_lock";
import { loadActiveBelief } from "@/tools/causal_belief_tools";

export const searchCausalCognitionTool = {
	type: "function" as const,
	function: {
		name: "search_causal_cognition",
		description:
			"Search persistent causal knowledge and current mechanism notes. Prefer consulting relevant stored knowledge when it can improve current reasoning. " +
			"Search long-term Cognition experience and the causal concept index, or expand one graph frontier from returned concept IDs. " +
			"Choose keywords freely from the task, tools, API behavior, action dependencies, mechanisms, or suspected missing causes. Retrieve experience whenever it could improve a plan, avoid a repeated error, or guide validation.",
		parameters: {
			type: "object",
			properties: {
				keywords: {
					type: "array",
					items: { type: "string" },
					description:
						"Up to 8 free index terms. Searches experience titles/summaries/evidence, labels, aliases, keywords, descriptions, mechanisms, and contexts.",
				},
				temporal_context: {type: "string", description: "Optional natural-language horizon and applicability conditions for written mechanism notes."},
				as_of: {type: "string", description: "For written notes, return the latest version available at or before this ISO timestamp with timezone; defaults to now."},
				concept_ids: {
					type: "array",
					items: { type: "string" },
					description:
						"Exact concept IDs from a previous index/frontier result. Their active incident edges are expanded by one step.",
				},
				known_concept_ids: {
					type: "array",
					items: { type: "string" },
					description:
						"Concept IDs already inspected in this reasoning chain; suppress them from the next frontier.",
				},
				direction: {
					type: "string",
					enum: ["upstream", "downstream", "both"],
				},
				context: { type: "array", items: { type: "string" } },
				concept_limit: { type: "integer", minimum: 1, maximum: 12 },
				relation_limit: { type: "integer", minimum: 1, maximum: 20 },
				byte_limit: {
					type: "integer",
					minimum: 1000,
					maximum: 6400,
					description: "Maximum UTF-8 bytes returned; default 3200.",
				},
			},
			additionalProperties: false,
		},
	},
};

export const updateCausalCognitionTool = {
	type: "function" as const,
	function: {
		name: "update_causal_cognition",
		description:
			"Save useful long-term experience (procedures, API semantics, preconditions, side effects, error recovery, or bounded observations) through experience, even without a causal graph. " +
			"For causal relation claims, Lift selected final directed Causal Belief relations with their evidence and confidence. Also maintain concept aliases/keywords, merge duplicates, retract claims, and remove isolated concepts. " +
			"Use direct structured arguments; do not encode another JSON object inside a string.",
		parameters: {
			type: "object",
			properties: {
				experience: {
					type: "object",
					description: "One useful long-term lesson. State applicability and uncertainty in summary; cite observed actions/results or tool documentation in evidence. A single episode can qualify with its scope made explicit. Saved independently of Belief edges and causal Lift.",
					properties: {
						title: { type: "string" },
						summary: { type: "string" },
						keywords: { type: "array", items: { type: "string" } },
						context: { type: "array", items: { type: "string" }, description: "Application, environment/version, or task scope where this lesson applies." },
						evidence: { type: "string" },
					},
					required: ["title", "summary", "keywords", "context", "evidence"],
					additionalProperties: false,
				},
				variable_aliases: {
					type: "object",
					additionalProperties: { type: "string" },
					description: "Optional mapping from a paraphrased or translated endpoint to a current Belief variable name. Case, punctuation and unique shortened names are resolved automatically. Variable description and source may be empty.",
				},
				episode_id: { type: "string" },
				context: {
					type: "array",
					items: { type: "string" },
					description:
						"Optional stable taxonomy tags only, each beginning domain: or scope:. Never put dates, years, months, task/run IDs, cutoffs, or resolution markers here; keep time in the episode objective/createdAt.",
				},
				summary: { type: "string" },
				lift_all_directed: {
					type: "boolean",
					description:
						"Explicitly opt in to lifting every final directed Belief edge as hypothesized. Prefer a selected relations list.",
				},
				groundings: {
					type: "array",
					items: {
						type: "object",
						properties: {
							variable: { type: "string" },
							concept_id: { type: "string" },
							kind: {
								type: "string",
								enum: ["exact", "alias", "proxy", "hypothesis", "ambiguous"],
							},
							confidence: { type: "number", minimum: 0, maximum: 1 },
							rationale: { type: "string" },
						},
						required: ["variable", "concept_id", "kind", "confidence"],
					},
				},
				relations: {
					type: "array",
					items: {
						type: "object",
						properties: {
							from_variable: { type: "string" },
							to_variable: { type: "string" },
							verdict: {
								type: "string",
								enum: ["hypothesized", "supported", "contradicted"],
							},
							confidence: {
								type: "string",
								enum: ["low", "medium", "high"],
							},
							mechanism: {
								type: "string",
								description:
									"Required and non-empty for supported or contradicted claims.",
							},
							lag: {
								type: "string",
								enum: [...CAUSAL_RELATION_LAGS],
								description:
									"Required canonical temporal scale. Use unknown when the task cannot resolve it.",
							},
							note: {
								type: "string",
								description:
									"Episode-specific evidence note; required and non-empty for supported or contradicted claims.",
							},
						},
						required: ["from_variable", "to_variable", "verdict", "lag"],
					},
				},
				concept_updates: {
					type: "array",
					items: {
						type: "object",
						properties: {
							concept_id: { type: "string" },
							label: { type: "string" },
							description: { type: "string" },
							add_aliases: { type: "array", items: { type: "string" } },
							remove_aliases: { type: "array", items: { type: "string" } },
							add_keywords: { type: "array", items: { type: "string" } },
							remove_keywords: { type: "array", items: { type: "string" } },
						},
						required: ["concept_id"],
					},
				},
				merge_concepts: {
					type: "array",
					items: {
						type: "object",
						properties: {
							source_concept_id: { type: "string" },
							target_concept_id: { type: "string" },
						},
						required: ["source_concept_id", "target_concept_id"],
					},
				},
				retract_claims: {
					type: "array",
					items: {
						type: "object",
						properties: {
							relation_id: { type: "string" },
							episode_id: { type: "string" },
						},
						required: ["relation_id", "episode_id"],
					},
				},
				remove_concept_ids: { type: "array", items: { type: "string" } },
				auto_merge: { type: "boolean" },
			},
			additionalProperties: false,
		},
	},
};

function parseJsonObject(
	value: string,
	label: string,
): Record<string, unknown> {
	let parsed: unknown;
	try {
		parsed = JSON.parse(value);
	} catch (error) {
		throw new Error(`${label} must be valid JSON: ${String(error)}`);
	}
	if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
		throw new Error(`${label} must be a JSON object`);
	}
	return parsed as Record<string, unknown>;
}

function inputObject(
	value: unknown,
	legacyKey?: string,
): Record<string, unknown> {
	if (typeof value === "string")
		return parseJsonObject(value, legacyKey ?? "input");
	if (!value || typeof value !== "object" || Array.isArray(value)) {
		throw new Error("Tool input must be an object");
	}
	const object = value as Record<string, unknown>;
	if (legacyKey && typeof object[legacyKey] === "string") {
		return parseJsonObject(object[legacyKey] as string, legacyKey);
	}
	return object;
}

function stringArray(value: unknown): string[] {
	return Array.isArray(value)
		? value.map((item) => String(item).trim()).filter(Boolean)
		: [];
}

function optionalString(value: unknown): string | undefined {
	return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function objectArray(value: unknown): Record<string, unknown>[] {
	return Array.isArray(value)
		? value.filter(
				(item): item is Record<string, unknown> =>
					Boolean(item) && typeof item === "object" && !Array.isArray(item),
			)
		: [];
}

const GROUNDING_KINDS = new Set<GroundingKind>([
	"exact",
	"alias",
	"proxy",
	"hypothesis",
	"ambiguous",
]);

function parseGroundings(value: unknown): CausalGrounding[] {
	const groundings: CausalGrounding[] = [];
	for (const object of objectArray(value)) {
		const variable = optionalString(object["variable"]);
		const conceptId = optionalString(
			object["concept_id"] ?? object["conceptId"],
		);
		const rawKind = optionalString(object["kind"]);
		const confidence = Number(object["confidence"] ?? 0.5);
		if (
			!variable ||
			!conceptId ||
			!rawKind ||
			!GROUNDING_KINDS.has(rawKind as GroundingKind)
		) {
			continue;
		}
		groundings.push({
			variable,
			conceptId,
			kind: rawKind as GroundingKind,
			confidence: Number.isFinite(confidence) ? confidence : 0.5,
			...(optionalString(object["rationale"])
				? { rationale: optionalString(object["rationale"]) }
				: {}),
		});
	}
	return groundings;
}

function parseRelations(value: unknown): LiftRelationInput[] {
	const relations: LiftRelationInput[] = [];
	for (const object of objectArray(value)) {
		const fromVariable = optionalString(
			object["from_variable"] ?? object["fromVariable"],
		);
		const toVariable = optionalString(
			object["to_variable"] ?? object["toVariable"],
		);
		const verdict = optionalString(object["verdict"]);
		if (
			!fromVariable ||
			!toVariable ||
			!verdict ||
			!["hypothesized", "supported", "contradicted"].includes(verdict)
		) {
			continue;
		}
		const confidence = optionalString(object["confidence"]);
		relations.push({
			fromVariable,
			toVariable,
			verdict: verdict as LiftRelationInput["verdict"],
			...(confidence && ["low", "medium", "high"].includes(confidence)
				? { confidence: confidence as LiftRelationInput["confidence"] }
				: {}),
			...(optionalString(object["mechanism"])
				? { mechanism: optionalString(object["mechanism"]) }
				: {}),
			...(optionalString(object["lag"])
				? { lag: optionalString(object["lag"]) }
				: {}),
			...(optionalString(object["note"])
				? { note: optionalString(object["note"]) }
				: {}),
		});
	}
	return relations;
}

function parseConceptUpdates(value: unknown): CausalConceptUpdateInput[] {
	return objectArray(value)
		.map((object): CausalConceptUpdateInput | undefined => {
			const conceptId = optionalString(
				object["concept_id"] ?? object["conceptId"],
			);
			if (!conceptId) return undefined;
			return {
				conceptId,
				...(optionalString(object["label"])
					? { label: optionalString(object["label"]) }
					: {}),
				...(typeof object["description"] === "string"
					? { description: object["description"] as string }
					: {}),
				addAliases: stringArray(object["add_aliases"] ?? object["addAliases"]),
				removeAliases: stringArray(
					object["remove_aliases"] ?? object["removeAliases"],
				),
				addKeywords: stringArray(
					object["add_keywords"] ?? object["addKeywords"],
				),
				removeKeywords: stringArray(
					object["remove_keywords"] ?? object["removeKeywords"],
				),
			};
		})
		.filter((item): item is CausalConceptUpdateInput => item !== undefined);
}

function parseMerges(value: unknown): CausalConceptMergeInput[] {
	return objectArray(value)
		.map((object): CausalConceptMergeInput | undefined => {
			const sourceConceptId = optionalString(
				object["source_concept_id"] ?? object["sourceConceptId"],
			);
			const targetConceptId = optionalString(
				object["target_concept_id"] ?? object["targetConceptId"],
			);
			return sourceConceptId && targetConceptId
				? { sourceConceptId, targetConceptId }
				: undefined;
		})
		.filter((item): item is CausalConceptMergeInput => item !== undefined);
}

function parseRetractions(value: unknown): RelationClaimRetractionInput[] {
	return objectArray(value)
		.map((object): RelationClaimRetractionInput | undefined => {
			const relationId = optionalString(
				object["relation_id"] ?? object["relationId"],
			);
			const episodeId = optionalString(
				object["episode_id"] ?? object["episodeId"],
			);
			return relationId && episodeId ? { relationId, episodeId } : undefined;
		})
		.filter((item): item is RelationClaimRetractionInput => item !== undefined);
}

function conceptLabels(graph: CausalCognitionGraphData): Map<string, string> {
	return new Map(graph.concepts.map((concept) => [concept.id, concept.label]));
}

function claimSummary(relation: CausalRelation): string {
	return relation.claims
		.map(
			(claim) =>
				`${claim.verdict[0]?.toUpperCase() ?? "?"}:${claim.confidence ?? "high"}@${claim.episodeId}`,
		)
		.join(",");
}

function appendWithinBudget(
	lines: string[],
	block: string[],
	byteLimit: number,
): boolean {
	const candidate = [...lines, ...block].join("\n");
	if (Buffer.byteLength(candidate, "utf8") > byteLimit) return false;
	lines.push(...block);
	return true;
}

function utf8Excerpt(value: string, byteLimit: number): string {
	if (Buffer.byteLength(value, "utf8") <= byteLimit) return value;
	let output = "";
	let bytes = 0;
	for (const char of value) {
		bytes += Buffer.byteLength(char, "utf8");
		if (bytes > byteLimit - 3) break;
		output += char;
	}
	return output + "…";
}

function renderSearchResult(
	result: CausalCognitionSearchResult,
	graph: CausalCognitionGraphData,
	byteLimit: number,
): string {
	const labels = conceptLabels(graph);
	const lines = [
		"# Causal Cognition index / frontier",
		"",
		`- Query keywords: ${result.queryKeywords.join(", ") || "none"}`,
		`- Expanded seeds: ${result.seedConceptIds.join(", ") || "none"}`,
		`- Visited active incident relations: ${result.visitedRelationCount}`,
		"",
	];
	let byteTruncated = false;
	if (result.experienceMatches.length > 0) {
		appendWithinBudget(lines, ["## Experience memory (check applicability in this task)", ""], byteLimit);
		for (const { episode, score } of result.experienceMatches) {
			const block = [
				`- **${episode.experience!.title}** (\`${episode.id}\`) score=${score.toFixed(3)}; context=${episode.context.join(", ")}; recorded=${episode.createdAt}`,
				`  Lesson: ${episode.summary}`,
				`  Evidence: ${episode.experience!.evidence}`,
			];
			if (!appendWithinBudget(lines, block, byteLimit)) {
				byteTruncated = true;
				const available = byteLimit - Buffer.byteLength(lines.join("\n"), "utf8") - 120;
				if (available >= 300) {
					appendWithinBudget(lines, [
						utf8Excerpt(block[0]!, Math.floor(available * 0.3)),
						"  Lesson: " + utf8Excerpt(episode.summary ?? "", Math.floor(available * 0.4)),
						"  Evidence: " + utf8Excerpt(episode.experience!.evidence, Math.floor(available * 0.2)),
					], byteLimit);
				}
				break;
			}
		}
	}
	if (result.indexMatches.length > 0) {
		appendWithinBudget(lines, ["## Index matches", ""], byteLimit);
		for (const match of result.indexMatches) {
			const ok = appendWithinBudget(
				lines,
				[
					`- **${match.label}** (\`${match.conceptId}\`) score=${match.score.toFixed(3)}; aliases=[${match.aliases.join(", ")}]; keywords=[${match.keywords.join(", ")}]; active degree=${match.activeIncoming} in/${match.activeOutgoing} out; matched=${match.matchedBy.join(" | ")}`,
				],
				byteLimit,
			);
			if (!ok) {
				byteTruncated = true;
				break;
			}
		}
		appendWithinBudget(lines, [""], byteLimit);
	}
	if (result.relations.length > 0) {
		appendWithinBudget(lines, ["## One-step causal frontier", ""], byteLimit);
		for (const match of result.relations) {
			const relation = match.relation;
			const ok = appendWithinBudget(
				lines,
				[
					`- \`${relation.id}\`: **${labels.get(relation.causeId) ?? relation.causeId}** --> **${labels.get(relation.effectId) ?? relation.effectId}**; posterior=${match.assessment.posterior.toFixed(3)} log-odds=${match.assessment.logOdds.toFixed(2)} active=${match.assessment.active}; prior=${match.assessment.priorLogOdds.toFixed(2)} support-log-BF=${match.assessment.supportLogBayesFactor.toFixed(2)} contradiction-log-BF=${match.assessment.contradictionLogBayesFactor.toFixed(2)}; context=${relation.contexts.join(",") || "unspecified"}; context-match=${match.contextSimilarity.toFixed(2)}; lag=${relation.lag ?? "unspecified"}; claims=[${claimSummary(relation)}]${relation.mechanism ? `; mechanism=${relation.mechanism}` : ""}`,
				],
				byteLimit,
			);
			if (!ok) {
				byteTruncated = true;
				break;
			}
		}
		appendWithinBudget(lines, [""], byteLimit);
	}
	if (result.frontier.length > 0) {
		appendWithinBudget(
			lines,
			["## Frontier for optional follow-up", ""],
			byteLimit,
		);
		for (const concept of result.frontier) {
			const ok = appendWithinBudget(
				lines,
				[
					`- **${concept.label}** (\`${concept.conceptId}\`); aliases=[${concept.aliases.join(", ")}]; keywords=[${concept.keywords.join(", ")}]; via=[${concept.viaRelationIds.join(", ")}]`,
				],
				byteLimit,
			);
			if (!ok) {
				byteTruncated = true;
				break;
			}
		}
	}
	for (const warning of result.warnings) {
		appendWithinBudget(lines, [`- Warning: ${warning}`], byteLimit);
	}
	if (result.truncated || byteTruncated) {
		appendWithinBudget(
			lines,
			["", "- Result truncated by the explicit relation/byte budget."],
			byteLimit,
		);
	}
	return lines.join("\n");
}

function renderWrittenNotes(matches: ReturnType<CognitionNoteStore["search"]>, byteLimit: number): string {
	const lines = ["# Written Causal Cognition (check applicability)", ""];
	let truncated = false;
	for (const {record} of matches) {
		const header = utf8Excerpt(`- \`${record.id}\` v${record.version}: **${utf8Excerpt(record.title, 120)}** (recorded=${record.available_at})`, Math.min(300, Math.floor(byteLimit / 3)));
		const remaining = byteLimit - Buffer.byteLength(lines.join("\n") + header, "utf8") - 80;
		if (remaining < 64) { truncated = true; break; }
		const note = utf8Excerpt(record.note, remaining);
		if (!appendWithinBudget(lines, [header, "  Note: " + note], byteLimit)) { truncated = true; break; }
		if (note !== record.note) truncated = true;
	}
	if (truncated) appendWithinBudget(lines, ["- Written notes truncated by the byte budget."], byteLimit);
	return lines.join("\n");
}

export async function searchCausalCognitionMemory(
	input: unknown,
): Promise<string> {
	try {
		const raw = inputObject(input, "request");
		const direction = optionalString(raw["direction"]);
		const store = new CausalCognitionGraphStore();
		const graph = store.load();
		const byteLimitRaw = Number(raw["byte_limit"] ?? raw["byteLimit"] ?? 3200);
		const byteLimit = Number.isFinite(byteLimitRaw)
			? Math.max(1000, Math.min(6400, Math.round(byteLimitRaw)))
			: 3200;
		const result = store.search({
			keywords: stringArray(raw["keywords"]),
			conceptIds: stringArray(raw["concept_ids"] ?? raw["conceptIds"]),
			knownConceptIds: stringArray(
				raw["known_concept_ids"] ?? raw["knownConceptIds"],
			),
			direction: ["upstream", "downstream", "both"].includes(direction ?? "")
				? (direction as "upstream" | "downstream" | "both")
				: "both",
			context: stringArray(raw["context"]),
			conceptLimit: Number(raw["concept_limit"] ?? raw["conceptLimit"] ?? 6),
			relationLimit: Number(raw["relation_limit"] ?? raw["relationLimit"] ?? 8),
		});
		const notes = new CognitionNoteStore().search(stringArray(raw["keywords"]),
			optionalString(raw["temporal_context"]) ?? "", optionalString(raw["as_of"]));
		if (notes.length === 0) return renderSearchResult(result, graph, byteLimit);
		const written = renderWrittenNotes(notes, Math.floor(byteLimit / 2));
		const separator = "\n\n";
		const graphBudget = byteLimit - Buffer.byteLength(written + separator, "utf8");
		return utf8Excerpt(written + separator + renderSearchResult(result, graph, graphBudget), byteLimit);
	} catch (error) {
		return `Error: Causal Cognition search failed: ${String(error)}`;
	}
}

export async function updateCausalCognition(input: unknown): Promise<string> {
	try {
		const raw = inputObject(input, "content");
		const store = new CausalCognitionGraphStore();
		return withFileLockSync(store.graphPath, () => {
		let graph = store.load();
		const relations = parseRelations(raw["relations"]);
		const liftAll =
			raw["lift_all_directed"] === true || raw["liftAllDirected"] === true;
		const wantsLift = Array.isArray(raw["relations"]) || liftAll;
		let liftSummary: string[] = [];
		let experienceSummary: string[] = [];
		const warnings: string[] = [];
		if (wantsLift) {
			const belief = loadActiveBelief();
			if (!belief) {
				return "Error: No active Causal Belief is available for Lift.";
			}
			if (Array.isArray(raw["relations"]) && relations.length === 0) {
				return "Error: relations was supplied but no valid relation claim was parsed.";
			}
			const lifted = liftCausalBelief(graph, belief, {
				...(optionalString(raw["episode_id"] ?? raw["episodeId"])
					? { episodeId: optionalString(raw["episode_id"] ?? raw["episodeId"]) }
					: {}),
				variableAliases: raw["variable_aliases"] && typeof raw["variable_aliases"] === "object" && !Array.isArray(raw["variable_aliases"])
					? Object.fromEntries(Object.entries(raw["variable_aliases"]).filter((entry): entry is [string, string] => typeof entry[1] === "string"))
					: {},
				groundings: parseGroundings(raw["groundings"]),
				context: stringArray(raw["context"]),
				...(liftAll && !Array.isArray(raw["relations"]) ? {} : { relations }),
				...(optionalString(raw["summary"])
					? { summary: optionalString(raw["summary"]) }
					: {}),
			});
			graph = lifted.graph;
			warnings.push(...lifted.warnings);
			if (
				lifted.createdRelationIds.length === 0 &&
				lifted.updatedRelationIds.length === 0
			) {
				return [
					"Error: Causal Cognition Lift was rejected; no graph changes were saved.",
					...warnings.map((warning) => `- Warning: ${warning}`),
				].join("\n");
			}
			liftSummary = [
				`- Episode: \`${lifted.episode.id}\``,
				`- Lift created ${lifted.createdConceptIds.length} concepts and ${lifted.createdRelationIds.length} relation variants; updated ${lifted.updatedRelationIds.length}; auto-merged ${lifted.mergedConcepts.length} duplicate concepts.`,
			];
		}
		if (raw["experience"] !== undefined) {
			const value = raw["experience"];
			if (!value || typeof value !== "object" || Array.isArray(value)) {
				return "Error: experience must be a structured object.";
			}
			const experience = value as Record<string, unknown>;
			const remembered = rememberCognitionExperience(graph, {
				title: optionalString(experience["title"]) ?? "",
				summary: optionalString(experience["summary"]) ?? "",
				keywords: stringArray(experience["keywords"]),
				context: stringArray(experience["context"]),
				evidence: optionalString(experience["evidence"]) ?? "",
			}, loadActiveBelief());
			graph = remembered.graph;
			experienceSummary = [`- Experience saved: \`${remembered.episode.id}\``];
		}
		const maintenanceInput = {
			conceptUpdates: parseConceptUpdates(
				raw["concept_updates"] ?? raw["conceptUpdates"],
			),
			merges: parseMerges(raw["merge_concepts"] ?? raw["mergeConcepts"]),
			retractClaims: parseRetractions(
				raw["retract_claims"] ?? raw["retractClaims"],
			),
			removeConceptIds: stringArray(
				raw["remove_concept_ids"] ?? raw["removeConceptIds"],
			),
			autoMerge: raw["auto_merge"] !== false && raw["autoMerge"] !== false,
		};
		const hasMaintenance =
			maintenanceInput.conceptUpdates.length > 0 ||
			maintenanceInput.merges.length > 0 ||
			maintenanceInput.retractClaims.length > 0 ||
			maintenanceInput.removeConceptIds.length > 0 ||
			wantsLift || experienceSummary.length > 0;
		if (!hasMaintenance) {
			return "Error: Supply experience, selected Lift relations, or a cognition maintenance operation.";
		}
		const maintained = maintainCausalCognition(graph, maintenanceInput);
		warnings.push(...maintained.warnings);
		store.save(maintained.graph);
		return [
			"✓ Updated Causal Cognition.",
			...liftSummary,
			...experienceSummary,
			`- Concept metadata updated: ${maintained.updatedConceptIds.length}`,
			`- Concepts merged: ${maintained.mergedConcepts.length}`,
			`- Claims retracted: ${maintained.retractedClaimCount}`,
			`- Empty relations removed: ${maintained.removedRelationIds.length}`,
			`- Isolated concepts removed: ${maintained.removedConceptIds.length}`,
			`- Graph size: ${maintained.graph.concepts.length} concepts, ${maintained.graph.relations.length} relation variants, ${maintained.graph.episodes.length} episodes`,
			...warnings.map((warning) => `- Warning: ${warning}`),
		].join("\n");
		});
	} catch (error) {
		return `Error: Causal Cognition update failed: ${String(error)}`;
	}
}
