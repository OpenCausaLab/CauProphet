import { z } from "zod";
import { sanitizeEvidenceText } from "./evidence";
import { findTrustedSource, normalizeSourceDomain } from "./trusted_sources";
import type {
	GeneralResearchTaskProfile,
	ResearchFreshness,
	ResearchQuery,
	ResearchRisk,
	RetrievalStage,
	SourceRoute,
} from "./types";

const MAX_SOURCES = 5;
const MAX_QUERIES = 6;

function stringArrayValue(value: unknown): unknown {
	if (typeof value !== "string") return value;
	return value.trim() ? [value] : [];
}

function compactSourcesValue(value: unknown): unknown {
	if (typeof value !== "string") return value;
	return value
		.split(/[,，\s]+/)
		.map((item) => item.trim())
		.filter(Boolean);
}

const OptionalPlannerStringsSchema = z.preprocess(
	stringArrayValue,
	z.array(z.string().min(1).max(500)).max(12),
);
const RequiredPlannerStringsSchema = z.preprocess(
	stringArrayValue,
	z.array(z.string().min(1).max(500)).min(1).max(12),
);
const StopConditionsSchema = z.preprocess(
	stringArrayValue,
	z.array(z.string().min(1).max(500)).min(1).max(10),
);

const PlannedSourceSchema = z.object({
	name: z.string().min(1).max(160),
	domain: z.string().max(253),
	authorityClaim: z.enum(["primary", "specialist", "open-web"]),
	access: z.enum([
		"direct-api",
		"direct-page",
		"document",
		"browser",
		"login",
		"web-search",
	]),
	requiresAuth: z.union([z.boolean(), z.literal("unknown")]),
	purpose: z.string().min(1).max(500),
	verification: z.string().min(1).max(500),
});

const PlannedQuerySchema = z.object({
	stage: z.enum([
		"primary-evidence",
		"current-status",
		"background",
		"cross-check",
		"counterevidence",
	]),
	query: z.string().min(1).max(500),
	rationale: z.string().min(1).max(500),
});

export const ResearchRequestSchema = z.object({
	question: z.string().min(1).max(4_000),
	intent: z.string().min(1).max(500),
	freshness: z.enum(["live", "recent", "stable", "historical"]),
	risk: z.enum(["standard", "high-stakes"]),
	sources: z.array(PlannedSourceSchema).min(1).max(MAX_SOURCES),
	constraints: OptionalPlannerStringsSchema,
	unresolvedFacts: RequiredPlannerStringsSchema,
	queries: z.array(PlannedQuerySchema).min(1).max(MAX_QUERIES),
	stopConditions: StopConditionsSchema,
});

const CompactSourceSchema = z.union([
	z.string().min(1).max(253),
	z.object({
		domain: z.string().min(1).max(253),
		name: z.string().min(1).max(160).optional(),
		purpose: z.string().min(1).max(500).optional(),
	}),
]);

const CompactResearchRequestSchema = z.object({
	question: z.string().min(1).max(4_000),
	intent: z.string().min(1).max(500).optional(),
	freshness: z.enum(["live", "recent", "stable", "historical"]).optional(),
	risk: z.enum(["standard", "high-stakes"]).optional(),
	sources: z.preprocess(
		compactSourcesValue,
		z.array(CompactSourceSchema).min(1).max(MAX_SOURCES),
	),
	constraints: OptionalPlannerStringsSchema.optional().default([]),
	unresolvedFacts: RequiredPlannerStringsSchema.optional(),
	stopConditions: StopConditionsSchema.optional(),
});

export type ResearchRequest = z.infer<typeof ResearchRequestSchema>;

export interface ValidatedResearchPlan {
	profile: GeneralResearchTaskProfile;
	sources: SourceRoute[];
	queries: ResearchQuery[];
	routing: {
		mode: "llm" | "fallback";
		model?: string;
		warnings: string[];
		rejectedSources: string[];
	};
}

export type ResearchPlanningFunction = (
	input: string,
	now: Date,
) => Promise<ValidatedResearchPlan>;

const STOPWORDS = new Set([
	"a",
	"an",
	"and",
	"are",
	"as",
	"at",
	"be",
	"by",
	"for",
	"from",
	"how",
	"in",
	"is",
	"it",
	"of",
	"on",
	"or",
	"the",
	"to",
	"what",
	"when",
	"which",
	"who",
	"will",
	"with",
]);

function normalizeQuestion(question: string): string {
	return question.replace(/\s+/g, " ").trim();
}

function extractKeywords(question: string): string[] {
	const unique: string[] = [];
	const add = (token: string): void => {
		const normalized = token.toLowerCase();
		if (
			!normalized ||
			STOPWORDS.has(normalized) ||
			unique.includes(normalized) ||
			unique.length >= 24
		)
			return;
		unique.push(normalized);
	};
	for (const token of question.match(/[a-z0-9][a-z0-9._%-]{1,}/gi) ?? []) {
		add(token);
	}
	for (const sequence of question.match(/[\u3400-\u9fff]+/g) ?? []) {
		const characters = [...sequence];
		if (characters.length <= 6) add(sequence);
		for (const size of [2, 3]) {
			for (let index = 0; index + size <= characters.length; index++) {
				add(characters.slice(index, index + size).join(""));
			}
		}
	}
	return unique;
}

function extractCutoff(
	question: string,
	now: Date,
): Pick<GeneralResearchTaskProfile, "cutoff" | "cutoffSource"> {
	const marker =
		/(?:\b(?:information\s+)?cutoff\b|\bas[ -]of\b|\busing (?:only )?information (?:available )?(?:through|before)\b|截至(?:时间)?|信息截止(?:时间)?)[\s:：-]*([^\n;；]+)/i;
	const match = question.match(marker)?.[1]?.trim();
	if (match) {
		const iso = match.match(
			/^([12]\d{3}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?)/,
		)?.[1];
		const english = match.match(
			/^([A-Za-z]{3,9}\s+\d{1,2},?\s+[12]\d{3}(?:\s+\d{1,2}:\d{2}(?:\s*(?:UTC|GMT|[AP]M))?)?)/i,
		)?.[1];
		const chinese = match.match(
			/^([12]\d{3})年(\d{1,2})月(\d{1,2})日(?:\s*(\d{1,2})[:：](\d{2}))?/,
		);
		const candidate =
			iso ??
			english ??
			(chinese
				? `${chinese[1]}-${chinese[2]?.padStart(2, "0")}-${chinese[3]?.padStart(2, "0")}T${(chinese[4] ?? "00").padStart(2, "0")}:${chinese[5] ?? "00"}:00+08:00`
				: undefined);
		const parsed = candidate ? new Date(candidate) : new Date(Number.NaN);
		if (!Number.isNaN(parsed.getTime())) {
			return { cutoff: parsed.toISOString(), cutoffSource: "explicit" };
		}
	}
	return { cutoff: now.toISOString(), cutoffSource: "retrieval-time" };
}

function fallbackFreshness(question: string): ResearchFreshness {
	if (
		/\b(now|right now|today|tomorrow|live|currently|current status)\b|现在|今天|明天|实时|余票|余号/i.test(
			question,
		)
	)
		return "live";
	if (
		/\b(latest|recent|this week|this month)\b|最新|近期|本周|本月/i.test(
			question,
		)
	)
		return "recent";
	if (
		/\b(historical|history|previous years?|archive)\b|历史|历年|档案/i.test(
			question,
		)
	)
		return "historical";
	return "stable";
}

function compactQuestion(question: string): string {
	return question.length <= 360 ? question : `${question.slice(0, 357)}...`;
}

function safetyFloorRisk(
	question: string,
	requestedRisk: ResearchRisk,
): ResearchRisk {
	if (requestedRisk === "high-stakes") return requestedRisk;
	const highImpact =
		/\b(?:CVE|vulnerabilit(?:y|ies)|exploit|security patch|fixed package|drug interaction|drug|medicat(?:ion|e)|medicine|dose|dosage|diagnos(?:is|e)|treatment|patient|prescription|legal advice|law|regulation|tax|court|contract|compliance|investment advice|securities|loan|mortgage)\b|安全漏洞|漏洞修复|补丁版本|药物相互作用|药物|药品|用药|剂量|诊断|治疗|患者|同服|一起吃|服用|法律|法规|税务|合同|合规|投资建议|贷款/i;
	return highImpact.test(question) ? "high-stakes" : requestedRisk;
}

function numericAtoms(value: string): Set<string> {
	return new Set(
		(value.match(/\d+(?:\.\d+)?/g) ?? []).map((token) => {
			const number = Number(token);
			return Number.isFinite(number) ? String(number) : token;
		}),
	);
}

function hasUnattributedNumber(value: string, question: string): boolean {
	const allowed = numericAtoms(question);
	for (const atom of numericAtoms(value)) {
		if (atom !== "0" && !allowed.has(atom)) return true;
	}
	return false;
}

function stripAnswerExamples(value: string): string {
	return value
		.replace(/\((?:e\.g\.|for example|例如|比如)[^)]*\)/gi, "")
		.replace(
			/(?:e\.g\.|for example|例如|比如)\s*[:：,，]?[^;；。]*[;；。]?/gi,
			"",
		)
		.replace(/\s+/g, " ")
		.trim();
}

function safePlannerText(
	value: string,
	question: string,
	field: string,
	warnings: string[],
): string | undefined {
	const cleaned = sanitizeEvidenceText(stripAnswerExamples(value), 500);
	if (!cleaned) return undefined;
	if (hasUnattributedNumber(cleaned, question)) {
		const answerShaped =
			/\b(?:answer|fixed|version|threshold|price|amount|availability|equals?)\b|答案|修复版本|阈值|价格|金额|余票|余号/i.test(
				cleaned,
			);
		if (field === "query" && !answerShaped) {
			warnings.push(
				"The routing query contains numbers not present in the question; retain them as unverified search tokens, not facts.",
			);
			return cleaned;
		}
		warnings.push(`The routing plan field ${field} contained answer-like numbers absent from the question and was discarded.`);
		return undefined;
	}
	return cleaned;
}

function deduplicate(values: string[]): string[] {
	return [...new Set(values.map((value) => value.trim()).filter(Boolean))];
}

function stageForRoute(route: SourceRoute): RetrievalStage {
	return route.authority === "primary" ? "primary-evidence" : "cross-check";
}

function buildQueries(
	request: ResearchRequest,
	question: string,
	sources: SourceRoute[],
	warnings: string[],
): ResearchQuery[] {
	const planned: ResearchQuery[] = [];
	for (const candidate of request.queries) {
		const query = safePlannerText(candidate.query, question, "query", warnings);
		const rationale = safePlannerText(
			candidate.rationale,
			question,
			"query rationale",
			warnings,
		);
		if (query && rationale) {
			planned.push({ stage: candidate.stage, query, rationale });
		}
	}

	const trusted = sources
		.filter(
			(route) =>
				route.verified === true &&
				route.authority !== "open-web" &&
				route.domains.length > 0,
		)
		.sort((left, right) => {
			if (left.authority === right.authority) return 0;
			return left.authority === "primary" ? -1 : 1;
		})
		.slice(0, 2);
	const siteBudget = trusted.length;
	const plannedBudget = Math.max(1, MAX_QUERIES - siteBudget);
	const selected: ResearchQuery[] = [];
	const addPlanned = (item: ResearchQuery | undefined): void => {
		if (!item || selected.includes(item) || selected.length >= plannedBudget)
			return;
		selected.push(item);
	};
	addPlanned(
		planned.find(
			(item) =>
				item.stage === "primary-evidence" || item.stage === "current-status",
		),
	);
	addPlanned(planned.find((item) => item.stage === "cross-check"));
	addPlanned(planned.find((item) => item.stage === "counterevidence"));
	for (const item of planned) addPlanned(item);

	const queries = [...selected];
	const minimizedIntent = safePlannerText(
		request.intent,
		question,
		"search intent",
		warnings,
	);
	const privacySafeBase = compactQuestion(
		(minimizedIntent ?? planned[0]?.query ?? question)
			.replace(/(?:^|\s)site:[a-z0-9.-]+(?=\s|$)/gi, " ")
			.replace(/\s+/g, " ")
			.trim(),
	);
	for (const source of trusted) {
		const domain = source.domains[0];
		if (!domain) continue;
		if (queries.some((item) => item.query.includes(`site:${domain}`))) continue;
		queries.push({
			stage: stageForRoute(source),
			query: `${privacySafeBase} site:${domain}`,
			rationale: `Locate evidence within ${source.name} after verification against the trusted domain registry.`,
		});
	}

	const fallbackStage: RetrievalStage =
		request.freshness === "live" || request.freshness === "recent"
			? "current-status"
			: "primary-evidence";
	if (queries.length < 2) {
		queries.push({
			stage: fallbackStage,
			query: `${privacySafeBase} official source latest`,
			rationale: "Discover current primary-source pages or official documentation when the exact source is unavailable.",
		});
	}
	if (queries.length < 3) {
		queries.push({
			stage: "cross-check",
			query: `${privacySafeBase} independent verification source`,
			rationale: "Use independent sources to check conflicts in scope, dates, versions, and definitions.",
		});
	}

	const seen = new Set<string>();
	return queries
		.filter((item) => {
			const key = item.query.toLowerCase().replace(/\s+/g, " ").trim();
			if (seen.has(key)) return false;
			seen.add(key);
			return true;
		})
		.slice(0, MAX_QUERIES);
}

function validateSources(
	request: ResearchRequest,
	question: string,
	warnings: string[],
	rejectedSources: string[],
): SourceRoute[] {
	const sources: SourceRoute[] = [];
	const seen = new Set<string>();
	for (const candidate of request.sources) {
		const domain = normalizeSourceDomain(candidate.domain);
		if (!domain) {
			rejectedSources.push(sanitizeEvidenceText(candidate.name, 120));
			continue;
		}
		const trusted = findTrustedSource(domain);
		const routeKey = trusted ? `registry:${trusted.id}` : `candidate:${domain}`;
		if (seen.has(routeKey)) continue;
		seen.add(routeKey);
		const purpose =
			safePlannerText(
				candidate.purpose,
				question,
				"source purpose",
				warnings,
			) ?? "Retrieve fields needed to answer the original question.";
		const verification =
			safePlannerText(
				candidate.verification,
				question,
				"source verification",
				warnings,
			) ??
			"Verify the domain, scope, timestamp, and record identity before use.";
		sources.push({
			id: trusted ? `registry:${trusted.id}` : `candidate:${domain}`,
			name: trusted?.name ?? sanitizeEvidenceText(candidate.name, 160),
			authority: trusted?.authority ?? candidate.authorityClaim,
			authorityClaim: candidate.authorityClaim,
			access: trusted?.access ?? candidate.access,
			domains: trusted ? deduplicate([domain, ...trusted.domains]) : [domain],
			matchSubdomains: trusted?.matchSubdomains,
			purpose: trusted ? `${trusted.purpose} Planned use: ${purpose}` : purpose,
			queryHint: purpose,
			...(trusted?.entryUrl ? { entryUrl: trusted.entryUrl } : {}),
			...(trusted?.fetchEntry ? { fetchEntry: true } : {}),
			verified: true,
			origin: trusted ? "registry" : "llm",
			requiresAuth: trusted
				? trusted.access === "login"
					? true
					: trusted.access === "browser"
						? "unknown"
						: false
				: candidate.requiresAuth,
			verification,
		});
	}
	sources.push({
		id: "open-web",
		name: "Configured open-web discovery",
		authority: "open-web",
		access: "web-search",
		domains: [],
		purpose:
			"Discover canonical pages and fill gaps when selected sources are blocked or incomplete.",
		queryHint: "official source independent verification",
		verified: false,
		origin: "fallback",
		requiresAuth: false,
		verification:
			"Search results remain untrusted until their real URL and evidence are checked.",
	});
	return sources;
}

export function validateResearchRequest(
	value: unknown,
	now = new Date(),
): ValidatedResearchPlan {
	const request = ResearchRequestSchema.parse(value);
	const question = normalizeQuestion(request.question);
	const warnings: string[] = [];
	const rejectedSources: string[] = [];
	const sources = validateSources(request, question, warnings, rejectedSources);
	const intent =
		safePlannerText(request.intent, question, "intent", warnings) ?? question;
	const filterTexts = (values: string[], field: string): string[] =>
		deduplicate(
			values
				.map((item) => safePlannerText(item, question, field, warnings))
				.filter((item): item is string => Boolean(item)),
		);
	const constraints = filterTexts(request.constraints, "constraint");
	const unresolvedFacts = filterTexts(
		request.unresolvedFacts,
		"unresolved fact",
	);
	const stopConditions = filterTexts(request.stopConditions, "stop condition");
	const risk = safetyFloorRisk(question, request.risk);
	if (risk !== request.risk) {
		warnings.push(
			"The deterministic safety baseline classified this task as high-stakes; the LLM cannot lower the risk level for medical, legal, financial, or security topics.",
		);
	}
	if (
		risk === "high-stakes" &&
		!sources.some((source) => source.authority === "primary")
	) {
		warnings.push(
			"This high-stakes task has no verified primary-source domain; do not give definitive advice on this basis.",
		);
		stopConditions.push(
			"Stop if no verified first-party source can be obtained.",
		);
	}
	return {
		profile: {
			question,
			intent,
			freshness: request.freshness,
			risk,
			...extractCutoff(question, now),
			keywords: extractKeywords(question),
			constraints,
			unresolvedFacts:
				unresolvedFacts.length > 0
					? unresolvedFacts
					: ["The factual answer remains unresolved until retrieval succeeds."],
			stopConditions:
				stopConditions.length > 0
					? deduplicate(stopConditions)
					: [
							"Stop if the relevant source, date, version, or entity cannot be verified.",
						],
			routingMode: "llm",
		},
		sources,
		queries: buildQueries(request, question, sources, warnings),
		routing: {
			mode: "llm",
			model: "calling-agent",
			warnings: deduplicate(warnings),
			rejectedSources: deduplicate(rejectedSources),
		},
	};
}

export function buildFallbackPlan(
	question: string,
	now = new Date(),
	reason = "No valid LLM source plan was supplied",
): ValidatedResearchPlan {
	const normalized = normalizeQuestion(question);
	const freshness = fallbackFreshness(normalized);
	const risk = safetyFloorRisk(normalized, "standard");
	return {
		profile: {
			question: normalized,
			intent: normalized,
			freshness,
			risk,
			...extractCutoff(normalized, now),
			keywords: extractKeywords(normalized),
			constraints: [],
			unresolvedFacts: [
				"The factual answer and authoritative source remain unresolved.",
			],
			stopConditions: [
				"Do not present search snippets alone as a verified final answer.",
			],
			routingMode: "fallback",
		},
		sources: [
			{
				id: "open-web",
				name: "Configured open-web discovery",
				authority: "open-web",
				access: "web-search",
				domains: [],
				purpose: "Discover likely first-party and independent sources.",
				queryHint: "official source independent verification",
				verified: false,
				origin: "fallback",
				requiresAuth: false,
				verification: "Treat every result as untrusted until checked.",
			},
		],
		queries: [
			{
				stage:
					freshness === "live" || freshness === "recent"
						? "current-status"
						: "primary-evidence",
				query: `${compactQuestion(normalized)} official source latest`,
				rationale: "Discover potential primary sources and current pages.",
			},
			{
				stage: "cross-check",
				query: `${compactQuestion(normalized)} independent verification source`,
				rationale: "Seek independent corroboration and identify conflicting definitions.",
			},
		],
		routing: {
			mode: "fallback",
			warnings: [sanitizeEvidenceText(reason, 300)],
			rejectedSources: [],
		},
	};
}

function compactResearchRequest(value: unknown): ResearchRequest | undefined {
	const result = CompactResearchRequestSchema.safeParse(value);
	if (!result.success) return undefined;
	const data = result.data;
	const question = normalizeQuestion(data.question);
	const freshness = data.freshness ?? fallbackFreshness(question);
	const stage: RetrievalStage =
		freshness === "live" || freshness === "recent"
			? "current-status"
			: "primary-evidence";
	return ResearchRequestSchema.parse({
		question,
		intent: data.intent ?? question,
		freshness,
		risk: data.risk ?? "standard",
		sources: data.sources.map((source) => {
			const domain = typeof source === "string" ? source : source.domain;
			return {
				name: typeof source === "string" ? domain : (source.name ?? domain),
				domain,
				authorityClaim: "primary",
				access: "web-search",
				requiresAuth: "unknown",
				purpose:
					typeof source === "string"
						? "Retrieve the fields needed for the original question"
						: (source.purpose ??
							"Retrieve the fields needed for the original question"),
				verification:
					"Verify domain ownership, record identity, scope, version, and timestamp",
			};
		}),
		constraints: data.constraints,
		unresolvedFacts: data.unresolvedFacts ?? [
			"The factual answer remains unknown until retrieval succeeds",
		],
		queries: [
			{
				stage,
				query: question,
				rationale:
					"Retrieve current evidence for the original question without assuming an answer",
			},
		],
		stopConditions: data.stopConditions ?? [
			"Stop if the relevant source, entity, date, or version cannot be verified",
		],
	});
}

function siteRestrictedRequest(input: string): ResearchRequest | undefined {
	const domains = [...input.matchAll(/(?:^|\s)site:([a-z0-9.-]+)(?=\s|$)/gi)]
		.map((match) => match[1] ?? "")
		.filter(Boolean);
	if (domains.length === 0) return undefined;
	const question = normalizeQuestion(
		input.replace(/(?:^|\s)site:[a-z0-9.-]+(?=\s|$)/gi, " "),
	);
	return compactResearchRequest({ question, sources: domains });
}

function looseQuestion(value: unknown, fallback: string): string {
	if (
		value &&
		typeof value === "object" &&
		"question" in value &&
		typeof value.question === "string"
	) {
		return value.question;
	}
	return fallback;
}

/**
 * Production uses the already-running main agent as the semantic router. It
 * passes a JSON source plan in the existing query string, avoiding a second
 * external LLM request. Plain strings remain backward compatible and use the
 * generic discovery fallback.
 */
export async function planResearchRequest(
	input: string,
	now = new Date(),
): Promise<ValidatedResearchPlan> {
	const trimmed = input.trim();
	if (!trimmed.startsWith("{")) {
		const siteRequest = siteRestrictedRequest(trimmed);
		return siteRequest
			? validateResearchRequest(siteRequest, now)
			: buildFallbackPlan(trimmed, now);
	}
	let parsed: unknown;
	try {
		parsed = JSON.parse(trimmed);
		const full = ResearchRequestSchema.safeParse(parsed);
		if (full.success) return validateResearchRequest(full.data, now);
		const compact = compactResearchRequest(parsed);
		if (compact) return validateResearchRequest(compact, now);
		throw full.error;
	} catch (error) {
		const detail = error instanceof Error ? error.message : String(error);
		return buildFallbackPlan(
			looseQuestion(parsed, trimmed),
			now,
			`Invalid LLM source plan; using generic discovery fallback: ${sanitizeEvidenceText(detail, 220)}`,
		);
	}
}
