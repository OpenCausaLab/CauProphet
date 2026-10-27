import type {
	EvidenceCard,
	EvidenceProfile,
	ExcludedEvidence,
	RetrievalStage,
	SearchResult,
	SourceAuthority,
	SourceRoute,
	TemporalStatus,
} from "./types";

const SPECIALIST_DOMAINS = [
	"reuters.com",
	"apnews.com",
	"bloomberg.com",
	"ft.com",
	"wsj.com",
	"espn.com",
	"fangraphs.com",
	"baseball-reference.com",
	"theathletic.com",
	"nature.com",
	"science.org",
];

const TRACKING_PARAMETERS = new Set([
	"fbclid",
	"gclid",
	"mc_cid",
	"mc_eid",
	"ref",
	"source",
	"spm",
]);

const STAGE_ROLE: Record<RetrievalStage, EvidenceCard["role"]> = {
	"resolution-anchor": "resolver",
	"current-signals": "driver",
	"historical-baseline": "base-rate",
	"market-consensus": "external-forecast",
	challenge: "counterevidence",
	"primary-evidence": "answer-source",
	"current-status": "current-status",
	background: "context",
	"cross-check": "corroboration",
	counterevidence: "counterevidence",
};

export function decodeHtml(value: string): string {
	const named: Record<string, string> = {
		amp: "&",
		apos: "'",
		gt: ">",
		lt: "<",
		nbsp: " ",
		quot: '"',
	};
	const decodeCodePoint = (digits: string, radix: number): string => {
		const codePoint = Number.parseInt(digits, radix);
		return Number.isInteger(codePoint) &&
			codePoint >= 0 &&
			codePoint <= 0x10ffff
			? String.fromCodePoint(codePoint)
			: "";
	};
	return value
		.replace(/&#(\d+);/g, (_match, digits: string) =>
			decodeCodePoint(digits, 10),
		)
		.replace(/&#x([0-9a-f]+);/gi, (_match, digits: string) =>
			decodeCodePoint(digits, 16),
		)
		.replace(
			/&([a-z]+);/gi,
			(match, name: string) => named[name.toLowerCase()] ?? match,
		);
}

export function normalizeText(value: string): string {
	return decodeHtml(value)
		.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
		.replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ")
		.replace(/<[^>]+>/g, " ")
		.split("")
		.map((character) => {
			const code = character.charCodeAt(0);
			return (code < 32 &&
				character !== "\t" &&
				character !== "\n" &&
				character !== "\r") ||
				code === 127
				? " "
				: character;
		})
		.join("")
		.replace(/\s+/g, " ")
		.trim();
}

export function sanitizeEvidenceText(value: string, maxLength = 700): string {
	const normalized = normalizeText(value)
		.replace(
			/\b(?:ignore|disregard|override)\b.{0,80}\b(?:instruction|prompt|system message)\b/gi,
			"[instruction-like text removed]",
		)
		.replace(/\b(system|developer|assistant)\s*:/gi, "$1 label:");
	if (normalized.length <= maxLength) return normalized;
	return `${normalized.slice(0, Math.max(0, maxLength - 3)).trimEnd()}...`;
}

export function canonicalizeUrl(rawUrl: string): string {
	try {
		const url = new URL(rawUrl);
		url.hash = "";
		for (const parameter of [...url.searchParams.keys()]) {
			if (
				parameter.toLowerCase().startsWith("utm_") ||
				TRACKING_PARAMETERS.has(parameter.toLowerCase())
			) {
				url.searchParams.delete(parameter);
			}
		}
		if (url.pathname.length > 1)
			url.pathname = url.pathname.replace(/\/+$/, "");
		url.hostname = url.hostname.toLowerCase();
		return url.toString();
	} catch {
		return rawUrl.trim();
	}
}

export function getDomain(rawUrl: string): string {
	try {
		return new URL(rawUrl).hostname.toLowerCase().replace(/^www\./, "");
	} catch {
		return "unknown";
	}
}

function domainMatches(domain: string, expected: string): boolean {
	const normalized = expected.toLowerCase().replace(/^www\./, "");
	return domain === normalized || domain.endsWith(`.${normalized}`);
}

function routeMatchesDomain(
	route: SourceRoute,
	domain: string,
	expected: string,
): boolean {
	const normalized = expected.toLowerCase().replace(/^www\./, "");
	return route.matchSubdomains === false
		? domain === normalized
		: domainMatches(domain, normalized);
}

export function sourceTierForUrl(
	url: string,
	routes: SourceRoute[],
): SourceAuthority {
	let parsed: URL;
	try {
		parsed = new URL(url);
	} catch {
		return "open-web";
	}
	if (parsed.protocol !== "https:") return "open-web";
	const domain = parsed.hostname.toLowerCase().replace(/^www\./, "");
	for (const route of routes) {
		if (route.verified !== true || route.authority === "open-web") continue;
		if (
			route.domains.some((expected) =>
				routeMatchesDomain(route, domain, expected),
			)
		)
			return route.authority;
	}
	if (SPECIALIST_DOMAINS.some((expected) => domainMatches(domain, expected)))
		return "specialist";
	return "open-web";
}

function normalizePublishedAt(value?: string): string | undefined {
	if (!value) return undefined;
	const parsed = new Date(value);
	return Number.isNaN(parsed.getTime()) ? undefined : parsed.toISOString();
}

function temporalStatus(
	publishedAt: string | undefined,
	cutoff: string,
): TemporalStatus {
	if (!publishedAt) return "unknown";
	return new Date(publishedAt).getTime() <= new Date(cutoff).getTime()
		? "pre-cutoff"
		: "post-cutoff";
}

function tokenAppears(text: string, token: string): boolean {
	if (token.length <= 2 && /^[a-z]+$/i.test(token)) {
		return new RegExp(
			`\\b${token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`,
			"i",
		).test(text);
	}
	return text.includes(token);
}

function relevanceScore(
	result: SearchResult,
	profile: EvidenceProfile,
): number {
	if (profile.keywords.length === 0) return 0.35;
	const haystack =
		`${result.title} ${result.snippet} ${getDomain(result.url)}`.toLowerCase();
	const matched = profile.keywords.filter((keyword) =>
		tokenAppears(haystack, keyword),
	).length;
	return Math.min(1, matched / Math.min(10, profile.keywords.length));
}

function recencyScore(
	status: TemporalStatus,
	publishedAt: string | undefined,
	cutoff: string,
): number {
	if (status === "post-cutoff") return 0;
	if (!publishedAt) return 0.4;
	const ageDays = Math.max(
		0,
		(new Date(cutoff).getTime() - new Date(publishedAt).getTime()) / 86_400_000,
	);
	if (ageDays <= 14) return 1;
	if (ageDays <= 90) return 0.8;
	if (ageDays <= 365) return 0.6;
	return 0.35;
}

function authorityScore(authority: SourceAuthority): number {
	if (authority === "primary") return 1;
	if (authority === "specialist") return 0.75;
	return 0.45;
}

function titleFingerprint(title: string): string {
	return sanitizeEvidenceText(title, 240)
		.toLowerCase()
		.replace(/[^a-z0-9\u3400-\u9fff]+/g, "")
		.slice(0, 160);
}

interface StagedResult {
	stage: RetrievalStage;
	result: SearchResult;
}

export function buildEvidenceLedger(
	stagedResults: StagedResult[],
	profile: EvidenceProfile,
	routes: SourceRoute[],
	retrievedAt: string,
	limit = 12,
): { evidence: EvidenceCard[]; excluded: ExcludedEvidence[] } {
	const candidates: EvidenceCard[] = [];
	const excluded: ExcludedEvidence[] = [];

	for (const { stage, result } of stagedResults) {
		if (!result.url || !result.title) continue;
		let parsedUrl: URL;
		try {
			parsedUrl = new URL(result.url);
		} catch {
			continue;
		}
		if (parsedUrl.protocol !== "https:") continue;
		const canonicalUrl = canonicalizeUrl(parsedUrl.toString());
		const publishedAt = normalizePublishedAt(result.publishedAt);
		const temporal = temporalStatus(publishedAt, profile.cutoff);
		if (temporal === "post-cutoff" && publishedAt) {
			excluded.push({
				title: sanitizeEvidenceText(result.title, 240),
				url: canonicalUrl,
				publishedAt,
				reason: "post-cutoff",
			});
			continue;
		}
		if (
			temporal === "unknown" &&
			profile.cutoffSource === "explicit" &&
			!profile.allowUndatedCurrentEvidence
		) {
			excluded.push({
				title: sanitizeEvidenceText(result.title, 240),
				url: canonicalUrl,
				reason: "unknown-time-with-explicit-cutoff",
			});
			continue;
		}

		const sourceTier = sourceTierForUrl(canonicalUrl, routes);
		const relevance = relevanceScore(result, profile);
		const score =
			0.58 * authorityScore(sourceTier) +
			0.3 * relevance +
			0.12 * recencyScore(temporal, publishedAt, profile.cutoff);
		candidates.push({
			id: "",
			stage,
			role: STAGE_ROLE[stage],
			title: sanitizeEvidenceText(result.title, 240),
			claim: sanitizeEvidenceText(result.snippet || result.title),
			url: canonicalUrl,
			domain: getDomain(canonicalUrl),
			sourceTier,
			provider: result.provider,
			...(publishedAt ? { publishedAt } : {}),
			retrievedAt,
			temporalStatus: temporal,
			relevance: Number(relevance.toFixed(3)),
			rankScore: Number(score.toFixed(3)),
		});
	}

	const bestByKey = new Map<string, EvidenceCard>();
	for (const card of candidates) {
		const canonicalUrl = canonicalizeUrl(card.url);
		const key =
			getDomain(canonicalUrl) === "unknown"
				? `${canonicalUrl}|${titleFingerprint(card.title)}`
				: canonicalUrl;
		const previous = bestByKey.get(key);
		if (!previous || card.rankScore > previous.rankScore)
			bestByKey.set(key, card);
	}

	const evidence = [...bestByKey.values()]
		.sort((left, right) => right.rankScore - left.rankScore)
		.slice(0, limit)
		.map((card, index) => ({ ...card, id: `E${index + 1}` }));

	const uniqueExcluded = [
		...new Map(
			excluded.map((item) => [
				`${item.url}|${item.publishedAt ?? "unknown"}|${item.reason}`,
				item,
			]),
		).values(),
	];
	return { evidence, excluded: uniqueExcluded };
}
