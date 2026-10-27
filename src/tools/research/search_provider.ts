import { decodeHtml, normalizeText, sanitizeEvidenceText } from "./evidence";
import type { SearchBatch, SearchResult } from "./types";

const SEARCH_TIMEOUT_MS = 15_000;

function requestSignal(): AbortSignal {
	return AbortSignal.timeout(SEARCH_TIMEOUT_MS);
}

function isHttpsUrl(value: unknown): value is string {
	if (typeof value !== "string") return false;
	try {
		return new URL(value).protocol === "https:";
	} catch {
		return false;
	}
}

function normalizedHostname(value: string): string | null {
	try {
		return new URL(value).hostname.toLowerCase().replace(/^www\./, "");
	} catch {
		return null;
	}
}

function siteDomains(query: string): string[] {
	return [
		...new Set(
			[...query.matchAll(/(?:^|\s)site:([a-z0-9.-]+)(?=\s|$)/gi)]
				.map((match) => match[1]?.toLowerCase().replace(/^www\./, ""))
				.filter((domain): domain is string => Boolean(domain)),
		),
	];
}

/**
 * Search providers occasionally ignore a `site:` operator while still
 * returning non-empty results. Treat those off-domain hits as a miss so the
 * configured fallback can try to honor the trusted-source route.
 */
export function enforceSiteDirectedResults(
	batch: SearchBatch,
	query: string,
): SearchBatch {
	const domains = siteDomains(query);
	if (domains.length === 0) return batch;
	const results = batch.results.filter((result) => {
		const hostname = normalizedHostname(result.url);
		return (
			hostname !== null &&
			domains.some(
				(domain) => hostname === domain || hostname.endsWith(`.${domain}`),
			)
		);
	});
	if (results.length === batch.results.length) return batch;
	const dropped = batch.results.length - results.length;
	return {
		...batch,
		results,
		error: [
			batch.error,
			`${dropped} off-domain result(s) ignored for site-directed query`,
		]
			.filter(Boolean)
			.join("; "),
	};
}

function parseDate(value: unknown): string | undefined {
	if (typeof value !== "string" || value.length === 0) return undefined;
	const parsed = new Date(value);
	return Number.isNaN(parsed.getTime()) ? undefined : parsed.toISOString();
}

function decodeDuckDuckGoUrl(value: string): string {
	const decoded = decodeHtml(value);
	try {
		const parsed = new URL(decoded, "https://html.duckduckgo.com");
		const redirected = parsed.searchParams.get("uddg");
		return redirected ?? parsed.toString();
	} catch {
		return decoded;
	}
}

function parseDuckDuckGo(html: string, limit: number): SearchResult[] {
	const anchors = [
		...html.matchAll(
			/<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi,
		),
	];
	const results: SearchResult[] = [];
	for (let index = 0; index < Math.min(limit, anchors.length); index++) {
		const anchor = anchors[index];
		if (!anchor?.[1] || !anchor[2]) continue;
		const start = (anchor.index ?? 0) + anchor[0].length;
		const end = anchors[index + 1]?.index ?? html.length;
		const block = html.slice(start, end);
		const snippetMatch = block.match(
			/<(?:a|div)[^>]*class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\/(?:a|div)>/i,
		);
		results.push({
			title: sanitizeEvidenceText(anchor[2], 240),
			url: decodeDuckDuckGoUrl(anchor[1]),
			snippet: sanitizeEvidenceText(snippetMatch?.[1] ?? ""),
			provider: "duckduckgo",
		});
	}
	return results.filter((result) => isHttpsUrl(result.url));
}

async function duckDuckGoSearch(
	query: string,
	limit: number,
): Promise<SearchBatch> {
	try {
		const response = await fetch(
			`https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`,
			{
				headers: {
					Accept: "text/html,application/xhtml+xml",
					"User-Agent": "CausalClaw/0.1 research-search",
				},
				signal: requestSignal(),
			},
		);
		if (!response.ok) {
			return {
				provider: "duckduckgo",
				query,
				results: [],
				error: `HTTP ${response.status}`,
			};
		}
		const results = parseDuckDuckGo(await response.text(), limit);
		return {
			provider: "duckduckgo",
			query,
			results,
			...(results.length === 0 ? { error: "No results found." } : {}),
		};
	} catch (error) {
		return { provider: "duckduckgo", query, results: [], error: String(error) };
	}
}

async function braveSearch(
	query: string,
	apiKey: string,
	limit: number,
): Promise<SearchBatch> {
	try {
		const response = await fetch(
			`https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=${limit}`,
			{
				headers: {
					Accept: "application/json",
					"Accept-Encoding": "gzip",
					"X-Subscription-Token": apiKey,
				},
				signal: requestSignal(),
			},
		);
		if (!response.ok) {
			const body = normalizeText((await response.text()).slice(0, 300));
			return {
				provider: "brave",
				query,
				results: [],
				error: `HTTP ${response.status}: ${body}`,
			};
		}
		const data = (await response.json()) as {
			web?: {
				results?: Array<{
					title?: string;
					url?: string;
					description?: string;
					profile?: { long_name?: string };
					page_age?: string;
				}>;
			};
		};
		const results = (data.web?.results ?? [])
			.filter((result) => Boolean(result.title) && isHttpsUrl(result.url))
			.slice(0, limit)
			.map(
				(result): SearchResult => ({
					title: sanitizeEvidenceText(result.title ?? "", 240),
					url: result.url ?? "",
					snippet: sanitizeEvidenceText(result.description ?? ""),
					provider: "brave",
					...(result.profile?.long_name
						? { siteName: sanitizeEvidenceText(result.profile.long_name, 120) }
						: {}),
					...(parseDate(result.page_age)
						? { publishedAt: parseDate(result.page_age) }
						: {}),
				}),
			);
		return {
			provider: "brave",
			query,
			results,
			...(results.length === 0 ? { error: "No results found." } : {}),
		};
	} catch (error) {
		return { provider: "brave", query, results: [], error: String(error) };
	}
}

async function bochaSearch(
	query: string,
	apiKey: string,
	limit: number,
): Promise<SearchBatch> {
	try {
		const response = await fetch("https://api.bochaai.com/v1/web-search", {
			method: "POST",
			headers: {
				Authorization: `Bearer ${apiKey}`,
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ query, freshness: "noLimit", count: limit }),
			signal: requestSignal(),
		});
		if (!response.ok) {
			const body = normalizeText((await response.text()).slice(0, 300));
			return {
				provider: "bocha",
				query,
				results: [],
				error: `HTTP ${response.status}: ${body}`,
			};
		}
		const data = (await response.json()) as {
			data?: {
				webPages?: {
					value?: Array<{
						name?: string;
						url?: string;
						snippet?: string;
						siteName?: string;
						datePublished?: string;
						dateLastCrawled?: string;
					}>;
				};
			};
		};
		const results = (data.data?.webPages?.value ?? [])
			.filter((result) => Boolean(result.name) && isHttpsUrl(result.url))
			.slice(0, limit)
			.map((result): SearchResult => {
				const publishedAt = parseDate(result.datePublished);
				return {
					title: sanitizeEvidenceText(result.name ?? "", 240),
					url: result.url ?? "",
					snippet: sanitizeEvidenceText(result.snippet ?? ""),
					provider: "bocha",
					...(result.siteName
						? { siteName: sanitizeEvidenceText(result.siteName, 120) }
						: {}),
					...(publishedAt ? { publishedAt } : {}),
				};
			});
		return {
			provider: "bocha",
			query,
			results,
			...(results.length === 0 ? { error: "No results found." } : {}),
		};
	} catch (error) {
		return { provider: "bocha", query, results: [], error: String(error) };
	}
}

async function selectedProvider(
	query: string,
	limit: number,
): Promise<SearchBatch> {
	try {
		const { GlobalState } = await import("@/core/state");
		const config = GlobalState.getInstance().config;
		if (config.searchProvider === "brave" && config.braveApiKey) {
			return braveSearch(query, config.braveApiKey, limit);
		}
		if (config.searchProvider === "bocha" && config.bochaApiKey) {
			return bochaSearch(query, config.bochaApiKey, limit);
		}
	} catch {
		// Config is optional for the zero-key DuckDuckGo fallback.
	}
	return duckDuckGoSearch(query, limit);
}

export async function searchOpenWeb(
	query: string,
	limit = 5,
): Promise<SearchBatch> {
	const boundedLimit = Math.max(1, Math.min(10, Math.round(limit)));
	const primary = enforceSiteDirectedResults(
		await selectedProvider(query, boundedLimit),
		query,
	);
	if (primary.results.length > 0 || primary.provider === "duckduckgo")
		return primary;

	const fallback = enforceSiteDirectedResults(
		await duckDuckGoSearch(query, boundedLimit),
		query,
	);
	return {
		...fallback,
		provider: `${primary.provider}->duckduckgo`,
		...(fallback.results.length === 0
			? {
					error: `${primary.provider}: ${primary.error ?? "no results"}; duckduckgo: ${fallback.error ?? "no results"}`,
				}
			: {
					error: `${primary.provider}: ${primary.error ?? "no results"}; used DuckDuckGo fallback`,
				}),
	};
}
