import { normalizeText, sanitizeEvidenceText } from "./evidence";
import type { SearchResult, SourceRoute } from "./types";

export type FetchFunction = (
	input: string | URL | Request,
	init?: RequestInit,
) => Promise<Response>;

export interface DirectFetchOutcome {
	result?: SearchResult;
	error?: string;
}

const MAX_CONTENT_LENGTH = 3_000_000;

function domainMatches(domain: string, expected: string): boolean {
	const normalized = expected.toLowerCase().replace(/^www\./, "");
	return domain === normalized || domain.endsWith(`.${normalized}`);
}

function routeAllowsDomain(route: SourceRoute, domain: string): boolean {
	return route.domains.some((expected) => {
		const normalized = expected.toLowerCase().replace(/^www\./, "");
		return route.matchSubdomains === false
			? domain === normalized
			: domainMatches(domain, normalized);
	});
}

function validateRouteUrl(route: SourceRoute): URL | undefined {
	if (!route.entryUrl || route.domains.length === 0) return undefined;
	try {
		const url = new URL(route.entryUrl);
		if (url.protocol !== "https:") return undefined;
		const domain = url.hostname.toLowerCase().replace(/^www\./, "");
		if (!routeAllowsDomain(route, domain)) return undefined;
		return url;
	} catch {
		return undefined;
	}
}

function metaContent(html: string, names: string[]): string | undefined {
	for (const name of names) {
		const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
		const patterns = [
			new RegExp(
				`<meta[^>]+(?:name|property)=["']${escaped}["'][^>]+content=["']([^"']+)["'][^>]*>`,
				"i",
			),
			new RegExp(
				`<meta[^>]+content=["']([^"']+)["'][^>]+(?:name|property)=["']${escaped}["'][^>]*>`,
				"i",
			),
		];
		for (const pattern of patterns) {
			const value = html.match(pattern)?.[1];
			if (value) return value;
		}
	}
	return undefined;
}

function normalizeDate(value?: string): string | undefined {
	if (!value) return undefined;
	const parsed = new Date(value);
	return Number.isNaN(parsed.getTime()) ? undefined : parsed.toISOString();
}

function summarizeHtml(html: string): {
	title: string;
	snippet: string;
	publishedAt?: string;
} {
	const title =
		metaContent(html, ["og:title", "twitter:title"]) ??
		html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1] ??
		"Official source";
	const description = metaContent(html, [
		"description",
		"og:description",
		"twitter:description",
	]);
	const body =
		html.match(/<main\b[^>]*>([\s\S]*?)<\/main>/i)?.[1] ??
		html.match(/<body\b[^>]*>([\s\S]*?)<\/body>/i)?.[1] ??
		html;
	const published =
		metaContent(html, [
			"article:published_time",
			"date",
			"datePublished",
			"DC.date.issued",
		]) ??
		html.match(/["']datePublished["']\s*:\s*["']([^"']+)["']/i)?.[1] ??
		html.match(/<time[^>]+datetime=["']([^"']+)["']/i)?.[1];
	const publishedAt = normalizeDate(published);
	const summaryText = description ? `${description} ${body}` : body;
	return {
		title: sanitizeEvidenceText(title, 240),
		snippet: sanitizeEvidenceText(summaryText, 1_800),
		...(publishedAt ? { publishedAt } : {}),
	};
}

function summarizeJson(value: unknown): string {
	if (
		value &&
		typeof value === "object" &&
		"dates" in value &&
		Array.isArray((value as { dates?: unknown[] }).dates)
	) {
		const dates = (value as { dates: Array<Record<string, unknown>> }).dates;
		const games = dates.flatMap((date) =>
			Array.isArray(date.games)
				? (date.games as Array<Record<string, unknown>>)
				: [],
		);
		const summaries = games.slice(0, 8).map((game) => {
			const teams = game.teams as
				| {
						away?: { team?: { name?: string } };
						home?: { team?: { name?: string } };
				  }
				| undefined;
			const venue = (game.venue as { name?: string } | undefined)?.name;
			const status = (game.status as { detailedState?: string } | undefined)
				?.detailedState;
			return [
				teams?.away?.team?.name,
				"at",
				teams?.home?.team?.name,
				venue ? `(${venue})` : "",
				status ?? "",
			]
				.filter(Boolean)
				.join(" ");
		});
		if (summaries.length > 0) return summaries.join("; ");
	}
	return sanitizeEvidenceText(JSON.stringify(value), 900);
}

const MAX_REDIRECTS = 3;

function isAllowlistedUrl(route: SourceRoute, url: URL): boolean {
	const domain = url.hostname.toLowerCase().replace(/^www\./, "");
	return url.protocol === "https:" && routeAllowsDomain(route, domain);
}

async function fetchWithAllowlistedRedirects(
	route: SourceRoute,
	initialUrl: URL,
	fetchFunction: FetchFunction,
): Promise<{ response: Response; resolvedUrl: URL }> {
	let currentUrl = initialUrl;
	for (let redirectCount = 0; redirectCount <= MAX_REDIRECTS; redirectCount++) {
		const response = await fetchFunction(currentUrl, {
			headers: {
				Accept: "text/html,application/json;q=0.9,*/*;q=0.5",
				"User-Agent": "CausalClaw/0.1 research-search",
			},
			redirect: "manual",
			signal: AbortSignal.timeout(12_000),
		});
		if (response.status >= 300 && response.status < 400) {
			const location = response.headers.get("location");
			if (!location) throw new Error("redirect missing Location header");
			if (redirectCount >= MAX_REDIRECTS) {
				throw new Error("too many redirects");
			}
			const redirected = new URL(location, currentUrl);
			if (!isAllowlistedUrl(route, redirected)) {
				throw new Error("rejected redirect outside allowlisted domains");
			}
			currentUrl = redirected;
			continue;
		}
		if (response.url) {
			const reportedUrl = new URL(response.url);
			if (!isAllowlistedUrl(route, reportedUrl)) {
				throw new Error("rejected response outside allowlisted domains");
			}
			currentUrl = reportedUrl;
		}
		return { response, resolvedUrl: currentUrl };
	}
	throw new Error("too many redirects");
}

async function readBoundedBody(response: Response): Promise<Uint8Array> {
	if (!response.body) return new Uint8Array();
	const reader = response.body.getReader();
	const chunks: Uint8Array[] = [];
	let total = 0;
	try {
		while (true) {
			const { done, value } = await reader.read();
			if (done) break;
			if (!value) continue;
			total += value.byteLength;
			if (total > MAX_CONTENT_LENGTH) {
				await reader.cancel("response exceeds size limit");
				throw new Error("response exceeds size limit");
			}
			chunks.push(value);
		}
	} finally {
		reader.releaseLock();
	}
	const body = new Uint8Array(total);
	let offset = 0;
	for (const chunk of chunks) {
		body.set(chunk, offset);
		offset += chunk.byteLength;
	}
	return body;
}

async function fetchAllowlistedUrl(
	route: SourceRoute,
	url: URL,
	fetchFunction: FetchFunction,
): Promise<DirectFetchOutcome> {
	try {
		const { response, resolvedUrl } = await fetchWithAllowlistedRedirects(
			route,
			url,
			fetchFunction,
		);
		if (!response.ok) {
			return { error: `${route.name}: HTTP ${response.status}` };
		}
		const contentLength = Number(response.headers.get("content-length") ?? 0);
		if (contentLength > MAX_CONTENT_LENGTH) {
			return { error: `${route.name}: response exceeds size limit` };
		}
		const body = await readBoundedBody(response);
		const text = new TextDecoder().decode(body);
		const contentType =
			response.headers.get("content-type")?.toLowerCase() ?? "";
		if (contentType.includes("application/json")) {
			const data = JSON.parse(text) as unknown;
			return {
				result: {
					title: route.name,
					url: resolvedUrl.toString(),
					snippet: summarizeJson(data),
					provider: `direct:${route.id}`,
					siteName: route.name,
				},
			};
		}

		const page = summarizeHtml(text);
		if (!page.snippet) {
			return { error: `${route.name}: empty page after extraction` };
		}
		return {
			result: {
				title: page.title || route.name,
				url: resolvedUrl.toString(),
				snippet: page.snippet,
				provider: `direct:${route.id}`,
				siteName: route.name,
				...(page.publishedAt ? { publishedAt: page.publishedAt } : {}),
			},
		};
	} catch (error) {
		return {
			error: `${route.name}: ${normalizeText(String(error)).slice(0, 220)}`,
		};
	}
}

export async function fetchOfficialEntry(
	route: SourceRoute,
	fetchFunction: FetchFunction = fetch,
): Promise<DirectFetchOutcome> {
	if (!route.fetchEntry) return {};
	const url = validateRouteUrl(route);
	if (!url) {
		return {
			error: `${route.name}: rejected invalid or non-allowlisted entry URL`,
		};
	}
	return fetchAllowlistedUrl(route, url, fetchFunction);
}

function trustedRouteForResult(
	result: SearchResult,
	routes: SourceRoute[],
): { route: SourceRoute; url: URL } | undefined {
	let url: URL;
	try {
		url = new URL(result.url);
	} catch {
		return undefined;
	}
	if (url.protocol !== "https:") return undefined;
	const domain = url.hostname.toLowerCase().replace(/^www\./, "");
	const route = routes.find(
		(candidate) =>
			candidate.verified === true &&
			candidate.authority !== "open-web" &&
			candidate.access !== "browser" &&
			candidate.access !== "login" &&
			candidate.access !== "web-search" &&
			routeAllowsDomain(candidate, domain),
	);
	return route ? { route, url } : undefined;
}

/**
 * Fetch a concrete search result only when its hostname maps to a trusted route
 * and its planned access mode supports non-interactive HTTP. Browser/login
 * sources stay as explicit handoff requirements rather than fake direct fetches.
 */
export async function fetchTrustedSearchResult(
	result: SearchResult,
	routes: SourceRoute[],
	fetchFunction: FetchFunction = fetch,
): Promise<DirectFetchOutcome> {
	const match = trustedRouteForResult(result, routes);
	if (!match) return {};
	return fetchAllowlistedUrl(match.route, match.url, fetchFunction);
}
