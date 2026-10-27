import type { SourceAccess, SourceAuthority } from "./types";

export interface TrustedSourceDefinition {
	id: string;
	name: string;
	domains: string[];
	authority: Exclude<SourceAuthority, "open-web">;
	access: SourceAccess;
	purpose: string;
	/** Defaults to true. Set false for broad public suffixes such as gov.cn. */
	matchSubdomains?: boolean;
	entryUrl?: string;
	fetchEntry?: boolean;
}

/**
 * This is an access/metadata overlay, not a query classifier or allowlist. The
 * main agent chooses sources and authority tiers; registry matches only enrich
 * known domains with stable entry points and access hints.
 */
export const TRUSTED_SOURCES: TrustedSourceDefinition[] = [
	{
		id: "ons",
		name: "UK Office for National Statistics",
		domains: ["ons.gov.uk"],
		authority: "primary",
		access: "direct-page",
		purpose:
			"Official UK statistics releases, datasets, methods, and revisions.",
		entryUrl: "https://www.ons.gov.uk/releases",
		fetchEntry: true,
	},
	{
		id: "federal-reserve",
		name: "Federal Reserve",
		domains: ["federalreserve.gov"],
		authority: "primary",
		access: "direct-page",
		purpose: "Official US monetary-policy documents, calendars, and data.",
		entryUrl: "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
		fetchEntry: true,
	},
	{
		id: "bls",
		name: "US Bureau of Labor Statistics",
		domains: ["bls.gov", "api.bls.gov"],
		authority: "primary",
		access: "direct-api",
		purpose: "Official US labor and price statistics.",
	},
	{
		id: "eurostat",
		name: "Eurostat",
		domains: ["eurostat.ec.europa.eu"],
		authority: "primary",
		access: "direct-api",
		purpose: "Official European Union statistics and metadata.",
	},
	{
		id: "us-census",
		name: "US Census Bureau",
		domains: ["census.gov", "api.census.gov"],
		authority: "primary",
		access: "direct-api",
		purpose: "Official US population and economic data.",
	},
	{
		id: "sec-edgar",
		name: "SEC EDGAR",
		domains: ["sec.gov", "data.sec.gov"],
		authority: "primary",
		access: "direct-api",
		purpose: "Official US corporate filings and structured filing data.",
	},
	{
		id: "china-tax",
		name: "State Taxation Administration of China",
		domains: ["chinatax.gov.cn"],
		authority: "primary",
		access: "direct-page",
		purpose:
			"Official Chinese tax laws, notices, interpretations, and validity dates.",
	},
	{
		id: "china-government",
		name: "The State Council of China",
		domains: ["gov.cn"],
		authority: "primary",
		matchSubdomains: false,
		access: "direct-page",
		purpose: "Official Chinese government policies and documents.",
	},
	{
		id: "crossref",
		name: "Crossref",
		domains: ["crossref.org", "api.crossref.org"],
		authority: "primary",
		access: "direct-api",
		purpose: "DOI metadata and registered update relationships.",
	},
	{
		id: "doi",
		name: "DOI Foundation resolver",
		domains: ["doi.org"],
		authority: "primary",
		access: "direct-page",
		purpose: "Canonical DOI resolution to the registered landing page.",
	},
	{
		id: "datacite",
		name: "DataCite",
		domains: ["datacite.org", "api.datacite.org"],
		authority: "primary",
		access: "direct-api",
		purpose:
			"Registered DOI metadata, especially for datasets and research objects.",
	},
	{
		id: "who",
		name: "World Health Organization",
		domains: ["who.int"],
		authority: "primary",
		access: "direct-page",
		purpose: "Official global public-health guidance, reports, and alerts.",
	},
	{
		id: "dailymed",
		name: "DailyMed",
		domains: ["dailymed.nlm.nih.gov"],
		authority: "primary",
		access: "direct-page",
		purpose: "Current FDA-submitted drug labeling.",
	},
	{
		id: "fda",
		name: "US Food and Drug Administration",
		domains: ["fda.gov"],
		authority: "primary",
		access: "direct-page",
		purpose: "Official US drug, device, safety, and regulatory information.",
	},
	{
		id: "rust",
		name: "The Rust Project",
		domains: ["rust-lang.org", "blog.rust-lang.org", "doc.rust-lang.org"],
		authority: "primary",
		access: "direct-page",
		purpose: "Official Rust releases, documentation, and release notes.",
	},
	{
		id: "pytorch",
		name: "PyTorch",
		domains: ["pytorch.org", "docs.pytorch.org"],
		authority: "primary",
		access: "direct-page",
		purpose:
			"Official PyTorch releases, compatibility, and installation guidance.",
	},
	{
		id: "ubuntu-security",
		name: "Ubuntu Security",
		domains: ["ubuntu.com"],
		authority: "primary",
		access: "direct-page",
		purpose: "Vendor CVE status and fixed package versions by Ubuntu release.",
	},
	{
		id: "launchpad",
		name: "Launchpad",
		domains: ["launchpad.net"],
		authority: "primary",
		access: "direct-page",
		purpose: "Canonical Ubuntu package publication and build records.",
	},
	{
		id: "nvd",
		name: "US National Vulnerability Database",
		domains: ["nvd.nist.gov"],
		authority: "specialist",
		access: "direct-page",
		purpose:
			"Independent CVE enrichment and affected-configuration cross-checking.",
	},
	{
		id: "china-weather",
		name: "China Meteorological Administration",
		domains: ["cma.gov.cn", "weather.cma.cn"],
		authority: "primary",
		access: "browser",
		purpose: "Official Chinese weather observations, forecasts, and warnings.",
	},
	{
		id: "nmc",
		name: "National Meteorological Centre of China",
		domains: ["nmc.cn"],
		authority: "primary",
		access: "direct-page",
		purpose: "Official national and local weather forecasts and warnings.",
	},
	{
		id: "china-railway",
		name: "China Railway 12306",
		domains: ["12306.cn"],
		authority: "primary",
		access: "browser",
		purpose: "Official train schedules, fares, and live ticket inventory.",
	},
	{
		id: "pumch",
		name: "Peking Union Medical College Hospital",
		domains: ["pumch.cn"],
		authority: "primary",
		access: "browser",
		purpose: "Official clinic schedules and appointment guidance.",
	},
	{
		id: "shanghai-museum",
		name: "Shanghai Museum",
		domains: ["shanghaimuseum.net"],
		authority: "primary",
		access: "browser",
		purpose: "Official opening hours, closure notices, and reservations.",
	},
	{
		id: "china-eastern",
		name: "China Eastern Airlines",
		domains: ["ceair.com"],
		authority: "primary",
		access: "browser",
		purpose: "First-party flight status and service information.",
	},
	{
		id: "binance",
		name: "Binance",
		domains: ["binance.com", "api.binance.com"],
		authority: "primary",
		access: "direct-api",
		purpose: "First-party exchange-specific market data.",
	},
	{
		id: "jma",
		name: "Japan Meteorological Agency",
		domains: ["jma.go.jp"],
		authority: "primary",
		access: "direct-page",
		purpose:
			"Official Japanese earthquake and weather observations and alerts.",
	},
	{
		id: "usgs-earthquakes",
		name: "US Geological Survey",
		domains: ["usgs.gov", "earthquake.usgs.gov"],
		authority: "primary",
		access: "direct-api",
		purpose:
			"Authoritative earthquake event feeds, magnitudes, and update times.",
	},
	{
		id: "mlb-stats",
		name: "MLB Stats",
		domains: ["mlb.com", "statsapi.mlb.com"],
		authority: "primary",
		access: "direct-api",
		purpose: "First-party baseball schedules, rosters, status, and statistics.",
	},
	{
		id: "ufc",
		name: "Ultimate Fighting Championship",
		domains: ["ufc.com"],
		authority: "primary",
		access: "browser",
		purpose:
			"First-party UFC event cards, bout order, rankings, replacements, and official results.",
	},
	{
		id: "ufcstats",
		name: "UFCStats",
		domains: ["ufcstats.com"],
		authority: "specialist",
		access: "direct-page",
		purpose:
			"UFC event and fighter records with bout-level historical performance statistics.",
	},
	{
		id: "cbs-sports-ufc",
		name: "CBS Sports UFC",
		domains: ["cbssports.com"],
		authority: "specialist",
		access: "direct-page",
		purpose:
			"Independent UFC card listings, records, market odds, and implied win probabilities.",
		entryUrl: "https://www.cbssports.com/ufc/schedule/",
		fetchEntry: true,
	},
	{
		id: "sherdog",
		name: "Sherdog",
		domains: ["sherdog.com"],
		authority: "specialist",
		access: "direct-page",
		purpose:
			"Independent MMA event cards, fighter records, bout order, and late card changes.",
	},
	{
		id: "tapology",
		name: "Tapology",
		domains: ["tapology.com"],
		authority: "specialist",
		access: "browser",
		purpose:
			"Independent MMA event cards, community predictions, records, and card-change tracking.",
	},
	{
		id: "mma-fighting",
		name: "MMA Fighting",
		domains: ["mmafighting.com"],
		authority: "specialist",
		access: "direct-page",
		purpose:
			"Current UFC event schedules, main-card ordering, bout replacements, and fight-week reporting.",
		entryUrl: "https://www.mmafighting.com/schedule/ufc",
		fetchEntry: true,
	},
	{
		id: "mma-mania",
		name: "MMA Mania",
		domains: ["mmamania.com"],
		authority: "specialist",
		access: "direct-page",
		purpose:
			"Dated UFC fight-card updates, bout ordering, replacements, and market previews.",
		entryUrl: "https://www.mmamania.com/ufc-mma-schedule",
		fetchEntry: true,
	},
	{
		id: "spacex",
		name: "SpaceX",
		domains: ["spacex.com"],
		authority: "primary",
		access: "direct-page",
		purpose: "First-party launch mission pages and status updates.",
	},
];

export function normalizeSourceDomain(value: string): string | undefined {
	const trimmed = value.trim().toLowerCase();
	if (!trimmed) return undefined;
	try {
		const url = new URL(
			trimmed.includes("://") ? trimmed : `https://${trimmed}`,
		);
		if (url.username || url.password || url.port || url.pathname !== "/") {
			return undefined;
		}
		const domain = url.hostname.replace(/^www\./, "");
		if (
			!domain.includes(".") ||
			domain === "localhost" ||
			domain.endsWith(".local") ||
			!/^([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$/i.test(domain)
		) {
			return undefined;
		}
		return domain;
	} catch {
		return undefined;
	}
}

function domainMatches(candidate: string, expected: string): boolean {
	return candidate === expected || candidate.endsWith(`.${expected}`);
}

export function findTrustedSource(
	domain: string,
): TrustedSourceDefinition | undefined {
	const normalized = normalizeSourceDomain(domain);
	if (!normalized) return undefined;
	const matches = TRUSTED_SOURCES.flatMap((source) =>
		source.domains
			.filter((expected) =>
				source.matchSubdomains === false
					? normalized === expected
					: domainMatches(normalized, expected),
			)
			.map((expected) => ({ source, specificity: expected.length })),
	);
	matches.sort((left, right) => right.specificity - left.specificity);
	return matches[0]?.source;
}
