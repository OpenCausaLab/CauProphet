export type RetrievalStage =
	| "resolution-anchor"
	| "current-signals"
	| "historical-baseline"
	| "market-consensus"
	| "challenge"
	| "primary-evidence"
	| "current-status"
	| "background"
	| "cross-check"
	| "counterevidence";

export type SourceAuthority = "primary" | "specialist" | "open-web";

export type SourceAccess =
	| "direct-api"
	| "direct-page"
	| "document"
	| "web-search"
	| "browser"
	| "login";

export type TemporalStatus = "pre-cutoff" | "post-cutoff" | "unknown";

export type ResearchFreshness = "live" | "recent" | "stable" | "historical";

export type ResearchRisk = "standard" | "high-stakes";

export type RoutingMode = "llm" | "fallback";

/** General research request and source-selection context. */
export interface GeneralResearchTaskProfile {
	question: string;
	intent: string;
	freshness: ResearchFreshness;
	risk: ResearchRisk;
	cutoff: string;
	cutoffSource: "explicit" | "retrieval-time";
	keywords: string[];
	constraints: string[];
	unresolvedFacts: string[];
	stopConditions: string[];
	routingMode: RoutingMode;
}

export interface EvidenceProfile {
	question: string;
	cutoff: string;
	cutoffSource: "explicit" | "retrieval-time";
	keywords: string[];
	/** Live, near-now retrieval may use current pages without publish metadata. */
	allowUndatedCurrentEvidence?: boolean;
}

export interface SourceRoute {
	id: string;
	name: string;
	authority: SourceAuthority;
	access: SourceAccess;
	domains: string[];
	matchSubdomains?: boolean;
	purpose: string;
	queryHint: string;
	entryUrl?: string;
	fetchEntry?: boolean;
	authorityClaim?: SourceAuthority;
	verified?: boolean;
	origin?: "llm" | "registry" | "fallback";
	requiresAuth?: boolean | "unknown";
	verification?: string;
}

export interface ResearchQuery {
	stage: RetrievalStage;
	query: string;
	rationale: string;
}

export interface SearchResult {
	title: string;
	url: string;
	snippet: string;
	provider: string;
	siteName?: string;
	publishedAt?: string;
}

export interface SearchBatch {
	provider: string;
	query: string;
	results: SearchResult[];
	error?: string;
}

export interface EvidenceCard {
	id: string;
	stage: RetrievalStage;
	role:
		| "resolver"
		| "driver"
		| "base-rate"
		| "external-forecast"
		| "counterevidence"
		| "answer-source"
		| "current-status"
		| "context"
		| "corroboration";
	title: string;
	claim: string;
	url: string;
	domain: string;
	sourceTier: SourceAuthority;
	provider: string;
	publishedAt?: string;
	retrievedAt: string;
	temporalStatus: TemporalStatus;
	relevance: number;
	rankScore: number;
}

export interface ExcludedEvidence {
	title: string;
	url: string;
	publishedAt?: string;
	reason: "post-cutoff" | "unknown-time-with-explicit-cutoff";
}

export interface ResearchPacket {
	profile: GeneralResearchTaskProfile;
	sources: SourceRoute[];
	queries: ResearchQuery[];
	routing: {
		mode: RoutingMode;
		model?: string;
		warnings: string[];
		rejectedSources: string[];
	};
	provider: string;
	evidence: EvidenceCard[];
	excluded: ExcludedEvidence[];
	errors: string[];
	warnings: string[];
}
