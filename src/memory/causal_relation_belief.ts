/**
 * Bayesian claim aggregation for one context-qualified causal relation.
 *
 * Hypothesized claims set a prior log-odds; only the strongest hypothesis is
 * used because repeated proposals are not independent evidence. Supported and
 * contradicted claims contribute positive and negative log Bayes factors.
 */

export type CausalRelationVerdict =
	| "hypothesized"
	| "supported"
	| "contradicted";

export type CausalRelationClaimConfidence = "low" | "medium" | "high";

export interface CausalRelationBeliefClaim {
	verdict: CausalRelationVerdict;
	/** Historical claims without a level retain the previous high-confidence reading. */
	confidence?: CausalRelationClaimConfidence;
}

export interface CausalRelationBeliefAssessment {
	/** Strongest hypothesis expressed directly in log prior odds. */
	priorLogOdds: number;
	/** Sum of positive log Bayes factors from supporting claims. */
	supportLogBayesFactor: number;
	/** Magnitude of negative log Bayes factors from contradicting claims. */
	contradictionLogBayesFactor: number;
	/** Posterior log-odds after aggregating all claims. */
	logOdds: number;
	/** Bayesian posterior used to rank relation variants. */
	posterior: number;
	/** Retrieval gate; equivalent to posterior > 0.5. */
	active: boolean;
}

export const CAUSAL_RELATION_HYPOTHESIS_LOG_ODDS: Readonly<
	Record<CausalRelationClaimConfidence, number>
> = {
	low: 0.25,
	medium: 0.75,
	high: 1.25,
};

export const CAUSAL_RELATION_CLAIM_LOG_BAYES_FACTOR: Readonly<
	Record<CausalRelationClaimConfidence, number>
> = {
	low: 0.5,
	medium: 1,
	high: 2,
};

/** A relation with no explicit hypothesis starts with a weak positive prior. */
export const DEFAULT_CAUSAL_RELATION_PRIOR_LOG_ODDS = 0.25;

/** posterior > 0.5 is exactly logOdds > 0. */
export const CAUSAL_RELATION_RETRIEVAL_LOG_ODDS_THRESHOLD = 0;

function confidence(
	claim: CausalRelationBeliefClaim,
): CausalRelationClaimConfidence {
	return claim.confidence ?? "high";
}

function sigmoid(value: number): number {
	if (value >= 0) {
		const exponent = Math.exp(-value);
		return 1 / (1 + exponent);
	}
	const exponent = Math.exp(value);
	return exponent / (1 + exponent);
}

/**
 * Compute posterior log-odds for one relation variant (same directed endpoints,
 * context, and lag):
 *
 *   L = h + sum(log BF support) - sum(log BF contradiction)
 *   p = sigmoid(L)
 */
export function assessCausalRelationBelief(
	claims: readonly CausalRelationBeliefClaim[],
): CausalRelationBeliefAssessment {
	let priorLogOdds = DEFAULT_CAUSAL_RELATION_PRIOR_LOG_ODDS;
	let strongestHypothesis = 0;
	let supportLogBayesFactor = 0;
	let contradictionLogBayesFactor = 0;

	for (const claim of claims) {
		const level = confidence(claim);
		if (claim.verdict === "hypothesized") {
			strongestHypothesis = Math.max(
				strongestHypothesis,
				CAUSAL_RELATION_HYPOTHESIS_LOG_ODDS[level],
			);
		} else if (claim.verdict === "supported") {
			supportLogBayesFactor += CAUSAL_RELATION_CLAIM_LOG_BAYES_FACTOR[level];
		} else if (claim.verdict === "contradicted") {
			contradictionLogBayesFactor +=
				CAUSAL_RELATION_CLAIM_LOG_BAYES_FACTOR[level];
		}
	}

	if (strongestHypothesis > 0) priorLogOdds = strongestHypothesis;
	const logOdds =
		priorLogOdds + supportLogBayesFactor - contradictionLogBayesFactor;
	const posterior = sigmoid(logOdds);

	return {
		priorLogOdds,
		supportLogBayesFactor,
		contradictionLogBayesFactor,
		logOdds,
		posterior,
		active: logOdds > CAUSAL_RELATION_RETRIEVAL_LOG_ODDS_THRESHOLD,
	};
}
