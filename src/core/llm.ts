import OpenAI from "openai";
import type {
	ChatCompletion,
	ChatCompletionCreateParamsNonStreaming,
} from "openai/resources/chat/completions";
import { GlobalState } from "@/core/state";

/** Retry transient connection/rate-limit/server failures before surfacing them. */
export const LLM_MAX_RETRIES = 5;

const SAFE_ERROR_TYPES = new Set([
	"Error", "TypeError", "AbortError", "TimeoutError", "APIError",
	"APIConnectionError", "APIConnectionTimeoutError", "APIUserAbortError",
	"BadRequestError", "AuthenticationError", "PermissionDeniedError",
	"NotFoundError", "ConflictError", "UnprocessableEntityError",
	"RateLimitError", "InternalServerError",
]);
const SAFE_TRANSPORT_CODES = new Set([
	"ECONNRESET", "ECONNREFUSED", "ECONNABORTED", "ETIMEDOUT", "EPIPE",
	"ENOTFOUND", "EAI_AGAIN", "ERR_SOCKET_CLOSED", "ERR_STREAM_PREMATURE_CLOSE",
	"UND_ERR_CONNECT_TIMEOUT", "UND_ERR_HEADERS_TIMEOUT", "UND_ERR_BODY_TIMEOUT",
	"UND_ERR_SOCKET", "ERR_TLS_CERT_ALTNAME_INVALID", "CERT_HAS_EXPIRED",
	"UNABLE_TO_VERIFY_LEAF_SIGNATURE", "DEPTH_ZERO_SELF_SIGNED_CERT",
	"ConnectionClosed", "TimeoutError",
]);

/** Preserve transport evidence without serializing provider messages or secrets. */
export function describeModelError(error: unknown): string {
	const chain: Array<Record<string, string | number>> = [];
	let current = error;
	for (let depth = 0; depth < 3 && current && typeof current === "object"; depth++) {
		const value = current as Record<string, unknown>;
		const type = current.constructor?.name ?? "Error";
		const entry: Record<string, string | number> = {
			type: SAFE_ERROR_TYPES.has(type) ? type : "UnknownError",
		};
		if (typeof value.name === "string" && SAFE_ERROR_TYPES.has(value.name)) {
			entry.name = value.name;
		}
		if (typeof value.code === "string" && SAFE_TRANSPORT_CODES.has(value.code)) {
			entry.code = value.code;
		}
		if (typeof value.status === "number" && Number.isInteger(value.status) && value.status >= 100 && value.status <= 599) {
			entry.status = value.status;
		}
		chain.push(entry);
		current = value.cause;
	}
	return JSON.stringify(chain.length ? chain : [{ type: "UnknownError" }]);
}

let openaiInstance: OpenAI | null = null;
/** Track the config values used to create the current instance for invalidation. */
let _cachedBaseUrl: string | undefined;
let _cachedApiKey: string | undefined;

/** Call this after saving new config to force a fresh client on next use. */
export function resetOpenAIClient(): void {
	openaiInstance = null;
	_cachedBaseUrl = undefined;
	_cachedApiKey = undefined;
}

export function getOpenAIClient(): OpenAI {
	const config = GlobalState.getInstance().config;

	// Invalidate if Base URL or API Key changed since last creation
	if (
		openaiInstance === null ||
		config.openaiBaseUrl !== _cachedBaseUrl ||
		config.openaiApiKey !== _cachedApiKey
	) {
		if (!config.openaiApiKey) {
			throw new Error(
				"OpenAI API Key is missing. Please configure it in Settings.",
			);
		}
		openaiInstance = new OpenAI({
			apiKey: config.openaiApiKey,
			baseURL: config.openaiBaseUrl,
			timeout: 900000, // 900 seconds — long reasoning requests may be slow to first token under load
			// The SDK retries connection errors, timeouts, 408/409/429, and 5xx
			// responses with exponential backoff. Five retries means an initial
			// request plus up to five recovery attempts before the agent loop stops.
			maxRetries: LLM_MAX_RETRIES,
		});
		_cachedBaseUrl = config.openaiBaseUrl;
		_cachedApiKey = config.openaiApiKey;
	}

	return openaiInstance;
}

export function getConfiguredModel(): string {
	return GlobalState.getInstance().config.model;
}

/** Completion surface for OpenAI-compatible model endpoints. */
export async function createModelCompletion(
	params: Omit<ChatCompletionCreateParamsNonStreaming, "model"> & {
		model?: string;
	},
	requestOptions?: OpenAI.RequestOptions,
): Promise<ChatCompletion> {
	const model = params.model ?? getConfiguredModel();
	return getOpenAIClient().chat.completions.create(
		{ ...params, model } as ChatCompletionCreateParamsNonStreaming,
		requestOptions,
	);
}

// 仅作测试用
export async function chatStateless(prompt: string): Promise<string | null> {
	try {
		const response = await createModelCompletion({
			model: getConfiguredModel(),
			messages: [{ role: "user", content: prompt }],
		});
		return response.choices[0]?.message?.content || null;
	} catch (error) {
		console.error("Failed to call LLM:", (error as Error).message);
		return null;
	}
}
