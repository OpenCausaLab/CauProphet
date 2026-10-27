import { z } from "zod";

export const ConfigSchema = z.object({
	openaiApiKey: z.string().optional(),
	openaiBaseUrl: z.string().default("https://api.openai.com/v1"),
	model: z.string().default("gpt-4o"),
	workspace: z.string().optional(),
	/** Model's max context window in tokens. Compression triggers at 80%. Default: 128k. */
	contextWindow: z.number().int().positive().default(128_000),
	/** Web search provider: "ddg" (DuckDuckGo, free, no key), "brave" (Brave Search API), or "bocha" (博查 AI). */
	searchProvider: z.enum(["ddg", "brave", "bocha"]).default("ddg"),
	/** Brave Search API key. Only needed when searchProvider is "brave". */
	braveApiKey: z.string().optional(),
	/** 博查 (Bocha AI) API key. Only needed when searchProvider is "bocha". */
	bochaApiKey: z.string().optional(),
});

export type Config = z.infer<typeof ConfigSchema>;
