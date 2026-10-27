import { researchSearch } from "@/tools/research/hybrid_research";

// --- 通用网络研究工具 (General Web Research Tool) ---
// 接口仍只有一个 query 字符串。主 Agent 已经理解用户问题，因此由它在
// query 中提供结构化来源计划；工具负责验证、搜索、证据整理与 cutoff 审计。

export const webSearchTool = {
	type: "function" as const,
	function: {
		name: "web_search",
		description:
			'General web research. Before calling, choose 1–5 specific source domains based on the question; do not guess the answer. Pass query as a compact JSON string, for example {"question":"full question","intent":"minimal search terms with private context removed","sources":["official.example","specialist.example"],"freshness":"live|recent|stable|historical","risk":"standard|high-stakes"}. Use hostnames only in sources. Prioritize the official institution, product maintainer, or regulator responsible for the fact, then independent specialist sources. The tool independently checks authority, access method, and cutoff, and fetches trusted text where direct access is allowed. Mark medical, legal, financial, and security questions high-stakes. Plain strings are supported for compatibility and use a generic discovery fallback.',
		parameters: {
			type: "object",
			properties: {
				query: {
					type: "string",
					description:
						"Prefer a compact JSON string with question, a sources hostname array, freshness, and risk. Optional intent should contain minimal search terms with private context removed. Do not insert unverified values, versions, prices, availability, or conclusions. Complex tasks may add constraints, unresolvedFacts, queries, and stopConditions.",
				},
			},
			required: ["query"],
		},
	},
};

/** 与既有 orchestrator 完全兼容的单参数入口。 */
export async function webSearch(query: string): Promise<string> {
	return researchSearch(query);
}
