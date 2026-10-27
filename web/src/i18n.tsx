import { createContext, useContext, useState } from "react";
import type { ReactNode } from "react";

// 1. Define translation dictionaries
const translations = {
  zh: {
    // Common
    cancel: "取消",
    save: "保存",
    saved: "已保存",
    failed: "失败",
    loading: "加载中…",

    // Top Nav
    "nav.sessionTitleBtn": "点击重命名对话",
    "nav.historyTooltip": "历史对话",
    "nav.settingsTooltip": "设置",
    "nav.graphTab": "因果图",
    "nav.cognitionTab": "因果认知",
    "nav.stateTab": "状态",

    // State panel
    "state.empty": "无活动因果信念 (Causal Belief)",
    "state.variables": "变量 ({count})",
    "state.effects": "效应估计 ({count})",
    "state.name": "名称",
    "state.role": "角色",
    "state.type": "类型",
    "state.treatmentOutcome": "干预因子 → 结果因子",
    "state.ate": "ATE",
    "state.effect": "效应 / 估计器",
    "state.validity": "结论有效性",
    "state.unknownEstimator": "未知估计器",
    "state.effectStatus.decision-grade": "可用于决策",
    "state.effectStatus.caution": "谨慎使用",
    "state.effectStatus.sensitivity-only": "仅敏感性分析",
    "state.effectStatus.inconclusive": "结论不确定",
    "state.effectStatus.refuted": "诊断未通过",
    "state.effectStatus.unknown": "有效性未知",
    "state.refuted": "是否已反驳?",
    "state.yes": "是",
    "state.no": "否",
    "state.notes": "备注",
    "state.placeholder": "智能体在工作时会自动填充变量、边和效应估计。",

    // Chat Panel
    "chat.connected": "已连接",
    "chat.connecting": "连接中…",
    "chat.contextPct": "上下文: {pct}%",
    "chat.compactTooltip": "压缩上下文 — 总结历史消息以释放上下文窗口",
    "chat.newSessionTooltip": "新会话",
    "chat.historyBanner": "显示此会话的最后 {count} 条消息",
    "chat.emptyPlaceholder": "知其然，更知其所以然。无论是生活中的十万个为什么、深度逻辑推演，还是复杂决策与数据探索，尽管向我提问。",
    "chat.placeholder": "向 CauProphet 提问，或探索任何一个为什么…",
    "chat.placeholderBusy": "智能体正在思考…",
    "chat.inputFooter": "回车发送 · Shift+Enter 换行",
    "chat.uploadTooltip": "上传文件到工作区",
    "chat.copyTooltip": "复制原始回复",
    "chat.interruptTooltip": "中断当前处理",
    "chat.thinking": "思考过程",
    "chat.compactWarning": "上下文历史已压缩。概要已添加到对话的最前面。",
    "chat.compactMinTurnsWarning": "对话轮数不足，暂不需要压缩（需要超过 {count} 轮最近的用户发言）。",
    "chat.interrupted": "此回复被中断，部分结果已保存。您可以继续发送消息让智能体接着完成。",
    "chat.suggestion1": "💡 为什么有些习惯很难彻底改掉？",
    "chat.suggestion2": "🌧️ 为什么夏天雷雨过后空气格外清新？",
    "chat.suggestion3": "⏳ 如果推行每周4天工作制，会带来什么影响？",
    "chat.suggestion4": "📊 帮我探索手头数据背后的真实机制与驱动因子",

    // Graph Panel
    "graph.title": "因果图",
    "graph.vars": "{count} 变量",
    "graph.edges": "{count} 边",
    "graph.score": "评分: {score}",
    "graph.empty": "暂无因果图",
    "graph.emptySub": "当智能体发现因果结构时，图会显示在这里",
    "graph.directed": "有向边",
    "graph.undirected": "无向边",
    "graph.bidirected": "双向边",
    "graph.partial": "部分有向",

    // Settings Panel
    "settings.title": "设置",
    "settings.subtitle": "配置您的 AI 后端和工作空间。",
    "settings.llmProvider": "大语言模型提供商",
    "settings.baseUrl": "接口根地址 (Base URL)",
    "settings.baseUrlHint": "兼容 OpenAI 的 API 端点（例如通过 LM Studio 或 Ollama 运行的本地模型）",
    "settings.apiKey": "API 密钥 (API Key)",
    "settings.apiKeyStoredHint": "保存在本地的 ~/.cauprophet/config.json",
    "settings.hideKey": "隐藏密钥",
    "settings.showKey": "显示密钥",
    "settings.model": "模型 (Model)",
    "settings.customModel": "— 自定义模型 ID…",
    "settings.customModelPlaceholder": "例如 llama-3.3-70b-instruct",
    "settings.presets": "预设",
    "settings.workspace": "工作空间 (Workspace)",
    "settings.workspacePath": "工作空间路径 (Workspace Path)",
    "settings.workspacePathHint": "智能体运行的项目的绝对路径",
    "settings.context": "上下文 (Context)",
    "settings.contextWindow": "上下文窗口大小 (Tokens)",
    "settings.contextWindowHint": "设置为您模型的最大上下文长度。当使用率达到 80% 时，CauProphet 会自动压缩历史记录。",
    "settings.testConnection": "测试连接",
    "settings.loadingConfig": "正在加载配置…",
    "settings.closeSettings": "关闭设置",
    "settings.webSearch": "网页搜索 (Web Search)",
    "settings.searchProvider": "搜索引擎",
    "settings.searchProviderHint": "选择网页搜索后端。DuckDuckGo 免费无需密钥，Brave 和博查需要 API 密钥。",
    "settings.braveApiKey": "Brave Search API 密钥",
    "settings.bochaApiKey": "博查 API 密钥",

    // Sidebar
    "sidebar.conversations": "历史对话",
    "sidebar.new": "新建",
    "sidebar.titlePlaceholder": "对话标题（可选）",
    "sidebar.create": "创建",
    "sidebar.empty": "暂无对话。",
    "sidebar.deleteTooltip": "删除对话",
    "sidebar.yesterday": "昨天",
    "sidebar.daysAgo": "{count}天前",
    "sidebar.turnMsg": "{count} 条消息",
    "sidebar.turnMsgs": "{count} 条消息",
  },
  en: {
    // Common
    cancel: "Cancel",
    save: "Save",
    saved: "Saved",
    failed: "Failed",
    loading: "Loading…",

    // Top Nav
    "nav.sessionTitleBtn": "Click to rename conversation",
    "nav.historyTooltip": "Session history",
    "nav.settingsTooltip": "Settings",
    "nav.graphTab": "Graph",
    "nav.cognitionTab": "Cognition",
    "nav.stateTab": "State",

    // State panel
    "state.empty": "No active Causal Belief",
    "state.variables": "Variables ({count})",
    "state.effects": "Effect Estimates ({count})",
    "state.name": "Name",
    "state.role": "Role",
    "state.type": "Type",
    "state.treatmentOutcome": "Treatment → Outcome",
    "state.ate": "ATE",
    "state.effect": "Effect / estimator",
    "state.validity": "Decision validity",
    "state.unknownEstimator": "Unknown estimator",
    "state.effectStatus.decision-grade": "Decision-grade",
    "state.effectStatus.caution": "Use with caution",
    "state.effectStatus.sensitivity-only": "Sensitivity only",
    "state.effectStatus.inconclusive": "Inconclusive",
    "state.effectStatus.refuted": "Diagnostic failed",
    "state.effectStatus.unknown": "Validity unknown",
    "state.refuted": "Refuted?",
    "state.yes": "Yes",
    "state.no": "No",
    "state.notes": "Notes",
    "state.placeholder": "The agent will populate variables, edges, and effect estimates as it works.",

    // Chat Panel
    "chat.connected": "Connected",
    "chat.connecting": "Connecting…",
    "chat.contextPct": "Context: {pct}%",
    "chat.compactTooltip": "Compact context — summarise old messages to free up context window",
    "chat.newSessionTooltip": "New session",
    "chat.historyBanner": "Showing last {count} messages from this session",
    "chat.emptyPlaceholder": "Going beyond plausibility by knowing why. Ask me anything from curious 'why' questions to deep logical reasoning, data analysis, and decision exploration.",
    "chat.placeholder": "Message CauProphet, or explore any 'why' question…",
    "chat.placeholderBusy": "Agent is thinking…",
    "chat.inputFooter": "Enter to send · Shift+Enter for newline",
    "chat.uploadTooltip": "Upload file to workspace",
    "chat.copyTooltip": "Copy raw response",
    "chat.interruptTooltip": "Interrupt current processing",
    "chat.thinking": "Thinking",
    "chat.compactWarning": "Context history has been compressed. A summary has been prepended to the conversation.",
    "chat.compactMinTurnsWarning": "Not enough conversation turns to compact yet (need more than {count} recent user turns).",
    "chat.interrupted": "This response was interrupted. Partial results have been saved. You can send another message to continue.",
    "chat.suggestion1": "💡 Why are some habits so hard to break?",
    "chat.suggestion2": "🌧️ Why does the air smell fresh after a thunderstorm?",
    "chat.suggestion3": "⏳ What happens if a company adopts a 4-day work week?",
    "chat.suggestion4": "📊 Help me uncover underlying mechanisms in my dataset",

    // Graph Panel
    "graph.title": "Causal Graph",
    "graph.vars": "{count} vars",
    "graph.edges": "{count} edges",
    "graph.score": "score: {score}",
    "graph.empty": "No causal graph yet",
    "graph.emptySub": "The graph will appear as the agent discovers causal structure",
    "graph.directed": "Directed",
    "graph.undirected": "Undirected",
    "graph.bidirected": "Bidirected",
    "graph.partial": "Partial",

    // Settings Panel
    "settings.title": "Settings",
    "settings.subtitle": "Configure your AI backend and workspace.",
    "settings.llmProvider": "LLM Provider",
    "settings.baseUrl": "Base URL",
    "settings.baseUrlHint": "OpenAI-compatible API endpoint (e.g., for local models via LM Studio or Ollama)",
    "settings.apiKey": "API Key",
    "settings.apiKeyStoredHint": "Stored locally in ~/.cauprophet/config.json",
    "settings.hideKey": "Hide key",
    "settings.showKey": "Show key",
    "settings.model": "Model",
    "settings.customModel": "— Custom model ID…",
    "settings.customModelPlaceholder": "e.g. llama-3.3-70b-instruct",
    "settings.presets": "Presets",
    "settings.workspace": "Workspace",
    "settings.workspacePath": "Workspace Path",
    "settings.workspacePathHint": "Absolute path to the project directory the agent will operate in",
    "settings.context": "Context",
    "settings.contextWindow": "Context Window Size (tokens)",
    "settings.contextWindowHint": "Set to your model's maximum context length. CauProphet will automatically compress history when usage reaches 80%.",
    "settings.testConnection": "Test connection",
    "settings.loadingConfig": "Loading configuration…",
    "settings.closeSettings": "Close settings",
    "settings.webSearch": "Web Search",
    "settings.searchProvider": "Search Engine",
    "settings.searchProviderHint": "Choose the web search backend. DuckDuckGo is free and requires no key. Brave and Bocha require API keys.",
    "settings.braveApiKey": "Brave Search API Key",
    "settings.bochaApiKey": "Bocha API Key",

    // Sidebar
    "sidebar.conversations": "Conversations",
    "sidebar.new": "New",
    "sidebar.titlePlaceholder": "Conversation title (optional)",
    "sidebar.create": "Create",
    "sidebar.empty": "No conversations yet.",
    "sidebar.deleteTooltip": "Delete conversation",
    "sidebar.yesterday": "Yesterday",
    "sidebar.daysAgo": "{count}d ago",
    "sidebar.turnMsg": "{count} msg",
    "sidebar.turnMsgs": "{count} msgs",
  }
};

export type Locale = "zh" | "en";
export type TranslationKeys = keyof typeof translations.zh;

interface I18nContextType {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: TranslationKeys, variables?: Record<string, string | number>) => string;
}

const I18nContext = createContext<I18nContextType | null>(null);

const STORAGE_KEY = "causalclaw_locale";

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "zh" || saved === "en") return saved;
    // Use English on first launch; preserve an explicitly saved language choice.
    return "en";
  });

  const setLocale = (l: Locale) => {
    setLocaleState(l);
    localStorage.setItem(STORAGE_KEY, l);
  };

  const t = (key: TranslationKeys, variables?: Record<string, string | number>) => {
    let message = translations[locale][key] || translations.en[key] || String(key);
    if (variables) {
      Object.entries(variables).forEach(([k, v]) => {
        message = message.replace(`{${k}}`, String(v));
      });
    }
    return message;
  };

  return (
    <I18nContext.Provider value={{ locale, setLocale, t }}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n() {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error("useI18n must be used within an I18nProvider");
  }
  return context;
}
