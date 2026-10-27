// Shared types mirroring src/memory/scratchpad.ts and server protocol.
// Kept in sync manually — no build-time sharing needed.

export interface CausalVariable {
  name: string;
  description: string;
  type: "continuous" | "categorical" | "binary" | "unknown";
  role: "treatment" | "outcome" | "confounder" | "mediator" | "instrument" | "other";
  source: string;
}

export interface CausalEdge {
  from: string;
  to: string;
  edgeType: "directed" | "undirected" | "bidirected" | "partially_directed";
  confidence: "high" | "medium" | "low";
  method?: string;
  rationale?: string;
}

export interface EffectIntervention {
  type: string;
  shiftDelta?: number;
  treatmentValue?: string | number | boolean;
  controlValue?: string | number | boolean;
  targetPopulation?: string;
  nRows?: number;
}

export interface EffectValidity {
  status: "valid" | "caution" | "sensitivity_only" | "inconclusive" | "unknown";
  decisionGrade: boolean;
  reasons: string[];
}

export interface EffectEstimate {
  treatment: string;
  outcome: string;
  ate: number | null;
  ci_lower?: number;
  ci_upper?: number;
  p_value?: number;
  estimator?: string;
  requestedMethod?: string;
  estimatorFamily?: string;
  estimateRole?: string;
  functionalForm?: string;
  estimand?: string;
  intervention?: EffectIntervention;
  adjustmentSet?: string[];
  effectModifiers?: string[];
  cate?: Record<string, unknown>;
  modelComparison?: Record<string, unknown> | Array<Record<string, unknown>>;
  supportDiagnostics?: Record<string, unknown>;
  validity?: EffectValidity;
  uncertainty?: Record<string, unknown>;
  warnings?: string[];
  refuted: boolean;
  refutation_note?: string;
}

export interface CausalBelief {
  id: string;
  title: string;
  objective: string;
  dataPath?: string;
  variables: CausalVariable[];
  edges: CausalEdge[];
  effects: EffectEstimate[];
  notes: string;
  score?: number;
  lastUpdated: string;
}

export type CausalScratchpad = CausalBelief;

export type AgentMode = "full";

export interface BeliefSummary {
  id: string;
  title: string;
  objective: string;
  lastUpdated: string;
  isActive: boolean;
}

export type ScratchpadSummary = BeliefSummary;

export interface PersistedToolCallRecord {
  tool: string;
  detail?: string;
}

/** A persisted chat turn stored in a Session. */
export interface PersistedChatTurn {
  role: "user" | "assistant";
  content: string | null;
  timestamp: string;
  think?: string;
  toolCalls?: PersistedToolCallRecord[];
  blocks?: MessageBlock[];
  /**
   * Lifecycle status of this turn.
   * - `"streaming"`: The agent loop was interrupted before completion.
   * - `"done"`: The agent loop completed normally.
   * When absent, the turn is assumed to be done (backwards compatibility).
   */
  status?: "streaming" | "done";
}

/** A conversation session (independent from scratchpads). */
export interface Session {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  agentMode?: AgentMode;
  turns: PersistedChatTurn[];
  activeScratchpadId?: string;
  contextTokens?: number;
  compactSummary?: {
    generatedAt: string;
    coversUpToIndex: number;
    content: string;
    turnCount: number;
  };
  historyStartIndex?: number;
}

export interface SessionSummary {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  turnCount: number;
  isActive: boolean;
  isRunning?: boolean;
  agentMode?: AgentMode;
}

/** Response when switching to an existing session (includes last 20 turns + linked scratchpad). */
export interface SessionSwitchResult extends Session {
  recentTurns: PersistedChatTurn[];
  scratchpad: CausalScratchpad | null;
  contextTokens?: number;
  contextWindow?: number;
  isRunning?: boolean;
}

export interface AppConfig {
  openaiBaseUrl: string;
  openaiApiKey: string;
  model: string;
  workspace: string;
  contextWindow: number;
  searchProvider: "ddg" | "brave" | "bocha";
  braveApiKey?: string;
  bochaApiKey?: string;
}

export interface CausalCognitionIndexConcept {
  id: string;
  label: string;
  aliases: string[];
  keywords: string[];
  description?: string;
  activeIncoming: number;
  activeOutgoing: number;
}

export type CausalCognitionRelationVerdict =
  | "hypothesized"
  | "supported"
  | "contradicted";

export type CausalCognitionRelationConfidence = "low" | "medium" | "high";

export interface CausalCognitionRelationClaim {
  episodeId: string;
  verdict: CausalCognitionRelationVerdict;
  confidence?: CausalCognitionRelationConfidence;
  note?: string;
}

export interface CausalCognitionRelationAssessment {
  priorLogOdds: number;
  supportLogBayesFactor: number;
  contradictionLogBayesFactor: number;
  logOdds: number;
  posterior: number;
  active: boolean;
}

export interface CausalCognitionIndexRelation {
  id: string;
  causeId: string;
  effectId: string;
  mechanism?: string;
  contexts: string[];
  lag?: string;
  claims: CausalCognitionRelationClaim[];
  assessment: CausalCognitionRelationAssessment;
}

export interface CausalCognitionIndexResponse {
  source: "active";
  sourceLabel: string;
  storageRelativePath: string;
  updatedAt: string | null;
  schemaVersion: number;
  counts: {
    concepts: number;
    relationVariants: number;
    activeRelations: number;
    relationClaims: number;
    episodes: number;
  };
  concepts: CausalCognitionIndexConcept[];
  relations: CausalCognitionIndexRelation[];
}

// WebSocket message types — must match server/index.ts protocol
export type ServerMessage =
  | { type: "think"; content: string }
  | { type: "content"; content: string }
  | { type: "tool_start"; tool: string; detail?: string }
  | { type: "tool_end"; tool: string }
  | { type: "warn"; content: string }
  | { type: "scratchpad_update"; scratchpad: CausalScratchpad }
  | { type: "done"; scratchpad: CausalScratchpad | null; contextTokens: number; contextWindow: number; sessionId?: string }
  | { type: "job"; sessionId: string; state: "running" | "done" | "error" | "aborted"; startedAt?: string; error?: string }
  | { type: "error"; content: string };

export interface ClientChatMessage {
  type: "chat";
  text: string;
  streaming: boolean;
  sessionId?: string;
}

export interface ClientResetMessage {
  type: "reset";
  objective?: string;
  sessionId?: string;
  agentMode?: AgentMode;
}

export interface ClientWatchMessage {
  type: "watch";
  sessionId: string;
}

export type ClientMessage = ClientChatMessage | ClientResetMessage | ClientWatchMessage;

// UI chat message (rendered in the chat panel)
export type ChatRole = "user" | "assistant";

export interface ThinkBlock { kind: "think"; content: string; }
export interface ContentBlock { kind: "content"; content: string; }
export interface ToolBlock { kind: "tool"; tool: string; state: "running" | "done"; detail?: string; }
export interface WarnBlock { kind: "warn"; content: string; }
export type MessageBlock = ThinkBlock | ContentBlock | ToolBlock | WarnBlock;

export interface ChatMessage {
  id: string;
  role: ChatRole;
  blocks: MessageBlock[];
  timestamp: number;
}
