/**
 * api.ts — REST API helpers for the CausalClaw backend server.
 */
import type {
  AppConfig,
  CausalBelief,
  CausalScratchpad,
  CausalCognitionIndexResponse,
  BeliefSummary,
  ScratchpadSummary,
  Session,
  SessionSummary,
	SessionSwitchResult,
} from "./types";

const BASE = "";  // relative URL — proxied by Vite dev server to localhost:3001

async function apiFetch<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, opts);
  if (!res.ok) throw new Error(`API ${path} failed: ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  // ---- Config ----
  getConfig: () => apiFetch<AppConfig>("/api/config"),

  saveConfig: (cfg: Partial<AppConfig>) =>
    apiFetch<{ ok: boolean }>("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
    }),

  // ---- Sessions ----
  getActiveSession: () => apiFetch<SessionSwitchResult>("/api/session/active"),

  listSessions: () => apiFetch<SessionSummary[]>("/api/sessions"),

	newSession: (title?: string) =>
    apiFetch<SessionSwitchResult>("/api/session/new", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ title }),
    }),

  switchSession: (id: string) =>
    apiFetch<SessionSwitchResult>(`/api/session/${id}`, { method: "POST" }),

  renameSession: (id: string, title: string) =>
    apiFetch<Session>(`/api/session/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    }),

  deleteSession: (id: string) =>
    apiFetch<{ ok: boolean }>(`/api/session/${id}`, { method: "DELETE" }),

  // ---- Causal Belief / Scratchpad ----
  getCausalBelief: () => apiFetch<CausalBelief | null>("/api/causal-belief"),
  getScratchpad: () => apiFetch<CausalScratchpad | null>("/api/scratchpad"),

  getCausalCognitionIndex: () =>
    apiFetch<CausalCognitionIndexResponse>("/api/causal-cognition/index"),

  listCausalBeliefs: () => apiFetch<BeliefSummary[]>("/api/causal-beliefs"),
  listScratchpads: () => apiFetch<ScratchpadSummary[]>("/api/scratchpads"),

  newCausalBelief: (title: string, objective: string) =>
    apiFetch<CausalBelief>("/api/causal-belief/new", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, objective }),
    }),
  newScratchpad: (title: string, objective: string) =>
    apiFetch<CausalScratchpad>("/api/scratchpad/new", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, objective }),
    }),

  switchCausalBelief: (id: string) =>
    apiFetch<CausalBelief>(`/api/causal-belief/${id}`, { method: "POST" }),
  switchScratchpad: (id: string) =>
    apiFetch<CausalScratchpad>(`/api/scratchpad/${id}`, { method: "POST" }),

  // ---- Compact ----
  compactSession: () =>
    apiFetch<{ ok: boolean; ran: boolean; contextTokens?: number; contextWindow?: number }>("/api/session/compact", { method: "POST" }),
};
