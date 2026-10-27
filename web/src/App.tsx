import { Fragment, useState, useEffect, useRef, useCallback } from "react";
import { BrainCircuit, Settings, History, Network } from "lucide-react";
import { ChatPanel } from "./components/ChatPanel";
import { CateDetails } from "./components/CateDetails";
import { CognitionIndexPanel } from "./components/CognitionIndexPanel";
import { GraphPanel } from "./components/GraphPanel";
import { SettingsPanel } from "./components/SettingsPanel";
import { Sidebar } from "./components/Sidebar";
import type { CausalScratchpad, PersistedChatTurn, SessionSwitchResult } from "./types";
import { formatEffectIntervention, presentEffect } from "./effectPresentation";
import { api } from "./api";
import { useI18n } from "./i18n";
import "./App.css";

type SidebarView = "settings" | "sessions";
type RightTab = "graph" | "cognition" | "scratchpad";

export default function App() {
  const { locale, setLocale, t } = useI18n();
  const [scratchpad, setScratchpad] = useState<CausalScratchpad | null>(null);
  const [sidebarView, setSidebarView] = useState<SidebarView | null>(null);
  const [rightTab, setRightTab] = useState<RightTab>("graph");
  /** Messages to pre-populate when switching sessions (last N from history). */
  const [initialMessages, setInitialMessages] = useState<PersistedChatTurn[]>([]);
  /** Changing this key causes ChatPanel to remount with new initialMessages. */
  const [chatKey, setChatKey] = useState(0);
  const [initialContextTokens, setInitialContextTokens] = useState<number | undefined>(undefined);
  const [initialContextWindow, setInitialContextWindow] = useState<number | undefined>(undefined);
  const [initialRunning, setInitialRunning] = useState(false);
  /** Active conversation session ID — used by Sidebar to highlight active session. */
  const [currentSessionId, setCurrentSessionId] = useState<string | undefined>(undefined);
  /** Editable title of the current session. */
  const [sessionTitle, setSessionTitle] = useState(locale === "zh" ? "新对话" : "New Conversation");
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [editTitleValue, setEditTitleValue] = useState("");
  /** Left pane width as percentage (drag-resizable, 25%–75%). Default 50%. */
  const [leftWidth, setLeftWidth] = useState(50);
  const dragRef = useRef(false);
  const dragStartX = useRef(0);
  const dragStartW = useRef(50);

  const onDragStart = useCallback((e: React.MouseEvent) => {
    dragRef.current = true;
    dragStartX.current = e.clientX;
    dragStartW.current = leftWidth;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, [leftWidth]);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragRef.current) return;
      const body = document.querySelector(".app-body") as HTMLElement;
      const total = body?.offsetWidth ?? window.innerWidth;
      const dx = e.clientX - dragStartX.current;
      const pct = dragStartW.current + (dx / total) * 100;
      setLeftWidth(Math.min(Math.max(pct, 25), 75));
    };
    const onUp = () => {
      if (!dragRef.current) return;
      dragRef.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, []);


  // Load initial session + scratchpad from server
  useEffect(() => {
    api.getActiveSession()
      .then((res) => {
        setCurrentSessionId(res.id);
        setSessionTitle(res.title);
        setInitialMessages(res.recentTurns ?? []);
        setScratchpad(res.scratchpad ?? null);
        setInitialContextTokens(res.contextTokens);
        setInitialContextWindow(res.contextWindow);
        setInitialRunning(res.isRunning === true);
        setChatKey((k) => k + 1);
      })
      .catch(() => {
        // Fallback
        api.getScratchpad()
          .then((pad) => { if (pad) setScratchpad(pad); })
          .catch(() => {/* server might not be running yet */});
      });
  }, []);

  const toggleSidebar = (view: SidebarView) => {
    setSidebarView((cur) => (cur === view ? null : view));
  };

  const handleSessionSwitch = (result: SessionSwitchResult) => {
    setCurrentSessionId(result.id);
    setSessionTitle(result.title);
    setInitialMessages(result.recentTurns ?? []);
    setScratchpad(result.scratchpad ?? null);
    setInitialContextTokens(result.contextTokens);
    setInitialContextWindow(result.contextWindow);
    setInitialRunning(result.isRunning === true);
    setChatKey((k) => k + 1);
  };

  const handleTitleEditStart = () => {
    setEditTitleValue(sessionTitle);
    setIsEditingTitle(true);
  };

  const handleTitleSave = async () => {
    const newTitle = editTitleValue.trim() || (locale === "zh" ? "新对话" : "New Conversation");
    setIsEditingTitle(false);
    setSessionTitle(newTitle);
    if (currentSessionId) {
      try { await api.renameSession(currentSessionId, newTitle); } catch { /* non-fatal */ }
    }
  };

  const handleTitleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") handleTitleSave();
    if (e.key === "Escape") setIsEditingTitle(false);
  };



  return (
    <div className="app-root">
      {/* ---- Top Nav ---- */}
      <header className="app-nav">
        <div className="app-nav-brand">
          <img src="/logo_horizontal.png" className="app-nav-logo" alt="CauProphet Logo" />
          <span className="app-nav-title">CauProphet</span>
        </div>

        {/* Inline-editable session title (center) */}
        <div className="app-nav-session-title">
          {isEditingTitle ? (
            <input
              id="session-title-input"
              className="app-session-title-input"
              value={editTitleValue}
              onChange={(e) => setEditTitleValue(e.target.value)}
              onBlur={handleTitleSave}
              onKeyDown={handleTitleKeyDown}
              autoFocus
              onFocus={(e) => e.target.select()}
              maxLength={80}
            />
          ) : (
            <button
              className="app-session-title-btn"
              onClick={handleTitleEditStart}
              title={t("nav.sessionTitleBtn")}
            >
              {sessionTitle}
              <span className="app-session-title-edit-hint">✏</span>
            </button>
          )}
        </div>

        {/* Scratchpad status pill */}
        {scratchpad && (
          <div className="app-nav-status">
            <span className="app-status-label">
              {scratchpad.title || (locale === "zh" ? "无标题" : "Untitled")}
            </span>
          </div>
        )}

        <div className="app-nav-actions">
          <button
            className="btn btn-ghost text-xs lang-btn"
            style={{ fontWeight: 600, padding: "0 8px", minWidth: 40, marginRight: 4 }}
            onClick={() => setLocale(locale === "en" ? "zh" : "en")}
            title={locale === "en" ? "Switch to Chinese" : "切换为英文"}
          >
            {locale === "en" ? "ZH" : "EN"}
          </button>
          <button
            className={`btn btn-ghost btn-icon ${sidebarView === "sessions" ? "nav-btn-active" : ""}`}
            onClick={() => toggleSidebar("sessions")}
            title={t("nav.historyTooltip")}
          >
            <History size={16} />
          </button>
          <button
            className={`btn btn-ghost btn-icon ${sidebarView === "settings" ? "nav-btn-active" : ""}`}
            onClick={() => toggleSidebar("settings")}
            title={t("nav.settingsTooltip")}
          >
            <Settings size={16} />
          </button>
        </div>
      </header>

      {/* ---- Main layout ---- */}
      <div className="app-body">
        {/* Left: Chat — key forces remount on session switch */}
        <div className="app-chat-pane" style={{ width: `${leftWidth}%` }}>
          <ChatPanel
            key={chatKey}
            onScratchpadUpdate={setScratchpad}
            initialMessages={initialMessages}
            initialContextTokens={initialContextTokens}
            initialContextWindow={initialContextWindow}
            sessionId={currentSessionId}
            initialRunning={initialRunning}
            onSessionReset={(sessionId) => {
              if (sessionId) {
                api.getActiveSession().then(handleSessionSwitch).catch(() => {
                  setCurrentSessionId(sessionId);
                  setSessionTitle(locale === "zh" ? "新对话" : "New Conversation");
                });
                return;
              }
              setCurrentSessionId(undefined);
              setSessionTitle(locale === "zh" ? "新对话" : "New Conversation");
              setInitialContextTokens(0);
              setInitialContextWindow(128000);
              setInitialRunning(false);
            }}
          />
        </div>

        {/* Drag handle */}
        <div
          className="app-resize-handle"
          onMouseDown={onDragStart}
          title={locale === "zh" ? "拖拽以调整大小" : "Drag to resize"}
        />
        {/* Right: Graph + (optional) scratchpad state */}
        <div className="app-right-pane">
          {/* Right tab bar */}
          <div className="app-right-tabs">
            <button
              className={`app-tab ${rightTab === "graph" ? "active" : ""}`}
              onClick={() => setRightTab("graph")}
            >
              <Network size={13} /> {t("nav.graphTab")}
            </button>
            <button
              className={`app-tab ${rightTab === "cognition" ? "active" : ""}`}
              onClick={() => setRightTab("cognition")}
            >
              <BrainCircuit size={13} /> {t("nav.cognitionTab")}
            </button>
            <button
              className={`app-tab ${rightTab === "scratchpad" ? "active" : ""}`}
              onClick={() => setRightTab("scratchpad")}
            >
              📋 {t("nav.stateTab")}
            </button>
          </div>

          <div className="app-right-content">
            {rightTab === "graph" && <GraphPanel scratchpad={scratchpad} />}
            {rightTab === "cognition" && <CognitionIndexPanel />}
            {rightTab === "scratchpad" && <ScratchpadState scratchpad={scratchpad} />}
          </div>
        </div>
      </div>

      {/* ---- Sidebar overlay (settings + sessions) ---- */}
      {sidebarView === "settings" ? (
        <div className="sidebar-overlay" onClick={() => setSidebarView(null)}>
          <div className="sidebar sidebar-settings" onClick={(e) => e.stopPropagation()}>
            <SettingsPanel onClose={() => setSidebarView(null)} />
          </div>
        </div>
      ) : (
        <Sidebar
          view={sidebarView === "sessions" ? "sessions" : null}
          onClose={() => setSidebarView(null)}
          onSessionSwitch={handleSessionSwitch}
          currentSessionId={currentSessionId}
        />
      )}
    </div>
  );
}

// ---- Minimal Scratchpad State panel ----
function ScratchpadState({ scratchpad }: { scratchpad: CausalScratchpad | null }) {
  const { locale, t } = useI18n();

  if (!scratchpad) {
    return (
      <div className="state-empty">
        <div className="text-muted text-sm">{t("state.empty")}</div>
      </div>
    );
  }

  const { variables, effects } = scratchpad;

  return (
    <div className="state-panel">
      {/* Header */}
      <div className="state-header">
        <div className="state-title">{scratchpad.title || (locale === "zh" ? "无标题" : "Untitled")}</div>
        {scratchpad.objective && (
          <div className="state-objective text-secondary text-sm">{scratchpad.objective}</div>
        )}
        {scratchpad.score !== undefined && (
          <div className="flex gap-2" style={{ marginTop: 6 }}>
            <span className="badge badge-green">BIC {scratchpad.score.toFixed(2)}</span>
          </div>
        )}
      </div>

      {/* Variables */}
      {variables.length > 0 && (
        <StateSection title={t("state.variables", { count: variables.length })}>
          <table className="state-table">
            <thead>
              <tr>
                <th>{t("state.name")}</th>
                <th>{t("state.role")}</th>
                <th>{t("state.type")}</th>
              </tr>
            </thead>
            <tbody>
              {variables.map((v) => (
                <tr key={v.name}>
                  <td className="font-mono" style={{ color: "var(--text-code)" }}>{v.name}</td>
                  <td>
                    <span
                      className="badge"
                      style={{
                        background: ROLE_BG[v.role] ?? "var(--bg-elevated)",
                        color: ROLE_FG[v.role] ?? "var(--text-muted)",
                      }}
                    >
                      {v.role}
                    </span>
                  </td>
                  <td className="text-muted text-sm">{v.type}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </StateSection>
      )}

      {/* Effect estimates */}
      {effects.length > 0 && (
        <StateSection title={t("state.effects", { count: effects.length })}>
          <table className="state-table">
            <thead>
              <tr>
                <th>{t("state.treatmentOutcome")}</th>
                <th>{t("state.effect")}</th>
                <th>{t("state.validity")}</th>
              </tr>
            </thead>
            <tbody>
              {effects.map((e, i) => {
                const presentation = presentEffect(e);
                const badgeClass = presentation.tone === "valid"
                  ? "badge-green"
                  : presentation.tone === "warning"
                    ? "badge-yellow"
                    : presentation.tone === "danger"
                      ? "badge-red"
                      : "badge-muted";
                return (
                  <Fragment key={`${e.treatment}-${e.outcome}-${i}`}>
                    <tr className={presentation.decisionGrade ? "" : "state-row-warn"}>
                      <td>
                        <div className="font-mono text-sm">{e.treatment} → {e.outcome}</div>
                        <div className="state-effect-meta">
                          {e.estimand ?? "total"} · {formatEffectIntervention(e)}
                          {e.effectModifiers?.length ? ` · CATE by ${e.effectModifiers.join(", ")}` : ""}
                        </div>
                      </td>
                      <td>
                        <div className="font-mono text-sm">{e.ate !== null && Number.isFinite(e.ate) ? e.ate.toFixed(4) : "N/A"}</div>
                        <div className="state-effect-meta">
                          {e.estimatorFamily ?? e.estimator ?? e.requestedMethod ?? t("state.unknownEstimator")}
                          {e.estimateRole ? ` · ${e.estimateRole}` : ""}
                        </div>
                      </td>
                      <td>
                        <span className={`badge ${badgeClass}`}>
                          {presentation.decisionGrade ? "✓" : "⚠"} {t(`state.effectStatus.${presentation.status}`)}
                        </span>
                      </td>
                    </tr>
                    {presentation.details.length > 0 && (
                      <tr className="state-effect-details">
                        <td colSpan={3}>⚠ {presentation.details.join("; ")}</td>
                      </tr>
                    )}
                    {e.cate && (
                      <tr className="state-cate-details">
                        <td colSpan={3}><CateDetails cate={e.cate} /></td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </StateSection>
      )}

      {/* Notes */}
      {scratchpad.notes.trim() && (
        <StateSection title={t("state.notes")}>
          <div className="text-sm" style={{ padding: "8px 12px", whiteSpace: "pre-wrap" }}>
            {scratchpad.notes}
          </div>
        </StateSection>
      )}

      {variables.length === 0 && effects.length === 0 && !scratchpad.notes.trim() && (
        <div className="state-placeholder">
          <div className="text-muted text-sm" style={{ padding: "20px 16px" }}>
            {t("state.placeholder")}
          </div>
        </div>
      )}
    </div>
  );
}

function StateSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="state-section">
      <div className="state-section-title">{title}</div>
      {children}
    </div>
  );
}

const ROLE_BG: Record<string, string> = {
  treatment:          "rgba(247,162,79,0.15)",
  outcome:            "rgba(79,247,162,0.15)",
  confounder:         "rgba(247,111,111,0.15)",
  mediator:           "rgba(180,140,255,0.15)",
  instrument:         "rgba(79,214,247,0.15)",
};

const ROLE_FG: Record<string, string> = {
  treatment:          "#f7a24f",
  outcome:            "#4ff7a2",
  confounder:         "#f76f6f",
  mediator:           "#b48cff",
  instrument:         "#4fd6f7",
};
