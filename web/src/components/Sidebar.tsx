import { useState, useEffect } from "react";
import { History, X, Plus, ChevronRight, MessageSquare, Trash2 } from "lucide-react";
import { api } from "../api";
import type { SessionSummary, SessionSwitchResult } from "../types";
import { useI18n } from "../i18n";
import type { Locale } from "../i18n";
import "./Sidebar.css";

interface SidebarProps {
  view: "sessions" | null;
  onClose: () => void;
  /** Callback when user switches to a different session. */
  onSessionSwitch: (result: SessionSwitchResult) => void;
  currentSessionId?: string;
}

export function Sidebar({ view, onClose, onSessionSwitch, currentSessionId }: SidebarProps) {
  if (!view) return null;

  return (
    <div className="sidebar-overlay" onClick={onClose}>
      <div className="sidebar" onClick={(e) => e.stopPropagation()}>
        <div className="sidebar-close">
          <button className="btn btn-ghost btn-icon" onClick={onClose}>
            <X size={15} />
          </button>
        </div>
        <SessionsPanel
          onSessionSwitch={(result) => { onSessionSwitch(result); onClose(); }}
          currentSessionId={currentSessionId}
        />
      </div>
    </div>
  );
}

// ---- Sessions Panel ----
function SessionsPanel({
  onSessionSwitch,
  currentSessionId,
}: {
  onSessionSwitch: (result: SessionSwitchResult) => void;
  currentSessionId?: string;
}) {
  const { locale, t } = useI18n();
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [showCreate, setShowCreate] = useState(false);

  useEffect(() => {
    let disposed = false;
    const refresh = () => {
      api.listSessions()
        .then((next) => { if (!disposed) setSessions(next); })
        .catch(() => { if (!disposed) setSessions([]); })
        .finally(() => { if (!disposed) setLoading(false); });
    };
    refresh();
    const timer = window.setInterval(refresh, 2000);
    return () => { disposed = true; window.clearInterval(timer); };
  }, []);

  const handleSwitch = async (id: string) => {
    try {
      const result = await api.switchSession(id);
      onSessionSwitch(result);
      setSessions((prev) => prev.map((s) => ({ ...s, isActive: s.id === id })));
    } catch (e) {
      console.error("Failed to switch session:", e);
    }
  };

  const handleCreate = async () => {
    const title = newTitle.trim() || (locale === "zh" ? "新对话" : "New Conversation");
    setCreating(true);
    try {
      const session = await api.newSession(title);
      onSessionSwitch({ ...session, recentTurns: [], scratchpad: null });
      setSessions((prev) => [
        { id: session.id, title: session.title, createdAt: session.createdAt, updatedAt: session.updatedAt, turnCount: 0, isActive: true },
        ...prev.map((s) => ({ ...s, isActive: false })),
      ]);
      setShowCreate(false);
      setNewTitle("");
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      await api.deleteSession(id);
      const remaining = sessions.filter((s) => s.id !== id);
      setSessions(remaining);
      // If we deleted the active session, switch to another or create a fresh one
      if (id === currentSessionId) {
        if (remaining.length > 0) {
          handleSwitch(remaining[0]!.id);
        } else {
          handleCreate();
        }
      }
    } catch (err) {
      console.error("Failed to delete session:", err);
    }
  };

  return (
    <div className="sessions-panel">
      <div className="sessions-header">
        <div className="flex items-center gap-2">
          <History size={14} style={{ color: "var(--accent)" }} />
          <span className="font-semibold" style={{ fontSize: 14 }}>{t("sidebar.conversations")}</span>
        </div>
        <button className="btn btn-ghost" style={{ fontSize: 12 }} onClick={() => setShowCreate((v) => !v)}>
          <Plus size={13} /> {t("sidebar.new")}
        </button>
      </div>

      {showCreate && (
        <div className="sessions-create">
          <input
            className="input"
            placeholder={t("sidebar.titlePlaceholder")}
            value={newTitle}
            onChange={(e) => setNewTitle(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") handleCreate(); }}
            autoFocus
          />
          <div className="flex gap-2">
            <button
              className="btn btn-primary flex-1"
              onClick={handleCreate}
              disabled={creating}
            >
              {creating ? <span className="spinner" /> : <Plus size={13} />}
              {t("sidebar.create")}
            </button>
            <button className="btn btn-ghost" onClick={() => setShowCreate(false)}>{t("cancel")}</button>
          </div>
        </div>
      )}

      <div className="sessions-list">
        {loading && (
          <div className="sessions-loading">
            <span className="spinner" /> {t("loading")}
          </div>
        )}
        {!loading && sessions.length === 0 && (
          <div className="sessions-empty">
            <MessageSquare size={24} style={{ opacity: 0.2, marginBottom: 8 }} />
            {t("sidebar.empty")}
          </div>
        )}
        {sessions.map((s) => (
          <button
            key={s.id}
            className={`session-item ${s.id === currentSessionId ? "active" : ""}`}
            onClick={() => handleSwitch(s.id)}
          >
            <div className="session-item-header">
              <span className="session-title truncate">{s.title || (locale === "zh" ? "无标题" : "Untitled")}</span>
              <div className="session-item-actions">
                {s.isRunning && <span className="session-running-indicator" title={locale === "zh" ? "正在运行" : "Running"} aria-label={locale === "zh" ? "正在运行" : "Running"}><span className="spinner" /></span>}
                <button
                  className="btn btn-ghost btn-icon session-delete-btn"
                  onClick={(e) => handleDelete(e, s.id)}
                  title={t("sidebar.deleteTooltip")}
                >
                  <Trash2 size={13} />
                </button>
                <ChevronRight size={13} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
              </div>
            </div>
            <div className="session-meta">
              <span className="badge badge-muted">
                {s.turnCount === 1 ? t("sidebar.turnMsg", { count: s.turnCount }) : t("sidebar.turnMsgs", { count: s.turnCount })}
              </span>
              <span className="text-sm text-muted">{formatDate(s.updatedAt, locale, t)}</span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

function formatDate(iso: string, locale: Locale, t: any): string {
  const d = new Date(iso);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffDays = Math.floor(diffMs / 86400000);
  if (diffDays === 0) return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (diffDays === 1) return locale === "zh" ? "昨天" : "Yesterday";
  if (diffDays < 7) return t("sidebar.daysAgo", { count: diffDays });
  return d.toLocaleDateString();
}
