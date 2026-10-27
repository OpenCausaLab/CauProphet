import React, { useState, useRef, useEffect, useCallback, useId } from "react";
import { Send, RotateCcw, ChevronDown, ChevronRight, Wrench, Archive, Paperclip, Copy, Check, Square } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import mermaid from "mermaid";
import "katex/dist/katex.min.css";
import type { ChatMessage, ServerMessage, CausalScratchpad, MessageBlock, PersistedChatTurn } from "../types";
import { api } from "../api";
import { useI18n } from "../i18n";
import "./ChatPanel.css";

// Initialize mermaid with custom variables matching index.css theme colors
mermaid.initialize({
  startOnLoad: false,
  theme: "dark",
  securityLevel: "loose",
  themeVariables: {
    background: "#1a1e28", // matches --bg-surface
    primaryColor: "#20263a", // matches --bg-elevated
    primaryTextColor: "#e8ecf4", // matches --text-primary
    secondaryColor: "#13161e", // matches --bg-panel
    lineColor: "#8b94ab", // matches --text-secondary
    signalColor: "#4f6ef7", // matches --accent
    signalTextColor: "#e8ecf4",
    transitionDuration: "0.15s",
  }
});

interface MermaidProps {
  chart: string;
}

const Mermaid = React.memo(function Mermaid({ chart }: MermaidProps) {
  const [svg, setSvg] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const uniqueId = useId().replace(/:/g, "");
  const containerId = useRef(`mermaid-${uniqueId}`);

  useEffect(() => {
    let active = true;
    async function renderChart() {
      try {
        const { svg: renderedSvg } = await mermaid.render(containerId.current, chart);
        if (active) {
          setSvg(renderedSvg);
          setError(null);
        }
      } catch (err: unknown) {
        console.error("Mermaid render error:", err);
        if (active) {
          setError("Failed to render Mermaid chart. Check syntax.");
        }
        const element = document.getElementById(containerId.current);
        if (element) element.remove();
      }
    }
    renderChart();
    return () => {
      active = false;
    };
  }, [chart]);

  if (error) {
    return (
      <div className="mermaid-error">
        <pre style={{ margin: 0, padding: "8px", background: "rgba(247, 111, 111, 0.1)", border: "1px solid var(--red)", borderRadius: "var(--radius-md)", color: "var(--red)", fontSize: "11px", overflowX: "auto" }}>{chart}</pre>
      </div>
    );
  }

  if (!svg) {
    return (
      <div className="mermaid-loading" style={{ display: "flex", justifyContent: "center", padding: "12px" }}>
        <span className="spinner" />
      </div>
    );
  }

  return (
    <div 
      className="mermaid-chart" 
      style={{ 
        width: "100%", 
        overflowX: "auto", 
        padding: "10px", 
        background: "var(--bg-panel)", 
        borderRadius: "var(--radius-md)", 
        border: "1px solid var(--border)",
        margin: "8px 0"
      }} 
      dangerouslySetInnerHTML={{ __html: svg }} 
    />
  );
});

const markdownComponents = {
  pre({ children, ...props }: React.HTMLProps<HTMLPreElement>) {
    const codeElement = React.Children.toArray(children).find(
      (child) => React.isValidElement(child) && child.type === "code"
    );
    if (React.isValidElement(codeElement)) {
      const className = (codeElement.props as { className?: string }).className || "";
      const match = /language-(\w+)/.exec(className);
      if (match && match[1] === "mermaid") {
        const chart = String((codeElement.props as { children?: React.ReactNode }).children).replace(/\n$/, "");
        return <Mermaid chart={chart} />;
      }
    }
    return <pre {...props}>{children}</pre>;
  }
};

const WS_URL = `ws://${window.location.host}/ws`;

interface ChatPanelProps {
  onScratchpadUpdate: (pad: CausalScratchpad | null) => void;
  /** Pre-loaded recent messages when restoring a session. */
  initialMessages?: PersistedChatTurn[];
  /** Called when the user starts a new session (reset button). */
  onSessionReset?: (sessionId?: string) => void;
  initialContextTokens?: number;
  initialContextWindow?: number;
  /** Session id used to route this socket to a detached worker. */
  sessionId?: string;
  /** True when the selected session already has a worker in progress. */
  initialRunning?: boolean;
}

let msgIdCounter = 0;
const newId = () => String(++msgIdCounter);

/** Convert persisted turns into renderable ChatMessage objects. */
function turnsToChatMessages(turns: PersistedChatTurn[], running = false): ChatMessage[] {
  return turns
    .filter((t) => t.content !== null || t.think || t.toolCalls?.length || t.blocks?.length || (running && t.role === "assistant" && t.status === "streaming"))
    .map((t) => {
      const isInterrupted = !running && t.role === "assistant" && t.status === "streaming";

      if (t.blocks && t.blocks.length > 0) {
        // Structured block-based persistence (supports intermediate text/thoughts/tool timelines)
        const blocks = [...t.blocks];
        if (isInterrupted) {
          blocks.push({ kind: "warn", content: "__INTERRUPTED__" });
        }
        return {
          id: newId(),
          role: t.role,
          blocks,
          timestamp: new Date(t.timestamp).getTime(),
        };
      }

      // [ABANDONED FALLBACK] Legacy non-sequential representation
      // Parses flat text fields (think/toolCalls/content) which do not preserve exact timeline order.
      const blocks: MessageBlock[] = [];
      if (t.think) {
        blocks.push({ kind: "think", content: t.think });
      }
      for (const tc of t.toolCalls ?? []) {
        blocks.push({ kind: "tool", tool: tc.tool, state: "done", detail: tc.detail });
      }
      if (t.content) {
        blocks.push({ kind: "content", content: t.content });
      }
      if (isInterrupted) {
        blocks.push({ kind: "warn", content: "__INTERRUPTED__" });
      }
      return {
        id: newId(),
        role: t.role,
        blocks,
        timestamp: new Date(t.timestamp).getTime(),
      };
    });
}

export function ChatPanel({
  onScratchpadUpdate,
  initialMessages = [],
  onSessionReset,
  initialContextTokens,
  initialContextWindow,
  sessionId,
  initialRunning = false,
}: ChatPanelProps) {
  const { locale, t } = useI18n();
  const [messages, setMessages] = useState<ChatMessage[]>(() => turnsToChatMessages(initialMessages, initialRunning));
  const [hasHistory] = useState(() => initialMessages.length > 0);
  const [input, setInput] = useState("");
  const [connected, setConnected] = useState(false);
  const [agentBusy, setAgentBusy] = useState(initialRunning);
  const [compacting, setCompacting] = useState(false);
  const [compactDone, setCompactDone] = useState(false);
  const [tokenUsage, setTokenUsage] = useState<{ used: number; limit: number } | null>(() => {
    if (initialContextTokens !== undefined && initialContextWindow !== undefined && initialContextWindow > 0) {
      return { used: initialContextTokens, limit: initialContextWindow };
    }
    return null;
  });

  useEffect(() => {
    if (initialContextTokens !== undefined && initialContextWindow !== undefined && initialContextWindow > 0) {
      setTokenUsage({ used: initialContextTokens, limit: initialContextWindow });
    } else {
      setTokenUsage(null);
    }
  }, [initialContextTokens, initialContextWindow]);

  // Auto-resize textarea based on content
  useEffect(() => {
    const textarea = inputRef.current;
    if (!textarea) return;

    // Reset height to calculate scrollHeight correctly
    textarea.style.height = "auto";

    // Set height based on scrollHeight, capped at ~4 lines of text
    // Line height is 1.5, font-size is 13.5px -> ~20.25px per line.
    // Padding top & bottom is 9px each -> 18px total padding.
    // 4 lines = 4 * 20.25px + 18px = 99px.
    const maxLinesHeight = 99;
    const newHeight = Math.min(textarea.scrollHeight, maxLinesHeight);

    textarea.style.height = `${newHeight}px`;
  }, [input]);

  const wsRef = useRef<WebSocket | null>(null);
  const onSessionResetRef = useRef(onSessionReset);
  useEffect(() => { onSessionResetRef.current = onSessionReset; }, [onSessionReset]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const pendingMsgIdRef = useRef<string | null>(null);

  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const uploadFile = useCallback(async (file: File) => {
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.error) {
        alert(`Upload failed: ${data.error}`);
      } else {
        setInput((prev) => {
          const suffix = prev ? ` ` : ``;
          return prev + suffix + file.name;
        });
      }
    } catch (err) {
      alert(`Upload error: ${String(err)}`);
    } finally {
      setUploading(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      uploadFile(e.dataTransfer.files[0]);
    }
  }, [uploadFile]);

  const triggerFileUpload = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      uploadFile(e.target.files[0]);
    }
  }, [uploadFile]);

  // ---- Message state helpers ----
  const appendBlock = useCallback((msgId: string | null, block: MessageBlock) => {
    if (!msgId) return;
    setMessages((prev) => prev.map((m) =>
      m.id === msgId ? { ...m, blocks: [...m.blocks, block] } : m
    ));
  }, []);

  const appendOrMergeContent = useCallback((msgId: string | null, content: string) => {
    if (!msgId) return;
    setMessages((prev) => prev.map((m) => {
      if (m.id !== msgId) return m;
      const lastBlock = m.blocks[m.blocks.length - 1];
      if (lastBlock?.kind === "content") {
        const newBlocks = [...m.blocks];
        newBlocks[newBlocks.length - 1] = { kind: "content", content: lastBlock.content + content };
        return { ...m, blocks: newBlocks };
      }
      return { ...m, blocks: [...m.blocks, { kind: "content", content }] };
    }));
  }, []);

  const setToolDone = useCallback((msgId: string | null, tool: string) => {
    if (!msgId) return;
    setMessages((prev) => prev.map((m) => {
      if (m.id !== msgId) return m;
      // Find the last running tool block for this tool and mark it done
      const blocks = [...m.blocks];
      for (let i = blocks.length - 1; i >= 0; i--) {
        const b = blocks[i];
        if (b && b.kind === "tool" && b.tool === tool && b.state === "running") {
          blocks[i] = { ...b, state: "done" };
          break;
        }
      }
      return { ...m, blocks };
    }));
  }, []);

  // ---- Server message handler ----
  const handleServerMessage = useCallback((msg: ServerMessage) => {
    const id = pendingMsgIdRef.current;

    if (msg.type === "job") {
      if (!sessionId || msg.sessionId !== sessionId) return;
      const running = msg.state === "running";
      if (running && !pendingMsgIdRef.current) {
        setMessages((prev) => {
          const lastAssistant = [...prev].reverse().find((item) => item.role === "assistant");
          if (lastAssistant) pendingMsgIdRef.current = lastAssistant.id;
          return prev;
        });
      }
      setAgentBusy(running);
      if (!running) pendingMsgIdRef.current = null;
      return;
    }

    if (msg.type === "done") {
      setAgentBusy(false);
      pendingMsgIdRef.current = null;
      onScratchpadUpdate(msg.scratchpad);
      if (msg.sessionId) onSessionResetRef.current?.(msg.sessionId);
      // Update token usage display from real API-reported values
      if (msg.contextTokens !== undefined && msg.contextWindow > 0) {
        setTokenUsage({ used: msg.contextTokens, limit: msg.contextWindow });
      }
      return;
    }

    if (msg.type === "error") {
      setAgentBusy(false);
      appendBlock(id, { kind: "warn", content: `${t("failed")}: ${msg.content}` });
      return;
    }

    if (msg.type === "think") {
      appendBlock(id, { kind: "think", content: msg.content });
      return;
    }

    if (msg.type === "content") {
      appendOrMergeContent(id, msg.content);
      return;
    }

    if (msg.type === "tool_start") {
      appendBlock(id, { kind: "tool", tool: msg.tool, state: "running", detail: msg.detail });
      return;
    }

    if (msg.type === "tool_end") {
      setToolDone(id, msg.tool);
      return;
    }

    if (msg.type === "scratchpad_update") {
      onScratchpadUpdate(msg.scratchpad);
      return;
    }

    if (msg.type === "warn") {
      appendBlock(id, { kind: "warn", content: msg.content });
    }
  }, [sessionId, onScratchpadUpdate, appendBlock, appendOrMergeContent, setToolDone]);

  // ---- WebSocket lifecycle ----
  const connect = useCallback(function doConnect() {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      setConnected(true);
      if (sessionId) {
        ws.send(JSON.stringify({ type: "watch", sessionId }));
      }
    };

    ws.onclose = () => {
      setConnected(false);
      // Reconnect after 2s
      setTimeout(doConnect, 2000);
    };

    ws.onerror = () => {
      ws.close();
    };

    ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data as string) as ServerMessage;
      handleServerMessage(msg);
    };

    wsRef.current = ws;
  }, [sessionId, handleServerMessage]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

  // ---- Send message ----
  const sendMessage = () => {
    const text = input.trim();
    if (!text || agentBusy || !connected) return;

    // Add user message
    const userMsg: ChatMessage = {
      id: newId(),
      role: "user",
      blocks: [{ kind: "content", content: text }],
      timestamp: Date.now(),
    };

    // Create empty assistant message
    const assistantId = newId();
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: "assistant",
      blocks: [],
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    pendingMsgIdRef.current = assistantId;
    setInput("");
    setAgentBusy(true);

    wsRef.current?.send(JSON.stringify({ type: "chat", text, sessionId, streaming: true }));
  };

  const handleReset = () => {
    if (agentBusy) return;
    wsRef.current?.send(JSON.stringify({ type: "reset", sessionId }));
    setMessages([]);
  };

  const handleInterrupt = () => {
    if (!agentBusy) return;
    wsRef.current?.send(JSON.stringify({ type: "interrupt", sessionId }));
  };

  const handleCompact = async () => {
    if (agentBusy || compacting) return;
    setCompacting(true);
    try {
      const result = await api.compactSession();
      if (result.ran) {
        setCompactDone(true);
        setTimeout(() => setCompactDone(false), 2500);
        if (result.contextTokens !== undefined && result.contextWindow !== undefined && result.contextWindow > 0) {
          setTokenUsage({ used: result.contextTokens, limit: result.contextWindow });
        }
        setMessages((prev) => [
          ...prev,
          {
            id: newId(),
            role: "assistant" as const,
            blocks: [{ kind: "warn" as const, content: t("chat.compactWarning") }],
            timestamp: Date.now(),
          },
        ]);
      } else {
        // Not enough turns to compress — show a subtle note
        setMessages((prev) => [
          ...prev,
          {
            id: newId(),
            role: "assistant" as const,
            blocks: [{ kind: "warn" as const, content: t("chat.compactMinTurnsWarning", { count: 5 }) }],
            timestamp: Date.now(),
          },
        ]);
      }
    } catch {
      // non-fatal — likely a timeout on very slow models; operation may have succeeded server-side
    } finally {
      setCompacting(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="chat-panel">
      {/* Header */}
      <div className="chat-header">
        <div className="chat-status">
          <span className={`chat-ws-dot ${connected ? "connected" : "disconnected"}`} />
          <span className="text-sm text-secondary">
            {connected ? t("chat.connected") : t("chat.connecting")}
          </span>
        </div>
        <div className="chat-header-actions">
          {/* Token usage progress bar */}
          {tokenUsage && (() => {
            const pct = Math.round((tokenUsage.used / tokenUsage.limit) * 100);
            const color = pct >= 80 ? "var(--red)" : pct >= 65 ? "var(--yellow)" : "var(--green)";
            const tokenCountUsed = tokenUsage.used.toLocaleString();
            const tokenCountLimit = tokenUsage.limit.toLocaleString();
            return (
              <div
                className="token-usage-bar-container"
                title={locale === "zh" ? `已用 ~${tokenCountUsed} / ${tokenCountLimit} tokens (${pct}%)` : `~${tokenCountUsed} / ${tokenCountLimit} tokens used (${pct}%)`}
              >
                <div className="token-usage-bar-label">
                  {t("chat.contextPct", { pct })}
                </div>
                <div className="token-usage-bar-bg">
                  <div
                    className="token-usage-bar-fill"
                    style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: color }}
                  />
                </div>
              </div>
            );
          })()}
          <button
            className={`btn btn-ghost btn-icon ${compactDone ? "text-success" : ""}`}
            onClick={handleCompact}
            disabled={agentBusy || compacting}
            title={t("chat.compactTooltip")}
          >
            {compacting ? <span className="spinner" /> : <Archive size={14} />}
          </button>
          <button
            className="btn btn-ghost btn-icon"
            onClick={handleReset}
            disabled={agentBusy}
            title={t("chat.newSessionTooltip")}
          >
            <RotateCcw size={14} />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {hasHistory && (
          <div className="chat-history-banner">
            🕐 {t("chat.historyBanner", { count: messages.length })}
          </div>
        )}
        {messages.length === 0 && (
          <div className="chat-empty">
            <img src="/logo-short.png" className="chat-empty-icon" alt="CauProphet Logo" />
            <div className="chat-empty-title">CauProphet</div>
            <div className="chat-empty-subtitle">
              {t("chat.emptyPlaceholder")}
            </div>
            <div className="chat-empty-suggestions">
              {(["chat.suggestion1", "chat.suggestion2", "chat.suggestion3", "chat.suggestion4"] as const).map((key) => (
                <button
                  key={key}
                  type="button"
                  className="chat-suggestion-chip"
                  onClick={() => {
                    setInput(t(key));
                    inputRef.current?.focus();
                  }}
                >
                  {t(key)}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div
        className={`chat-input-area ${dragActive ? "drag-active" : ""}`}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
      >
        <div className="chat-input-row">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            style={{ display: "none" }}
          />
          <button
            type="button"
            className="btn btn-ghost btn-icon chat-upload-btn"
            onClick={triggerFileUpload}
            disabled={agentBusy || !connected || uploading}
            title={t("chat.uploadTooltip")}
          >
            {uploading ? <span className="spinner" style={{ width: 14, height: 14 }} /> : <Paperclip size={15} />}
          </button>
          <textarea
            ref={inputRef}
            className="chat-textarea"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={agentBusy ? t("chat.placeholderBusy") : t("chat.placeholder")}
            disabled={agentBusy || !connected || uploading}
            rows={1}
          />
          {agentBusy ? (
            <button
              className="btn chat-send-btn chat-stop-btn"
              onClick={handleInterrupt}
              title={t("chat.interruptTooltip")}
              aria-label={t("chat.interruptTooltip")}
            >
              <span className="chat-stop-spinner spinner" />
              <Square size={13} className="chat-stop-icon" fill="currentColor" />
            </button>
          ) : (
            <button
              className="btn btn-primary chat-send-btn"
              onClick={sendMessage}
              disabled={!input.trim() || !connected}
            >
              <Send size={15} />
            </button>
          )}
        </div>
        <div className="chat-input-footer">
          <span className="text-sm text-muted">
            {t("chat.inputFooter")}
          </span>
        </div>
      </div>
    </div>
  );
}

// ---- Message Bubble ----
function MessageBubble({ message }: { message: ChatMessage }) {
  const { t } = useI18n();
  const [copied, setCopied] = useState(false);

  // Concatenate the raw markdown text of all content blocks (original, un-rendered).
  const rawText = message.blocks
    .filter((b) => b.kind === "content")
    .map((b) => b.content)
    .join("\n\n");

  const showCopy = message.role === "assistant" && rawText.trim().length > 0;

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(rawText);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API unavailable (e.g. non-secure context) — fall back to a hidden textarea.
      const ta = document.createElement("textarea");
      ta.value = rawText;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand("copy");
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      } catch {
        // give up silently
      }
      document.body.removeChild(ta);
    }
  }, [rawText]);

  return (
    <div className={`chat-message ${message.role}`}>
      <div className="chat-message-inner">
        {message.blocks.map((block, i) => (
          <BlockRenderer key={i} block={block} />
        ))}
        {message.role === "assistant" && message.blocks.length === 0 && (
          <div className="chat-typing">
            <span className="typing-dot" />
            <span className="typing-dot" />
            <span className="typing-dot" />
          </div>
        )}
        {showCopy && (
          <button
            className={`chat-copy-btn ${copied ? "copied" : ""}`}
            onClick={handleCopy}
            title={t("chat.copyTooltip")}
            aria-label={t("chat.copyTooltip")}
          >
            {copied ? <Check size={13} /> : <Copy size={13} />}
          </button>
        )}
      </div>
    </div>
  );
}

// ---- Block Renderer ----
function BlockRenderer({ block }: { block: MessageBlock }) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);

  if (block.kind === "content") {
    return (
      <div className="chat-content markdown">
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeKatex]}
          components={markdownComponents}
        >
          {block.content}
        </ReactMarkdown>
      </div>
    );
  }

  if (block.kind === "think") {
    return (
      <div className="chat-think">
        <button className="chat-think-toggle" onClick={() => setOpen((v) => !v)}>
          {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          <span>{t("chat.thinking")}</span>
        </button>
        {open && (
          <div className="chat-think-body text-secondary">
            {block.content}
          </div>
        )}
      </div>
    );
  }

  if (block.kind === "tool") {
    const hasDetail = !!block.detail;
    return (
      <div className={`chat-tool-container ${block.state}`}>
        {hasDetail ? (
          <button className="chat-tool-toggle" onClick={() => setOpen((v) => !v)}>
            {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            <Wrench size={12} />
            <span className="font-mono">{block.tool}</span>
            {block.state === "running" ? (
              <span className="spinner" style={{ marginLeft: "auto" }} />
            ) : (
              <span className="chat-tool-done">✓</span>
            )}
          </button>
        ) : (
          <div className="chat-tool">
            <Wrench size={12} />
            <span className="font-mono">{block.tool}</span>
            {block.state === "running" ? (
              <span className="spinner" style={{ marginLeft: "auto" }} />
            ) : (
              <span className="chat-tool-done">✓</span>
            )}
          </div>
        )}
        {hasDetail && open && (
          <pre className="chat-tool-detail font-mono">
            {block.detail}
          </pre>
        )}
      </div>
    );
  }

  if (block.kind === "warn") {
    if (block.content === "__INTERRUPTED__") {
      return (
        <div className="chat-interrupted">
          ⚠ {t("chat.interrupted")}
        </div>
      );
    }
    return (
      <div className="chat-warn">
        ⚠ {block.content}
      </div>
    );
  }

  return null;
}
