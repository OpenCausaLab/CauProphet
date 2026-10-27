import { useEffect, useState } from "react";
import { Save, RefreshCw, CheckCircle, AlertCircle, Eye, EyeOff, Server, Key, Cpu, FolderOpen, Layers, X, Search } from "lucide-react";
import { api } from "../api";
import type { AppConfig } from "../types";
import { useI18n } from "../i18n";
import "./SettingsPanel.css";

interface SettingsPanelProps {
  onClose?: () => void;
}

const COMMON_MODELS = [
  "gpt-4o",
  "gpt-4o-mini",
  "gpt-4-turbo",
  "o1-preview",
  "o1-mini",
  "claude-3-5-sonnet-20241022",
  "claude-3-opus-20240229",
  "deepseek-chat",
  "deepseek-reasoner",
  "qwen-plus",
];

export function SettingsPanel({ onClose }: SettingsPanelProps) {
  const { locale, t } = useI18n();
  const [cfg, setCfg] = useState<AppConfig>({
    openaiBaseUrl: "https://api.openai.com/v1",
    openaiApiKey: "",
    model: "gpt-4o",
    workspace: "",
    contextWindow: 128000,
    searchProvider: "ddg",
    braveApiKey: "",
    bochaApiKey: "",
  });
  const [rawKey, setRawKey] = useState(""); // unmasked key field
  const [showKey, setShowKey] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; msg: string } | null>(null);
  const [saveResult, setSaveResult] = useState<{ ok: boolean } | null>(null);
  const [useCustomModel, setUseCustomModel] = useState(false);

  useEffect(() => {
    api.getConfig()
      .then((c) => {
        setCfg((defaults) => ({
          ...defaults,
          ...c,
        }));
        setRawKey(c.openaiApiKey); // will be "••••xxxx" initially
        setUseCustomModel(!COMMON_MODELS.includes(c.model));
      })
      .catch(() => {/* server may not be running yet */})
      .finally(() => setLoading(false));
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setSaveResult(null);
    try {
      const payload: Partial<AppConfig> = {
        openaiBaseUrl: cfg.openaiBaseUrl,
        model: cfg.model,
        workspace: cfg.workspace,
        contextWindow: cfg.contextWindow,
        searchProvider: cfg.searchProvider,
      };
      // Only send key if user typed a real one (not the masked placeholder)
      if (rawKey && !rawKey.startsWith("••••")) {
        payload.openaiApiKey = rawKey;
      }
      if (cfg.braveApiKey && !cfg.braveApiKey.startsWith("••••")) {
        payload.braveApiKey = cfg.braveApiKey;
      }
      if (cfg.bochaApiKey && !cfg.bochaApiKey.startsWith("••••")) {
        payload.bochaApiKey = cfg.bochaApiKey;
      }
      await api.saveConfig(payload);
      setSaveResult({ ok: true });
      setTimeout(() => setSaveResult(null), 2500);
    } catch {
      setSaveResult({ ok: false });
    } finally {
      setSaving(false);
    }
  };

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      // Request goes to Bun server (server-side fetch), avoiding browser CORS restrictions.
      const res = await fetch("/api/test-connection", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          baseUrl: cfg.openaiBaseUrl,
          // Send real key if user typed one; otherwise send stored key via server config
          apiKey: rawKey.startsWith("••••") ? undefined : rawKey,
        }),
      });
      const data = await res.json() as { ok: boolean; message: string };
      setTestResult({ ok: data.ok, msg: data.message });
    } catch (e) {
      // Likely server not running
      setTestResult({
        ok: false,
        msg: locale === "zh"
          ? `无法连接到 CauProphet 服务端。'bun run start' 是否正在运行？(${String(e)})`
          : `Cannot reach CauProphet server. Is 'bun run start' running? (${String(e)})`
      });
    } finally {
      setTesting(false);
    }
  };

  if (loading) {
    return (
      <div className="settings-loading">
        <div className="spinner" />
        <span>{t("settings.loadingConfig")}</span>
      </div>
    );
  }

  return (
    <div className="settings-panel">
      <div className="settings-header">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", width: "100%" }}>
          <h2>{t("settings.title")}</h2>
          {onClose && (
            <button className="btn btn-ghost btn-icon" onClick={onClose} title={t("settings.closeSettings")} style={{ padding: 4 }}>
              <X size={15} />
            </button>
          )}
        </div>
        <p className="text-secondary text-sm">{t("settings.subtitle")}</p>
      </div>

      <div className="settings-body">
        {/* --- LLM Provider --- */}
        <section className="settings-section">
          <div className="settings-section-title">
            <Server size={14} />
            {t("settings.llmProvider")}
          </div>

          <div className="settings-field">
            <label className="settings-label">{t("settings.baseUrl")}</label>
            <div className="settings-hint">{t("settings.baseUrlHint")}</div>
            <input
              className="input"
              type="url"
              value={cfg.openaiBaseUrl}
              onChange={(e) => setCfg((c) => ({ ...c, openaiBaseUrl: e.target.value }))}
              placeholder="https://api.openai.com/v1"
            />
          </div>

          <div className="settings-field">
            <label className="settings-label">
              <Key size={13} /> {t("settings.apiKey")}
            </label>
            <div className="settings-hint">{t("settings.apiKeyStoredHint")}</div>
            <div className="settings-key-row">
              <input
                className="input"
                type={showKey ? "text" : "password"}
                value={rawKey}
                onChange={(e) => setRawKey(e.target.value)}
                placeholder="sk-..."
                spellCheck={false}
              />
              <button
                className="btn btn-ghost btn-icon"
                onClick={() => setShowKey((v) => !v)}
                title={showKey ? t("settings.hideKey") : t("settings.showKey")}
              >
                {showKey ? <EyeOff size={15} /> : <Eye size={15} />}
              </button>
            </div>
          </div>
        </section>

        {/* --- Model --- */}
        <section className="settings-section">
          <div className="settings-section-title">
            <Cpu size={14} />
            {t("settings.model")}
          </div>

          <div className="settings-field">
            <label className="settings-label">{t("settings.model")}</label>
            {!useCustomModel ? (
              <div className="settings-model-row">
                <select
                  className="input settings-select"
                  value={cfg.model}
                  onChange={(e) => {
                    if (e.target.value === "__custom__") {
                      setUseCustomModel(true);
                    } else {
                      setCfg((c) => ({ ...c, model: e.target.value }));
                    }
                  }}
                >
                  {COMMON_MODELS.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                  <option value="__custom__">{t("settings.customModel")}</option>
                </select>
              </div>
            ) : (
              <div className="settings-model-row">
                <input
                  className="input"
                  type="text"
                  value={cfg.model}
                  onChange={(e) => setCfg((c) => ({ ...c, model: e.target.value }))}
                  placeholder={t("settings.customModelPlaceholder")}
                />
                <button
                  className="btn btn-ghost text-sm"
                  onClick={() => {
                    setUseCustomModel(false);
                    setCfg((c) => ({ ...c, model: COMMON_MODELS[0]! }));
                  }}
                >
                  {t("settings.presets")}
                </button>
              </div>
            )}
          </div>
        </section>

        {/* --- Workspace --- */}
        <section className="settings-section">
          <div className="settings-section-title">
            <FolderOpen size={14} />
            {t("settings.workspace")}
          </div>

          <div className="settings-field">
            <label className="settings-label">{t("settings.workspacePath")}</label>
            <div className="settings-hint">{t("settings.workspacePathHint")}</div>
            <input
              className="input font-mono"
              type="text"
              value={cfg.workspace}
              onChange={(e) => setCfg((c) => ({ ...c, workspace: e.target.value }))}
              placeholder="/home/user/my-project"
            />
          </div>
        </section>

        {/* --- Context Compression --- */}
        <section className="settings-section">
          <div className="settings-section-title">
            <Layers size={14} />
            {t("settings.context")}
          </div>

          <div className="settings-field">
            <label className="settings-label">{t("settings.contextWindow")}</label>
            <div className="settings-hint">{t("settings.contextWindowHint")}</div>
            <input
              id="settings-context-window"
              className="input"
              type="number"
              min={4000}
              step={1000}
              value={cfg.contextWindow ?? 128000}
              onChange={(e) =>
                setCfg((c) => ({ ...c, contextWindow: Number(e.target.value) || 128000 }))
              }
            />
          </div>
        </section>

        {/* --- Web Search --- */}
        <section className="settings-section">
          <div className="settings-section-title">
            <Search size={14} />
            {t("settings.webSearch")}
          </div>

          <div className="settings-field">
            <label className="settings-label">{t("settings.searchProvider")}</label>
            <div className="settings-hint">{t("settings.searchProviderHint")}</div>
            <select
              className="input settings-select"
              value={cfg.searchProvider}
              onChange={(e) =>
                setCfg((c) => ({ ...c, searchProvider: e.target.value as AppConfig["searchProvider"] }))
              }
            >
              <option value="ddg">DuckDuckGo (Free)</option>
              <option value="brave">Brave Search</option>
              <option value="bocha">{locale === "zh" ? "博查 (Bocha AI)" : "Bocha AI"}</option>
            </select>
          </div>

          {cfg.searchProvider === "brave" && (
            <div className="settings-field">
              <label className="settings-label">
                <Key size={13} /> {t("settings.braveApiKey")}
              </label>
              <input
                className="input"
                type="password"
                value={cfg.braveApiKey ?? ""}
                onChange={(e) => setCfg((c) => ({ ...c, braveApiKey: e.target.value }))}
                placeholder="BSA..."
                spellCheck={false}
              />
            </div>
          )}

          {cfg.searchProvider === "bocha" && (
            <div className="settings-field">
              <label className="settings-label">
                <Key size={13} /> {t("settings.bochaApiKey")}
              </label>
              <input
                className="input"
                type="password"
                value={cfg.bochaApiKey ?? ""}
                onChange={(e) => setCfg((c) => ({ ...c, bochaApiKey: e.target.value }))}
                placeholder="sk-..."
                spellCheck={false}
              />
            </div>
          )}
        </section>

        {/* --- Connection test --- */}
        {testResult && (
          <div className={`settings-test-result ${testResult.ok ? "ok" : "err"}`}>
            {testResult.ok
              ? <CheckCircle size={14} />
              : <AlertCircle size={14} />}
            {testResult.msg}
          </div>
        )}
      </div>

      {/* --- Footer actions --- */}
      <div className="settings-footer">
        <button
          className="btn btn-ghost"
          onClick={handleTest}
          disabled={testing}
        >
          {testing ? <span className="spinner" /> : <RefreshCw size={13} />}
          {t("settings.testConnection")}
        </button>

        <div className="flex gap-2 items-center">
          {saveResult && (
            <span className={`settings-save-status ${saveResult.ok ? "ok" : "err"}`}>
              {saveResult.ok ? <CheckCircle size={13} /> : <AlertCircle size={13} />}
              {saveResult.ok ? t("saved") : t("failed")}
            </span>
          )}
          <button
            className="btn btn-primary"
            onClick={handleSave}
            disabled={saving}
          >
            {saving ? <span className="spinner" /> : <Save size={13} />}
            {t("save")}
          </button>
        </div>
      </div>
    </div>
  );
}
