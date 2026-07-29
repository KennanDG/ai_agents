import { useEffect, useMemo, useState } from "react";
import { CircleAlert, KeyRound, LoaderCircle, Save, X } from "lucide-react";
import {
  fetchAgentConfiguration,
  updateAgentConfiguration,
  type AgentConfiguration,
  type ChatProvider,
} from "../lib/adminApi";

type AgentSettingsModalProps = {
  open: boolean;
  apiBaseUrl: string;
  apiKey: string;
  onClose: () => void;
};

const PROVIDERS: ChatProvider[] = ["groq", "deepseek", "openrouter", "openai"];

const providerLabel: Record<ChatProvider, string> = {
  groq: "Groq",
  deepseek: "DeepSeek",
  openrouter: "OpenRouter",
  openai: "OpenAI",
};

const emptySecrets = (): Record<ChatProvider, string> => ({
  groq: "",
  deepseek: "",
  openrouter: "",
  openai: "",
});

export const AgentSettingsModal = ({
  open,
  apiBaseUrl,
  apiKey,
  onClose,
}: AgentSettingsModalProps) => {
  const [configuration, setConfiguration] = useState<AgentConfiguration | null>(null);
  const [secrets, setSecrets] = useState(emptySecrets);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setMessage(null);

    fetchAgentConfiguration({ apiBaseUrl, apiKey })
      .then((result) => {
        if (!cancelled) setConfiguration(result);
      })
      .catch((reason) => {
        if (!cancelled) {
          setError(reason instanceof Error ? reason.message : "Failed to load agent configuration.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl, apiKey, open]);

  const changedSecrets = useMemo(
    () =>
      Object.fromEntries(
        PROVIDERS.flatMap((provider) => {
          const value = secrets[provider].trim();
          return value ? [[provider, value]] : [];
        }),
      ) as Partial<Record<ChatProvider, string>>,
    [secrets],
  );

  if (!open) return null;

  const update = <K extends keyof AgentConfiguration>(
    key: K,
    value: AgentConfiguration[K],
  ) => {
    setConfiguration((current) => (current ? { ...current, [key]: value } : current));
  };

  const save = async () => {
    if (!configuration) return;
    setSaving(true);
    setError(null);
    setMessage(null);

    try {
      const updated = await updateAgentConfiguration({
        apiBaseUrl,
        apiKey,
        configuration: {
          coding_provider: configuration.coding_provider,
          coding_model: configuration.coding_model,
          reasoning_provider: configuration.reasoning_provider,
          reasoning_model: configuration.reasoning_model,
          caption_model: configuration.caption_model,
          voice_chat_model: configuration.voice_chat_model,
          voice_stt_model: configuration.voice_stt_model,
          voice_tts_model: configuration.voice_tts_model,
          voice_tts_voice: configuration.voice_tts_voice,
          voice_tts_enabled: configuration.voice_tts_enabled,
          secrets: changedSecrets,
        },
      });
      setConfiguration(updated);
      setSecrets(emptySecrets());
      setMessage("Saved. New coding and voice runs will use this configuration.");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Failed to save agent configuration.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/65 p-4" role="presentation">
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="agent-settings-title"
        className="flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-xl border border-line-strong bg-panel-soft shadow-2xl"
      >
        <header className="flex h-12 shrink-0 items-center justify-between border-b border-line px-4">
          <div>
            <h2 id="agent-settings-title" className="text-sm font-semibold text-ink">
              Agent configuration
            </h2>
            <p className="text-[10px] text-muted">Model routing and backend-only provider credentials</p>
          </div>
          <button type="button" className="icon-button" aria-label="Close" onClick={onClose}>
            <X size={16} />
          </button>
        </header>

        <div className="min-h-0 flex-1 overflow-auto p-4">
          {loading ? (
            <div className="flex items-center gap-2 py-10 text-xs text-muted">
              <LoaderCircle size={14} className="animate-spin text-accent-light" />
              Loading agent configuration…
            </div>
          ) : configuration ? (
            <div className="space-y-5">
              <section className="rounded-lg border border-line bg-panel p-4">
                <h3 className="text-xs font-semibold text-ink">Coding agent models</h3>
                <div className="mt-3 grid gap-3 sm:grid-cols-2">
                  <label className="text-[10px] font-medium text-muted">
                    Coding provider
                    <select
                      value={configuration.coding_provider}
                      onChange={(event) => update("coding_provider", event.target.value as ChatProvider)}
                      className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/70"
                    >
                      {PROVIDERS.map((provider) => (
                        <option key={provider} value={provider}>{providerLabel[provider]}</option>
                      ))}
                    </select>
                  </label>
                  <label className="text-[10px] font-medium text-muted">
                    Coding model
                    <input
                      value={configuration.coding_model}
                      onChange={(event) => update("coding_model", event.target.value)}
                      className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/70"
                    />
                  </label>
                  <label className="text-[10px] font-medium text-muted">
                    Reasoning provider
                    <select
                      value={configuration.reasoning_provider}
                      onChange={(event) => update("reasoning_provider", event.target.value as ChatProvider)}
                      className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/70"
                    >
                      {PROVIDERS.map((provider) => (
                        <option key={provider} value={provider}>{providerLabel[provider]}</option>
                      ))}
                    </select>
                  </label>
                  <label className="text-[10px] font-medium text-muted">
                    Reasoning model
                    <input
                      value={configuration.reasoning_model}
                      onChange={(event) => update("reasoning_model", event.target.value)}
                      className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/70"
                    />
                  </label>
                  <label className="text-[10px] font-medium text-muted sm:col-span-2">
                    Vision / caption model
                    <input
                      value={configuration.caption_model}
                      onChange={(event) => update("caption_model", event.target.value)}
                      className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/70"
                    />
                  </label>
                </div>
              </section>

              <section className="rounded-lg border border-line bg-panel p-4">
                <h3 className="text-xs font-semibold text-ink">Voice agent models</h3>
                <p className="mt-1 text-[10px] text-muted">
                  The current voice transport uses Groq for chat, transcription, and speech.
                </p>
                <div className="mt-3 grid gap-3 sm:grid-cols-2">
                  <label className="text-[10px] font-medium text-muted">
                    Voice chat model
                    <input
                      value={configuration.voice_chat_model}
                      onChange={(event) => update("voice_chat_model", event.target.value)}
                      className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/70"
                    />
                  </label>
                  <label className="text-[10px] font-medium text-muted">
                    Speech-to-text model
                    <input
                      value={configuration.voice_stt_model}
                      onChange={(event) => update("voice_stt_model", event.target.value)}
                      className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/70"
                    />
                  </label>
                  <label className="text-[10px] font-medium text-muted">
                    Text-to-speech model
                    <input
                      value={configuration.voice_tts_model}
                      onChange={(event) => update("voice_tts_model", event.target.value)}
                      className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/70"
                    />
                  </label>
                  <label className="text-[10px] font-medium text-muted">
                    Voice
                    <input
                      value={configuration.voice_tts_voice}
                      onChange={(event) => update("voice_tts_voice", event.target.value)}
                      className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/70"
                    />
                  </label>
                  <label className="flex cursor-pointer items-center gap-2 text-[11px] text-muted sm:col-span-2">
                    <input
                      type="checkbox"
                      checked={configuration.voice_tts_enabled}
                      onChange={(event) => update("voice_tts_enabled", event.target.checked)}
                      className="accent-accent"
                    />
                    Enable voice replies
                  </label>
                </div>
              </section>

              <section className="rounded-lg border border-line bg-panel p-4">
                <div className="flex items-start gap-2">
                  <KeyRound size={14} className="mt-0.5 shrink-0 text-accent-light" />
                  <div>
                    <h3 className="text-xs font-semibold text-ink">Provider secrets</h3>
                    <p className="mt-1 text-[10px] leading-4 text-muted">
                      Secret values are never returned to the renderer. This patch keeps secrets in the backend process only; use AWS Secrets Manager or environment variables for durable storage.
                    </p>
                  </div>
                </div>
                <div className="mt-3 grid gap-3 sm:grid-cols-2">
                  {PROVIDERS.map((provider) => (
                    <label key={provider} className="text-[10px] font-medium text-muted">
                      {providerLabel[provider]} API key
                      <span className={`ml-2 text-[9px] ${configuration.secrets_configured[provider] ? "text-emerald-300" : "text-faint"}`}>
                        {configuration.secrets_configured[provider] ? "configured" : "not configured"}
                      </span>
                      <input
                        type="password"
                        autoComplete="new-password"
                        value={secrets[provider]}
                        onChange={(event) =>
                          setSecrets((current) => ({ ...current, [provider]: event.target.value }))
                        }
                        placeholder="Leave blank to keep current value"
                        className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/70"
                      />
                    </label>
                  ))}
                </div>
              </section>

              {message ? (
                <div className="rounded-md border border-emerald-500/20 bg-emerald-500/8 p-3 text-[11px] text-emerald-300">
                  {message}
                </div>
              ) : null}
              {error ? (
                <div className="flex items-start gap-2 rounded-md border border-rose-500/20 bg-rose-500/8 p-3 text-[11px] text-rose-300">
                  <CircleAlert size={14} className="mt-0.5 shrink-0" />
                  {error}
                </div>
              ) : null}
            </div>
          ) : null}
        </div>

        <footer className="flex shrink-0 items-center justify-end gap-2 border-t border-line px-4 py-3">
          <button type="button" className="secondary-button" onClick={onClose}>Cancel</button>
          <button
            type="button"
            className="primary-button"
            disabled={!configuration || saving}
            onClick={save}
          >
            {saving ? <LoaderCircle size={12} className="animate-spin" /> : <Save size={12} />}
            Save configuration
          </button>
        </footer>
      </section>
    </div>
  );
}
