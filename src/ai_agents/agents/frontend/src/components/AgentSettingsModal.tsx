import { useCallback, useEffect, useMemo, useState } from "react";
import { CircleAlert, KeyRound, LoaderCircle, RefreshCw, Save, X } from "lucide-react";
import {
  fetchAgentConfiguration,
  fetchAvailableModels,
  updateAgentConfiguration,
  type AgentConfiguration,
  type ChatProvider,
  type ModelCapability,
  type ModelCatalogResponse,
} from "../lib/adminApi";

type AgentSettingsModalProps = {
  open: boolean;
  apiBaseUrl: string;
  apiKey: string;
  onClose: () => void;
  onSaved?: (configuration: AgentConfiguration) => void;
};

type ProviderField =
  | "coding_provider"
  | "reasoning_provider"
  | "caption_provider"
  | "voice_chat_provider"
  | "voice_stt_provider"
  | "voice_tts_provider";

type ModelField =
  | "coding_model"
  | "reasoning_model"
  | "caption_model"
  | "voice_chat_model"
  | "voice_stt_model"
  | "voice_tts_model";

type ModelSlot = {
  providerField: ProviderField;
  modelField: ModelField;
  capability: ModelCapability;
};

const CHAT_PROVIDERS: ChatProvider[] = [
  "groq",
  "deepseek",
  "openrouter",
  "openai",
  "anthropic",
  "google",
];
const VISION_PROVIDERS: ChatProvider[] = ["groq", "openrouter", "openai", "anthropic", "google"];
const AUDIO_PROVIDERS: ChatProvider[] = ["groq", "openai"];

const PROVIDERS: ChatProvider[] = CHAT_PROVIDERS;

const providerLabel: Record<ChatProvider, string> = {
  groq: "Groq",
  deepseek: "DeepSeek",
  openrouter: "OpenRouter",
  openai: "OpenAI",
  anthropic: "Anthropic",
  google: "Google",
};

const emptySecrets = (): Record<ChatProvider, string> => ({
  groq: "",
  deepseek: "",
  openrouter: "",
  openai: "",
  anthropic: "",
  google: "",
});

const catalogKey = (provider: ChatProvider, capability: ModelCapability) =>
  `${provider}:${capability}`;

const modelOptions = (catalog: ModelCatalogResponse | undefined, currentModel: string) => {
  const models = catalog?.models ?? [];
  return currentModel && !models.includes(currentModel) ? [currentModel, ...models] : models;
};

const NumberInput = ({
  label,
  value,
  min,
  max,
  help,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  help: string;
  onChange: (value: number) => void;
}) => {
  const [displayValue, setDisplayValue] = useState(String(value));

  useEffect(() => {
    setDisplayValue(String(value));
  }, [value]);

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setDisplayValue(event.target.value);
  };

  const handleBlur = () => {
    const parsed = Number.parseInt(displayValue, 10);
    if (Number.isFinite(parsed)) {
      onChange(parsed);
    } else {
      setDisplayValue(String(value));
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      event.currentTarget.blur();
    }
  };

  const parsed = Number.parseInt(displayValue, 10);
  const isOutOfRange = Number.isFinite(parsed) && (parsed < min || parsed > max);

  return (
    <label className="text-[10px] font-medium text-muted">
      {label}
      <input
        type="text"
        inputMode="numeric"
        autoComplete="off"
        value={displayValue}
        onChange={handleChange}
        onBlur={handleBlur}
        onKeyDown={handleKeyDown}
        className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/70"
      />
      <span className="mt-1 block text-[9px] font-normal leading-4 text-faint">{help}</span>
      {isOutOfRange && (
        <span className="mt-1 block text-[9px] font-normal leading-4 text-rose-400">
          Must be between {min} and {max}.
        </span>
      )}
    </label>
  );
};

const ProviderSelect = ({
  label,
  value,
  providers,
  onChange,
}: {
  label: string;
  value: ChatProvider;
  providers: ChatProvider[];
  onChange: (provider: ChatProvider) => void;
}) => (
  <label className="text-[10px] font-medium text-muted">
    {label}
    <select
      value={value}
      onChange={(event) => onChange(event.target.value as ChatProvider)}
      className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/70"
    >
      {providers.map((provider) => (
        <option key={provider} value={provider}>
          {providerLabel[provider]}
        </option>
      ))}
    </select>
  </label>
);

const ModelSelect = ({
  label,
  value,
  catalog,
  loading,
  onChange,
  onRefresh,
}: {
  label: string;
  value: string;
  catalog?: ModelCatalogResponse;
  loading: boolean;
  onChange: (model: string) => void;
  onRefresh: () => void;
}) => {
  const options = modelOptions(catalog, value);

  return (
    <label className="text-[10px] font-medium text-muted">
      <span className="flex items-center justify-between gap-2">
        {label}
        <button
          type="button"
          className="inline-flex items-center gap-1 text-[9px] font-normal text-faint hover:text-ink"
          onClick={onRefresh}
          disabled={loading}
          title="Refresh models"
        >
          <RefreshCw size={10} className={loading ? "animate-spin" : ""} />
          Refresh
        </button>
      </span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={loading && options.length === 0}
        className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/70 disabled:opacity-60"
      >
        {options.length > 0 ? (
          options.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))
        ) : (
          <option value="">No compatible models found</option>
        )}
      </select>
      <span className="mt-1 block text-[9px] font-normal leading-4 text-faint">
        {loading
          ? "Loading models…"
          : catalog?.source === "live"
            ? "Live provider catalog"
            : catalog?.error ?? "Using the Default catalog"}
      </span>
    </label>
  );
};

export const AgentSettingsModal = ({
  open,
  apiBaseUrl,
  apiKey,
  onClose,
  onSaved,
}: AgentSettingsModalProps) => {
  const [configuration, setConfiguration] = useState<AgentConfiguration | null>(null);
  const [secrets, setSecrets] = useState(emptySecrets);
  const [githubToken, setGitHubToken] = useState("");
  const [catalogs, setCatalogs] = useState<Record<string, ModelCatalogResponse>>({});
  const [catalogLoading, setCatalogLoading] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const loadCatalog = useCallback(
    async (provider: ChatProvider, capability: ModelCapability) => {
      const key = catalogKey(provider, capability);
      setCatalogLoading((current) => ({ ...current, [key]: true }));
      try {
        const result = await fetchAvailableModels({
          apiBaseUrl,
          apiKey,
          provider,
          capability,
        });
        setCatalogs((current) => ({ ...current, [key]: result }));
        return result;
      } finally {
        setCatalogLoading((current) => ({ ...current, [key]: false }));
      }
    },
    [apiBaseUrl, apiKey],
  );

  const loadConfigurationCatalogs = useCallback(
    async (result: AgentConfiguration) => {
      const slots: ModelSlot[] = [
        { providerField: "coding_provider", modelField: "coding_model", capability: "chat" },
        { providerField: "reasoning_provider", modelField: "reasoning_model", capability: "chat" },
        { providerField: "caption_provider", modelField: "caption_model", capability: "vision" },
        { providerField: "voice_chat_provider", modelField: "voice_chat_model", capability: "chat" },
        { providerField: "voice_stt_provider", modelField: "voice_stt_model", capability: "stt" },
        { providerField: "voice_tts_provider", modelField: "voice_tts_model", capability: "tts" },
      ];

      await Promise.all(
        slots.map((slot) => loadCatalog(result[slot.providerField], slot.capability)),
      );
    },
    [loadCatalog],
  );

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setMessage(null);
    setGitHubToken("");

    fetchAgentConfiguration({ apiBaseUrl, apiKey })
      .then(async (result) => {
        if (cancelled) return;
        setConfiguration(result);
        await loadConfigurationCatalogs(result);
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
  }, [apiBaseUrl, apiKey, loadConfigurationCatalogs, open]);

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

  const changeProvider = async ({
    providerField,
    modelField,
    capability,
    provider,
  }: ModelSlot & { provider: ChatProvider }) => {
    update(providerField, provider);
    setError(null);

    try {
      const catalog = await loadCatalog(provider, capability);
      setConfiguration((current) => {
        if (!current || current[providerField] !== provider) return current;
        const currentModel = current[modelField];
        const nextModel = catalog.models.includes(currentModel)
          ? currentModel
          : catalog.models[0] ?? currentModel;
        return { ...current, [modelField]: nextModel };
      });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Failed to load provider models.");
    }
  };

  const catalogFor = (provider: ChatProvider, capability: ModelCapability) =>
    catalogs[catalogKey(provider, capability)];
  const catalogIsLoading = (provider: ChatProvider, capability: ModelCapability) =>
    Boolean(catalogLoading[catalogKey(provider, capability)]);

  const refreshCatalog = async (
    provider: ChatProvider,
    capability: ModelCapability,
    modelField: ModelField,
  ) => {
    setError(null);
    try {
      const catalog = await loadCatalog(provider, capability);
      setConfiguration((current) => {
        if (!current) return current;
        const currentModel = current[modelField];
        if (currentModel || catalog.models.length === 0) return current;
        return { ...current, [modelField]: catalog.models[0] };
      });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Failed to refresh models.");
    }
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
          caption_provider: configuration.caption_provider,
          caption_model: configuration.caption_model,
          voice_chat_provider: configuration.voice_chat_provider,
          voice_chat_model: configuration.voice_chat_model,
          voice_stt_provider: configuration.voice_stt_provider,
          voice_stt_model: configuration.voice_stt_model,
          voice_tts_provider: configuration.voice_tts_provider,
          voice_tts_model: configuration.voice_tts_model,
          voice_tts_voice: configuration.voice_tts_voice,
          voice_tts_enabled: configuration.voice_tts_enabled,
          coding_max_subtask_workers: configuration.coding_max_subtask_workers,
          coding_max_implementation_units: configuration.coding_max_implementation_units,
          coding_max_patch_retries_per_unit: configuration.coding_max_patch_retries_per_unit,
          coding_max_implementation_iterations: configuration.coding_max_implementation_iterations,
          coding_route_max_tokens: configuration.coding_route_max_tokens,
          coding_planner_max_tokens: configuration.coding_planner_max_tokens,
          coding_repo_navigation_max_tokens: configuration.coding_repo_navigation_max_tokens,
          coding_simple_patch_max_tokens: configuration.coding_simple_patch_max_tokens,
          coding_reconciliation_max_tokens: configuration.coding_reconciliation_max_tokens,
          coding_reconciliation_context_max_tokens: configuration.coding_reconciliation_context_max_tokens,
          coding_max_reasoning_reconciliations: configuration.coding_max_reasoning_reconciliations,
          coding_context_prompt_base_tokens: configuration.coding_context_prompt_base_tokens,
          coding_max_context_prompt_tokens: configuration.coding_max_context_prompt_tokens,
          coding_context_prompt_reserve_tokens: configuration.coding_context_prompt_reserve_tokens,
          coding_context_window_safety_tokens: configuration.coding_context_window_safety_tokens,
          coding_model_context_window_tokens: configuration.coding_model_context_window_tokens,
          reasoning_model_context_window_tokens: configuration.reasoning_model_context_window_tokens,
          coding_model_max_output_tokens: configuration.coding_model_max_output_tokens,
          reasoning_model_max_output_tokens: configuration.reasoning_model_max_output_tokens,
          secrets: changedSecrets,
          github_token: githubToken.trim() || undefined,
        },
      });
      setConfiguration(updated);
      onSaved?.(updated);
      setSecrets(emptySecrets());
      setGitHubToken("");
      setCatalogs({});
      await loadConfigurationCatalogs(updated);
      setMessage("Saved. New coding, vision, and voice runs will use this configuration.");
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
        className="flex max-h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-xl border border-line-strong bg-panel-soft shadow-2xl"
      >
        <header className="flex h-12 shrink-0 items-center justify-between border-b border-line px-4">
          <div>
            <h2 id="agent-settings-title" className="text-sm font-semibold text-ink">
              Agent configuration
            </h2>
            <p className="text-[10px] text-muted">
              Capability-aware model routing, runtime budgets, and backend-only credentials
            </p>
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
                  <ProviderSelect
                    label="Coding provider"
                    value={configuration.coding_provider}
                    providers={CHAT_PROVIDERS}
                    onChange={(provider) =>
                      void changeProvider({
                        providerField: "coding_provider",
                        modelField: "coding_model",
                        capability: "chat",
                        provider,
                      })
                    }
                  />
                  <ModelSelect
                    label="Coding model"
                    value={configuration.coding_model}
                    catalog={catalogFor(configuration.coding_provider, "chat")}
                    loading={catalogIsLoading(configuration.coding_provider, "chat")}
                    onChange={(model) => update("coding_model", model)}
                    onRefresh={() =>
                      void refreshCatalog(configuration.coding_provider, "chat", "coding_model")
                    }
                  />
                  <ProviderSelect
                    label="Reasoning provider"
                    value={configuration.reasoning_provider}
                    providers={CHAT_PROVIDERS}
                    onChange={(provider) =>
                      void changeProvider({
                        providerField: "reasoning_provider",
                        modelField: "reasoning_model",
                        capability: "chat",
                        provider,
                      })
                    }
                  />
                  <ModelSelect
                    label="Reasoning model"
                    value={configuration.reasoning_model}
                    catalog={catalogFor(configuration.reasoning_provider, "chat")}
                    loading={catalogIsLoading(configuration.reasoning_provider, "chat")}
                    onChange={(model) => update("reasoning_model", model)}
                    onRefresh={() =>
                      void refreshCatalog(configuration.reasoning_provider, "chat", "reasoning_model")
                    }
                  />
                  <ProviderSelect
                    label="Vision provider"
                    value={configuration.caption_provider}
                    providers={VISION_PROVIDERS}
                    onChange={(provider) =>
                      void changeProvider({
                        providerField: "caption_provider",
                        modelField: "caption_model",
                        capability: "vision",
                        provider,
                      })
                    }
                  />
                  <ModelSelect
                    label="Vision / caption model"
                    value={configuration.caption_model}
                    catalog={catalogFor(configuration.caption_provider, "vision")}
                    loading={catalogIsLoading(configuration.caption_provider, "vision")}
                    onChange={(model) => update("caption_model", model)}
                    onRefresh={() =>
                      void refreshCatalog(configuration.caption_provider, "vision", "caption_model")
                    }
                  />
                </div>
              </section>

              <section className="rounded-lg border border-line bg-panel p-4">
                <h3 className="text-xs font-semibold text-ink">Coding agent execution</h3>
                <p className="mt-1 text-[10px] leading-4 text-muted">
                  Worker count controls concurrency, implementation units and repair rounds control
                  how much total work a run can complete.
                </p>
                <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  <NumberInput
                    label="Subtask workers"
                    value={configuration.coding_max_subtask_workers}
                    min={1}
                    max={6}
                    help="Recommended: 3-4"
                    onChange={(value) => update("coding_max_subtask_workers", value)}
                  />
                  <NumberInput
                    label="Max implementation units"
                    value={configuration.coding_max_implementation_units}
                    min={1}
                    max={12}
                    help="Maximum independently implementable units in one plan."
                    onChange={(value) => update("coding_max_implementation_units", value)}
                  />
                  <NumberInput
                    label="Patch retries per unit"
                    value={configuration.coding_max_patch_retries_per_unit}
                    min={0}
                    max={4}
                    help="Default: 1 retry for a failed unit patch."
                    onChange={(value) => update("coding_max_patch_retries_per_unit", value)}
                  />
                  <NumberInput
                    label="Implementation repair rounds"
                    value={configuration.coding_max_implementation_iterations}
                    min={1}
                    max={8}
                    help="Default: 2 total implementation/repair iterations."
                    onChange={(value) => update("coding_max_implementation_iterations", value)}
                  />
                  <NumberInput
                    label="Router max tokens"
                    value={configuration.coding_route_max_tokens}
                    min={256}
                    max={2000}
                    help="Default: 900. Used only when LLM routing is enabled."
                    onChange={(value) => update("coding_route_max_tokens", value)}
                  />
                  <NumberInput
                    label="Planner max tokens"
                    value={configuration.coding_planner_max_tokens}
                    min={512}
                    max={6000}
                    help="Default: 3,000 for structured planning."
                    onChange={(value) => update("coding_planner_max_tokens", value)}
                  />
                  <NumberInput
                    label="Repo navigator max tokens"
                    value={configuration.coding_repo_navigation_max_tokens}
                    min={512}
                    max={4000}
                    help="Default: 1,600. Used only when LLM navigation is enabled."
                    onChange={(value) => update("coding_repo_navigation_max_tokens", value)}
                  />
                  <NumberInput
                    label="Patch worker max tokens"
                    value={configuration.coding_simple_patch_max_tokens}
                    min={2000}
                    max={16000}
                    help="Default: 8,000 per implementation-unit worker."
                    onChange={(value) => update("coding_simple_patch_max_tokens", value)}
                  />
                  <NumberInput
                    label="Reconciler max tokens"
                    value={configuration.coding_reconciliation_max_tokens}
                    min={2000}
                    max={32000}
                    help="Default: 10,000 for conflict reconciliation."
                    onChange={(value) => update("coding_reconciliation_max_tokens", value)}
                  />
                  <NumberInput
                    label="Reconciler context max tokens"
                    value={configuration.coding_reconciliation_context_max_tokens}
                    min={4000}
                    max={64000}
                    help="Default: 24,000 context tokens for reconciliation."
                    onChange={(value) => update("coding_reconciliation_context_max_tokens", value)}
                  />
                  <NumberInput
                    label="Max reasoning reconciliations"
                    value={configuration.coding_max_reasoning_reconciliations}
                    min={0}
                    max={3}
                    help="Default: 1 reasoning-model reconciliation pass."
                    onChange={(value) => update("coding_max_reasoning_reconciliations", value)}
                  />
                </div>
              </section>

              <section className="rounded-lg border border-line bg-panel p-4">
                <h3 className="text-xs font-semibold text-ink">Coding context and model budgets</h3>
                <p className="mt-1 text-[10px] leading-4 text-muted">
                  Prompt budgets are token-based. Context-window and output values are Default
                  limits used when a provider does not advertise a more specific model profile.
                </p>
                <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  <NumberInput
                    label="Base context prompt tokens"
                    value={configuration.coding_context_prompt_base_tokens}
                    min={4000}
                    max={64000}
                    help="Default: 16,000."
                    onChange={(value) => update("coding_context_prompt_base_tokens", value)}
                  />
                  <NumberInput
                    label="Max context prompt tokens"
                    value={configuration.coding_max_context_prompt_tokens}
                    min={8000}
                    max={128000}
                    help="Default: 32,000."
                    onChange={(value) => update("coding_max_context_prompt_tokens", value)}
                  />
                  <NumberInput
                    label="Context output reserve"
                    value={configuration.coding_context_prompt_reserve_tokens}
                    min={2000}
                    max={64000}
                    help="Default: 10,000 tokens reserved for model output."
                    onChange={(value) => update("coding_context_prompt_reserve_tokens", value)}
                  />
                  <NumberInput
                    label="Context safety reserve"
                    value={configuration.coding_context_window_safety_tokens}
                    min={1000}
                    max={32000}
                    help="Default: 6,000 safety tokens below the context window."
                    onChange={(value) => update("coding_context_window_safety_tokens", value)}
                  />
                  <NumberInput
                    label="Coding model context window"
                    value={configuration.coding_model_context_window_tokens}
                    min={16000}
                    max={2000000}
                    help="Default: 131,072 tokens."
                    onChange={(value) => update("coding_model_context_window_tokens", value)}
                  />
                  <NumberInput
                    label="Reasoning model context window"
                    value={configuration.reasoning_model_context_window_tokens}
                    min={16000}
                    max={2000000}
                    help="Default: 131,072 tokens."
                    onChange={(value) => update("reasoning_model_context_window_tokens", value)}
                  />
                  <NumberInput
                    label="Coding model max output"
                    value={configuration.coding_model_max_output_tokens}
                    min={2000}
                    max={128000}
                    help="Default: 32,000 tokens."
                    onChange={(value) => update("coding_model_max_output_tokens", value)}
                  />
                  <NumberInput
                    label="Reasoning model max output"
                    value={configuration.reasoning_model_max_output_tokens}
                    min={2000}
                    max={128000}
                    help="Default: 32,000 tokens."
                    onChange={(value) => update("reasoning_model_max_output_tokens", value)}
                  />
                </div>
              </section>

              <section className="rounded-lg border border-line bg-panel p-4">
                <h3 className="text-xs font-semibold text-ink">Voice agent models</h3>
                <p className="mt-1 text-[10px] leading-4 text-muted">
                  Chat, transcription, and speech synthesis are routed independently. Only providers
                  with the required audio capability are offered for STT and TTS.
                </p>
                <div className="mt-3 grid gap-3 sm:grid-cols-2">
                  <ProviderSelect
                    label="Voice chat provider"
                    value={configuration.voice_chat_provider}
                    providers={CHAT_PROVIDERS}
                    onChange={(provider) =>
                      void changeProvider({
                        providerField: "voice_chat_provider",
                        modelField: "voice_chat_model",
                        capability: "chat",
                        provider,
                      })
                    }
                  />
                  <ModelSelect
                    label="Voice chat model"
                    value={configuration.voice_chat_model}
                    catalog={catalogFor(configuration.voice_chat_provider, "chat")}
                    loading={catalogIsLoading(configuration.voice_chat_provider, "chat")}
                    onChange={(model) => update("voice_chat_model", model)}
                    onRefresh={() =>
                      void refreshCatalog(configuration.voice_chat_provider, "chat", "voice_chat_model")
                    }
                  />
                  <ProviderSelect
                    label="Speech-to-text provider"
                    value={configuration.voice_stt_provider}
                    providers={AUDIO_PROVIDERS}
                    onChange={(provider) =>
                      void changeProvider({
                        providerField: "voice_stt_provider",
                        modelField: "voice_stt_model",
                        capability: "stt",
                        provider,
                      })
                    }
                  />
                  <ModelSelect
                    label="Speech-to-text model"
                    value={configuration.voice_stt_model}
                    catalog={catalogFor(configuration.voice_stt_provider, "stt")}
                    loading={catalogIsLoading(configuration.voice_stt_provider, "stt")}
                    onChange={(model) => update("voice_stt_model", model)}
                    onRefresh={() =>
                      void refreshCatalog(configuration.voice_stt_provider, "stt", "voice_stt_model")
                    }
                  />
                  <ProviderSelect
                    label="Text-to-speech provider"
                    value={configuration.voice_tts_provider}
                    providers={AUDIO_PROVIDERS}
                    onChange={(provider) =>
                      void changeProvider({
                        providerField: "voice_tts_provider",
                        modelField: "voice_tts_model",
                        capability: "tts",
                        provider,
                      })
                    }
                  />
                  <ModelSelect
                    label="Text-to-speech model"
                    value={configuration.voice_tts_model}
                    catalog={catalogFor(configuration.voice_tts_provider, "tts")}
                    loading={catalogIsLoading(configuration.voice_tts_provider, "tts")}
                    onChange={(model) => update("voice_tts_model", model)}
                    onRefresh={() =>
                      void refreshCatalog(configuration.voice_tts_provider, "tts", "voice_tts_model")
                    }
                  />
                  <label className="text-[10px] font-medium text-muted">
                    Voice
                    <input
                      value={configuration.voice_tts_voice}
                      onChange={(event) => update("voice_tts_voice", event.target.value)}
                      className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/70"
                    />
                  </label>
                  <label className="flex cursor-pointer items-center gap-2 self-end pb-2 text-[11px] text-muted">
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
                      Live model discovery is account-aware when a key is configured. Without one,
                      the API returns a safe Default catalog. Secret values are never returned to
                      the renderer and remain session-only unless you configure environment variables
                      or Secrets Manager.
                    </p>
                  </div>
                </div>
                <div className="mt-3 rounded-md border border-line bg-surface/40 p-3">
                  <label className="text-[10px] font-medium text-muted">
                    GitHub token
                    <span
                      className={`ml-2 text-[9px] ${
                        configuration.github_token_configured ? "text-emerald-300" : "text-faint"
                      }`}
                    >
                      {configuration.github_token_configured ? "configured" : "not configured"}
                    </span>
                    <input
                      type="password"
                      autoComplete="new-password"
                      value={githubToken}
                      onChange={(event) => setGitHubToken(event.target.value)}
                      placeholder="Leave blank to keep current token"
                      className="mt-1 w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/70"
                    />
                  </label>
                  <p className="mt-1 text-[9px] leading-4 text-faint">
                    Used by repository import, status, push, and pull-request operations. The token
                    is never returned to the renderer and a value entered here is session-only.
                  </p>
                </div>
                <div className="mt-3 grid gap-3 sm:grid-cols-2">
                  {PROVIDERS.map((provider) => (
                    <label key={provider} className="text-[10px] font-medium text-muted">
                      {providerLabel[provider]} API key
                      <span
                        className={`ml-2 text-[9px] ${
                          configuration.secrets_configured[provider]
                            ? "text-emerald-300"
                            : "text-faint"
                        }`}
                      >
                        {configuration.secrets_configured[provider]
                          ? "configured"
                          : "not configured"}
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
          <button type="button" className="secondary-button" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="primary-button"
            disabled={!configuration || saving}
            onClick={() => void save()}
          >
            {saving ? <LoaderCircle size={12} className="animate-spin" /> : <Save size={12} />}
            Save configuration
          </button>
        </footer>
      </section>
    </div>
  );
};
