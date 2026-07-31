type ApiClientConfig = {
  apiBaseUrl: string;
  apiKey?: string;
};

const authHeaders = (apiKey?: string): HeadersInit =>
  apiKey ? { "x-api-key": apiKey } : {};

async function readJson<T>(response: Response): Promise<T> {
  if (response.ok) return (await response.json()) as T;

  let message = `${response.status} ${response.statusText}`;
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (body.detail) {
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    }
  } catch {
    // Preserve the HTTP status when a JSON error body is unavailable.
  }
  throw new Error(message);
}

export type ChatProvider =
  | "groq"
  | "deepseek"
  | "openrouter"
  | "openai"
  | "anthropic";

export type ModelCapability = "chat" | "vision" | "stt" | "tts";

export type AgentConfiguration = {
  coding_provider: ChatProvider;
  coding_model: string;
  reasoning_provider: ChatProvider;
  reasoning_model: string;
  caption_provider: ChatProvider;
  caption_model: string;
  voice_chat_provider: ChatProvider;
  voice_chat_model: string;
  voice_stt_provider: ChatProvider;
  voice_stt_model: string;
  voice_tts_provider: ChatProvider;
  voice_tts_model: string;
  voice_tts_voice: string;
  voice_tts_enabled: boolean;
  secrets_configured: Record<ChatProvider, boolean>;
  secrets_persistence: "session_only";
};

export type UpdateAgentConfiguration = Omit<
  AgentConfiguration,
  "secrets_configured" | "secrets_persistence"
> & {
  secrets?: Partial<Record<ChatProvider, string>>;
};

export type ModelCatalogResponse = {
  provider: ChatProvider;
  capability: ModelCapability;
  models: string[];
  source: "live" | "fallback";
  secret_configured: boolean;
  error?: string | null;
};

export const fetchAgentConfiguration = async ({
  apiBaseUrl,
  apiKey,
}: ApiClientConfig): Promise<AgentConfiguration> => {
  const response = await fetch(`${apiBaseUrl}/admin/agent-configuration`, {
    headers: authHeaders(apiKey),
  });
  return readJson<AgentConfiguration>(response);
};

export const updateAgentConfiguration = async ({
  apiBaseUrl,
  apiKey,
  configuration,
}: ApiClientConfig & {
  configuration: UpdateAgentConfiguration;
}): Promise<AgentConfiguration> => {
  const response = await fetch(`${apiBaseUrl}/admin/agent-configuration`, {
    method: "PUT",
    headers: {
      "content-type": "application/json",
      ...authHeaders(apiKey),
    },
    body: JSON.stringify(configuration),
  });
  return readJson<AgentConfiguration>(response);
};

export const fetchAvailableModels = async ({
  apiBaseUrl,
  apiKey,
  provider,
  capability,
}: ApiClientConfig & {
  provider: ChatProvider;
  capability: ModelCapability;
}): Promise<ModelCatalogResponse> => {
  const url = new URL("/admin/models", apiBaseUrl);
  url.searchParams.set("provider", provider);
  url.searchParams.set("capability", capability);
  const response = await fetch(url, { headers: authHeaders(apiKey) });
  return readJson<ModelCatalogResponse>(response);
};

export type AgentKind = "coding" | "voice";

export type SkillSummary = {
  agent: AgentKind;
  name: string;
  purpose: string;
  allowed_tools: string[];
  content: string;
  custom: boolean;
};

export const fetchSkills = async ({
  apiBaseUrl,
  apiKey,
  agent,
}: ApiClientConfig & { agent: AgentKind }): Promise<SkillSummary[]> => {
  const url = new URL("/admin/skills", apiBaseUrl);
  url.searchParams.set("agent", agent);
  const response = await fetch(url, { headers: authHeaders(apiKey) });
  return readJson<SkillSummary[]>(response);
};

export const saveSkill = async ({
  apiBaseUrl,
  apiKey,
  agent,
  name,
  content,
  overwrite,
}: ApiClientConfig & {
  agent: AgentKind;
  name: string;
  content: string;
  overwrite: boolean;
}): Promise<SkillSummary> => {
  const response = await fetch(`${apiBaseUrl}/admin/skills`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...authHeaders(apiKey),
    },
    body: JSON.stringify({ agent, name, content, overwrite }),
  });
  return readJson<SkillSummary>(response);
};

export const deleteSkill = async ({
  apiBaseUrl,
  apiKey,
  agent,
  name,
}: ApiClientConfig & {
  agent: AgentKind;
  name: string;
}): Promise<{ deleted: boolean }> => {
  const response = await fetch(
    `${apiBaseUrl}/admin/skills/${encodeURIComponent(agent)}/${encodeURIComponent(name)}`,
    {
      method: "DELETE",
      headers: authHeaders(apiKey),
    },
  );
  return readJson<{ deleted: boolean }>(response);
};

export type ToolSummary = {
  agent: AgentKind;
  name: string;
  module: string;
  purpose: string;
  status: "builtin" | "pending_review";
};

export const fetchTools = async ({
  apiBaseUrl,
  apiKey,
  agent,
}: ApiClientConfig & { agent: AgentKind }): Promise<ToolSummary[]> => {
  const url = new URL("/admin/tools", apiBaseUrl);
  url.searchParams.set("agent", agent);
  const response = await fetch(url, { headers: authHeaders(apiKey) });
  return readJson<ToolSummary[]>(response);
};

export const quarantineTool = async ({
  apiBaseUrl,
  apiKey,
  agent,
  name,
  purpose,
  source,
}: ApiClientConfig & {
  agent: AgentKind;
  name: string;
  purpose: string;
  source: string;
}): Promise<ToolSummary> => {
  const response = await fetch(`${apiBaseUrl}/admin/tools/quarantine`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...authHeaders(apiKey),
    },
    body: JSON.stringify({ agent, name, purpose, source }),
  });
  return readJson<ToolSummary>(response);
};
