import type { RepositoryFile, RepositoryTreeResponse } from "../types";

type ApiClientConfig = {
  apiBaseUrl: string;
  apiKey?: string;
};

type RepositoryRequest = ApiClientConfig & {
  repoRoot: string;
};

export type GitHubStatus = {
  connected: boolean;
  token_kind: "user" | "installation";
  account?: string | null;
};

export type GitHubRepositorySummary = {
  id: number;
  full_name: string;
  name: string;
  owner: string;
  private: boolean;
  default_branch: string;
  clone_url: string;
  html_url: string;
  updated_at?: string | null;
  permissions: {
    admin: boolean;
    maintain: boolean;
    push: boolean;
    triage: boolean;
    pull: boolean;
  };
};

export type GitHubRepositoryImportResponse = {
  full_name: string;
  ref: string;
  repo_root: string;
  reused_existing_checkout: boolean;
};

// TODO: Replace renderer API-key access with a short-lived backend session token.
const apiUrl = (
  path: string,
  config: ApiClientConfig,
  params: Record<string, string | number | undefined>,
) => {
  const url = new URL(path, config.apiBaseUrl);

  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) url.searchParams.set(key, String(value));
  }

  return url;
};

const authHeaders = (apiKey?: string): HeadersInit => {
  return apiKey ? { "x-api-key": apiKey } : {};
};

async function readJson<T>(response: Response): Promise<T> {
  if (response.ok) return (await response.json()) as T;

  let message = `${response.status} ${response.statusText}`;
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (body.detail) {
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    }
  } catch {
    // Keep the HTTP status message.
  }

  throw new Error(message);
}

export const fetchRepositoryTree = async ({
  apiBaseUrl,
  apiKey,
  repoRoot,
  maxDepth = 8,
  maxEntries = 1500,
}: RepositoryRequest & {
  maxDepth?: number;
  maxEntries?: number;
}): Promise<RepositoryTreeResponse> => {
  const url = apiUrl("/coding-agent/repository/tree", { apiBaseUrl, apiKey }, {
    repo_root: repoRoot,
    max_depth: maxDepth,
    max_entries: maxEntries,
  });

  const response = await fetch(url, { headers: authHeaders(apiKey) });
  return readJson<RepositoryTreeResponse>(response);
};

export const fetchRepositoryFile = async ({
  apiBaseUrl,
  apiKey,
  repoRoot,
  path,
}: RepositoryRequest & { path: string }): Promise<RepositoryFile> => {
  const url = apiUrl("/coding-agent/repository/file", { apiBaseUrl, apiKey }, {
    repo_root: repoRoot,
    path,
  });

  const response = await fetch(url, { headers: authHeaders(apiKey) });
  return readJson<RepositoryFile>(response);
};

export const fetchGitHubStatus = async ({
  apiBaseUrl,
  apiKey,
}: ApiClientConfig): Promise<GitHubStatus> => {
  const response = await fetch(apiUrl("/github/status", { apiBaseUrl, apiKey }, {}), {
    headers: authHeaders(apiKey),
  });
  return readJson<GitHubStatus>(response);
};

export const fetchGitHubRepositories = async ({
  apiBaseUrl,
  apiKey,
}: ApiClientConfig): Promise<GitHubRepositorySummary[]> => {
  const response = await fetch(
    apiUrl("/github/repositories", { apiBaseUrl, apiKey }, { per_page: 100 }),
    { headers: authHeaders(apiKey) },
  );
  return readJson<GitHubRepositorySummary[]>(response);
};

export const importGitHubRepository = async ({
  apiBaseUrl,
  apiKey,
  fullName,
  ref,
  refresh = false,
}: ApiClientConfig & {
  fullName: string;
  ref?: string | null;
  refresh?: boolean;
}): Promise<GitHubRepositoryImportResponse> => {
  const response = await fetch(apiUrl("/github/repositories/import", { apiBaseUrl, apiKey }, {}), {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...authHeaders(apiKey),
    },
    body: JSON.stringify({
      full_name: fullName,
      ref: ref ?? null,
      refresh,
    }),
  });

  return readJson<GitHubRepositoryImportResponse>(response);
};
