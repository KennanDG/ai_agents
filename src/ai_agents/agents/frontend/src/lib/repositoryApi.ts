import type { RepositoryFile, RepositoryTreeResponse } from "../types";

type ApiClientConfig = {
    apiBaseUrl: string;
    apiKey?: string;
};

type RepositoryRequest = ApiClientConfig & {
    repoRoot: string;
};



//TODO: Create Websocket token so api key does not sit in renderer
const apiUrl = (path: string, config: ApiClientConfig, params: Record<string, string | number | undefined>) => {
    const url = new URL(path, config.apiBaseUrl);

    for (const [key, value] of Object.entries(params)) {
        if (value !== undefined) url.searchParams.set(key, String(value));
    }

    // This mirrors the WebSocket auth style and keeps local dev simple.
    if (config.apiKey) url.searchParams.set("api_key", config.apiKey);

    return url;
}


const authHeaders = (apiKey?: string): HeadersInit => {
  return apiKey ? { "x-api-key": apiKey } : {};
}



async function readJson<T>(response: Response): Promise<T> {
    if (response.ok) return (await response.json()) as T;

    let message = `${response.status} ${response.statusText}`;
    try {
        const body = (await response.json()) as { detail?: unknown };
        if (body.detail) message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
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
}: RepositoryRequest & { maxDepth?: number; maxEntries?: number }): Promise<RepositoryTreeResponse>  => {
  
    const url = apiUrl("/coding-agent/repository/tree", { apiBaseUrl, apiKey }, {
        repo_root: repoRoot,
        max_depth: maxDepth,
        max_entries: maxEntries,
    });

    const response = await fetch(url, { headers: authHeaders(apiKey) });
    return readJson<RepositoryTreeResponse>(response);
}



export const fetchRepositoryFile = async ({
  apiBaseUrl,
  apiKey,
  repoRoot,
  path,
}: RepositoryRequest & { path: string }): Promise<RepositoryFile>  => {
    
    const url = apiUrl("/coding-agent/repository/file", { apiBaseUrl, apiKey }, {
        repo_root: repoRoot,
        path,
    });

    const response = await fetch(url, { headers: authHeaders(apiKey) });
    return readJson<RepositoryFile>(response);
}
