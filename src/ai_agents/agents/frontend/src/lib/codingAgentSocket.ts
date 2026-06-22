export type CodingAgentAttachedFile = {
  name: string;
  content?: string | null;
  data_url?: string | null;
  path?: string | null;
  source: "upload" | "repo";
  mime_type?: string | null;
  size?: number | null;
  truncated?: boolean | null;
};


export type CodingAgentRunRequest = {
  request: string;
  repo_root: string;
  workspace_root?: string | null;
  allow_write?: boolean;
  thread_id?: string | null;
  memory_user_id?: string | null;
  memory_namespace?: string | null;
  memory_enabled?: boolean | null;
  setup_memory?: boolean;
  attached_files?: CodingAgentAttachedFile[];
};


export type CodingAgentRunResult = {
  thread_id: string;
  status: string;
  report?: string | null;
  selected_skill?: string | null;
  route_confidence?: number | null;
  route_reason?: string | null;
  plan: string[];
  files_inspected: string[];
  patch_summary?: string | null;
  file_changes: Record<string, unknown>[];
  diffs: string[];
  validation_commands: string[];
  validation_results: Record<string, unknown>[];
  memory_enabled: boolean;
  memory_namespace?: string | null;
  long_term_memories: string[];
  memory_errors: string[];
  errors: string[];
  raw: Record<string, unknown>;
};

export type CodingAgentServerEvent =
  | {
      type: "session.ready";
      payload: {
        message: string;
        protocol_version: string;
      };
    }
  | {
      type: "run.started";
      run_id: string;
      thread_id: string;
      payload: {
        repo_root: string;
        workspace_root: string | null;
        allow_write: boolean;
      };
    }
  | {
      type: "node.completed";
      run_id: string;
      thread_id: string;
      node: string;
      payload: Record<string, unknown>;
    }
  | {
      type: "run.completed";
      run_id: string;
      thread_id: string;
      payload: CodingAgentRunResult;
    }
  | {
      type: "run.failed";
      run_id?: string | null;
      thread_id?: string | null;
      payload: {
        error: string;
        error_type?: string;
        details?: unknown;
      };
    }
  | {
      type: "pong";
    };

type CodingAgentSocketOptions = {
  apiBaseUrl: string;
  apiKey: string;
  onEvent: (event: CodingAgentServerEvent) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (event: Event) => void;
};




const makeSocketUrl = (apiBaseUrl: string, apiKey?: string) => {
  const url = new URL("/coding-agent/ws", apiBaseUrl);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  if (apiKey) url.searchParams.set("api_key", apiKey);
  return url;
}


export const createCodingAgentSocket = (options: CodingAgentSocketOptions) => {
  const socket = new WebSocket(makeSocketUrl(options.apiBaseUrl, options.apiKey));
  const pendingMessages: string[] = [];

  const sendMessage = (message: unknown) => {
    const serialized = JSON.stringify(message);

    if (socket.readyState === WebSocket.OPEN) {
      socket.send(serialized);
      return;
    }

    pendingMessages.push(serialized);
  }


  socket.addEventListener("open", () => {
    while (pendingMessages.length > 0) {
      const message = pendingMessages.shift();
      if (message) socket.send(message);
    }

    options.onOpen?.();
  });


  socket.addEventListener("message", (message) => {
    try {
      const event = JSON.parse(message.data) as CodingAgentServerEvent;
      options.onEvent(event);
    } catch (error) {
      console.error("Failed to parse coding agent WebSocket event.", error);
    }
  });


  socket.addEventListener("close", () => {
    options.onClose?.();
  });


  socket.addEventListener("error", (event) => {
    options.onError?.(event);
  });

  return {
    socket,

    run(request: CodingAgentRunRequest) {
      sendMessage({
        type: "run.request",
        payload: request,
      });
    },

    ping() {
      sendMessage({
        type: "ping",
        payload: {},
      });
    },

    close() {
      socket.close();
    },
  };
}


