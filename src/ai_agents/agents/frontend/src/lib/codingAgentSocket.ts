/**
 * Frontend WebSocket contract for the coding-agent API.
 *
 * Mirrors:
 * - api/api_schemas.py::CodingAgentAttachedFile
 * - api/api_schemas.py::CodingAgentRunRequest
 * - api/api_schemas.py::CodingAgentRunResult
 * - api/api_schemas.py::CodingAgentClientMessage
 * - api/api_schemas.py::CodingAgentServerEvent
 *
 * The discriminated client/server unions below are intentionally more specific
 * than the backend's generic `payload: dict[str, Any]` envelope. Their payload
 * shapes mirror what `api/coding_agent.py` actually reads and emits.
 */

export type CodingAgentAttachedFile = {
  name: string;
  content?: string | null;
  data_url?: string | null;
  path?: string | null;
  /** Backend default: "upload". */
  source?: "upload" | "repo";
  mime_type?: string | null;
  size?: number | null;
  truncated?: boolean | null;
};

export type CodingAgentTaskMode = "simple" | "standard" | "parallel";

export type ApprovalStatus = "not_required" | "pending" | "applied" | "rejected";

/**
 * Public implementation-unit payload carried in CodingAgentRunResult.
 * `api_schemas.py` deliberately exposes implementation units as dictionaries,
 * so known runtime keys are typed while unknown forward-compatible keys remain
 * allowed.
 */
export type CodingAgentImplementationUnit = Record<string, unknown> & {
  id?: string;
  objective?: string;
  acceptance_criteria?: string[];
  search_requests?: Record<string, unknown>[];
  candidate_paths?: string[];
  depends_on?: string[];
  validation_commands?: string[];

  /** Legacy aliases retained for persisted/older runs. */
  unit_id?: string;
  title?: string;
  target_paths?: string[];
  dependencies?: string[];
};

export type CodingAgentCompletionLedgerEntry = Record<string, unknown> & {
  status?: string;
  patch_retries?: number;
  generation?: number;
  message?: string;
};

export type CodingAgentCompletionLedger = Record<string, CodingAgentCompletionLedgerEntry>;

/**
 * Known runtime settings emitted by coding_agent.py. The API schema intentionally
 * keeps this dictionary open-ended, so additional backend settings remain valid.
 */
export type CodingAgentRuntimeSettings = Record<string, unknown> & {
  max_subtask_workers?: number;
  max_context_workers?: number;
  max_implementation_units?: number;
  max_patch_retries_per_unit?: number;
  max_implementation_iterations?: number;
  route_max_tokens?: number;
  planner_max_tokens?: number;
  repo_navigation_max_tokens?: number;
  simple_patch_max_tokens?: number;
  patch_max_tokens?: number;
  reconciliation_max_tokens?: number;
  reconciliation_context_max_tokens?: number;
  max_reasoning_reconciliations?: number;
  progress_max_tokens?: number;
  context_prompt_base_tokens?: number;
  max_context_prompt_tokens?: number;
  context_prompt_reserve_tokens?: number;
  context_window_safety_tokens?: number;
  coding_model_context_window_tokens?: number;
  reasoning_model_context_window_tokens?: number;
  coding_model_max_output_tokens?: number;
  reasoning_model_max_output_tokens?: number;
};

export type CodingAgentTokenBudgets = {
  route: number;
  planner: number;
  repo_navigation: number;
  patch_worker: number;
  reconciliation: number;
  reconciliation_context: number;
};

/** Mirrors api_schemas.py::CodingAgentRunRequest. */
export type CodingAgentRunRequest = {
  request: string;
  repo_root: string;
  workspace_root?: string | null;
  allow_write?: boolean;
  thread_id?: string | null;
  memory_user_id?: string | null;
  memory_namespace?: string | null;
  memory_enabled?: boolean | null;
  setup_memory?: boolean | null;

  /** Primary divide-and-conquer implementation loop limit. */
  max_implementation_iterations?: number | null;
  /** Legacy compatibility alias. */
  max_iterations?: number | null;

  /** Primary implementation-worker concurrency override. */
  subtask_worker_count?: number | null;
  /** Legacy compatibility alias. */
  subagent_count?: number | null;

  route_max_tokens?: number | null;
  planner_max_tokens?: number | null;
  repo_navigation_max_tokens?: number | null;
  simple_patch_max_tokens?: number | null;
  patch_max_tokens?: number | null;
  progress_max_tokens?: number | null;

  attached_files?: CodingAgentAttachedFile[];
};

/**
 * Mirrors api_schemas.py::CodingAgentRunResult as serialized by Pydantic
 * `model_dump()` for the `run.completed` event. Fields with backend defaults are
 * still present on the completed response; nullable fields therefore use `| null`
 * instead of being optional.
 */
export type CodingAgentRunResult = {
  thread_id: string;
  status: string;

  report: string | null;
  markdown_response: string | null;

  selected_skill: string | null;
  selected_skills: string[];
  task_mode: CodingAgentTaskMode | null;

  /** Legacy planning/context aliases retained by the backend. */
  subtasks: Record<string, unknown>[];
  context_worker_count: number;

  implementation_units: CodingAgentImplementationUnit[];
  completion_ledger: CodingAgentCompletionLedger;
  implementation_generation: number;
  implementation_iteration: number;
  max_implementation_iterations: number;
  subtask_worker_count: number;
  subtask_worker_results: Record<string, unknown>[];
  runtime_settings: CodingAgentRuntimeSettings;

  route_confidence: number | null;
  route_reason: string | null;

  plan: string[];
  files_inspected: string[];
  patch_summary: string | null;
  file_changes: Record<string, unknown>[];
  diffs: string[];

  validation_commands: string[];
  validation_results: Record<string, unknown>[];

  approval_required: boolean;
  approval_status: ApprovalStatus;
  blocking_validation_failed: boolean;
  advisory_validation_failed: boolean;
  applied_files: string[];

  memory_enabled: boolean;
  memory_namespace: string | null;
  long_term_memories: string[];
  memory_errors: string[];

  errors: string[];
  raw: Record<string, unknown>;
};

/**
 * Exact client messages understood by coding_agent.py inside the generic
 * CodingAgentClientMessage Pydantic envelope.
 */
export type CodingAgentClientMessage =
  | {
      type: "ping";
      payload: Record<string, never>;
    }
  | {
      type: "run.request";
      payload: CodingAgentRunRequest;
    }
  | {
      type: "run.apply.request";
      payload: {
        thread_id: string;
        paths?: string[] | null;
      };
    }
  | {
      type: "run.reject.request";
      payload: {
        thread_id: string;
      };
    };

/**
 * All CodingAgentServerEvent model dumps contain these envelope fields. They are
 * serialized by Pydantic, including null metadata values. Event-specific variants
 * narrow fields that coding_agent.py guarantees to be non-null.
 */
type CodingAgentServerEventEnvelope = {
  run_id: string | null;
  thread_id: string | null;
  node: string | null;
};

export type CodingAgentServerEvent =
  | (CodingAgentServerEventEnvelope & {
      type: "session.ready";
      payload: {
        message: string;
        protocol_version: string;
      };
    })
  | (CodingAgentServerEventEnvelope & {
      type: "run.started";
      run_id: string;
      thread_id: string;
      payload: {
        repo_root: string;
        workspace_root: string | null;
        allow_write: boolean;
        subtask_worker_count: number;
        /** Legacy alias emitted for older frontends. */
        subagent_count: number;
        max_implementation_iterations: number;
        token_budgets: CodingAgentTokenBudgets;
        runtime_settings: CodingAgentRuntimeSettings;
      };
    })
  | (CodingAgentServerEventEnvelope & {
      type: "node.completed";
      run_id: string;
      thread_id: string;
      node: string;
      payload: Record<string, unknown>;
    })
  | (CodingAgentServerEventEnvelope & {
      type: "run.completed";
      run_id: string;
      thread_id: string;
      payload: CodingAgentRunResult;
    })
  | (CodingAgentServerEventEnvelope & {
      type: "run.failed";
      payload: {
        error: string;
        error_type?: string;
        details?: unknown;
      };
    })
  | (CodingAgentServerEventEnvelope & {
      type: "run.approval_required";
      run_id: string;
      thread_id: string;
      payload: {
        thread_id: string;
        changed_paths: string[];
        blocking_validation_failed: boolean;
        advisory_validation_failed: boolean;
      };
    })
  | (CodingAgentServerEventEnvelope & {
      type: "run.applied";
      run_id: string;
      thread_id: string;
      payload: {
        thread_id: string;
        applied_files: string[];
        remaining_paths: string[];
        approval_status: "pending" | "applied";
      };
    })
  | (CodingAgentServerEventEnvelope & {
      type: "run.rejected";
      payload: {
        thread_id: string;
        approval_status: "rejected";
      };
    })
  | (CodingAgentServerEventEnvelope & {
      type: "pong";
      payload: Record<string, never>;
    });

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
};

export const createCodingAgentSocket = (options: CodingAgentSocketOptions) => {
  const socket = new WebSocket(makeSocketUrl(options.apiBaseUrl, options.apiKey));
  const pendingMessages: string[] = [];

  const sendMessage = (message: CodingAgentClientMessage) => {
    const serialized = JSON.stringify(message);

    if (socket.readyState === WebSocket.OPEN) {
      socket.send(serialized);
      return;
    }

    pendingMessages.push(serialized);
  };

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

    apply(threadId: string, paths?: string[]) {
      sendMessage({
        type: "run.apply.request",
        payload: {
          thread_id: threadId,
          paths,
        },
      });
    },

    reject(threadId: string) {
      sendMessage({
        type: "run.reject.request",
        payload: {
          thread_id: threadId,
        },
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
};
