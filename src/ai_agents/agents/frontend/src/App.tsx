import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
// import { ActivityBar } from "./components/ActivityBar";
import { DiffPanel } from "./components/DiffPanel";
import { OutputPanel } from "./components/OutputPanel";
import { Sidebar } from "./components/Sidebar";
import { TaskPanel } from "./components/TaskPanel";

import {
  createCodingAgentSocket,
  type CodingAgentAttachedFile,
  type CodingAgentRunResult,
  type CodingAgentServerEvent,
} from "./lib/codingAgentSocket";

import { fetchRepositoryFile, fetchRepositoryTree } from "./lib/repositoryApi";
import type { AgentMessage, AgentRunState, ChangeStatus, FileChange, RepositoryFile, RepositoryTreeEntry } from "./types";

const apiBaseUrl = import.meta.env.VITE_AI_AGENTS_API_BASE ?? "http://localhost:8000";
const apiKey = import.meta.env.VITE_AI_AGENTS_API_KEY ?? "";
const configuredRepoRoot = import.meta.env.VITE_CODING_AGENT_REPO_ROOT ?? ".";
const configuredWorkspaceRoot = import.meta.env.VITE_CODING_AGENT_WORKSPACE_ROOT ?? configuredRepoRoot;

const initialRunState: AgentRunState = {
  status: "connecting",
  plan: [],
  completedNodes: [],
  filesInspected: [],
  fileChanges: [],
  diffs: [],
  validationCommands: [],
  validationResults: [],
  approvalRequired: false,
  approvalStatus: "not_required",
  blockingValidationFailed: false,
  advisoryValidationFailed: false,
  appliedFiles: [],
  errors: [],
  logs: [],
};

const nowLabel = () => {
  return new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(new Date());
}

const asStringArray = (value: unknown): string[] | undefined => {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : undefined;
}

const asRecordArray = (value: unknown): Record<string, unknown>[] | undefined => {
  return Array.isArray(value) ? value.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object") : undefined;
}


const languageFromPath = (path: string) => {
  const extension = path.split(".").at(-1)?.toLowerCase();
  switch (extension) {
    case "c":
    case "cc":
    case "cpp":
    case "cxx":
    case "c++":
    case "h":
    case "hh":
    case "hpp":
    case "hxx":
      return "cpp";
    case "css":
      return "css";
    case "html":
      return "html";
    case "java":
      return "java";
    case "js":
    case "jsx":
      return "javascript";
    case "json":
      return "json";
    case "md":
      return "markdown";
    case "py":
      return "python";
    case "rs":
      return "rust";
    case "ts":
    case "tsx":
      return "typescript";
    case "yml":
    case "yaml":
      return "yaml";
    default:
      return "plaintext";
  }
}


const asChangeStatus = (value: unknown): ChangeStatus => {
  return value === "added" || value === "deleted" || value === "modified" ? value : "modified";
}


const asFileChanges = (value: unknown): FileChange[] | undefined => {
  if (!Array.isArray(value)) return undefined;

  return value.flatMap((item) => {
    if (!item || typeof item !== "object") return [];

    const record = item as Record<string, unknown>;
    const path = record.path ?? record.file_path ?? record.file;
    if (typeof path !== "string") return [];

    return [{
      path,
      status: asChangeStatus(record.status),
      additions: typeof record.additions === "number" ? record.additions : 0,
      deletions: typeof record.deletions === "number" ? record.deletions : 0,
      language: typeof record.language === "string" ? record.language : languageFromPath(path),
      original: typeof record.original === "string" ? record.original : typeof record.before === "string" ? record.before : "",
      modified: typeof record.modified === "string" ? record.modified : typeof record.after === "string" ? record.after : "",
    }];
  });
}


const mergeResult = (state: AgentRunState, result: CodingAgentRunResult): AgentRunState => {
  return {
    ...state,
    threadId: result.thread_id,
    selectedSkill: result.selected_skill,
    routeConfidence: result.route_confidence,
    routeReason: result.route_reason,
    plan: result.plan ?? state.plan,
    filesInspected: result.files_inspected ?? state.filesInspected,
    patchSummary: result.patch_summary,
    fileChanges: asFileChanges(result.file_changes) ?? state.fileChanges,
    diffs: result.diffs ?? state.diffs,
    validationCommands: result.validation_commands ?? state.validationCommands,
    validationResults: result.validation_results ?? state.validationResults,
    approvalRequired: Boolean(result.approval_required),
    approvalStatus: result.approval_status ?? state.approvalStatus,
    blockingValidationFailed: Boolean(result.blocking_validation_failed),
    advisoryValidationFailed: Boolean(result.advisory_validation_failed),
    appliedFiles: result.applied_files ?? state.appliedFiles,
    report: result.report,
    errors: result.errors ?? state.errors,
  };
}


const runReducer = (state: AgentRunState, event: CodingAgentServerEvent): AgentRunState => {
  switch (event.type) {
    case "session.ready":
      return {
        ...state,
        status: "ready",
        logs: [...state.logs, `[socket] ${event.payload.message}`],
      };

    case "run.started":
      return {
        ...initialRunState,
        status: "running",
        runId: event.run_id,
        threadId: event.thread_id,
        logs: [
          `[run] started ${event.thread_id}`,
          `[repo] ${event.payload.repo_root}`,
          `[mode] ${event.payload.allow_write ? "write" : "read-only"}`,
        ],
      };

    case "node.completed": {
      const payload = event.payload;
      const plan = asStringArray(payload.plan) ?? state.plan;
      const filesInspected = asStringArray(payload.files_inspected) ?? state.filesInspected;
      const fileChanges = asFileChanges(payload.file_changes) ?? state.fileChanges;
      const diffs = asStringArray(payload.diffs) ?? state.diffs;
      const validationCommands = asStringArray(payload.validation_commands) ?? state.validationCommands;
      const validationResults = asRecordArray(payload.validation_results) ?? state.validationResults;
      const errors = asStringArray(payload.errors) ?? state.errors;

      return {
        ...state,
        runId: event.run_id,
        threadId: event.thread_id,
        plan,
        filesInspected,
        fileChanges,
        diffs,
        validationCommands,
        validationResults,
        errors,
        selectedSkill: typeof payload.selected_skill === "string" ? payload.selected_skill : state.selectedSkill,
        routeConfidence: typeof payload.route_confidence === "number" ? payload.route_confidence : state.routeConfidence,
        routeReason: typeof payload.route_reason === "string" ? payload.route_reason : state.routeReason,
        patchSummary: typeof payload.patch_summary === "string" ? payload.patch_summary : state.patchSummary,
        report: typeof payload.report === "string" ? payload.report : state.report,
        completedNodes: [...state.completedNodes, event.node],
        logs: [...state.logs, `[node] completed ${event.node}`],
      };
    }

    case "run.completed":
      return {
        ...mergeResult(state, event.payload),
        status: "completed",
        runId: event.run_id,
        threadId: event.thread_id,
        logs: [...state.logs, `[run] completed ${event.thread_id}`],
      };

    case "run.failed":
      return {
        ...state,
        status: "failed",
        runId: event.run_id,
        threadId: event.thread_id,
        errors: [...state.errors, event.payload.error],
        logs: [...state.logs, `[error] ${event.payload.error}`],
      };

    case "run.approval_required":
      return {
        ...state,
        status: "approval_pending",
        runId: event.run_id,
        threadId: event.thread_id,
        approvalRequired: true,
        approvalStatus: "pending",
        blockingValidationFailed: event.payload.blocking_validation_failed,
        advisoryValidationFailed: event.payload.advisory_validation_failed,
        logs: [...state.logs, "[approval] waiting for user approval"],
      };

    case "run.applied":
      return {
        ...state,
        status: event.payload.approval_status === "applied" ? "applied" : "approval_pending",
        approvalStatus: event.payload.approval_status,
        appliedFiles: [...state.appliedFiles, ...event.payload.applied_files],
        logs: [
          ...state.logs,
          `[approval] applied ${event.payload.applied_files.length} file(s)`,
        ],
      };

    case "run.rejected":
      return {
        ...state,
        status: "rejected",
        approvalStatus: "rejected",
        approvalRequired: false,
        logs: [...state.logs, "[approval] rejected changes"],
      };


    case "pong":
      return { ...state, logs: [...state.logs, "[socket] pong"] };

    default:
      return state;
  }
}


const App = () => {
  const [activePath, setActivePath] = useState<string | null>(null);
  const [activeFile, setActiveFile] = useState<RepositoryFile | null>(null);
  const [allowWrite, setAllowWrite] = useState(true);

  const [fileLoading, setFileLoading] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);
  
  const [memoryEnabled, setMemoryEnabled] = useState(true);
  const [messages, setMessages] = useState<AgentMessage[]>([]);

  // const [outputOpen, setOutputOpen] = useState(false);

  const [repoRoot, setRepoRoot] = useState(configuredRepoRoot);
  const [repoEntries, setRepoEntries] = useState<RepositoryTreeEntry[]>([]);
  const [repoLoading, setRepoLoading] = useState(false);
  const [repoError, setRepoError] = useState<string | null>(null);
  const [run, dispatchRun] = useReducer(runReducer, initialRunState);

  // const [sidebarOpen, setSidebarOpen] = useState(true);
  const socketRef = useRef<ReturnType<typeof createCodingAgentSocket> | null>(null);

  
  const activeChange = useMemo(
    () => run.fileChanges.find((change) => change.path === activePath && (change.original || change.modified)) ?? null,
    [activePath, run.fileChanges],
  );

  const repoName = useMemo(() => repoRoot.split(/[\\/]/).filter(Boolean).at(-1) ?? "repository", [repoRoot]);


  const refreshRepository = useCallback(async () => {
    setRepoLoading(true);
    setRepoError(null);

    try {
      const tree = await fetchRepositoryTree({ apiBaseUrl, apiKey, repoRoot: configuredRepoRoot });
      setRepoRoot(tree.repo_root);
      setRepoEntries(tree.entries);

      const firstFile = tree.entries.find((entry) => entry.kind === "file");
      setActivePath((current) => current ?? firstFile?.path ?? null);
    } catch (error) {
      setRepoError(error instanceof Error ? error.message : "Failed to load repository.");
    } finally {
      setRepoLoading(false);
    }
  }, []);



  useEffect(() => {
    void refreshRepository();
  }, [refreshRepository]);



  useEffect(() => {
    if (!activePath) {
      setActiveFile(null);
      return;
    }

    const abortController = new AbortController();
    setActiveFile(null);
    setFileLoading(true);
    setFileError(null);

    fetchRepositoryFile({ apiBaseUrl, apiKey, repoRoot, path: activePath })
      .then((file) => {
        if (!abortController.signal.aborted) setActiveFile(file);
      })
      .catch((error) => {
        if (!abortController.signal.aborted) setFileError(error instanceof Error ? error.message : "Failed to load file.");
      })
      .finally(() => {
        if (!abortController.signal.aborted) setFileLoading(false);
      });

    return () => abortController.abort();
  }, [activePath, repoRoot]);



  useEffect(() => {
    const client = createCodingAgentSocket({
      apiBaseUrl,
      apiKey,
      onEvent: (event) => {
        dispatchRun(event);

        if (event.type === "run.completed" && event.payload.report) {
          setMessages((current) => [
            ...current,
            { id: crypto.randomUUID(), role: "agent", body: event.payload.report ?? "Run completed.", time: nowLabel() },
          ]);
        }

        if (event.type === "run.failed") {
          setMessages((current) => [
            ...current,
            { id: crypto.randomUUID(), role: "agent", body: event.payload.error, time: nowLabel() },
          ]);
        }
      },
      onOpen: () => {
        console.log("Coding agent socket connected.");
      },
      onClose: () => {
        dispatchRun({ type: "run.failed", payload: { error: "Coding agent socket closed." } });
      },
      onError: (event) => {
        console.error("Coding agent socket error.", event);
      },
    });

    socketRef.current = client;

    return () => {
      client.close();
      socketRef.current = null;
    };
  }, []);




  const approveAllChanges = () => {
    if (!run.threadId) return;
    socketRef.current?.apply(run.threadId);
  };

  const approveFileChange = (path: string) => {
    if (!run.threadId) return;
    socketRef.current?.apply(run.threadId, [path]);
  };

  const rejectChanges = () => {
    if (!run.threadId) return;
    socketRef.current?.reject(run.threadId);
  };



  
  const submitPrompt = (prompt: string, attachedFiles: CodingAgentAttachedFile[] = []) => {
    const attachmentLabel =
      attachedFiles.length > 0
        ? `\n\nAttached files:\n${attachedFiles.map((file) => `- ${file.name}`).join("\n")}`
        : "";

    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: "user",
        body: prompt + attachmentLabel,
        time: nowLabel(),
      },
    ]);

    socketRef.current?.run({
      thread_id: run.threadId,
      request: prompt,
      repo_root: repoRoot,
      workspace_root: configuredWorkspaceRoot === configuredRepoRoot ? repoRoot : configuredWorkspaceRoot,
      allow_write: allowWrite,
      memory_enabled: memoryEnabled,
      attached_files: attachedFiles,
      max_iterations: 3,
    });
  };



  return (
    <main className="flex h-dvh min-h-0 min-w-0 overflow-hidden bg-canvas text-ink">
      {/* <ActivityBar /> */}


      <Sidebar
        repoName={repoName}
        repoRoot={repoRoot}
        entries={repoEntries}
        changes={run.fileChanges}
        activePath={activePath}
        isLoading={repoLoading}
        error={repoError}
        onSelect={setActivePath}
        onRefresh={refreshRepository}
      />

      {/* {sidebarOpen ? (
        <div className="hidden lg:flex">
          <Sidebar
            repoName={repoName}
            repoRoot={repoRoot}
            entries={repoEntries}
            changes={run.fileChanges}
            activePath={activePath}
            isLoading={repoLoading}
            error={repoError}
            onSelect={setActivePath}
            onRefresh={refreshRepository}
          />
        </div>
        
      ) : null} */}

      <div className="flex min-h-0 w-90 shrink-0 flex-col border-r border-line bg-panel-soft">
        <div className="flex shrink-0 items-center gap-3 border-b border-line px-3 py-2">
          <label className="flex cursor-pointer items-center gap-1.5 text-xs text-ink-soft">
            <input
              type="checkbox"
              checked={allowWrite}
              onChange={() => setAllowWrite(!allowWrite)}
              className="accent-accent"
            />
            Allow Write
          </label>

          <label className="flex cursor-pointer items-center gap-1.5 text-xs text-ink-soft">
            <input
              type="checkbox"
              checked={memoryEnabled}
              onChange={() => setMemoryEnabled(!memoryEnabled)}
              className="accent-accent"
            />
            Memory Enabled
          </label>
        </div>

        <TaskPanel
          messages={messages}
          run={run}
          onSubmit={submitPrompt}
          allowWrite={allowWrite}
          activePath={activePath}
          activeFile={activeFile}
          onApproveAll={approveAllChanges}
          onRejectChanges={rejectChanges}
        />
      </div>

      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <DiffPanel
          file={activeFile}
          change={activeChange}
          isLoading={fileLoading}
          error={fileError}
          canApprove={run.approvalStatus === "pending"}
          onAcceptFile={approveFileChange}
          onRejectChanges={rejectChanges}
        />

        <OutputPanel run={run} />
        
        {/* {outputOpen || run.errors.length > 0 ? (
          <OutputPanel run={run} />
        ) : null} */}
      </div>
    </main>
  );
}

export default App;
