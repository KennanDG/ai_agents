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

import {
  fetchGitHubBranches,
  fetchGitHubRepositories,
  fetchGitHubStatus,
  fetchRepositoryFile,
  fetchRepositoryTree,
  importGitHubRepository,
  type GitHubBranchSummary,
  type GitHubRepositorySummary,
} from "./lib/repositoryApi";
import { submitVoiceTurn } from "./lib/voiceAgentApi";
import type { AgentMessage, AgentRunState, ChangeStatus, FileChange, RepositoryFile, RepositoryTreeEntry } from "./types";

const apiBaseUrl = import.meta.env.VITE_AI_AGENTS_API_BASE ?? "http://0.0.0.0:8000";
const apiKey = import.meta.env.VITE_AI_AGENTS_API_KEY ?? "";
const configuredRepoRoot : string = import.meta.env.VITE_CODING_AGENT_REPO_ROOT ?? ".";
const configuredWorkspaceRoot : string = import.meta.env.VITE_CODING_AGENT_WORKSPACE_ROOT ?? configuredRepoRoot;

const createRunState = (status: AgentRunState["status"] = "connecting"): AgentRunState => ({
  status,
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
});

const initialRunState = createRunState();

const nowLabel = () => {
  return new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(new Date());
}

const base64AudioToObjectUrl = (base64: string, mimeType: string) => {
  const binary = window.atob(base64);
  const bytes = new Uint8Array(binary.length);

  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }

  return URL.createObjectURL(new Blob([bytes], { type: mimeType }));
};

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
        ...createRunState("running"),
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
  const [githubRepositories, setGitHubRepositories] = useState<GitHubRepositorySummary[]>([]);
  const [githubLoading, setGitHubLoading] = useState(false);
  const [githubError, setGitHubError] = useState<string | null>(null);
  const [selectedGitHubRepository, setSelectedGitHubRepository] = useState<string | null>(null);
  const [currentBranch, setCurrentBranch] = useState<string | null>(null);
  const [branches, setBranches] = useState<GitHubBranchSummary[]>([]);
  const [branchesLoading, setBranchesLoading] = useState(false);
  const [branchesError, setBranchesError] = useState<string | null>(null);
  const [run, dispatchRun] = useReducer(runReducer, initialRunState);

  const [voiceSessionId, setVoiceSessionId] = useState<string | null>(null);
  const [voiceHistory, setVoiceHistory] = useState<AgentMessage[]>([]);
  const [voiceReplyUrl, setVoiceReplyUrl] = useState<string | null>(null);
  const voiceReplyUrlRef = useRef<string | null>(null);

  // const [sidebarOpen, setSidebarOpen] = useState(true);
  const socketRef = useRef<ReturnType<typeof createCodingAgentSocket> | null>(null);
  const newThreadForNextRunRef = useRef(false);
  const activeRunMessageIdRef = useRef<string | null>(null);

  
  const activeChange = useMemo(
    () => run.fileChanges.find((change) => change.path === activePath && (change.original || change.modified)) ?? null,
    [activePath, run.fileChanges],
  );

  const repoName = useMemo(() => repoRoot.split(/[\\/]/).filter(Boolean).at(-1) ?? "repository", [repoRoot]);
  const effectiveWorkspaceRoot = selectedGitHubRepository
    ? repoRoot
    : configuredWorkspaceRoot === configuredRepoRoot
      ? repoRoot
      : configuredWorkspaceRoot;

  const loadRepository = useCallback(async (targetRoot: string) => {
    setRepoLoading(true);
    setRepoError(null);

    try {
      const tree = await fetchRepositoryTree({ apiBaseUrl, apiKey, repoRoot: targetRoot });
      setRepoRoot(tree.repo_root);
      setRepoEntries(tree.entries);

      const firstFile = tree.entries.find((entry) => entry.kind === "file");
      setActivePath(firstFile?.path ?? null);
      setActiveFile(null);
    } catch (error) {
      setRepoError(error instanceof Error ? error.message : "Failed to load repository.");
    } finally {
      setRepoLoading(false);
    }
  }, []);

  const refreshRepository = useCallback(async () => {
    await loadRepository(repoRoot);
  }, [loadRepository, repoRoot]);

  const refreshGitHubRepositories = useCallback(async () => {
    setGitHubLoading(true);
    setGitHubError(null);

    try {
      const status = await fetchGitHubStatus({ apiBaseUrl, apiKey });
      if (!status.connected) {
        setGitHubRepositories([]);
        setGitHubError("GitHub is not configured on the backend.");
        return;
      }

      const repositories = await fetchGitHubRepositories({ apiBaseUrl, apiKey });
      setGitHubRepositories(repositories);
    } catch (error) {
      setGitHubRepositories([]);
      setGitHubError(error instanceof Error ? error.message : "Failed to load GitHub repositories.");
    } finally {
      setGitHubLoading(false);
    }
  }, []);

  const selectGitHubRepository = useCallback(async (fullName: string) => {
    const repository = githubRepositories.find((item) => item.full_name === fullName);
    if (!repository) {
      setGitHubError(`Repository is not available: ${fullName}`);
      return;
    }

    setGitHubLoading(true);
    setGitHubError(null);

    try {
      const imported = await importGitHubRepository({
        apiBaseUrl,
        apiKey,
        fullName,
        ref: repository.default_branch,
      });
      setSelectedGitHubRepository(imported.full_name);
      setCurrentBranch(imported.ref);
      newThreadForNextRunRef.current = true;
      await loadRepository(imported.repo_root);
    } catch (error) {
      setGitHubError(error instanceof Error ? error.message : "Failed to import GitHub repository.");
    } finally {
      setGitHubLoading(false);
    }
  }, [githubRepositories, loadRepository]);

  const useLocalRepository = useCallback(async () => {
    setSelectedGitHubRepository(null);
    setCurrentBranch(null);
    newThreadForNextRunRef.current = true;
    await loadRepository(configuredRepoRoot);
  }, [loadRepository]);

  const switchBranch = useCallback(async (branchName: string) => {
    if (!selectedGitHubRepository) return;
    setRepoLoading(true);
    setRepoError(null);
    try {
      const imported = await importGitHubRepository({
        apiBaseUrl,
        apiKey,
        fullName: selectedGitHubRepository,
        ref: branchName,
        refresh: true,
      });
      setCurrentBranch(imported.ref);
      newThreadForNextRunRef.current = true;
      await loadRepository(imported.repo_root);
    } catch (error) {
      setRepoError(error instanceof Error ? error.message : "Failed to switch branch.");
    } finally {
      setRepoLoading(false);
    }
  }, [selectedGitHubRepository, loadRepository, apiBaseUrl, apiKey]);

  useEffect(() => {
    void loadRepository(configuredRepoRoot);
    void refreshGitHubRepositories();
  }, [loadRepository, refreshGitHubRepositories]);

  useEffect(() => {
    if (!selectedGitHubRepository) {
      setBranches([]);
      setCurrentBranch(null);
      return;
    }
    let cancelled = false;
    setBranchesLoading(true);
    setBranchesError(null);
    const load = async () => {
      try {
        const list = await fetchGitHubBranches({ apiBaseUrl, apiKey, fullName: selectedGitHubRepository });
        if (!cancelled) {
          setBranches(list);
        }
      } catch (error) {
        if (!cancelled) {
          setBranchesError(error instanceof Error ? error.message : "Failed to load branches");
          setBranches([]);
        }
      } finally {
        if (!cancelled) setBranchesLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [selectedGitHubRepository, apiBaseUrl, apiKey]);



  useEffect(() => {
    return () => {
      if (voiceReplyUrlRef.current) {
        URL.revokeObjectURL(voiceReplyUrlRef.current);
      }
    };
  }, []);



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

        const activeRunMessageId = activeRunMessageIdRef.current;
        if (activeRunMessageId) {
          setMessages((current) =>
            current.map((message) =>
              message.id === activeRunMessageId && message.run
                ? { ...message, run: runReducer(message.run, event) }
                : message,
            ),
          );
        }

        if (event.type === "run.started") {
          newThreadForNextRunRef.current = false;
        }

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




  const submitVoiceAudio = async (
    audio: Blob,
    promptText: string,
    attachedFiles: CodingAgentAttachedFile[],
  ): Promise<boolean> => {
    try {
      const response = await submitVoiceTurn({
        apiBaseUrl,
        apiKey,
        audio,
        sessionId: voiceSessionId,
        history: voiceHistory,
        promptText,
        attachedFiles,
        repoRoot,
        workspaceRoot: effectiveWorkspaceRoot,
        activePath,
        allowWrite,
      });

      setVoiceSessionId(response.session_id);

      const draftContext = promptText.trim() ? `\n\nTyped draft:\n${promptText.trim()}` : "";

      const attachmentContext = attachedFiles.length > 0
        ? `\n\nAttached files:\n${attachedFiles.map((file) => `- ${file.name}`).join("\n")}`
        : "";


      const userVoiceMessage: AgentMessage = {
        id: crypto.randomUUID(),
        role: "user",
        body: `${response.transcript ? `🎙️ ${response.transcript}` : "🎙️ Voice input"}${draftContext}${attachmentContext}`,
        time: nowLabel(),
      };
      
      const agentVoiceMessage: AgentMessage = {
        id: crypto.randomUUID(),
        role: "agent",
        body: response.reply_text,
        time: nowLabel(),
      };

      setMessages((current) => [...current, userVoiceMessage, agentVoiceMessage]);
      setVoiceHistory((current) => [...current, userVoiceMessage, agentVoiceMessage].slice(-6));

      if (response.audio_base64) {
        const nextUrl = base64AudioToObjectUrl(
          response.audio_base64,
          response.audio_mime_type ?? "audio/wav",
        );

        if (voiceReplyUrlRef.current) {
          URL.revokeObjectURL(voiceReplyUrlRef.current);
        }

        voiceReplyUrlRef.current = nextUrl;
        setVoiceReplyUrl(nextUrl);
      }

      if (response.errors.length > 0) {
        setMessages((current) => [
          ...current,
          {
            id: crypto.randomUUID(),
            role: "agent",
            body: `Voice warning:\n${response.errors.join("\n")}`,
            time: nowLabel(),
          },
        ]);
      }

      if (response.status === "ready" && response.coding_request) {
        activeRunMessageIdRef.current = userVoiceMessage.id;
        setMessages((current) => [
          ...current.map((message) =>
            message.id === userVoiceMessage.id
              ? { ...message, run: createRunState("running") }
              : message,
          ),
          {
            id: crypto.randomUUID(),
            role: "agent",
            body: `Handing this to the coding agent:\n\n${response.coding_request}`,
            time: nowLabel(),
          },
        ]);

        setVoiceHistory([]);
        setVoiceSessionId(null);
        runCodingAgent(response.coding_request, attachedFiles);
        return true;
      }

      return false;
      
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "agent",
          body: error instanceof Error ? error.message : "Voice agent failed.",
          time: nowLabel(),
        },
      ]);
      return false;
    }
  };


  const runCodingAgent = (request: string, attachedFiles: CodingAgentAttachedFile[] = []) => {
    socketRef.current?.run({
      thread_id: newThreadForNextRunRef.current ? null : run.threadId,
      request,
      repo_root: repoRoot,
      workspace_root: effectiveWorkspaceRoot,
      allow_write: allowWrite,
      memory_enabled: memoryEnabled,
      attached_files: attachedFiles,
      max_iterations: 3,
    });
  };

  
  const submitPrompt = (prompt: string, attachedFiles: CodingAgentAttachedFile[] = []) => {
    const attachmentLabel =
      attachedFiles.length > 0
        ? `\n\nAttached files:\n${attachedFiles.map((file) => `- ${file.name}`).join("\n")}`
        : "";
    const messageId = crypto.randomUUID();

    activeRunMessageIdRef.current = messageId;
    setMessages((current) => [
      ...current,
      {
        id: messageId,
        role: "user",
        body: prompt + attachmentLabel,
        time: nowLabel(),
        run: createRunState("running"),
      },
    ]);

    runCodingAgent(prompt, attachedFiles);
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
        agentRunning={run.status === "running"}
        error={repoError}
        onSelect={setActivePath}
        onRefresh={refreshRepository}
        githubRepositories={githubRepositories}
        selectedGitHubRepository={selectedGitHubRepository}
        githubLoading={githubLoading}
        githubError={githubError}
        onSelectGitHubRepository={selectGitHubRepository}
        onUseLocalRepository={useLocalRepository}
        onRefreshGitHubRepositories={refreshGitHubRepositories}
        branches={branches}
        currentBranch={currentBranch}
        branchesLoading={branchesLoading}
        branchesError={branchesError}
        onSwitchBranch={switchBranch}
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
          onVoiceAudio={submitVoiceAudio}
          voiceReplyUrl={voiceReplyUrl}
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
