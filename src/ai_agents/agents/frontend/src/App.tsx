import { type MouseEvent as ReactMouseEvent, useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { ActivityBar, type ActivityAction, type ActivityView } from "./components/ActivityBar";
import { AgentSettingsModal } from "./components/AgentSettingsModal";
import { SkillsPage } from "./components/SkillsPage";
import { SourceControlPage } from "./components/SourceControlPage";
import { DiffPanel } from "./components/DiffPanel";
import { OutputPanel } from "./components/OutputPanel";
import { Sidebar } from "./components/Sidebar";
import { TaskPanel } from "./components/TaskPanel";

import {
  createCodingAgentSocket,
  type CodingAgentAttachedFile,
  type CodingAgentCompletionLedger,
  type CodingAgentImplementationUnit,
  type CodingAgentRunResult,
  type CodingAgentServerEvent,
  type CodingAgentTaskMode,
} from "./lib/codingAgentSocket";

import {
  commitGitHubChanges,
  createGitHubBranch,
  createGitHubPullRequest,
  fetchGitHubBranches,
  fetchGitHubRepositories,
  fetchGitHubRepositoryStatus,
  fetchGitHubStatus,
  fetchRepositoryFile,
  fetchRepositoryTree,
  importGitHubRepository,
  pullGitHubBranch,
  pushGitHubBranch,
  testGitHubConnection,
  type GitHubBranchSummary,
  type GitHubRepositoryStatus,
  type GitHubRepositorySummary,
} from "./lib/repositoryApi";
import { selectVoiceContextAttachments, submitVoiceTurn } from "./lib/voiceAgentApi";
import {
  fetchAgentConfiguration,
  type AgentConfiguration,
} from "./lib/adminApi";
import type { AgentMessage, AgentRunState, ChangeStatus, FileChange, RepositoryFile, RepositoryTreeEntry } from "./types";

const apiBaseUrl = import.meta.env.VITE_AI_AGENTS_API_BASE ?? "http://0.0.0.0:8000";
const apiKey = import.meta.env.VITE_AI_AGENTS_API_KEY ?? "";
const configuredRepoRoot : string = import.meta.env.VITE_CODING_AGENT_REPO_ROOT ?? ".";
const configuredWorkspaceRoot : string = import.meta.env.VITE_CODING_AGENT_WORKSPACE_ROOT ?? configuredRepoRoot;

type DivideConquerRunState = AgentRunState & {
  selectedSkills?: string[];
  taskMode?: CodingAgentTaskMode | null;
  implementationUnits?: CodingAgentImplementationUnit[];
  completionLedger?: CodingAgentCompletionLedger;
  implementationGeneration?: number;
  implementationIteration?: number;
  maxImplementationIterations?: number;
  subtaskWorkerCount?: number;
  subtaskWorkerResults?: Record<string, unknown>[];
  contextWorkerCount?: number;
  runtimeSettings?: Record<string, unknown>;
};

const createRunState = (status: AgentRunState["status"] = "connecting"): DivideConquerRunState => ({
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
  selectedSkills: [],
  taskMode: null,
  implementationUnits: [],
  completionLedger: {},
  implementationGeneration: 0,
  implementationIteration: 0,
  maxImplementationIterations: 0,
  subtaskWorkerCount: 0,
  subtaskWorkerResults: [],
  contextWorkerCount: 0,
  runtimeSettings: {},
});

const initialRunState = createRunState();

type RunAction = CodingAgentServerEvent | { type: "session.reset" };

type GitHubCommitReceipt = {
  branch: string;
  commitSha: string;
  committedFiles: string[];
};

const nowLabel = () => {
  return new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(new Date());
}

const clampSize = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

const startPanelResize = (
  event: ReactMouseEvent,
  startSize: number,
  setSize: (size: number) => void,
  min: number,
  max: number,
) => {
  event.preventDefault();
  const startX = event.clientX;
  const onMove = (move: MouseEvent) => {
    setSize(clampSize(startSize + (move.clientX - startX), min, max));
  };
  const onUp = () => {
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
  };
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
};

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

const asRecord = (value: unknown): Record<string, unknown> | undefined => {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

const asNumber = (value: unknown): number | undefined => {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

const asTaskMode = (value: unknown): CodingAgentTaskMode | undefined => {
  return value === "simple" || value === "standard" || value === "parallel" ? value : undefined;
}

const mergeWorkerResults = (
  current: Record<string, unknown>[] = [],
  incoming: Record<string, unknown>[] | undefined,
): Record<string, unknown>[] => {
  if (!incoming) return current;

  const merged = new Map<string, Record<string, unknown>>();
  const add = (item: Record<string, unknown>, fallbackIndex: number) => {
    const id = item.unit_id ?? item.id ?? item.subtask_id ?? item.worker_id;
    const generation = item.generation ?? "";
    const key = typeof id === "string" ? `${generation}:${id}` : `anonymous:${fallbackIndex}:${JSON.stringify(item)}`;
    merged.set(key, item);
  };

  current.forEach(add);
  incoming.forEach((item, index) => add(item, current.length + index));
  return [...merged.values()];
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


const mergeResult = (state: AgentRunState, result: CodingAgentRunResult): DivideConquerRunState => {
  const current = state as DivideConquerRunState;

  return {
    ...state,
    threadId: result.thread_id,
    selectedSkill: result.selected_skill,
    selectedSkills: result.selected_skills ?? current.selectedSkills ?? [],
    taskMode: result.task_mode ?? current.taskMode ?? null,
    routeConfidence: result.route_confidence,
    routeReason: result.route_reason,
    plan: result.plan ?? state.plan,
    implementationUnits: result.implementation_units ?? current.implementationUnits ?? [],
    completionLedger: result.completion_ledger ?? current.completionLedger ?? {},
    implementationGeneration: result.implementation_generation ?? current.implementationGeneration ?? 0,
    implementationIteration: result.implementation_iteration ?? current.implementationIteration ?? 0,
    maxImplementationIterations: result.max_implementation_iterations ?? current.maxImplementationIterations ?? 0,
    subtaskWorkerCount: result.subtask_worker_count ?? current.subtaskWorkerCount ?? 0,
    subtaskWorkerResults: result.subtask_worker_results ?? current.subtaskWorkerResults ?? [],
    contextWorkerCount: result.context_worker_count ?? current.contextWorkerCount ?? 0,
    runtimeSettings: result.runtime_settings ?? current.runtimeSettings ?? {},
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
    markdown_response: result.markdown_response,
    errors: result.errors ?? state.errors,
  };
}


const runReducer = (state: AgentRunState, event: RunAction): DivideConquerRunState => {
  switch (event.type) {
    case "session.reset":
      return createRunState("ready");

    case "session.ready":
      return {
        ...state,
        status: "ready",
        logs: [...state.logs, `[socket] ${event.payload.message}`],
      };

    case "run.started": {
      const workerCount = event.payload.subtask_worker_count ?? event.payload.subagent_count;
      const maxImplementationIterations = event.payload.max_implementation_iterations;

      return {
        ...createRunState("running"),
        runId: event.run_id,
        threadId: event.thread_id,
        subtaskWorkerCount: workerCount ?? 0,
        maxImplementationIterations: maxImplementationIterations ?? 0,
        runtimeSettings: event.payload.runtime_settings ?? {},
        logs: [
          `[run] started ${event.thread_id}`,
          `[repo] ${event.payload.repo_root}`,
          `[mode] ${event.payload.allow_write ? "write" : "read-only"}`,
          ...(workerCount != null ? [`[workers] ${workerCount} implementation worker(s)`] : []),
          ...(maxImplementationIterations != null
            ? [`[iterations] max ${maxImplementationIterations} implementation iteration(s)`]
            : []),
        ],
      };
    }

    case "node.completed": {
      const payload = event.payload;
      const plan = asStringArray(payload.plan) ?? state.plan;
      const filesInspected = asStringArray(payload.files_inspected) ?? state.filesInspected;
      const fileChanges = asFileChanges(payload.file_changes) ?? state.fileChanges;
      const diffs = asStringArray(payload.diffs) ?? state.diffs;
      const validationCommands = asStringArray(payload.validation_commands) ?? state.validationCommands;
      const validationResults = asRecordArray(payload.validation_results) ?? state.validationResults;
      const errors = asStringArray(payload.errors) ?? state.errors;
      const current = state as DivideConquerRunState;
      const implementationUnits = (asRecordArray(payload.implementation_units) as CodingAgentImplementationUnit[] | undefined)
        ?? current.implementationUnits
        ?? [];
      const completionLedger = (asRecord(payload.completion_ledger) as CodingAgentCompletionLedger | undefined)
        ?? current.completionLedger
        ?? {};
      const incomingWorkerResults = asRecordArray(payload.subtask_worker_results);
      const subtaskWorkerResults = mergeWorkerResults(current.subtaskWorkerResults, incomingWorkerResults);

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
        selectedSkills: asStringArray(payload.selected_skills) ?? current.selectedSkills ?? [],
        taskMode: asTaskMode(payload.task_mode) ?? current.taskMode ?? null,
        implementationUnits,
        completionLedger,
        implementationGeneration: asNumber(payload.implementation_generation) ?? current.implementationGeneration ?? 0,
        implementationIteration: asNumber(payload.implementation_iteration) ?? current.implementationIteration ?? 0,
        maxImplementationIterations: asNumber(payload.max_implementation_iterations) ?? current.maxImplementationIterations ?? 0,
        subtaskWorkerCount: asNumber(payload.subtask_worker_count) ?? current.subtaskWorkerCount ?? 0,
        subtaskWorkerResults,
        contextWorkerCount: asNumber(payload.context_worker_count) ?? current.contextWorkerCount ?? 0,
        runtimeSettings: asRecord(payload.runtime_settings) ?? current.runtimeSettings ?? {},
        routeConfidence: typeof payload.route_confidence === "number" ? payload.route_confidence : state.routeConfidence,
        routeReason: typeof payload.route_reason === "string" ? payload.route_reason : state.routeReason,
        patchSummary: typeof payload.patch_summary === "string" ? payload.patch_summary : state.patchSummary,
        report: typeof payload.report === "string" ? payload.report : state.report,
        markdown_response: typeof payload.markdown_response === "string" ? payload.markdown_response : state.markdown_response,
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
  const [activeView, setActiveView] = useState<ActivityView>("explorer");
  const [agentSettingsOpen, setAgentSettingsOpen] = useState(false);
  const [agentConfiguration, setAgentConfiguration] = useState<AgentConfiguration | null>(null);
  const [activePath, setActivePath] = useState<string | null>(null);
  const [activeFile, setActiveFile] = useState<RepositoryFile | null>(null);
  const [allowWrite, setAllowWrite] = useState(true);
  const [sidebarWidth, setSidebarWidth] = useState(288);
  const [taskPanelWidth, setTaskPanelWidth] = useState(360);
  const [diffPanelHidden, setDiffPanelHidden] = useState(false);

  const [fileLoading, setFileLoading] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);
  
  const [memoryEnabled, setMemoryEnabled] = useState(true);
  const [messages, setMessages] = useState<AgentMessage[]>([]);

  // const [outputOpen, setOutputOpen] = useState(false);

  const [repoRoot, setRepoRoot] = useState(configuredRepoRoot);
  const [localRepoRoot, setLocalRepoRoot] = useState(configuredRepoRoot);
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
  const [githubRepositoryStatus, setGitHubRepositoryStatus] = useState<GitHubRepositoryStatus | null>(null);
  const [githubActionLoading, setGitHubActionLoading] = useState<string | null>(null);
  const [githubActionMessage, setGitHubActionMessage] = useState<string | null>(null);
  const [githubActionError, setGitHubActionError] = useState<string | null>(null);
  const [githubPullRequestUrl, setGitHubPullRequestUrl] = useState<string | null>(null);
  const [lastGitHubCommit, setLastGitHubCommit] = useState<GitHubCommitReceipt | null>(null);
  const [agentSessionKey, setAgentSessionKey] = useState(0);
  const [run, dispatchRun] = useReducer(runReducer, initialRunState);
  const [changes, setChanges] = useState<FileChange[]>([]);
  const previousRunStatusRef = useRef(run.status);

  useEffect(() => {
    if (run.status === "running" && previousRunStatusRef.current !== "running") {
      setChanges([]);
    } else {
      setChanges(run.fileChanges);
    }
    previousRunStatusRef.current = run.status;
  }, [run.fileChanges, run.status]);

  const clearChanges = useCallback(() => {
    setChanges([]);
  }, []);

  const [voiceSessionId, setVoiceSessionId] = useState<string | null>(null);
  const [voiceHistory, setVoiceHistory] = useState<AgentMessage[]>([]);
  const [voiceReplyUrl, setVoiceReplyUrl] = useState<string | null>(null);
  const voiceReplyUrlRef = useRef<string | null>(null);

  // const [sidebarOpen, setSidebarOpen] = useState(true);
  const socketRef = useRef<ReturnType<typeof createCodingAgentSocket> | null>(null);
  const newThreadForNextRunRef = useRef(false);
  const activeRunMessageIdRef = useRef<string | null>(null);

  const resetAgentWorkspace = useCallback(() => {
    dispatchRun({ type: "session.reset" });
    setMessages([]);
    setVoiceHistory([]);
    setVoiceSessionId(null);
    setLastGitHubCommit(null);
    activeRunMessageIdRef.current = null;
    newThreadForNextRunRef.current = true;

    if (voiceReplyUrlRef.current) {
      URL.revokeObjectURL(voiceReplyUrlRef.current);
      voiceReplyUrlRef.current = null;
    }
    setVoiceReplyUrl(null);
    setAgentSessionKey((current) => current + 1);
  }, []);

  
  const activeChange = useMemo(
    () => changes.find((change) => change.path === activePath && (change.original || change.modified)) ?? null,
    [activePath, changes],
  );

  const repoName = useMemo(() => repoRoot.split(/[\\/]/).filter(Boolean).at(-1) ?? "repository", [repoRoot]);
  const selectedGitHubRepositorySummary = useMemo(
    () => githubRepositories.find((repository) => repository.full_name === selectedGitHubRepository) ?? null,
    [githubRepositories, selectedGitHubRepository],
  );
  const committablePaths = useMemo(() => {
    if (!githubRepositoryStatus) return [];
    const changedPaths = new Set([
      ...githubRepositoryStatus.staged_files,
      ...githubRepositoryStatus.unstaged_files,
      ...githubRepositoryStatus.untracked_files,
    ]);
    return [...new Set(run.appliedFiles)].filter((path) => changedPaths.has(path));
  }, [githubRepositoryStatus, run.appliedFiles]);
  const effectiveWorkspaceRoot =
    repoRoot === configuredRepoRoot && configuredWorkspaceRoot !== configuredRepoRoot
      ? configuredWorkspaceRoot
      : repoRoot;

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
      return tree.repo_root;
    } catch (error) {
      setRepoError(error instanceof Error ? error.message : "Failed to load repository.");
      return null;
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

  const loadGitHubRepositoryStatus = useCallback(async (fullName: string) => {
    try {
      const status = await fetchGitHubRepositoryStatus({ apiBaseUrl, apiKey, fullName });
      setGitHubRepositoryStatus(status);
      setCurrentBranch(status.branch);
      return status;
    } catch (error) {
      setGitHubRepositoryStatus(null);
      setGitHubActionError(error instanceof Error ? error.message : "Failed to load GitHub repository status.");
      return null;
    }
  }, []);

  const clearGitHubActionFeedback = () => {
    setGitHubActionMessage(null);
    setGitHubActionError(null);
    setGitHubPullRequestUrl(null);
  };

  const selectGitHubRepository = useCallback(async (fullName: string) => {
    const repository = githubRepositories.find((item) => item.full_name === fullName);
    if (!repository) {
      setGitHubActionError(`Repository is not available: ${fullName}`);
      return;
    }

    setGitHubActionLoading("repository");
    clearGitHubActionFeedback();

    try {
      const imported = await importGitHubRepository({
        apiBaseUrl,
        apiKey,
        fullName,
        ref: repository.default_branch,
      });
      setSelectedGitHubRepository(imported.full_name);
      setCurrentBranch(imported.ref);
      await loadRepository(imported.repo_root);
      await loadGitHubRepositoryStatus(imported.full_name);
      resetAgentWorkspace();

      const feedback = [`Using ${imported.full_name} on ${imported.ref}.`];
      if (imported.saved_previous_changes && imported.previous_ref) {
        feedback.push(`Saved local changes from ${imported.previous_ref} for later restoration.`);
      }
      if (imported.restored_target_changes) {
        feedback.push(`Restored the saved local changes for ${imported.ref}.`);
      }
      setGitHubActionMessage(feedback.join(" "));
    } catch (error) {
      setGitHubActionError(error instanceof Error ? error.message : "Failed to import GitHub repository.");
    } finally {
      setGitHubActionLoading(null);
    }
  }, [githubRepositories, loadGitHubRepositoryStatus, loadRepository, resetAgentWorkspace]);

  const selectLocalRepository = useCallback(async (targetRoot: string) => {
    const normalizedRoot = targetRoot.trim();
    if (!normalizedRoot) {
      setGitHubActionError("Enter a local directory path.");
      return false;
    }

    setGitHubActionLoading("local-repository");
    clearGitHubActionFeedback();

    try {
      const resolvedRoot = await loadRepository(normalizedRoot);
      if (!resolvedRoot) {
        setGitHubActionError("The selected directory could not be opened by the backend.");
        return false;
      }

      setLocalRepoRoot(resolvedRoot);
      setSelectedGitHubRepository(null);
      setCurrentBranch(null);
      setGitHubRepositoryStatus(null);
      resetAgentWorkspace();
      setGitHubActionMessage(`Using local repository ${resolvedRoot}.`);
      return true;
    } finally {
      setGitHubActionLoading(null);
    }
  }, [loadRepository, resetAgentWorkspace]);

  const useLocalRepository = useCallback(async () => {
    await selectLocalRepository(localRepoRoot);
  }, [localRepoRoot, selectLocalRepository]);

  const browseLocalRepository = useCallback(async () => {
    const selectDirectory = window.desktop?.selectDirectory;
    if (!selectDirectory) return null;

    const selectedPath = await selectDirectory({
      title: "Select repository root",
      defaultPath: localRepoRoot,
    });
    if (!selectedPath) return null;

    const selected = await selectLocalRepository(selectedPath);
    return selected ? selectedPath : null;
  }, [localRepoRoot, selectLocalRepository]);

  const switchBranch = useCallback(async (branchName: string) => {
    if (!selectedGitHubRepository || branchName === currentBranch) return;

    setGitHubActionLoading("switch");
    clearGitHubActionFeedback();
    try {
      const imported = await importGitHubRepository({
        apiBaseUrl,
        apiKey,
        fullName: selectedGitHubRepository,
        ref: branchName,
        refresh: true,
      });
      setCurrentBranch(imported.ref);
      await loadRepository(imported.repo_root);
      await loadGitHubRepositoryStatus(imported.full_name);
      resetAgentWorkspace();

      const feedback = [`Switched to ${imported.ref}.`];
      if (imported.saved_previous_changes && imported.previous_ref) {
        feedback.push(`Saved local changes from ${imported.previous_ref}.`);
      }
      if (imported.restored_target_changes) {
        feedback.push(`Restored the saved local changes for ${imported.ref}.`);
      }
      setGitHubActionMessage(feedback.join(" "));
    } catch (error) {
      setGitHubActionError(error instanceof Error ? error.message : "Failed to switch branch.");
    } finally {
      setGitHubActionLoading(null);
    }
  }, [currentBranch, selectedGitHubRepository, loadGitHubRepositoryStatus, loadRepository, resetAgentWorkspace]);

  const testSelectedGitHubConnection = useCallback(async () => {
    setGitHubActionLoading("test");
    clearGitHubActionFeedback();
    try {
      const result = await testGitHubConnection({
        apiBaseUrl,
        apiKey,
        fullName: selectedGitHubRepository,
      });
      setGitHubActionMessage(result.message);
      if (!result.connected) {
        setGitHubActionError("The API, Git transport, token permissions, and workspace checks did not all pass.");
      }
    } catch (error) {
      setGitHubActionError(error instanceof Error ? error.message : "GitHub connection test failed.");
    } finally {
      setGitHubActionLoading(null);
    }
  }, [selectedGitHubRepository]);

  const createAgentBranch = useCallback(async (branch: string) => {
    if (!selectedGitHubRepository) return;
    setGitHubActionLoading("branch");
    clearGitHubActionFeedback();
    try {
      const result = await createGitHubBranch({
        apiBaseUrl,
        apiKey,
        fullName: selectedGitHubRepository,
        branch,
        base: currentBranch,
      });
      setCurrentBranch(result.branch);
      setBranches((current) => current.some((item) => item.name === result.branch)
        ? current
        : [{ name: result.branch, sha: result.sha }, ...current]);
      setGitHubActionMessage(`Created and switched to ${result.branch}.`);
      await loadGitHubRepositoryStatus(selectedGitHubRepository);
      await loadRepository(repoRoot);
      resetAgentWorkspace();
    } catch (error) {
      setGitHubActionError(error instanceof Error ? error.message : "Failed to create branch.");
    } finally {
      setGitHubActionLoading(null);
    }
  }, [currentBranch, loadGitHubRepositoryStatus, loadRepository, repoRoot, resetAgentWorkspace, selectedGitHubRepository]);

  const pullCurrentGitHubBranch = useCallback(async () => {
    if (!selectedGitHubRepository) return;
    setGitHubActionLoading("pull");
    clearGitHubActionFeedback();
    try {
      const result = await pullGitHubBranch({ apiBaseUrl, apiKey, fullName: selectedGitHubRepository });
      setGitHubActionMessage(result.changed ? `Fast-forwarded ${result.branch}.` : `${result.branch} is already current.`);
      await loadGitHubRepositoryStatus(selectedGitHubRepository);
      await loadRepository(repoRoot);
    } catch (error) {
      setGitHubActionError(error instanceof Error ? error.message : "Failed to pull branch.");
    } finally {
      setGitHubActionLoading(null);
    }
  }, [loadGitHubRepositoryStatus, loadRepository, repoRoot, selectedGitHubRepository]);

  const commitAppliedGitHubChanges = useCallback(async (message: string) => {
    if (!selectedGitHubRepository) return false;
    if (committablePaths.length === 0) {
      setGitHubActionError("No approved and applied agent files are currently available to commit.");
      return false;
    }
    setGitHubActionLoading("commit");
    clearGitHubActionFeedback();
    try {
      const result = await commitGitHubChanges({
        apiBaseUrl,
        apiKey,
        fullName: selectedGitHubRepository,
        message,
        paths: committablePaths,
      });
      setLastGitHubCommit({
        branch: result.branch,
        commitSha: result.commit_sha,
        committedFiles: result.committed_files,
      });
      setGitHubActionMessage(`Committed ${result.committed_files.length} file(s) as ${result.commit_sha.slice(0, 7)}.`);
      await loadGitHubRepositoryStatus(selectedGitHubRepository);
      await loadRepository(repoRoot);
      return true;
    } catch (error) {
      setGitHubActionError(error instanceof Error ? error.message : "Failed to commit changes.");
      return false;
    } finally {
      setGitHubActionLoading(null);
    }
  }, [committablePaths, loadGitHubRepositoryStatus, loadRepository, repoRoot, selectedGitHubRepository]);

  const pushCurrentGitHubBranch = useCallback(async () => {
    if (!selectedGitHubRepository) return;
    setGitHubActionLoading("push");
    clearGitHubActionFeedback();
    try {
      const result = await pushGitHubBranch({ apiBaseUrl, apiKey, fullName: selectedGitHubRepository });
      setGitHubActionMessage(result.pushed ? `Pushed ${result.branch}.` : `${result.branch} was already pushed.`);
      await loadGitHubRepositoryStatus(selectedGitHubRepository);
    } catch (error) {
      setGitHubActionError(error instanceof Error ? error.message : "Failed to push branch.");
    } finally {
      setGitHubActionLoading(null);
    }
  }, [loadGitHubRepositoryStatus, selectedGitHubRepository]);

  const openGitHubPullRequest = useCallback(async ({
    title,
    body,
    base,
    draft,
  }: {
    title: string;
    body: string;
    base: string;
    draft: boolean;
  }) => {
    if (!selectedGitHubRepository) return false;
    setGitHubActionLoading("pr");
    clearGitHubActionFeedback();
    try {
      const result = await createGitHubPullRequest({
        apiBaseUrl,
        apiKey,
        fullName: selectedGitHubRepository,
        title,
        body,
        base,
        head: currentBranch,
        draft,
      });
      setGitHubPullRequestUrl(result.html_url);
      setGitHubActionMessage(result.created
        ? `Created ${result.draft ? "draft " : ""}PR #${result.number}.`
        : `PR #${result.number} is already open.`);
      return true;
    } catch (error) {
      setGitHubActionError(error instanceof Error ? error.message : "Failed to create pull request.");
      return false;
    } finally {
      setGitHubActionLoading(null);
    }
  }, [currentBranch, selectedGitHubRepository]);

  useEffect(() => {
    void loadRepository(configuredRepoRoot).then((resolvedRoot) => {
      if (resolvedRoot) setLocalRepoRoot(resolvedRoot);
    });
    void refreshGitHubRepositories();
    void fetchAgentConfiguration({ apiBaseUrl, apiKey })
      .then(setAgentConfiguration)
      .catch((error) => console.error("Failed to load agent execution settings.", error));
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
  }, [selectedGitHubRepository]);

  useEffect(() => {
    if (!selectedGitHubRepository) {
      setGitHubRepositoryStatus(null);
      return;
    }
    void loadGitHubRepositoryStatus(selectedGitHubRepository);
  }, [loadGitHubRepositoryStatus, selectedGitHubRepository, run.appliedFiles]);



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
            { id: crypto.randomUUID(), role: "agent", body: event.payload.markdown_response ?? "Run completed.", time: nowLabel() },
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

  // const approveFileChange = (path: string) => {
  //   if (!run.threadId) return;
  //   socketRef.current?.apply(run.threadId, [path]);
  // };

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
      // Keep the voice agent and downstream coding agent on the same attachment set.
      // Otherwise the voice handoff can be grounded in five files while the coding
      // run is unexpectedly classified/planned from a much larger set.
      const voiceAttachments = selectVoiceContextAttachments(attachedFiles);

      const response = await submitVoiceTurn({
        apiBaseUrl,
        apiKey,
        audio,
        sessionId: voiceSessionId,
        history: voiceHistory,
        promptText,
        attachedFiles: voiceAttachments,
        repoRoot,
        workspaceRoot: effectiveWorkspaceRoot,
        activePath,
        allowWrite,
      });

      setVoiceSessionId(response.session_id);

      const draftContext = promptText.trim() ? `\n\nTyped draft:\n${promptText.trim()}` : "";

      const attachmentContext = voiceAttachments.length > 0
        ? `\n\nAttached files:\n${voiceAttachments.map((file) => `- ${file.name}`).join("\n")}`
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
      setVoiceHistory((current) => [...current, userVoiceMessage, agentVoiceMessage].slice(-12));

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
        runCodingAgent(response.coding_request, voiceAttachments);
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
    const runRequest = {
      thread_id: newThreadForNextRunRef.current ? null : run.threadId,
      request,
      repo_root: repoRoot,
      workspace_root: effectiveWorkspaceRoot,
      allow_write: allowWrite,
      memory_enabled: memoryEnabled,
      attached_files: attachedFiles,
      // New divide-and-conquer names are primary; legacy aliases keep rolling
      // upgrades compatible with an older backend.
      max_implementation_iterations: 3,
      max_iterations: 3,
      subtask_worker_count: agentConfiguration?.coding_subagent_count,
      subagent_count: agentConfiguration?.coding_subagent_count,
      route_max_tokens: agentConfiguration?.coding_route_max_tokens,
      planner_max_tokens: agentConfiguration?.coding_planner_max_tokens,
      repo_navigation_max_tokens: agentConfiguration?.coding_repo_navigation_max_tokens,
      simple_patch_max_tokens: agentConfiguration?.coding_simple_patch_max_tokens,
      patch_max_tokens: agentConfiguration?.coding_patch_max_tokens,
      progress_max_tokens: agentConfiguration?.coding_progress_max_tokens,
    };
    clearChanges();
    socketRef.current?.run(runRequest);
  };

  
  const submitPrompt = (prompt: string, attachedFiles: CodingAgentAttachedFile[] = []) => {
    const attachmentLabel =
      attachedFiles.length > 0
        ? `\n\nAttached files:\n${attachedFiles.map((file) => `- ${file.name}`).join("\n")}`
        : "";
    const messageId = crypto.randomUUID();

    activeRunMessageIdRef.current = messageId;
    setMessages([
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


  const selectActivity = (action: ActivityAction) => {
    if (action === "agent") {
      setAgentSettingsOpen(true);
      return;
    }

    if (action === "explorer" || action === "source-control" || action === "skills") {
      setActiveView(action);
    }
  };


  return (
    <main className="flex h-dvh min-h-0 min-w-0 overflow-hidden bg-canvas text-ink">
      <ActivityBar
        activeView={activeView}
        agentSettingsOpen={agentSettingsOpen}
        onSelect={selectActivity}
      />

      {activeView === "explorer" ? (
        <>
          <Sidebar
            repoName={repoName}
            repoRoot={repoRoot}
            entries={repoEntries}
            changes={changes}
            activePath={activePath}
            isLoading={repoLoading}
            agentRunning={run.status === "running"}
            error={repoError}
            onSelect={setActivePath}
            onRefresh={refreshRepository}
            onClearChanges={clearChanges}
            width={sidebarWidth}
          />

          <div
            role="separator"
            aria-orientation="vertical"
            aria-label="Resize sidebar"
            className="w-1 shrink-0 cursor-col-resize hover:bg-accent/60"
            onMouseDown={(event) => startPanelResize(event, sidebarWidth, setSidebarWidth, 200, 480)}
          />

          <div
            style={diffPanelHidden ? undefined : { width: taskPanelWidth }}
            className={`flex min-h-0 flex-col border-r border-line bg-panel-soft ${diffPanelHidden ? "min-w-0 flex-1" : "shrink-0"}`}
          >
            <div className="flex shrink-0 items-center gap-3 border-b border-line px-3 py-2">
              <label className="flex cursor-pointer items-center gap-1.5 text-xs text-ink-soft">
                <input
                  type="checkbox"
                  checked={allowWrite}
                  onChange={() => setAllowWrite(!allowWrite)}
                  className="accent-accent"
                />
                Write
              </label>

              <label className="flex cursor-pointer items-center gap-1.5 text-xs text-ink-soft">
                <input
                  type="checkbox"
                  checked={memoryEnabled}
                  onChange={() => setMemoryEnabled(!memoryEnabled)}
                  className="accent-accent"
                />
                Memory
              </label>

              <button
                type="button"
                className="ml-auto rounded-md border border-line px-2 py-1 text-[10px] text-muted hover:border-accent/60 hover:text-ink"
                title={diffPanelHidden ? "Show the diff panel" : "Hide the diff panel and expand the task panel"}
                onClick={() => setDiffPanelHidden((current) => !current)}
              >
                {diffPanelHidden ? "Show Diffs" : "Expand"}
              </button>
            </div>

            <TaskPanel
              key={agentSessionKey}
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
              onResetSession={resetAgentWorkspace}
            />
          </div>

          {!diffPanelHidden && (
            <>
              <div
                role="separator"
                aria-orientation="vertical"
                aria-label="Resize task panel"
                className="w-1 shrink-0 cursor-col-resize hover:bg-accent/60"
                onMouseDown={(event) => startPanelResize(event, taskPanelWidth, setTaskPanelWidth, 260, 720)}
              />

              <div className="flex min-h-0 min-w-0 flex-1 flex-col">
                <DiffPanel
                  file={activeFile}
                  change={activeChange}
                  isLoading={fileLoading}
                  error={fileError}
                />
                <OutputPanel run={run} />
              </div>
            </>
          )}
        </>
      ) : activeView === "source-control" ? (
        <SourceControlPage
          repoRoot={repoRoot}
          localRepoRoot={localRepoRoot}
          localDirectoryPickerAvailable={Boolean(window.desktop?.selectDirectory)}
          githubRepositories={githubRepositories}
          selectedGitHubRepository={selectedGitHubRepository}
          githubLoading={githubLoading}
          githubError={githubError}
          onSelectGitHubRepository={selectGitHubRepository}
          onUseLocalRepository={useLocalRepository}
          onSelectLocalRepository={selectLocalRepository}
          onBrowseLocalRepository={browseLocalRepository}
          onRefreshGitHubRepositories={refreshGitHubRepositories}
          branches={branches}
          currentBranch={currentBranch}
          branchesLoading={branchesLoading}
          branchesError={branchesError}
          onSwitchBranch={switchBranch}
          defaultBranch={selectedGitHubRepositorySummary?.default_branch ?? null}
          repositoryPermissions={selectedGitHubRepositorySummary?.permissions ?? null}
          githubRepositoryStatus={githubRepositoryStatus}
          githubActionLoading={githubActionLoading}
          githubActionMessage={githubActionMessage}
          githubActionError={githubActionError}
          githubPullRequestUrl={githubPullRequestUrl}
          committableFileCount={committablePaths.length}
          lastCommit={lastGitHubCommit}
          onTestGitHubConnection={testSelectedGitHubConnection}
          onCreateGitHubBranch={createAgentBranch}
          onPullGitHubBranch={pullCurrentGitHubBranch}
          onCommitGitHubChanges={commitAppliedGitHubChanges}
          onPushGitHubBranch={pushCurrentGitHubBranch}
          onCreateGitHubPullRequest={openGitHubPullRequest}
        />
      ) : (
        <SkillsPage apiBaseUrl={apiBaseUrl} apiKey={apiKey} />
      )}

      <AgentSettingsModal
        open={agentSettingsOpen}
        apiBaseUrl={apiBaseUrl}
        apiKey={apiKey}
        onClose={() => setAgentSettingsOpen(false)}
        onSaved={setAgentConfiguration}
      />
    </main>
  );
}

export default App;
