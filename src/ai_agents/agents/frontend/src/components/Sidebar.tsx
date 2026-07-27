import { Fragment, useEffect, useMemo, useState, ReactNode } from "react";
import {
  ChevronDown,
  ChevronRight,
  Download,
  FileCode2,
  Folder,
  FolderGit2,
  GitBranchPlus,
  GitCommitHorizontal,
  GitPullRequest,
  LoaderCircle,
  PlugZap,
  RotateCcw,
  Upload,
} from "lucide-react";
import type { FileChange, RepositoryTreeEntry } from "../types";
import type {
  GitHubBranchSummary,
  GitHubRepositoryStatus,
  GitHubRepositorySummary,
} from "../lib/repositoryApi";

interface SidebarProps {
  repoName: string;
  repoRoot: string;
  entries: RepositoryTreeEntry[];
  changes: FileChange[];
  activePath: string | null;
  isLoading: boolean;
  agentRunning?: boolean;
  error?: string | null;
  onSelect: (path: string) => void;
  onRefresh: () => void;
  githubRepositories?: GitHubRepositorySummary[];
  selectedGitHubRepository?: string | null;
  githubLoading?: boolean;
  githubError?: string | null;
  onSelectGitHubRepository?: (fullName: string) => void;
  onUseLocalRepository?: () => void;
  onRefreshGitHubRepositories?: () => void;
  branches?: GitHubBranchSummary[];
  currentBranch?: string | null;
  branchesLoading?: boolean;
  branchesError?: string | null;
  onSwitchBranch?: (branchName: string) => void;
  defaultBranch?: string | null;
  repositoryPermissions?: GitHubRepositorySummary["permissions"] | null;
  githubRepositoryStatus?: GitHubRepositoryStatus | null;
  githubActionLoading?: string | null;
  githubActionMessage?: string | null;
  githubActionError?: string | null;
  githubPullRequestUrl?: string | null;
  committableFileCount?: number;
  onTestGitHubConnection?: () => void;
  onCreateGitHubBranch?: (branch: string) => void;
  onPullGitHubBranch?: () => void;
  onCommitGitHubChanges?: (message: string) => void;
  onPushGitHubBranch?: () => void;
  onCreateGitHubPullRequest?: (request: {
    title: string;
    body: string;
    base: string;
    draft: boolean;
  }) => void;
}

const statusColor = {
  modified: "text-amber-300",
  added: "text-emerald-300",
  deleted: "text-rose-300",
};

const formatBytes = (size?: number | null) => {
  if (!size) return "";
  if (size < 1024) return `${size}b`;
  if (size < 1024 * 1024) return `${Math.round(size / 1024)}kb`;
  return `${(size / 1024 / 1024).toFixed(1)}mb`;
}

export const Sidebar = ({
  repoName,
  repoRoot,
  entries,
  changes,
  activePath,
  isLoading,
  agentRunning = false,
  error,
  onSelect,
  onRefresh,
  githubRepositories = [],
  selectedGitHubRepository = null,
  githubLoading = false,
  githubError = null,
  onSelectGitHubRepository,
  onUseLocalRepository,
  onRefreshGitHubRepositories,
  branches = [],
  currentBranch = null,
  branchesLoading = false,
  branchesError = null,
  onSwitchBranch,
  defaultBranch = null,
  repositoryPermissions = null,
  githubRepositoryStatus = null,
  githubActionLoading = null,
  githubActionMessage = null,
  githubActionError = null,
  githubPullRequestUrl = null,
  committableFileCount = 0,
  onTestGitHubConnection,
  onCreateGitHubBranch,
  onPullGitHubBranch,
  onCommitGitHubChanges,
  onPushGitHubBranch,
  onCreateGitHubPullRequest,
}: SidebarProps) => {
  
  const fileEntries = entries.filter((entry) => entry.kind === "file");
  const [openDirs, setOpenDirs] = useState<Set<string>>(new Set());
  const [showChanges, setShowChanges] = useState(true);
  const [showFiles, setShowFiles] = useState(true);
  const [showGitHubActions, setShowGitHubActions] = useState(false);
  const [newBranch, setNewBranch] = useState("agent/");
  const [commitMessage, setCommitMessage] = useState("");
  const [pullRequestTitle, setPullRequestTitle] = useState("");
  const [pullRequestBody, setPullRequestBody] = useState("");
  const [pullRequestBase, setPullRequestBase] = useState(defaultBranch ?? "");
  const [draftPullRequest, setDraftPullRequest] = useState(true);
  const childrenByParent = useMemo(() => {
    const map = new Map<string, RepositoryTreeEntry[]>();
    for (const entry of entries) {
      const parentPath = entry.path.substring(0, entry.path.lastIndexOf('/'));
      if (!map.has(parentPath)) map.set(parentPath, []);
      map.get(parentPath)!.push(entry);
    }
    return map;
  }, [entries]);

  const rootEntries = useMemo(() => childrenByParent.get('') ?? [], [childrenByParent]);
  const availableBranches = useMemo(() => {
    if (!currentBranch || branches.some((branch) => branch.name === currentBranch)) return branches;
    return [{ name: currentBranch, sha: githubRepositoryStatus?.head_sha ?? "" }, ...branches];
  }, [branches, currentBranch, githubRepositoryStatus?.head_sha]);
  const canPush = Boolean(repositoryPermissions?.push);
  const onDefaultBranch = Boolean(currentBranch && defaultBranch && currentBranch === defaultBranch);
  const operationRunning = Boolean(githubActionLoading);

  useEffect(() => {
    setPullRequestBase(defaultBranch ?? "");
    setNewBranch("agent/");
    setCommitMessage("");
    setPullRequestTitle("");
    setPullRequestBody("");
    setDraftPullRequest(true);
  }, [defaultBranch, selectedGitHubRepository]);

  const renderTree = (nodes: RepositoryTreeEntry[]): ReactNode[] =>
    nodes.map((entry) => {
      const isFile = entry.kind === "file";
      const isDir = entry.kind === "directory";
      const active = activePath === entry.path;
      const Icon = isFile ? FileCode2 : Folder;
      const ChevronIcon = openDirs.has(entry.path) ? ChevronDown : ChevronRight;

      const handleClick = () => {
        if (isDir) {
          setOpenDirs(prev => {
            const next = new Set(prev);
            if (next.has(entry.path)) {
              next.delete(entry.path);
            } else {
              next.add(entry.path);
            }
            return next;
          });
        } else {
          onSelect(entry.path);
        }
      };

      const children = isDir && openDirs.has(entry.path) ? childrenByParent.get(entry.path) ?? [] : [];

      return (
        <Fragment key={`${entry.kind}:${entry.path}`}>
          <button
            type="button"
            onClick={handleClick}
            style={{ paddingLeft: `${12 + (entry.path.split('/').length - 1) * 16}px` }}
            className={`group flex w-full items-center gap-2 border-l-2 py-1.5 pr-3 text-left ${
              active ? "border-accent bg-selected" : "border-transparent hover:bg-hover"
            } ${isDir ? "cursor-pointer text-faint" : "cursor-pointer text-ink-soft"}`}
          >
            {isDir ? (
              <ChevronIcon size={12} className="text-muted" />
            ) : null}
            <Icon size={13} className={isFile ? "text-faint" : "text-accent-light"} />
            <span className="min-w-0 flex-1 truncate text-[12px]">{entry.name}</span>
            {isFile ? <span className="font-mono text-[8px] text-faint">{formatBytes(entry.size)}</span> : null}
          </button>
          {children.length > 0 && renderTree(children)}
        </Fragment>
      );
    });

  return (
    <aside className="flex w-72 shrink-0 flex-col border-r border-line bg-panel-soft">
      <div className="flex h-12 items-center justify-between border-b border-line px-3">
        <span className="text-[16px] font-semibold uppercase tracking-[0.16em] text-muted">Workspace</span>
        <button type="button" className="icon-button" aria-label="Refresh repository" title="Refresh repository" onClick={onRefresh}>
          <RotateCcw size={13} />
        </button>
      </div>

      <div className="border-b border-line p-3">
        <div className="flex items-center gap-2">
          <FolderGit2 size={15} className="shrink-0 text-accent-light" />
          <select
            value={selectedGitHubRepository ? `github:${selectedGitHubRepository}` : "local"}
            disabled={githubLoading}
            onChange={(event) => {
              const value = event.target.value;
              if (value === "local") {
                onUseLocalRepository?.();
              } else if (value.startsWith("github:")) {
                onSelectGitHubRepository?.(value.slice("github:".length));
              }
            }}
            className="min-w-0 flex-1 rounded-md border border-line bg-surface px-2.5 py-2 text-xs font-medium text-ink outline-none hover:border-line-strong focus:border-accent/70"
            aria-label="Select repository"
          >
            <option value="local">Local · {selectedGitHubRepository ? "configured workspace" : repoName}</option>
            {githubRepositories.map((repository) => (
              <option key={repository.id} value={`github:${repository.full_name}`}>
                GitHub · {repository.full_name}{repository.private ? " (private)" : ""}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="icon-button"
            aria-label="Refresh GitHub repositories"
            title="Refresh GitHub repositories"
            onClick={onRefreshGitHubRepositories}
            disabled={githubLoading}
          >
            <RotateCcw size={13} />
          </button>
        </div>
        <p className="mt-2 truncate px-1 font-mono text-[10px] text-faint">{repoRoot}</p>
        {githubLoading ? <p className="mt-2 px-1 text-[10px] text-muted">Loading GitHub repositories…</p> : null}
        {githubError ? <p className="mt-2 px-1 text-[10px] leading-4 text-rose-300">{githubError}</p> : null}
        {selectedGitHubRepository && branches && branches.length > 0 && (
          <div className="mt-2">
            <label className="block text-[10px] font-medium text-muted mb-1">Branch</label>
            <select
              value={currentBranch ?? ""}
              disabled={branchesLoading || isLoading}
              onChange={(event) => {
                const value = event.target.value;
                if (value) {
                  onSwitchBranch?.(value);
                }
              }}
              className="w-full rounded-md border border-line bg-surface px-2.5 py-2 text-xs text-ink outline-none hover:border-line-strong focus:border-accent/70"
              aria-label="Switch branch"
            >
              {availableBranches.map((branch) => (
                <option key={branch.name} value={branch.name}>
                  {branch.name}
                </option>
              ))}
            </select>
            {branchesLoading && <p className="mt-1 px-1 text-[10px] text-muted">Loading branches…</p>}
            {branchesError && <p className="mt-1 px-1 text-[10px] leading-4 text-rose-300">{branchesError}</p>}
          </div>
        )}

        {selectedGitHubRepository ? (
          <div className="mt-3 border-t border-line pt-3">
            <button
              type="button"
              onClick={() => setShowGitHubActions((current) => !current)}
              className="flex w-full items-center gap-2 text-left text-[10px] font-semibold uppercase tracking-wider text-muted hover:text-ink-soft"
            >
              {showGitHubActions ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              GitHub publish controls
              {githubActionLoading ? <LoaderCircle size={12} className="ml-auto animate-spin text-accent-light" /> : null}
            </button>

            {showGitHubActions ? (
              <div className="mt-2 space-y-2">
                <div className="rounded-md border border-line bg-surface/60 p-2 text-[10px] text-muted">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate font-mono text-ink-soft">{currentBranch ?? "No branch"}</span>
                    <span>{githubRepositoryStatus?.ahead ?? 0} ahead · {githubRepositoryStatus?.behind ?? 0} behind</span>
                  </div>
                  <div className="mt-1 flex items-center justify-between gap-2">
                    <span>{githubRepositoryStatus?.dirty ? "Uncommitted changes" : "Working tree clean"}</span>
                    <span>{canPush ? "Write access" : "Read-only"}</span>
                  </div>
                </div>

                {onDefaultBranch ? (
                  <p className="rounded-md border border-amber-400/30 bg-amber-400/5 px-2 py-1.5 text-[10px] leading-4 text-amber-200">
                    Direct agent commits to {defaultBranch} are blocked. Create an agent/* branch first.
                  </p>
                ) : null}

                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    className="secondary-button justify-center"
                    disabled={operationRunning}
                    onClick={onTestGitHubConnection}
                  >
                    <PlugZap size={12} /> Test
                  </button>
                  <button
                    type="button"
                    className="secondary-button justify-center"
                    disabled={operationRunning || Boolean(githubRepositoryStatus?.dirty)}
                    onClick={onPullGitHubBranch}
                    title={githubRepositoryStatus?.dirty ? "Commit or discard local changes before pulling." : "Fast-forward only pull"}
                  >
                    <Download size={12} /> Pull
                  </button>
                </div>

                <div className="flex gap-2">
                  <input
                    value={newBranch}
                    onChange={(event) => setNewBranch(event.target.value)}
                    className="min-w-0 flex-1 rounded-md border border-line bg-surface px-2 py-1.5 font-mono text-[10px] text-ink outline-none focus:border-accent/70"
                    placeholder="agent/feature-name"
                    aria-label="New GitHub branch"
                  />
                  <button
                    type="button"
                    className="icon-button shrink-0"
                    disabled={operationRunning || !canPush || newBranch.trim().length < 3}
                    onClick={() => onCreateGitHubBranch?.(newBranch.trim())}
                    title="Create and switch branch"
                  >
                    <GitBranchPlus size={13} />
                  </button>
                </div>

                <textarea
                  value={commitMessage}
                  onChange={(event) => setCommitMessage(event.target.value)}
                  className="min-h-14 w-full resize-y rounded-md border border-line bg-surface px-2 py-1.5 text-[10px] leading-4 text-ink outline-none focus:border-accent/70"
                  placeholder="Commit message"
                  maxLength={200}
                />
                <button
                  type="button"
                  className="secondary-button w-full justify-center"
                  disabled={operationRunning || !canPush || onDefaultBranch || committableFileCount === 0 || commitMessage.trim().length < 3}
                  onClick={() => onCommitGitHubChanges?.(commitMessage.trim())}
                >
                  <GitCommitHorizontal size={12} /> Commit {committableFileCount} applied file{committableFileCount === 1 ? "" : "s"}
                </button>

                <button
                  type="button"
                  className="secondary-button w-full justify-center"
                  disabled={operationRunning || !canPush || onDefaultBranch || Boolean(githubRepositoryStatus?.dirty)}
                  onClick={onPushGitHubBranch}
                >
                  <Upload size={12} /> Push current branch
                </button>

                <div className="border-t border-line pt-2">
                  <input
                    value={pullRequestTitle}
                    onChange={(event) => setPullRequestTitle(event.target.value)}
                    className="w-full rounded-md border border-line bg-surface px-2 py-1.5 text-[10px] text-ink outline-none focus:border-accent/70"
                    placeholder="Pull request title"
                    maxLength={256}
                  />
                  <div className="mt-2 flex gap-2">
                    <select
                      value={pullRequestBase}
                      onChange={(event) => setPullRequestBase(event.target.value)}
                      className="min-w-0 flex-1 rounded-md border border-line bg-surface px-2 py-1.5 text-[10px] text-ink outline-none focus:border-accent/70"
                      aria-label="Pull request base branch"
                    >
                      {availableBranches.map((branch) => (
                        <option key={`base:${branch.name}`} value={branch.name}>{branch.name}</option>
                      ))}
                    </select>
                    <label className="flex items-center gap-1 text-[10px] text-muted">
                      <input
                        type="checkbox"
                        checked={draftPullRequest}
                        onChange={(event) => setDraftPullRequest(event.target.checked)}
                        className="accent-accent"
                      />
                      Draft
                    </label>
                  </div>
                  <textarea
                    value={pullRequestBody}
                    onChange={(event) => setPullRequestBody(event.target.value)}
                    className="mt-2 min-h-16 w-full resize-y rounded-md border border-line bg-surface px-2 py-1.5 text-[10px] leading-4 text-ink outline-none focus:border-accent/70"
                    placeholder="What changed, why, impact, and validation"
                  />
                  <button
                    type="button"
                    className="primary-button mt-2 w-full justify-center"
                    disabled={
                      operationRunning
                      || !canPush
                      || !currentBranch
                      || currentBranch === pullRequestBase
                      || pullRequestTitle.trim().length < 3
                      || Boolean(githubRepositoryStatus?.dirty)
                    }
                    onClick={() => onCreateGitHubPullRequest?.({
                      title: pullRequestTitle.trim(),
                      body: pullRequestBody,
                      base: pullRequestBase,
                      draft: draftPullRequest,
                    })}
                  >
                    <GitPullRequest size={12} /> Create {draftPullRequest ? "draft " : ""}PR
                  </button>
                </div>

                {githubActionMessage ? <p className="text-[10px] leading-4 text-emerald-300">{githubActionMessage}</p> : null}
                {githubActionError ? <p className="text-[10px] leading-4 text-rose-300">{githubActionError}</p> : null}
                {githubPullRequestUrl ? (
                  <a
                    href={githubPullRequestUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="block truncate text-[10px] text-accent-light underline underline-offset-2"
                  >
                    Open pull request
                  </a>
                ) : null}
              </div>
            ) : null}
          </div>
        ) : null}
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        {(changes.length > 0 || agentRunning) && (
          <div className="border-b border-line py-2 max-h-64 overflow-y-auto">
            <button
              type="button"
              onClick={() => setShowChanges(prev => !prev)}
              className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-[11px] font-semibold uppercase tracking-wider text-muted hover:bg-hover"
            >
              {showChanges ? <ChevronDown size={12} className="text-muted" /> : <ChevronRight size={12} className="text-muted" />}
              <span>Agent changes · {changes.length}</span>
              {agentRunning ? (
                <LoaderCircle size={12} className="ml-auto animate-spin text-accent-light" aria-label="Agent is running" />
              ) : null}
            </button>
            {showChanges && agentRunning && changes.length === 0 ? (
              <div className="flex items-center gap-2 px-4 py-3 text-[10px] text-muted">
                <LoaderCircle size={12} className="animate-spin text-accent-light" />
                Generating file changes…
              </div>
            ) : null}
            {showChanges && changes.map((change) => {
              const fileName = change.path.split("/").at(-1);
              const folder = change.path.slice(0, -(fileName?.length ?? 0)).replace(/\/$/, "");

              return (
                <button
                  type="button"
                  key={change.path}
                  onClick={() => onSelect(change.path)}
                  className={`group flex w-full items-center gap-2 border-l-2 px-3 py-2 text-left ${
                    activePath === change.path ? "border-accent bg-selected" : "border-transparent hover:bg-hover"
                  }`}
                >
                  <span className={`font-mono text-[10px] font-bold uppercase ${statusColor[change.status]}`}>
                    {change.status[0]}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-xs text-ink">{fileName}</span>
                    <span className="block truncate text-[10px] text-faint">{folder}</span>
                  </span>
                  <span className="font-mono text-[9px] text-faint group-hover:text-muted">+{change.additions} −{change.deletions}</span>
                </button>
              );
            })}
          </div>
        )}
        <div className="py-2">
          <button
            type="button"
            onClick={() => setShowFiles(prev => !prev)}
            className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-[10px] font-semibold uppercase tracking-wider text-muted hover:bg-hover"
          >
            {showFiles ? <ChevronDown size={12} className="text-muted" /> : <ChevronRight size={12} className="text-muted" />}
            Files · {fileEntries.length}
          </button>

          {isLoading ? <p className="px-3 py-2 text-[11px] text-muted">Loading repository…</p> : null}
          {error ? <p className="px-3 py-2 text-[11px] leading-5 text-rose-300">{error}</p> : null}

          {showFiles && renderTree(rootEntries)}
        </div>
      </div>

      <div className="border-t border-line p-3 text-[10px] text-faint">
        <div className="flex items-center justify-between"><span>Platform</span><span className="font-mono text-muted">{window.desktop?.platform ?? "browser"}</span></div>
      </div>
    </aside>
  );
}
