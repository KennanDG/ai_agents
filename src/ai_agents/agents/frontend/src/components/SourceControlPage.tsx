import { type ReactNode, useEffect, useMemo, useState } from "react";
import {
  ArrowDownToLine,
  ArrowUpFromLine,
  CheckCircle2,
  CircleAlert,
  GitBranch,
  GitCommitHorizontal,
  GitPullRequest,
  LoaderCircle,
  Plus,
  RefreshCcw,
  ShieldCheck,
} from "lucide-react";
import type {
  GitHubBranchSummary,
  GitHubRepositoryStatus,
  GitHubRepositorySummary,
} from "../lib/repositoryApi";

type SourceControlPageProps = {
  repoName: string;
  repoRoot: string;
  githubRepositories: GitHubRepositorySummary[];
  selectedGitHubRepository: string | null;
  githubLoading: boolean;
  githubError: string | null;
  onSelectGitHubRepository: (fullName: string) => void;
  onUseLocalRepository: () => void;
  onRefreshGitHubRepositories: () => void;
  branches: GitHubBranchSummary[];
  currentBranch: string | null;
  branchesLoading: boolean;
  branchesError: string | null;
  onSwitchBranch: (branchName: string) => void;
  defaultBranch: string | null;
  repositoryPermissions: GitHubRepositorySummary["permissions"] | null;
  githubRepositoryStatus: GitHubRepositoryStatus | null;
  githubActionLoading: string | null;
  githubActionMessage: string | null;
  githubActionError: string | null;
  githubPullRequestUrl: string | null;
  committableFileCount: number;
  onTestGitHubConnection: () => void;
  onCreateGitHubBranch: (branch: string) => void;
  onPullGitHubBranch: () => void;
  onCommitGitHubChanges: (message: string) => void;
  onPushGitHubBranch: () => void;
  onCreateGitHubPullRequest: (request: {
    title: string;
    body: string;
    base: string;
    draft: boolean;
  }) => void;
};

const FieldLabel = ({ children }: { children: ReactNode }) => (
  <label className="mb-1 block text-[10px] font-semibold uppercase tracking-wider text-muted">
    {children}
  </label>
);

const StatusList = ({
  title,
  paths,
}: {
  title: string;
  paths: string[];
}) => (
  <div className="rounded-lg border border-line bg-panel p-3">
    <div className="mb-2 flex items-center justify-between">
      <span className="text-xs font-semibold text-ink-soft">{title}</span>
      <span className="rounded bg-surface px-1.5 py-0.5 font-mono text-[9px] text-muted">
        {paths.length}
      </span>
    </div>
    {paths.length > 0 ? (
      <div className="max-h-40 space-y-1 overflow-auto">
        {paths.map((path) => (
          <div key={path} className="truncate font-mono text-[10px] text-muted" title={path}>
            {path}
          </div>
        ))}
      </div>
    ) : (
      <p className="text-[10px] text-faint">None</p>
    )}
  </div>
);

export const SourceControlPage = ({
  repoName,
  repoRoot,
  githubRepositories,
  selectedGitHubRepository,
  githubLoading,
  githubError,
  onSelectGitHubRepository,
  onUseLocalRepository,
  onRefreshGitHubRepositories,
  branches,
  currentBranch,
  branchesLoading,
  branchesError,
  onSwitchBranch,
  defaultBranch,
  repositoryPermissions,
  githubRepositoryStatus,
  githubActionLoading,
  githubActionMessage,
  githubActionError,
  githubPullRequestUrl,
  committableFileCount,
  onTestGitHubConnection,
  onCreateGitHubBranch,
  onPullGitHubBranch,
  onCommitGitHubChanges,
  onPushGitHubBranch,
  onCreateGitHubPullRequest,
}: SourceControlPageProps) => {
  const [newBranch, setNewBranch] = useState("agent/");
  const [commitMessage, setCommitMessage] = useState("");
  const [prTitle, setPrTitle] = useState("");
  const [prBody, setPrBody] = useState("");
  const [prBase, setPrBase] = useState(defaultBranch ?? "main");
  const [draft, setDraft] = useState(true);

  useEffect(() => {
    if (defaultBranch) setPrBase(defaultBranch);
  }, [defaultBranch]);

  const selectedSummary = useMemo(
    () => githubRepositories.find((item) => item.full_name === selectedGitHubRepository) ?? null,
    [githubRepositories, selectedGitHubRepository],
  );

  const busy = githubActionLoading !== null;
  const canPush = Boolean(repositoryPermissions?.push && selectedGitHubRepository);
  const canCommit = canPush && committableFileCount > 0 && commitMessage.trim().length > 0;
  const canOpenPr = Boolean(
    canPush &&
      currentBranch &&
      prBase &&
      currentBranch !== prBase &&
      prTitle.trim() &&
      githubRepositoryStatus &&
      !githubRepositoryStatus.dirty,
  );

  return (
    <section className="flex min-h-0 min-w-0 flex-1 flex-col bg-canvas">
      <header className="flex h-12 shrink-0 items-center justify-between border-b border-line px-5">
        <div className="flex items-center gap-2">
          <GitBranch size={16} className="text-accent-light" />
          <h1 className="text-sm font-semibold text-ink">Source control</h1>
        </div>
        <button
          type="button"
          className="secondary-button"
          onClick={onRefreshGitHubRepositories}
          disabled={githubLoading}
        >
          {githubLoading ? <LoaderCircle size={12} className="animate-spin" /> : <RefreshCcw size={12} />}
          Refresh repositories
        </button>
      </header>

      <div className="min-h-0 flex-1 overflow-auto p-5">
        <div className="mx-auto grid max-w-6xl gap-4 lg:grid-cols-[minmax(0,1.2fr)_minmax(320px,0.8fr)]">
          <div className="space-y-4">
            <div className="rounded-xl border border-line bg-panel-soft p-4">
              <div className="mb-4 flex items-start justify-between gap-3">
                <div>
                  <h2 className="text-sm font-semibold text-ink">Repository</h2>
                  <p className="mt-1 text-[11px] text-muted">
                    Choose the managed checkout used by the coding agent and Git operations.
                  </p>
                </div>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={onTestGitHubConnection}
                  disabled={busy}
                >
                  {githubActionLoading === "test" ? (
                    <LoaderCircle size={12} className="animate-spin" />
                  ) : (
                    <ShieldCheck size={12} />
                  )}
                  Test connection
                </button>
              </div>

              <FieldLabel>Repository source</FieldLabel>
              <select
                value={selectedGitHubRepository ? `github:${selectedGitHubRepository}` : "local"}
                disabled={githubLoading || busy}
                onChange={(event) => {
                  const value = event.target.value;
                  if (value === "local") onUseLocalRepository();
                  else if (value.startsWith("github:")) {
                    onSelectGitHubRepository(value.slice("github:".length));
                  }
                }}
                className="w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none hover:border-line-strong focus:border-accent/70"
              >
                <option value="local">Local · {repoName}</option>
                {githubRepositories.map((repository) => (
                  <option key={repository.id} value={`github:${repository.full_name}`}>
                    GitHub · {repository.full_name}{repository.private ? " (private)" : ""}
                  </option>
                ))}
              </select>

              <p className="mt-2 truncate font-mono text-[10px] text-faint" title={repoRoot}>
                {repoRoot}
              </p>

              {githubError ? (
                <div className="mt-3 flex items-start gap-2 rounded-md border border-rose-500/20 bg-rose-500/8 p-2.5 text-[10px] leading-4 text-rose-300">
                  <CircleAlert size={13} className="mt-0.5 shrink-0" />
                  {githubError}
                </div>
              ) : null}

              {selectedGitHubRepository ? (
                <div className="mt-4 grid gap-3 sm:grid-cols-2">
                  <div>
                    <FieldLabel>Current branch</FieldLabel>
                    <select
                      value={currentBranch ?? ""}
                      disabled={branchesLoading || busy}
                      onChange={(event) => event.target.value && onSwitchBranch(event.target.value)}
                      className="w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none hover:border-line-strong focus:border-accent/70"
                    >
                      {branches.map((branch) => (
                        <option key={branch.name} value={branch.name}>
                          {branch.name}
                        </option>
                      ))}
                    </select>
                    {branchesError ? <p className="mt-1 text-[10px] text-rose-300">{branchesError}</p> : null}
                  </div>

                  <div>
                    <FieldLabel>Create agent branch</FieldLabel>
                    <div className="flex gap-2">
                      <input
                        value={newBranch}
                        onChange={(event) => setNewBranch(event.target.value)}
                        className="min-w-0 flex-1 rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/70"
                        placeholder="agent/feature-name"
                      />
                      <button
                        type="button"
                        className="secondary-button h-8"
                        disabled={!newBranch.trim() || busy || !repositoryPermissions?.push}
                        onClick={() => onCreateGitHubBranch(newBranch.trim())}
                      >
                        {githubActionLoading === "branch" ? (
                          <LoaderCircle size={12} className="animate-spin" />
                        ) : (
                          <Plus size={12} />
                        )}
                        Create
                      </button>
                    </div>
                  </div>
                </div>
              ) : null}

              {selectedSummary ? (
                <div className="mt-4 grid grid-cols-2 gap-2 text-[10px] text-muted sm:grid-cols-4">
                  <div className="rounded-md border border-line bg-surface p-2">
                    <span className="block text-faint">Default</span>
                    <span className="font-mono text-ink-soft">{selectedSummary.default_branch}</span>
                  </div>
                  <div className="rounded-md border border-line bg-surface p-2">
                    <span className="block text-faint">Push</span>
                    <span className="text-ink-soft">{selectedSummary.permissions.push ? "Allowed" : "Denied"}</span>
                  </div>
                  <div className="rounded-md border border-line bg-surface p-2">
                    <span className="block text-faint">Visibility</span>
                    <span className="text-ink-soft">{selectedSummary.private ? "Private" : "Public"}</span>
                  </div>
                  <div className="rounded-md border border-line bg-surface p-2">
                    <span className="block text-faint">Policy</span>
                    <span className="text-ink-soft">PR-first</span>
                  </div>
                </div>
              ) : null}
            </div>

            {selectedGitHubRepository ? (
              <div className="rounded-xl border border-line bg-panel-soft p-4">
                <div className="mb-4 flex items-center justify-between">
                  <div>
                    <h2 className="text-sm font-semibold text-ink">Working tree</h2>
                    <p className="mt-1 text-[11px] text-muted">
                      Only approved and applied agent files are eligible for the scoped commit action.
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={onPullGitHubBranch}
                      disabled={busy}
                    >
                      {githubActionLoading === "pull" ? (
                        <LoaderCircle size={12} className="animate-spin" />
                      ) : (
                        <ArrowDownToLine size={12} />
                      )}
                      Pull
                    </button>
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={onPushGitHubBranch}
                      disabled={busy || !canPush}
                    >
                      {githubActionLoading === "push" ? (
                        <LoaderCircle size={12} className="animate-spin" />
                      ) : (
                        <ArrowUpFromLine size={12} />
                      )}
                      Push
                    </button>
                  </div>
                </div>

                <div className="grid gap-3 md:grid-cols-3">
                  <StatusList title="Staged" paths={githubRepositoryStatus?.staged_files ?? []} />
                  <StatusList title="Unstaged" paths={githubRepositoryStatus?.unstaged_files ?? []} />
                  <StatusList title="Untracked" paths={githubRepositoryStatus?.untracked_files ?? []} />
                </div>

                <div className="mt-4 grid gap-3 sm:grid-cols-[1fr_auto]">
                  <div>
                    <FieldLabel>Commit message</FieldLabel>
                    <input
                      value={commitMessage}
                      onChange={(event) => setCommitMessage(event.target.value)}
                      className="w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/70"
                      placeholder="Describe the approved agent change"
                    />
                  </div>
                  <button
                    type="button"
                    className="primary-button mt-auto h-8"
                    disabled={!canCommit || busy}
                    onClick={() => onCommitGitHubChanges(commitMessage.trim())}
                  >
                    {githubActionLoading === "commit" ? (
                      <LoaderCircle size={12} className="animate-spin" />
                    ) : (
                      <GitCommitHorizontal size={12} />
                    )}
                    Commit {committableFileCount} file{committableFileCount === 1 ? "" : "s"}
                  </button>
                </div>
              </div>
            ) : null}
          </div>

          <div className="space-y-4">
            <div className="rounded-xl border border-line bg-panel-soft p-4">
              <div className="mb-4 flex items-center gap-2">
                <GitPullRequest size={15} className="text-accent-light" />
                <div>
                  <h2 className="text-sm font-semibold text-ink">Pull request</h2>
                  <p className="mt-1 text-[11px] text-muted">Push the agent branch before opening the PR.</p>
                </div>
              </div>

              <div className="space-y-3">
                <div>
                  <FieldLabel>Title</FieldLabel>
                  <input
                    value={prTitle}
                    onChange={(event) => setPrTitle(event.target.value)}
                    className="w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/70"
                    placeholder="Agent change summary"
                  />
                </div>
                <div>
                  <FieldLabel>Base branch</FieldLabel>
                  <select
                    value={prBase}
                    onChange={(event) => setPrBase(event.target.value)}
                    className="w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/70"
                  >
                    {branches.map((branch) => (
                      <option key={branch.name} value={branch.name}>
                        {branch.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <FieldLabel>Description</FieldLabel>
                  <textarea
                    value={prBody}
                    onChange={(event) => setPrBody(event.target.value)}
                    rows={7}
                    className="w-full resize-y rounded-md border border-line bg-surface px-3 py-2 text-xs leading-5 text-ink outline-none focus:border-accent/70"
                    placeholder="What changed, why, and how it was validated"
                  />
                </div>
                <label className="flex cursor-pointer items-center gap-2 text-[11px] text-muted">
                  <input
                    type="checkbox"
                    checked={draft}
                    onChange={(event) => setDraft(event.target.checked)}
                    className="accent-accent"
                  />
                  Open as draft
                </label>
                <button
                  type="button"
                  className="primary-button h-8 w-full justify-center"
                  disabled={!canOpenPr || busy}
                  onClick={() =>
                    onCreateGitHubPullRequest({
                      title: prTitle.trim(),
                      body: prBody,
                      base: prBase,
                      draft,
                    })
                  }
                >
                  {githubActionLoading === "pr" ? (
                    <LoaderCircle size={12} className="animate-spin" />
                  ) : (
                    <GitPullRequest size={12} />
                  )}
                  Create {draft ? "draft " : ""}pull request
                </button>
              </div>
            </div>

            <div className="rounded-xl border border-line bg-panel-soft p-4">
              <h2 className="text-sm font-semibold text-ink">Repository status</h2>
              <div className="mt-3 space-y-2 text-[11px]">
                <div className="flex justify-between gap-3">
                  <span className="text-muted">Branch</span>
                  <span className="font-mono text-ink-soft">{currentBranch ?? "Local only"}</span>
                </div>
                <div className="flex justify-between gap-3">
                  <span className="text-muted">Ahead / behind</span>
                  <span className="font-mono text-ink-soft">
                    {githubRepositoryStatus
                      ? `${githubRepositoryStatus.ahead} / ${githubRepositoryStatus.behind}`
                      : "—"}
                  </span>
                </div>
                <div className="flex justify-between gap-3">
                  <span className="text-muted">Working tree</span>
                  <span className="text-ink-soft">
                    {githubRepositoryStatus?.dirty ? "Dirty" : "Clean"}
                  </span>
                </div>
                <div className="flex justify-between gap-3">
                  <span className="text-muted">Default branch</span>
                  <span className="font-mono text-ink-soft">{defaultBranch ?? "—"}</span>
                </div>
              </div>
            </div>

            {githubActionMessage ? (
              <div className="flex items-start gap-2 rounded-xl border border-emerald-500/20 bg-emerald-500/8 p-3 text-[11px] leading-5 text-emerald-300">
                <CheckCircle2 size={14} className="mt-0.5 shrink-0" />
                <div>
                  <p>{githubActionMessage}</p>
                  {githubPullRequestUrl ? (
                    <a
                      href={githubPullRequestUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="mt-1 inline-block underline underline-offset-2"
                    >
                      Open pull request
                    </a>
                  ) : null}
                </div>
              </div>
            ) : null}

            {githubActionError ? (
              <div className="flex items-start gap-2 rounded-xl border border-rose-500/20 bg-rose-500/8 p-3 text-[11px] leading-5 text-rose-300">
                <CircleAlert size={14} className="mt-0.5 shrink-0" />
                {githubActionError}
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </section>
  );
}
