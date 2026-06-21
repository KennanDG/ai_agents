import { Fragment, useMemo, useState, ReactNode } from "react";
import { ChevronDown, ChevronRight, FileCode2, Folder, FolderGit2, RotateCcw } from "lucide-react";
import type { FileChange, RepositoryTreeEntry } from "../types";

interface SidebarProps {
  repoName: string;
  repoRoot: string;
  entries: RepositoryTreeEntry[];
  changes: FileChange[];
  activePath: string | null;
  isLoading: boolean;
  error?: string | null;
  onSelect: (path: string) => void;
  onRefresh: () => void;
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
  error,
  onSelect,
  onRefresh,
}: SidebarProps) => {
  
  const fileEntries = entries.filter((entry) => entry.kind === "file");
  const [openDirs, setOpenDirs] = useState<Set<string>>(new Set());
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
        <button type="button" className="flex w-full items-center gap-2 rounded-md border border-line bg-surface px-2.5 py-2 text-left hover:border-line-strong">
          <FolderGit2 size={15} className="text-accent-light" />
          <span className="min-w-0 flex-1 truncate text-xs font-medium text-ink">{repoName}</span>
          <ChevronDown size={13} className="text-muted" />
        </button>
        <p className="mt-2 truncate px-1 font-mono text-[10px] text-faint">{repoRoot}</p>
      </div>

      {changes.length > 0 ? (
        <div className="border-b border-line py-2">
          <div className="px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted">Agent changes · {changes.length}</div>
          {changes.map((change) => {
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
      ) : null}

      <div className="min-h-0 flex-1 overflow-auto py-2">
        <div className="flex items-center justify-between px-3 py-1.5">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-muted">Files · {fileEntries.length}</span>
        </div>

        {isLoading ? <p className="px-3 py-2 text-[11px] text-muted">Loading repository…</p> : null}
        {error ? <p className="px-3 py-2 text-[11px] leading-5 text-rose-300">{error}</p> : null}

        {renderTree(rootEntries)}
      </div>

      <div className="border-t border-line p-3 text-[10px] text-faint">
        <div className="flex items-center justify-between"><span>Platform</span><span className="font-mono text-muted">{window.desktop?.platform ?? "browser"}</span></div>
      </div>
    </aside>
  );
}
