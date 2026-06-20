import { ChevronDown, FilePlus2, FolderGit2, Plus, RotateCcw } from "lucide-react";
import type { FileChange } from "../types";

interface SidebarProps {
  changes: FileChange[];
  activePath: string;
  onSelect: (path: string) => void;
}

const statusColor = {
  modified: "text-amber-300",
  added: "text-emerald-300",
  deleted: "text-rose-300",
};

export function Sidebar({ changes, activePath, onSelect }: SidebarProps) {
  return (
    <aside className="flex w-72 shrink-0 flex-col border-r border-line bg-panel-soft">
      <div className="flex h-12 items-center justify-between border-b border-line px-3">
        <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted">Workspace</span>
        <button type="button" className="icon-button" aria-label="New task" title="New task">
          <Plus size={15} />
        </button>
      </div>

      <div className="border-b border-line p-3">
        <button type="button" className="flex w-full items-center gap-2 rounded-md border border-line bg-surface px-2.5 py-2 text-left hover:border-line-strong">
          <FolderGit2 size={15} className="text-accent-light" />
          <span className="min-w-0 flex-1 truncate text-xs font-medium text-ink">ai_agents</span>
          <ChevronDown size={13} className="text-muted" />
        </button>
        <p className="mt-2 truncate px-1 font-mono text-[10px] text-faint">~/projects/ai_agents</p>
      </div>

      <div className="min-h-0 flex-1 overflow-auto py-2">
        <div className="flex items-center justify-between px-3 py-1.5">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-muted">Changes · {changes.length}</span>
          <div className="flex gap-1">
            <button type="button" className="icon-button" aria-label="Add file" title="Add file"><FilePlus2 size={13} /></button>
            <button type="button" className="icon-button" aria-label="Refresh" title="Refresh"><RotateCcw size={13} /></button>
          </div>
        </div>
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

      <div className="border-t border-line p-3 text-[10px] text-faint">
        <div className="flex items-center justify-between"><span>Branch</span><span className="font-mono text-muted">feature/gui</span></div>
        <div className="mt-1.5 flex items-center justify-between"><span>Platform</span><span className="font-mono text-muted">{window.desktop?.platform ?? "browser"}</span></div>
      </div>
    </aside>
  );
}
