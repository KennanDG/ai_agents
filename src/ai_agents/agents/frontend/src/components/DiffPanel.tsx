import { DiffEditor } from "@monaco-editor/react";
import { Check, Columns2, FileCode2, MoreHorizontal, Undo2 } from "lucide-react";
import type { FileChange } from "../types";

interface DiffPanelProps {
  change: FileChange;
}

export function DiffPanel({ change }: DiffPanelProps) {
  return (
    <section className="flex min-h-0 min-w-0 flex-1 flex-col bg-canvas">
      <header className="flex h-12 shrink-0 items-center gap-2 border-b border-line bg-panel-soft px-3">
        <FileCode2 size={14} className="text-sky-300" />
        <span className="min-w-0 truncate font-mono text-[11px] text-ink-soft">{change.path}</span>
        <div className="ml-auto flex items-center gap-1.5">
          <span className="font-mono text-[10px] text-emerald-300">+{change.additions}</span>
          <span className="font-mono text-[10px] text-rose-300">−{change.deletions}</span>
          <button type="button" className="icon-button" aria-label="Side-by-side diff" title="Side-by-side diff"><Columns2 size={14} /></button>
          <button type="button" className="icon-button" aria-label="More actions" title="More actions"><MoreHorizontal size={15} /></button>
        </div>
      </header>

      <div className="min-h-0 flex-1">
        <DiffEditor
          key={change.path}
          original={change.original}
          modified={change.modified}
          language={change.language}
          theme="vs-dark"
          options={{
            automaticLayout: true,
            fontFamily: "JetBrains Mono, ui-monospace, SFMono-Regular, monospace",
            fontSize: 12,
            lineHeight: 20,
            minimap: { enabled: false },
            renderOverviewRuler: false,
            scrollBeyondLastLine: false,
            wordWrap: "on",
            padding: { top: 12, bottom: 12 },
            originalEditable: false,
          }}
        />
      </div>

      <footer className="flex h-12 shrink-0 items-center justify-between border-t border-line bg-panel-soft px-3">
        <p className="text-[10px] text-muted">Review this file before applying the patch.</p>
        <div className="flex gap-2">
          <button type="button" className="secondary-button"><Undo2 size={13} /> Reject</button>
          <button type="button" className="primary-button"><Check size={13} /> Accept file</button>
        </div>
      </footer>
    </section>
  );
}
