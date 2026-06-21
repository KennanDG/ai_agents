import { DiffEditor } from "@monaco-editor/react";
import { Check, Columns2, FileCode2, MoreHorizontal, Undo2 } from "lucide-react";
import type { FileChange, RepositoryFile } from "../types";


interface DiffPanelProps {
  file: RepositoryFile | null;
  change?: FileChange | null;
  isLoading?: boolean;
  error?: string | null;
}



export const DiffPanel = ({ file, change, isLoading = false, error }: DiffPanelProps) => {
  const path = change?.path ?? file?.path ?? "No file selected";
  const language = change?.language ?? file?.language ?? "plaintext";
  const original = change?.original ?? file?.content ?? "";
  const modified = change?.modified ?? file?.content ?? "";
  const additions = change?.additions ?? 0;
  const deletions = change?.deletions ?? 0;
  const hasChange = Boolean(change);

  return (
    <section className="flex min-h-0 min-w-0 flex-1 flex-col bg-canvas">
      <header className="flex h-12 shrink-0 items-center gap-2 border-b border-line bg-panel-soft px-3">
        <FileCode2 size={14} className="text-sky-300" />
        <span className="min-w-0 truncate font-mono text-[11px] text-ink-soft">{path}</span>
        <div className="ml-auto flex items-center gap-1.5">
          {hasChange ? (
            <>
              <span className="font-mono text-[10px] text-emerald-300">+{additions}</span>
              <span className="font-mono text-[10px] text-rose-300">−{deletions}</span>
            </>
          ) : null}
          <button type="button" className="icon-button" aria-label="Side-by-side diff" title="Side-by-side diff"><Columns2 size={14} /></button>
          <button type="button" className="icon-button" aria-label="More actions" title="More actions"><MoreHorizontal size={15} /></button>
        </div>
      </header>

      <div className="min-h-0 flex-1">
        {isLoading ? (
          <div className="grid h-full place-items-center text-xs text-muted">Loading file…</div>
        ) : error ? (
          <div className="grid h-full place-items-center px-8 text-center text-xs leading-6 text-rose-300">{error}</div>
        ) : file || change ? (
          <DiffEditor
            key={`${path}:${hasChange ? "change" : "file"}`}
            original={original}
            modified={modified}
            language={language}
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
        ) : (
          <div className="grid h-full place-items-center text-xs text-muted">Select a repository file to preview it.</div>
        )}
      </div>

      <footer className="flex h-12 shrink-0 items-center justify-between border-t border-line bg-panel-soft px-3">
        <p className="text-[10px] text-muted">
          {hasChange ? "Review this file before applying the patch." : "Repository preview. Agent changes will appear here when a run produces diffs."}
        </p>
        {hasChange ? (
          <div className="flex gap-2">
            <button type="button" className="secondary-button"><Undo2 size={13} /> Reject</button>
            <button type="button" className="primary-button"><Check size={13} /> Accept file</button>
          </div>
        ) : null}
      </footer>
    </section>
  );
}

