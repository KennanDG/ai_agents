import { DiffEditor } from "@monaco-editor/react";
import {
  Check,
  Clipboard,
  Columns2,
  Download,
  FileCode2,
  MoreHorizontal,
  Undo2,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { FileChange, RepositoryFile } from "../types";

interface DiffPanelProps {
  file: RepositoryFile | null;
  change?: FileChange | null;
  isLoading?: boolean;
  error?: string | null;
  canApprove?: boolean;
  onAcceptFile?: (path: string) => void;
  onRejectChanges?: () => void;
}

const modelPath = (side: "repo" | "sandbox", path: string) => {
  const normalized = path.replace(/\\/g, "/").replace(/^\/+/, "");
  const encoded = normalized
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/");

  return `file:///${side}/${encoded || "untitled.txt"}`;
};

export const DiffPanel = ({
  file,
  change,
  isLoading = false,
  error,
  canApprove = false,
  onAcceptFile,
  onRejectChanges,
}: DiffPanelProps) => {
  const path = change?.path ?? file?.path ?? "No file selected";
  const language = change?.language ?? file?.language ?? "plaintext";
  const original = change?.original ?? file?.content ?? "";
  const modified = change?.modified ?? file?.content ?? "";
  const additions = change?.additions ?? 0;
  const deletions = change?.deletions ?? 0;
  const hasChange = Boolean(change);

  const [renderSideBySide, setRenderSideBySide] = useState(true);
  const [ignoreTrimWhitespace, setIgnoreTrimWhitespace] = useState(false);
  const [moreOpen, setMoreOpen] = useState(false);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  const originalModelPath = useMemo(() => modelPath("repo", path), [path]);
  const modifiedModelPath = useMemo(() => modelPath("sandbox", path), [path]);

  useEffect(() => {
    if (!moreOpen) return;

    const closeOnOutsideClick = (event: MouseEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) {
        setMoreOpen(false);
      }
    };

    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMoreOpen(false);
    };

    window.addEventListener("mousedown", closeOnOutsideClick);
    window.addEventListener("keydown", closeOnEscape);

    return () => {
      window.removeEventListener("mousedown", closeOnOutsideClick);
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [moreOpen]);

  useEffect(() => {
    setActionMessage(null);
    setMoreOpen(false);
  }, [path]);

  const copyText = async (value: string, successMessage: string) => {
    try {
      await navigator.clipboard.writeText(value);
      setActionMessage(successMessage);
    } catch {
      setActionMessage("Clipboard access was blocked by the browser.");
    } finally {
      setMoreOpen(false);
    }
  };

  const downloadModifiedFile = () => {
    const fileName = path.split(/[\\/]/).at(-1) ?? "modified-file.txt";
    const url = URL.createObjectURL(new Blob([modified], { type: "text/plain;charset=utf-8" }));
    const anchor = document.createElement("a");

    anchor.href = url;
    anchor.download = fileName;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);

    setActionMessage(`Downloaded ${fileName}.`);
    setMoreOpen(false);
  };

  return (
    <section className="flex min-h-0 min-w-0 flex-1 flex-col bg-canvas">
      <header className="relative flex h-12 shrink-0 items-center gap-2 border-b border-line bg-panel-soft px-3">
        <FileCode2 size={14} className="text-sky-300" />
        <span className="min-w-0 truncate font-mono text-[11px] text-ink-soft">{path}</span>
        <div className="ml-auto flex items-center gap-1.5">
          {hasChange ? (
            <>
              <span className="font-mono text-[10px] text-emerald-300">+{additions}</span>
              <span className="font-mono text-[10px] text-rose-300">−{deletions}</span>
            </>
          ) : null}

          <button
            type="button"
            className={`icon-button ${renderSideBySide ? "bg-selected text-accent-light" : ""}`}
            aria-label={renderSideBySide ? "Use inline diff" : "Use side-by-side diff"}
            aria-pressed={renderSideBySide}
            title={renderSideBySide ? "Switch to inline diff" : "Switch to side-by-side diff"}
            onClick={() => setRenderSideBySide((current) => !current)}
            disabled={!file && !change}
          >
            <Columns2 size={14} />
          </button>

          <div ref={menuRef} className="relative">
            <button
              type="button"
              className={`icon-button ${moreOpen ? "bg-selected text-ink-soft" : ""}`}
              aria-label="More diff actions"
              aria-expanded={moreOpen}
              title="More actions"
              onClick={() => setMoreOpen((current) => !current)}
              disabled={!file && !change}
            >
              <MoreHorizontal size={15} />
            </button>

            {moreOpen ? (
              <div className="absolute right-0 top-8 z-30 w-52 overflow-hidden rounded-md border border-line-strong bg-panel shadow-2xl">
                <button
                  type="button"
                  className="flex w-full items-center gap-2 px-3 py-2 text-left text-[11px] text-ink-soft hover:bg-hover"
                  onClick={() => void copyText(path, "Copied file path.")}
                >
                  <Clipboard size={13} /> Copy file path
                </button>
                <button
                  type="button"
                  className="flex w-full items-center gap-2 px-3 py-2 text-left text-[11px] text-ink-soft hover:bg-hover"
                  onClick={() => void copyText(modified, "Copied modified contents.")}
                >
                  <Clipboard size={13} /> Copy modified contents
                </button>
                <button
                  type="button"
                  className="flex w-full items-center gap-2 px-3 py-2 text-left text-[11px] text-ink-soft hover:bg-hover"
                  onClick={downloadModifiedFile}
                >
                  <Download size={13} /> Download modified file
                </button>
                <label className="flex cursor-pointer items-center gap-2 border-t border-line px-3 py-2 text-[11px] text-ink-soft hover:bg-hover">
                  <input
                    type="checkbox"
                    checked={ignoreTrimWhitespace}
                    onChange={(event) => setIgnoreTrimWhitespace(event.target.checked)}
                    className="accent-accent"
                  />
                  Ignore trim whitespace
                </label>
              </div>
            ) : null}
          </div>
        </div>
      </header>

      <div className="min-h-0 flex-1">
        {(() => {
          if (isLoading) {
            return <div className="grid h-full place-items-center text-xs text-muted">Loading file…</div>;
          }
          if (error) {
            return <div className="grid h-full place-items-center px-8 text-center text-xs leading-6 text-rose-300">{error}</div>;
          }
          if (file || change) {
            return (
              <DiffEditor
                key={`${path}:${hasChange ? "change" : "file"}`}
                original={original}
                modified={modified}
                language={language}
                originalLanguage={language}
                modifiedLanguage={language}
                originalModelPath={originalModelPath}
                modifiedModelPath={modifiedModelPath}
                theme="vs-dark"
                beforeMount={(monaco) => {
                  const typescript = monaco.languages.typescript;
                  const compilerOptions = {
                    allowJs: true,
                    allowNonTsExtensions: true,
                    esModuleInterop: true,
                    jsx: typescript.JsxEmit.ReactJSX,
                    module: typescript.ModuleKind.ESNext,
                    moduleResolution: typescript.ModuleResolutionKind.NodeJs,
                    noEmit: true,
                    skipLibCheck: true,
                    target: typescript.ScriptTarget.ES2022,
                  };
                  const diagnosticsOptions = {
                    noSemanticValidation: true,
                    noSyntaxValidation: false,
                    noSuggestionDiagnostics: true,
                  };

                  typescript.typescriptDefaults.setCompilerOptions(compilerOptions);
                  typescript.javascriptDefaults.setCompilerOptions(compilerOptions);
                  typescript.typescriptDefaults.setDiagnosticsOptions(diagnosticsOptions);
                  typescript.javascriptDefaults.setDiagnosticsOptions(diagnosticsOptions);
                  typescript.typescriptDefaults.setEagerModelSync(true);
                  typescript.javascriptDefaults.setEagerModelSync(true);
                }}
                options={{
                  automaticLayout: true,
                  enableSplitViewResizing: true,
                  fontFamily: "JetBrains Mono, ui-monospace, SFMono-Regular, monospace",
                  fontSize: 12,
                  ignoreTrimWhitespace,
                  lineHeight: 20,
                  minimap: { enabled: false },
                  originalEditable: false,
                  padding: { top: 12, bottom: 12 },
                  renderOverviewRuler: false,
                  renderSideBySide,
                  scrollBeyondLastLine: false,
                  wordWrap: "on",
                }}
              />
            );
          }
          return <div className="grid h-full place-items-center text-xs text-muted">Select a repository file to preview it.</div>;
        })()}
      </div>

      <footer className="flex h-12 shrink-0 items-center justify-between gap-3 border-t border-line bg-panel-soft px-3">
        <p className="min-w-0 truncate text-[10px] text-muted">
          {actionMessage ?? (hasChange
            ? "Review this file before applying the patch."
            : "Repository preview. Agent changes will appear here when a run produces diffs.")}
        </p>
        {hasChange ? (
          <div className="flex shrink-0 gap-2">
            <button
              type="button"
              className="secondary-button"
              disabled={!canApprove}
              onClick={onRejectChanges}
            >
              <Undo2 size={13} /> Reject
            </button>

            <button
              type="button"
              className="primary-button"
              disabled={!canApprove || !change?.path}
              onClick={() => change?.path && onAcceptFile?.(change.path)}
            >
              <Check size={13} /> Accept file
            </button>
          </div>
        ) : null}
      </footer>
    </section>
  );
};
