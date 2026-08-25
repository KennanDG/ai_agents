import Editor from "@monaco-editor/react";
import { LoaderCircle, Save, X } from "lucide-react";
import { useEffect, useState } from "react";

type CodeEditorModalProps = {
  open: boolean;
  title: string;
  initialContent: string;
  onCancel: () => void;
  onSave: (content: string) => Promise<void>;
};

export const CodeEditorModal = ({
  open,
  title,
  initialContent,
  onCancel,
  onSave,
}: CodeEditorModalProps) => {
  const [content, setContent] = useState(initialContent);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setContent(initialContent);
    setError(null);
  }, [initialContent, open]);

  useEffect(() => {
    if (!open) return;

    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !saving) onCancel();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onCancel, open, saving]);

  if (!open) return null;

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      await onSave(content);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Failed to save tool file.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" role="presentation">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="code-editor-modal-title"
        className="flex h-[min(80vh,720px)] w-full max-w-4xl flex-col overflow-hidden rounded-lg border border-line bg-panel shadow-2xl"
      >
        <header className="flex shrink-0 items-center justify-between border-b border-line px-4 py-3">
          <h2 id="code-editor-modal-title" className="text-sm font-semibold text-ink">
            {title}
          </h2>
          <button
            type="button"
            className="icon-button"
            aria-label="Close code editor"
            disabled={saving}
            onClick={onCancel}
          >
            <X size={14} />
          </button>
        </header>

        <div className="min-h-0 flex-1">
          <Editor
            height="100%"
            language="python"
            theme="vs-dark"
            value={content}
            onChange={(value) => setContent(value ?? "")}
            options={{
              automaticLayout: true,
              minimap: { enabled: false },
              scrollBeyondLastLine: false,
              tabSize: 4,
            }}
          />
        </div>

        <footer className="shrink-0 border-t border-line px-4 py-3">
          {error ? (
            <p role="alert" className="mb-3 text-[11px] text-rose-300">
              {error}
            </p>
          ) : null}
          <div className="flex justify-end gap-2">
            <button type="button" className="secondary-button" disabled={saving} onClick={onCancel}>
              Cancel
            </button>
            <button
              type="button"
              className="primary-button"
              disabled={saving || !content.trim()}
              onClick={() => void save()}
            >
              {saving ? <LoaderCircle size={12} className="animate-spin" /> : <Save size={12} />}
              Save
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
};
