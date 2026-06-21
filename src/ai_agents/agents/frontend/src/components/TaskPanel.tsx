import { ArrowUp, Check, Circle, Paperclip, ShieldCheck, Sparkles } from "lucide-react";
import { type ChangeEvent, type SubmitEvent, useRef, useState } from "react";
import type { AgentMessage, AgentRunState } from "../types";

interface TaskPanelProps {
  messages: AgentMessage[];
  run: AgentRunState;
  onSubmit: (prompt: string) => void;
  allowWrite: boolean;
}

const statusClass = {
  disconnected: "border-rose-500/20 bg-rose-500/8 text-rose-300",
  connecting: "border-amber-500/20 bg-amber-500/8 text-amber-300",
  ready: "border-emerald-500/20 bg-emerald-500/8 text-emerald-300",
  running: "border-accent/20 bg-accent/8 text-accent-light",
  completed: "border-emerald-500/20 bg-emerald-500/8 text-emerald-300",
  failed: "border-rose-500/20 bg-rose-500/8 text-rose-300",
};

export const TaskPanel = ({ messages, run, onSubmit, allowWrite }: TaskPanelProps) => {
  const [prompt, setPrompt] = useState("");
  const [attachedFiles, setAttachedFiles] = useState<{ name: string; content: string }[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isRunning = run.status === "running";

  const handleSubmit = (event: SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = prompt.trim();
    if (!trimmed || isRunning) return;

    let finalPrompt = trimmed;
    if (attachedFiles.length > 0) {
      const fileContents = attachedFiles.map(f => `--- ${f.name} ---\n${f.content}`).join('\n');
      finalPrompt += '\n\nAttached files:\n' + fileContents;
    }

    onSubmit(finalPrompt);
    setPrompt("");
    setAttachedFiles([]);
  }

  const handleAttachClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const fileList: File[] = [];
    for (let i = 0; i < files.length; i++) {
      fileList.push(files[i]);
    }

    Promise.all(
      fileList.map(
        (file) =>
          new Promise<{ name: string; content: string }>((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => {
              resolve({ name: file.name, content: (e.target?.result as string) || '' });
            };
            reader.onerror = () => {
              resolve({ name: file.name, content: '' });
            };
            reader.readAsText(file);
          })
      )
    ).then((loadedFiles) => {
      const validFiles = loadedFiles.filter((f) => f.content !== '');
      if (validFiles.length > 0) {
        setAttachedFiles((prev) => [...prev, ...validFiles]);
      }
    });

    event.target.value = '';
  };

  return (
    <section className="flex min-h-0 flex-1 flex-col bg-panel-soft">
      <div className="flex h-12 shrink-0 items-center gap-2 border-b border-line px-4">
        <Sparkles size={15} className="text-accent-light" />
        <h1 className="text-xs font-semibold text-ink">Agent session</h1>
        <span className={`ml-auto rounded-full border px-2 py-0.5 text-[9px] font-medium uppercase ${statusClass[run.status]}`}>{run.status}</span>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <div className="mb-5 rounded-lg border border-line bg-surface p-3">
          <div className="mb-2.5 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider text-muted">
            <ShieldCheck size={13} /> Plan
          </div>
          {run.plan.length > 0 ? (
            <ol className="space-y-2">
              {run.plan.map((step, index) => {
                const done = run.completedNodes.length > index || run.status === "completed";
                return (
                  <li key={`${step}:${index}`} className="flex items-center gap-2 text-[11px] text-muted">
                    {done ? <Check size={12} className="text-emerald-300" /> : <Circle size={10} className="text-accent-light" />}
                    <span>{step}</span>
                  </li>
                );
              })}
            </ol>
          ) : (
            <p className="text-[11px] leading-5 text-faint">The plan will stream in after the routing and planning nodes run.</p>
          )}

          {run.selectedSkill ? (
            <p className="mt-3 border-t border-line pt-3 text-[10px] leading-5 text-faint">
              Skill: <span className="font-mono text-muted">{run.selectedSkill}</span>
              {run.routeConfidence != null ? <span> · confidence {Math.round(run.routeConfidence * 100)}%</span> : null}
            </p>
          ) : null}
        </div>

        <div className="space-y-4">
          {messages.map((message) => (
            <article key={message.id} className={message.role === "user" ? "message-user" : "message-agent"}>
              <div className="mb-1.5 flex items-center gap-2 text-[9px] font-semibold uppercase tracking-wider text-faint">
                <span>{message.role === "user" ? "You" : "Agent"}</span>
                <span>·</span>
                <time>{message.time}</time>
              </div>
              <p className="whitespace-pre-wrap wrap-break-word text-xs leading-5 text-ink-soft">
                {message.body}
              </p>
            </article>
          ))}
        </div>
      </div>

      <form className="shrink-0 border-t border-line p-3" onSubmit={handleSubmit}>
        <input
          type="file"
          ref={fileInputRef}
          className="sr-only"
          multiple
          onChange={handleFileChange}
        />
        <div className="rounded-lg border border-line-strong bg-surface p-2.5 focus-within:border-accent/70 focus-within:ring-1 focus-within:ring-accent/20">
          <label htmlFor="agent-prompt" className="sr-only">Message the coding agent</label>
          <textarea
            id="agent-prompt"
            rows={3}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Ask the agent to change your code…"
            className="w-full resize-none bg-transparent text-xs leading-5 text-ink outline-none placeholder:text-faint"
          />
          {attachedFiles.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {attachedFiles.map((file, idx) => (
                <span key={idx} className="inline-flex items-center gap-1 rounded bg-line px-1.5 py-0.5 text-[10px] text-muted">
                  {file.name}
                  <button
                    type="button"
                    className="text-faint hover:text-ink"
                    onClick={() => setAttachedFiles(prev => prev.filter((_, i) => i !== idx))}
                    aria-label="Remove file"
                  >
                    &times;
                  </button>
                </span>
              ))}
            </div>
          )}
          <div className="mt-2 flex items-center">
            <button type="button" className="icon-button" aria-label="Attach context" title="Attach context" onClick={handleAttachClick}><Paperclip size={14} /></button>
            <span className="ml-1 text-[9px] text-faint">{isRunning ? "Agent running" : allowWrite ? "Write mode" : "Read mode"}</span>
            <button
              type="submit"
              disabled={!prompt.trim() || isRunning}
              className="ml-auto grid size-7 place-items-center rounded-md bg-accent text-white shadow-glow hover:bg-accent-light disabled:cursor-not-allowed disabled:opacity-50"
              aria-label="Send prompt"
            >
              <ArrowUp size={15} />
            </button>
          </div>
        </div>
      </form>
    </section>
  );
}
