import { ArrowUp, Check, Circle, Paperclip, ShieldCheck, Sparkles } from "lucide-react";
import type { AgentMessage } from "../types";

interface TaskPanelProps {
  messages: AgentMessage[];
}

const plan = [
  { label: "Inspect validation flow", done: true },
  { label: "Update state and node", done: true },
  { label: "Add regression test", done: true },
  { label: "Awaiting approval", done: false },
];

export function TaskPanel({ messages }: TaskPanelProps) {
  return (
    <section className="flex min-h-0 w-90 shrink-0 flex-col border-r border-line bg-panel-soft">
      <div className="flex h-12 items-center gap-2 border-b border-line px-4">
        <Sparkles size={15} className="text-accent-light" />
        <h1 className="text-xs font-semibold text-ink">Agent session</h1>
        <span className="ml-auto rounded-full border border-emerald-500/20 bg-emerald-500/8 px-2 py-0.5 text-[9px] font-medium text-emerald-300">READY</span>
      </div>

      <div className="min-h-0 flex-1 overflow-auto px-4 py-4">
        <div className="mb-5 rounded-lg border border-line bg-surface p-3">
          <div className="mb-2.5 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider text-muted">
            <ShieldCheck size={13} /> Plan
          </div>
          <ol className="space-y-2">
            {plan.map((step) => (
              <li key={step.label} className="flex items-center gap-2 text-[11px] text-muted">
                {step.done ? <Check size={12} className="text-emerald-300" /> : <Circle size={10} className="text-accent-light" />}
                <span>{step.label}</span>
              </li>
            ))}
          </ol>
        </div>

        <div className="space-y-4">
          {messages.map((message) => (
            <article key={message.id} className={message.role === "user" ? "message-user" : "message-agent"}>
              <div className="mb-1.5 flex items-center gap-2 text-[9px] font-semibold uppercase tracking-wider text-faint">
                <span>{message.role === "user" ? "You" : "Agent"}</span>
                <span>·</span>
                <time>{message.time}</time>
              </div>
              <p className="text-xs leading-5 text-ink-soft">{message.body}</p>
            </article>
          ))}
        </div>
      </div>

      <form className="border-t border-line p-3" onSubmit={(event) => event.preventDefault()}>
        <div className="rounded-lg border border-line-strong bg-surface p-2.5 focus-within:border-accent/70 focus-within:ring-1 focus-within:ring-accent/20">
          <label htmlFor="agent-prompt" className="sr-only">Message the coding agent</label>
          <textarea id="agent-prompt" rows={3} placeholder="Ask the agent to change your code…" className="w-full resize-none bg-transparent text-xs leading-5 text-ink outline-none placeholder:text-faint" />
          <div className="mt-2 flex items-center">
            <button type="button" className="icon-button" aria-label="Attach context" title="Attach context"><Paperclip size={14} /></button>
            <span className="ml-1 text-[9px] text-faint">Write mode</span>
            <button type="submit" className="ml-auto grid size-7 place-items-center rounded-md bg-accent text-white shadow-glow hover:bg-accent-light" aria-label="Send prompt">
              <ArrowUp size={15} />
            </button>
          </div>
        </div>
      </form>
    </section>
  );
}
