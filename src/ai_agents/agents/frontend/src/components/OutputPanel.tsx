import { AlertTriangle, CheckCircle2, ChevronDown, TerminalSquare } from "lucide-react";
import type { AgentRunState } from "../types";

interface OutputPanelProps {
  run: AgentRunState;
}

export const OutputPanel = ({ run }: OutputPanelProps) => {
  const problemCount = run.errors.length;

  return (
    <section className="flex h-44 shrink-0 flex-col border-t border-line bg-[#080a0e]">
      <div className="flex h-9 items-center gap-5 border-b border-line px-3">
        <button type="button" className="output-tab output-tab-active">Terminal</button>
        <button type="button" className="output-tab">Validation</button>
        <button type="button" className="output-tab">Problems <span className="text-faint">{problemCount}</span></button>
        <button type="button" className="icon-button ml-auto" aria-label="Collapse output" title="Collapse output"><ChevronDown size={14} /></button>
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-3 font-mono text-[10px] leading-5 text-muted">
        {run.logs.length === 0 ? (
          <p className="flex items-center gap-2 text-faint"><TerminalSquare size={12} /> Waiting for agent output…</p>
        ) : (
          run.logs.map((line, index) => <p key={`${line}:${index}`}>{line}</p>)
        )}

        {run.validationCommands.map((command) => (
          <p key={command} className="mt-2 flex items-center gap-2 text-faint"><TerminalSquare size={12} /> {command}</p>
        ))}

        {run.validationResults.map((result, index) => {
          const passed = Boolean(result.passed ?? result.success);
          return (
            <p key={index} className={`mt-1 flex items-center gap-2 ${passed ? "text-emerald-300" : "text-rose-300"}`}>
              {passed ? <CheckCircle2 size={12} /> : <AlertTriangle size={12} />}
              {JSON.stringify(result)}
            </p>
          );
        })}

        {run.diffs.length > 0 ? (
          <pre className="mt-3 whitespace-pre-wrap border-t border-line pt-3 text-[10px] leading-5 text-ink-soft">{run.diffs.join("\n\n")}</pre>
        ) : null}
      </div>
    </section>
  );
}
