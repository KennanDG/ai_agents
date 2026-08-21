import { AlertTriangle, CheckCircle2, ChevronDown, ChevronUp, TerminalSquare } from "lucide-react";
import { type MouseEvent as ReactMouseEvent, useState } from "react";
import type { AgentRunState } from "../types";

interface OutputPanelProps {
  run: AgentRunState;
}

export const OutputPanel = ({ run }: OutputPanelProps) => {
  const [collapsed, setCollapsed] = useState(false);
  const [height, setHeight] = useState(176);
  const [activeTab, setActiveTab] = useState<"terminal" | "validation" | "problems">("terminal");
  const problemCount = run.errors.length;

  const startResize = (event: ReactMouseEvent) => {
    event.preventDefault();
    const startY = event.clientY;
    const startHeight = height;
    const onMove = (move: MouseEvent) => {
      const next = startHeight - (move.clientY - startY);
      setHeight(Math.min(Math.max(next, 80), 600));
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  return (
    <section style={{ height: collapsed ? 36 : height }} className="flex shrink-0 flex-col border-t border-line bg-[#080a0e]">
      {!collapsed && (
        <div
          role="separator"
          aria-orientation="horizontal"
          aria-label="Resize output panel"
          className="h-1 w-full shrink-0 cursor-row-resize hover:bg-accent/60"
          onMouseDown={startResize}
        />
      )}
      <div className="flex h-9 items-center gap-5 border-b border-line px-3">
        <button type="button" className={`output-tab ${activeTab === "terminal" ? "output-tab-active" : ""}`} onClick={() => setActiveTab("terminal")}>Terminal</button>
        <button type="button" className={`output-tab ${activeTab === "validation" ? "output-tab-active" : ""}`} onClick={() => setActiveTab("validation")}>Validation</button>
        <button type="button" className={`output-tab ${activeTab === "problems" ? "output-tab-active" : ""}`} onClick={() => setActiveTab("problems")}>Problems <span className="text-faint">{problemCount}</span></button>
        <button type="button" className="icon-button ml-auto" aria-label={collapsed ? "Expand output" : "Collapse output"} title={collapsed ? "Expand output" : "Collapse output"} onClick={() => setCollapsed(!collapsed)}>
  {collapsed ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
</button>
      </div>
      {!collapsed && (

        <div className="min-h-0 flex-1 overflow-auto p-3 font-mono text-[10px] leading-5 text-muted">
          {activeTab === "terminal" && (
            <>
              {run.logs.length === 0 ? (
                <p className="flex items-center gap-2 text-faint"><TerminalSquare size={12} /> Waiting for agent output…</p>
              ) : (
                run.logs.map((line, index) => <p key={`${line}:${index}`}>{line}</p>)
              )}
              {run.diffs.length > 0 && (
                <pre className="mt-3 whitespace-pre-wrap border-t border-line pt-3 text-[10px] leading-5 text-ink-soft">{run.diffs.join("\n\n")}</pre>
              )}
            </>
          )}
          {activeTab === "validation" && (
            <>
              {run.validationCommands.length === 0 && run.validationResults.length === 0 ? (
                <p className="flex items-center gap-2 text-faint"><TerminalSquare size={12} /> No validation results yet.</p>
              ) : (
                <>
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
                </>
              )}
            </>
          )}
          {activeTab === "problems" && (
            <>
              {problemCount === 0 ? (
                <p className="flex items-center gap-2 text-faint"><AlertTriangle size={12} /> No problems detected.</p>
              ) : (
                <>
                  {run.fileChanges.map((change) => (
                    <p key={change.path} className="mt-1 flex items-center gap-2 text-ink-soft">
                      <AlertTriangle size={12} className={change.status === "deleted" ? "text-rose-300" : "text-amber-300"} />
                      <span className="font-mono">{change.path}</span>
                      <span className="text-faint">({change.status})</span>
                    </p>
                  ))}
                  {run.errors.map((error, index) => (
                    <p key={index} className="mt-1 flex items-center gap-2 text-rose-300">
                      <AlertTriangle size={12} />
                      {error}
                    </p>
                  ))}
                </>
              )}
            </>
          )}
        </div>
      )}

    </section>
  );
}
