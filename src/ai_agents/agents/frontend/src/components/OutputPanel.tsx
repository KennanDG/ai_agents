import { CheckCircle2, ChevronDown, TerminalSquare } from "lucide-react";

export function OutputPanel() {
  return (
    <section className="flex h-40 shrink-0 flex-col border-t border-line bg-[#080a0e]">
      <div className="flex h-9 items-center gap-5 border-b border-line px-3">
        <button type="button" className="output-tab output-tab-active">Terminal</button>
        <button type="button" className="output-tab">Validation</button>
        <button type="button" className="output-tab">Problems <span className="text-faint">0</span></button>
        <button type="button" className="icon-button ml-auto" aria-label="Collapse output" title="Collapse output"><ChevronDown size={14} /></button>
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-3 font-mono text-[10px] leading-5 text-muted">
        <p className="flex items-center gap-2 text-faint"><TerminalSquare size={12} /> uv run pytest src/ai_agents/agents/coding/tests/test_validation.py</p>
        <p className="mt-1 text-ink-soft">============================= test session starts =============================</p>
        <p>collected 3 items</p>
        <p className="mt-1 flex items-center gap-2 text-emerald-300"><CheckCircle2 size={12} /> 3 passed in 0.42s</p>
      </div>
    </section>
  );
}
