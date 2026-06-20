import {
  Bot,
  FileCode2,
  GitBranch,
  Search,
  Settings,
  Sparkles,
} from "lucide-react";

const primaryActions = [
  { label: "Agent", icon: Bot, active: true },
  { label: "Explorer", icon: FileCode2 },
  { label: "Search", icon: Search },
  { label: "Source control", icon: GitBranch },
  { label: "Skills", icon: Sparkles },
];

export function ActivityBar() {
  return (
    <nav className="flex w-13 shrink-0 flex-col items-center border-r border-line bg-panel py-2" aria-label="Primary">
      <div className="mb-3 grid size-8 place-items-center rounded-lg bg-accent text-ink shadow-glow">
        <span className="font-mono text-sm font-bold">A</span>
      </div>

      <div className="flex flex-1 flex-col gap-1">
        {primaryActions.map(({ label, icon: Icon, active }) => (
          <button
            key={label}
            type="button"
            title={label}
            aria-label={label}
            aria-current={active ? "page" : undefined}
            className={`activity-button ${active ? "activity-button-active" : ""}`}
          >
            <Icon size={19} strokeWidth={1.7} />
          </button>
        ))}
      </div>

      <button type="button" title="Settings" aria-label="Settings" className="activity-button">
        <Settings size={19} strokeWidth={1.7} />
      </button>
    </nav>
  );
}
