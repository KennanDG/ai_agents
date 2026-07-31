import {
  Bot,
  FileCode2,
  GitBranch,
  Search,
  Settings,
  Sparkles,
} from "lucide-react";

export type ActivityView = "explorer" | "source-control" | "skills";
export type ActivityAction = ActivityView | "agent" | "search";

type ActivityBarProps = {
  activeView: ActivityView;
  agentSettingsOpen?: boolean;
  onSelect: (action: ActivityAction) => void;
};

const primaryActions = [
  { action: "explorer" as const, label: "Explorer", icon: FileCode2 },
  { action: "agent" as const, label: "Agent configuration", icon: Bot },
  { action: "search" as const, label: "Search", icon: Search, disabled: true },
  { action: "source-control" as const, label: "Source control", icon: GitBranch },
  { action: "skills" as const, label: "Skills", icon: Sparkles },
];

export const ActivityBar = ({
  activeView,
  agentSettingsOpen = false,
  onSelect,
}: ActivityBarProps) => {
  return (
    <nav
      className="flex w-13 shrink-0 flex-col items-center border-r border-line bg-panel py-2"
      aria-label="Primary"
    >
      <div className="mb-3 grid size-8 place-items-center rounded-lg bg-accent text-ink shadow-glow">
        <span className="font-mono text-sm font-bold">A</span>
      </div>

      <div className="flex flex-1 flex-col gap-1">
        {primaryActions.map(({ action, label, icon: Icon, disabled }) => {
          const active = action === activeView || (action === "agent" && agentSettingsOpen);

          return (
            <button
              key={action}
              type="button"
              title={disabled ? `${label} is not configured yet` : label}
              aria-label={label}
              aria-current={active ? "page" : undefined}
              aria-disabled={disabled || undefined}
              disabled={disabled}
              onClick={() => onSelect(action)}
              className={`activity-button ${active ? "activity-button-active" : ""} disabled:cursor-not-allowed disabled:opacity-35`}
            >
              <Icon size={19} strokeWidth={1.7} />
            </button>
          );
        })}
      </div>

      <button
        type="button"
        title="Agent configuration"
        aria-label="Agent configuration"
        onClick={() => onSelect("agent")}
        className={`activity-button ${agentSettingsOpen ? "activity-button-active" : ""}`}
      >
        <Settings size={19} strokeWidth={1.7} />
      </button>
    </nav>
  );
}
