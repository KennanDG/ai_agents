export type ChangeStatus = "modified" | "added" | "deleted";

export interface FileChange {
  path: string;
  status: ChangeStatus;
  additions: number;
  deletions: number;
  language: string;
  original: string;
  modified: string;
}

export interface RepositoryTreeEntry {
  path: string;
  name: string;
  kind: "file" | "directory";
  depth: number;
  size?: number | null;
}

export interface RepositoryTreeResponse {
  repo_root: string;
  entries: RepositoryTreeEntry[];
}

export interface RepositoryFile {
  repo_root: string;
  path: string;
  language: string;
  content: string;
  size: number;
}

export interface AgentMessage {
  id: string;
  role: "user" | "agent";
  body: string;
  time: string;
}

export type AgentRunStatus =
  | "disconnected"
  | "connecting"
  | "ready"
  | "running"
  | "completed"
  | "failed";

export interface AgentRunState {
  status: AgentRunStatus;
  runId?: string | null;
  threadId?: string | null;
  selectedSkill?: string | null;
  routeConfidence?: number | null;
  routeReason?: string | null;
  plan: string[];
  completedNodes: string[];
  filesInspected: string[];
  patchSummary?: string | null;
  fileChanges: FileChange[];
  diffs: string[];
  validationCommands: string[];
  validationResults: Record<string, unknown>[];
  report?: string | null;
  errors: string[];
  logs: string[];
}

declare global {
  interface Window {
    desktop?: {
      platform: string;
    };
  }
}
