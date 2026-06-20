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

export interface AgentMessage {
  id: string;
  role: "user" | "agent";
  body: string;
  time: string;
}
