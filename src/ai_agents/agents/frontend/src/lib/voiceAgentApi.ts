import type { AgentMessage } from "../types";

export type VoiceAgentTurnResponse = {
    session_id: string;
    transcript: string;
    reply_text: string;
    status: "clarifying" | "ready" | "error";
    coding_request?: string | null;
    audio_mime_type?: string | null;
    audio_base64?: string | null;
    errors: string[];
};


type SubmitVoiceTurnArgs = {
    apiBaseUrl: string;
    apiKey: string;
    audio: Blob;
    sessionId?: string | null;
    history: AgentMessage[];
    repoRoot: string;
    workspaceRoot?: string | null;
    activePath?: string | null;
    allowWrite: boolean;
};


const toVoiceHistory = (messages: AgentMessage[]) => {
    return messages.slice(-12).map((message) => ({
        role: message.role === "agent" ? "assistant" : "user",
        content: message.body,
    }));
};


export const submitVoiceTurn = async ({
    apiBaseUrl,
    apiKey,
    audio,
    sessionId,
    history,
    repoRoot,
    workspaceRoot,
    activePath,
    allowWrite,
}: SubmitVoiceTurnArgs): Promise<VoiceAgentTurnResponse> => {
  
    const form = new FormData();

    form.append("audio", audio, "voice-input.webm");
    form.append("history_json", JSON.stringify(toVoiceHistory(history)));
    form.append("repo_root", repoRoot);
    form.append("allow_write", String(allowWrite));

    if (sessionId) form.append("session_id", sessionId);
    if (workspaceRoot) form.append("workspace_root", workspaceRoot);
    if (activePath) form.append("active_path", activePath);

    const response = await fetch(`${apiBaseUrl}/voice-agent/turn`, {
        method: "POST",
        headers: apiKey ? { "x-api-key": apiKey } : undefined,
        body: form,
    });

    if (!response.ok) {
        const detail = await response.text();
        throw new Error(`Voice agent failed: ${response.status} ${detail}`);
    }

    return response.json() as Promise<VoiceAgentTurnResponse>;
};