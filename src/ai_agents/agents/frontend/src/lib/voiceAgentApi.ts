import type { AgentMessage } from "../types";
import type { CodingAgentAttachedFile } from "./codingAgentSocket";

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
    promptText: string;
    attachedFiles: CodingAgentAttachedFile[];
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


const toVoiceAttachments = (attachedFiles: CodingAgentAttachedFile[]) => {
    let remainingContentChars = 60_000;

    return attachedFiles.slice(0, 5).map((file) => {
        const rawContent = file.data_url ? "" : (file.content ?? "");
        const maxChars = Math.min(20_000, Math.max(0, remainingContentChars));
        const content = rawContent.slice(0, maxChars);
        remainingContentChars -= content.length;

        return {
            name: file.name,
            source: file.source,
            path: file.path ?? null,
            mime_type: file.mime_type ?? null,
            size: file.size ?? null,
            content: content || null,
            has_image_data: Boolean(file.data_url),
            content_truncated: Boolean(file.truncated) || content.length < rawContent.length,
        };
    });
};


export const submitVoiceTurn = async ({
    apiBaseUrl,
    apiKey,
    audio,
    sessionId,
    history,
    promptText,
    attachedFiles,
    repoRoot,
    workspaceRoot,
    activePath,
    allowWrite,
}: SubmitVoiceTurnArgs): Promise<VoiceAgentTurnResponse> => {
  
    const form = new FormData();

    form.append("audio", audio, "voice-input.webm");
    form.append("history_json", JSON.stringify(toVoiceHistory(history)));
    form.append("prompt_text", promptText);
    form.append("attached_files_json", JSON.stringify(toVoiceAttachments(attachedFiles)));
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
