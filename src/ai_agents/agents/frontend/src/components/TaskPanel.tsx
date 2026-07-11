import { ArrowUp, Check, Circle, Mic, Paperclip, ShieldCheck, Sparkles, Square } from "lucide-react";
import { type ChangeEvent, type ClipboardEvent, type DragEvent, type SubmitEvent, useRef, useState } from "react";
import type { AgentMessage, AgentRunState, RepositoryFile } from "../types";
import type { CodingAgentAttachedFile } from "../lib/codingAgentSocket";

interface TaskPanelProps {
  messages: AgentMessage[];
  run: AgentRunState;
  onSubmit: (prompt: string, attachedFiles: CodingAgentAttachedFile[]) => void;
  onVoiceAudio?: (audio: Blob) => Promise<void> | void;
  voiceReplyUrl?: string | null;
  onApproveAll: () => void;
  onRejectChanges: () => void;
  allowWrite: boolean;
  activePath?: string | null;
  activeFile?: RepositoryFile | null;
}




/*
 =============   Helpers  =============
*/

const statusClass = {
  disconnected: "border-rose-500/20 bg-rose-500/8 text-rose-300",
  connecting: "border-amber-500/20 bg-amber-500/8 text-amber-300",
  ready: "border-emerald-500/20 bg-emerald-500/8 text-emerald-300",
  running: "border-accent/20 bg-accent/8 text-accent-light",
  completed: "border-emerald-500/20 bg-emerald-500/8 text-emerald-300",
  approval_pending: "border-accent/20 bg-accent/8 text-accent-light",
  applied: "border-emerald-500/20 bg-emerald-500/8 text-emerald-300",
  rejected: "border-rose-500/20 bg-rose-500/8 text-rose-300",
  failed: "border-rose-500/20 bg-rose-500/8 text-rose-300",
};


const MAX_TEXT_ATTACHMENT_BYTES = 1_000_000;
const MAX_IMAGE_ATTACHMENT_BYTES = 5_000_000;

const IMAGE_MIME_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);
const TEXT_FILE_EXTENSIONS = /\.(c|cpp|cc|cxx|c\+\+|h|hpp|hh|hxx|css|csv|html|java|js|jsx|json|md|py|rs|sql|toml|ts|tsx|txt|xml|ya?ml)$/i;


const isSupportedImage = (file: File) => {
  return IMAGE_MIME_TYPES.has(file.type) || /\.(png|jpe?g|webp)$/i.test(file.name);
};


const isSupportedText = (file: File) => {
  return file.type.startsWith("text/") || TEXT_FILE_EXTENSIONS.test(file.name);
};


const readAsText = (file: File) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (event) => resolve(typeof event.target?.result === "string" ? event.target.result : "");
    reader.onerror = () => reject(new Error(`Failed to read ${file.name}.`));
    reader.readAsText(file);
  });


const readAsDataUrl = (file: File) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (event) => resolve(typeof event.target?.result === "string" ? event.target.result : "");
    reader.onerror = () => reject(new Error(`Failed to read ${file.name}.`));
    reader.readAsDataURL(file);
  });



/*
   =============  Component  =============
*/
export const TaskPanel = ({ messages, run, onSubmit, onVoiceAudio, voiceReplyUrl, onApproveAll, onRejectChanges, allowWrite, activePath, activeFile }: TaskPanelProps) => {
  const [prompt, setPrompt] = useState("");
  const [attachedFiles, setAttachedFiles] = useState<CodingAgentAttachedFile[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [attachmentError, setAttachmentError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isRunning = run.status === "running";
  const dragCounter = useRef(0);
  const [isRecording, setIsRecording] = useState(false);
  const [voiceError, setVoiceError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);




  const attachActiveRepoFile = () => {
    if (!activePath) {
      setAttachmentError("Open a repository file before attaching it as repo context.");
      return;
    }

    setAttachmentError(null);

    setAttachedFiles((prev) => {
      if (prev.some((file) => file.source === "repo" && file.path === activePath)) {
        return prev;
      }

      return [
        ...prev,
        {
          name: activePath.split("/").at(-1) ?? activePath,
          path: activePath,
          source: "repo",
          content: null,
          data_url: null,
          mime_type: null,
          size: activeFile?.size ?? null,
        },
      ];
    });
  };




  const processFiles = (files: File[]) => {
    if (!files || files.length === 0) return;

    setAttachmentError(null);

    Promise.all(
      files.map(async (file): Promise<CodingAgentAttachedFile | null> => {
        try {
          if (isSupportedImage(file)) {
            if (file.size > MAX_IMAGE_ATTACHMENT_BYTES) {
              setAttachmentError(`${file.name} is too large. Image uploads are limited to 5 MB.`);
              return null;
            }

            const dataUrl = await readAsDataUrl(file);

            if (!dataUrl.startsWith("data:image/")) {
              setAttachmentError(`${file.name} could not be read as an image.`);
              return null;
            }

            return {
              name: file.name,
              content: null,
              data_url: dataUrl,
              source: "upload",
              mime_type: file.type || null,
              size: file.size,
            };
          }


          if (!isSupportedText(file)) {
            setAttachmentError(`${file.name} is not a supported text or image attachment.`);
            return null;
          }


          if (file.size > MAX_TEXT_ATTACHMENT_BYTES) {
            setAttachmentError(`${file.name} is too large. Text uploads are limited to 1 MB.`);
            return null;
          }

          const content = await readAsText(file);

          if (!content.trim()) {
            setAttachmentError(`${file.name} is empty or could not be read as text.`);
            return null;
          }

          return {
            name: file.name,
            content,
            data_url: null,
            source: "upload",
            mime_type: file.type || null,
            size: file.size,
          };
        } catch (error) {
          setAttachmentError(error instanceof Error ? error.message : `Failed to read ${file.name}.`);
          return null;
        }
      })
    ).then((loadedFiles) => {
      const validFiles = loadedFiles.filter((file): file is CodingAgentAttachedFile => file !== null);

      if (validFiles.length > 0) {
        setAttachedFiles((prev) => [...prev, ...validFiles]);
      }
    });
  };




  const handleSubmit = (event: SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();

    const trimmed = prompt.trim();
    if (!trimmed || isRunning) return;

    onSubmit(trimmed, attachedFiles);

    setPrompt("");
    setAttachedFiles([]);
  };



  const handleAttachClick = () => {
    fileInputRef.current?.click();
  };
  
  

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    processFiles(Array.from(files));
    event.target.value = '';
  };



  const handleDragEnter = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current += 1;
    if (e.dataTransfer?.types?.includes('Files')) {
      setIsDragOver(true);
    }
  };



  const handleDragLeave = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current -= 1;
    if (dragCounter.current === 0) {
      setIsDragOver(false);
    }
  };


  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };


  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    dragCounter.current = 0;
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      processFiles(Array.from(files));
    }
  };

  const handlePaste = (e: ClipboardEvent<HTMLTextAreaElement>) => {
    const items = e.clipboardData?.items;
    if (!items || items.length === 0) return;

    const imageItems: DataTransferItem[] = [];
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.startsWith("image/")) {
        imageItems.push(items[i]);
      }
    }

    if (imageItems.length > 0) {
      e.preventDefault();
      e.stopPropagation();

      const files = imageItems
        .map(item => item.getAsFile())
        .filter((file): file is File => file !== null);

      if (files.length > 0) {
        processFiles(files);
      }
    }
  };



  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
  };

  const startRecording = async () => {
    if (!onVoiceAudio || isRunning) return;

    setVoiceError(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setVoiceError("Microphone access is not available in this browser.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      audioChunksRef.current = [];

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "";

      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        setIsRecording(false);

        mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;

        const audio = new Blob(audioChunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });

        audioChunksRef.current = [];

        if (audio.size > 0) {
          await onVoiceAudio(audio);
        }
      };

      recorder.start();
      setIsRecording(true);
    } catch (error) {
      setIsRecording(false);
      setVoiceError(error instanceof Error ? error.message : "Could not access microphone.");
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
      return;
    }

    void startRecording();
  };



  return (
    <section className="flex min-h-0 flex-1 flex-col bg-panel-soft">

      <div className="flex h-12 shrink-0 items-center gap-2 border-b border-line px-4">
        <Sparkles size={15} className="text-accent-light" />
        <h1 className="text-xs font-semibold text-ink">Agent session</h1>
        <span className={`ml-auto rounded-full border px-2 py-0.5 text-[9px] font-medium uppercase ${statusClass[run.status]}`}>{run.status}</span>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">

        <div className="mb-5 rounded-lg border border-line bg-surface p-3">
          <div className="mb-2.5 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider text-muted">
            <ShieldCheck size={13} /> Plan
          </div>
          {run.plan.length > 0 ? (
            <ol className="space-y-2">
              {run.plan.map((step, index) => {
                const done = run.completedNodes.length > index || run.status === "completed";
                return (
                  <li key={`${step}:${index}`} className="flex items-center gap-2 text-[11px] text-muted">
                    {done ? <Check size={12} className="text-emerald-300" /> : <Circle size={10} className="text-accent-light" />}
                    <span>{step}</span>
                  </li>
                );
              })}
            </ol>
          ) : (
            <p className="text-[11px] leading-5 text-faint">The plan will stream in after the routing and planning nodes run.</p>
          )}

          {run.selectedSkill ? (
            <p className="mt-3 border-t border-line pt-3 text-[10px] leading-5 text-faint">
              Skill: <span className="font-mono text-muted">{run.selectedSkill}</span>
              {run.routeConfidence != null ? <span> · confidence {Math.round(run.routeConfidence * 100)}%</span> : null}
            </p>
          ) : null}
        </div>

        <div className="space-y-4">
          {run.approvalStatus === "pending" ? (
            <div className="mb-4 rounded-lg border border-amber-400/30 bg-amber-400/10 p-3">
              <p className="text-[11px] leading-5 text-amber-100">
                The agent produced file changes. Validation results are available, but final repo write is waiting for your approval.
                {run.blockingValidationFailed ? " Blocking validation failed, so review carefully before applying." : ""}
                {run.advisoryValidationFailed ? " Advisory validation warnings were reported." : ""}
              </p>

              <div className="mt-3 flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  className="inline-flex items-center justify-center whitespace-nowrap rounded-md bg-accent px-4 py-2 text-xs font-medium text-white shadow-glow transition-colors hover:bg-accent-light focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
                  onClick={onApproveAll}
                >
                  Apply all changes
                </button>
                <button
                  type="button"
                  className="inline-flex items-center justify-center whitespace-nowrap rounded-md border border-rose-500/40 bg-surface px-4 py-2 text-xs font-medium text-ink transition-colors hover:border-rose-500/60 hover:bg-rose-500/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500/50"
                  onClick={onRejectChanges}
                >
                  Reject changes
                </button>
              </div>
            </div>
          ) : null}

          {messages.map((message) => (
            <article key={message.id} className={message.role === "user" ? "message-user" : "message-agent"}>
              <div className="mb-1.5 flex items-center gap-2 text-[9px] font-semibold uppercase tracking-wider text-faint">
                <span>{message.role === "user" ? "You" : "Agent"}</span>
                <span>·</span>
                <time>{message.time}</time>
              </div>
              <p className="whitespace-pre-wrap wrap-break-word text-xs leading-5 text-ink-soft">
                {message.body}
              </p>
            </article>
          ))}

          {voiceReplyUrl ? (
            <div className="rounded-lg border border-line bg-surface p-2">
              <p className="mb-1 text-[9px] font-semibold uppercase tracking-wider text-faint">Voice reply</p>
              <audio
                key={voiceReplyUrl}
                src={voiceReplyUrl}
                controls
                autoPlay
                preload="auto"
                className="h-8 w-full"
                aria-label="Play the latest voice-agent reply"
              />
            </div>
          ) : null}
          
        </div>

      </div>


      <form className="shrink-0 border-t border-line p-3" onSubmit={handleSubmit}>
        
        <input
          type="file"
          ref={fileInputRef}
          className="sr-only"
          multiple
          accept=".c,.c++,.cc,.cpp,.cxx,.h,.hh,.hpp,.hxx,.java,.rs,.css,.csv,.html,.js,.jsx,.json,.md,.py,.sql,.toml,.ts,.tsx,.txt,.xml,.yaml,.yml,image/png,image/jpeg,image/webp"
          onChange={handleFileChange}
        />

        <div
          onDragOver={handleDragOver}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`relative ${isDragOver ? "ring-2 ring-accent/30" : ""}`}
        >

          <div className="rounded-lg border border-line-strong bg-surface p-2.5 focus-within:border-accent/70 focus-within:ring-1 focus-within:ring-accent/20">
            
            <label htmlFor="agent-prompt" className="sr-only">Message the coding agent</label>
            <textarea
              id="agent-prompt"
              rows={3}
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  const form = event.currentTarget.closest('form');
                  if (form) form.requestSubmit();
                }
              }}
              onPaste={handlePaste}
              placeholder="Ask the agent to change your code…"
              className="w-full resize-none bg-transparent text-xs leading-5 text-ink outline-none placeholder:text-faint"
            />

            {attachedFiles.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1">
                {attachedFiles.map((file, idx) => (
                  <span key={idx} className="inline-flex items-center gap-1 rounded bg-line px-1.5 py-0.5 text-[10px] text-muted">
                    {file.data_url ? "Image: " : ""}{file.name}
                    <button
                      type="button"
                      className="text-faint hover:text-ink"
                      onClick={() => setAttachedFiles(prev => prev.filter((_, i) => i !== idx))}
                      aria-label="Remove file"
                    >
                      &times;
                    </button>
                  </span>
                ))}
              </div>
            )}
            {attachmentError ? (
              <p className="mt-2 text-[10px] leading-4 text-rose-300">{attachmentError}</p>
            ) : null}
            {voiceError ? (
              <p className="mt-2 text-[10px] leading-4 text-rose-300">{voiceError}</p>
            ) : null}
            <div className="mt-2 flex items-center">
              <button
                type="button"
                className="icon-button"
                aria-label="Attach local file"
                title="Attach local file"
                onClick={handleAttachClick}
              >
                <Paperclip size={14} />
              </button>

              <button
                type="button"
                className={`ml-1 icon-button ${isRecording ? "text-rose-300" : ""}`}
                aria-label={isRecording ? "Stop voice recording" : "Start voice recording"}
                title={isRecording ? "Stop voice recording" : "Start voice recording"}
                onClick={toggleRecording}
                disabled={isRunning}
              >
                {isRecording ? <Square size={13} /> : <Mic size={14} />}
              </button>

              <button
                type="button"
                className="ml-1 rounded-md border border-line px-2 py-1 text-[10px] text-muted hover:border-accent/60 hover:text-ink"
                aria-label="Attach open repository file"
                title="Attach open repository file"
                onClick={attachActiveRepoFile}
              >
                Attach open file
              </button>

              <span className="ml-1 text-[9px] text-faint">{isRunning ? "Agent running" : allowWrite ? "Write mode" : "Read mode"}</span>
              
              <button
                type="submit"
                disabled={!prompt.trim() || isRunning}
                className="ml-auto grid size-7 place-items-center rounded-md bg-accent text-white shadow-glow hover:bg-accent-light disabled:cursor-not-allowed disabled:opacity-50"
                aria-label="Send prompt"
              >
                <ArrowUp size={15} />
              </button>
              
            </div>

          </div>

          {isDragOver && (
            <div className="absolute inset-0 z-10 flex items-center justify-center rounded-lg border-2 border-dashed border-accent/60 bg-accent/10 pointer-events-none">
              <span className="text-[11px] font-semibold text-accent-light">Drop files to attach</span>
            </div>
          )}

        </div>
      </form>
    </section>
  );
}
