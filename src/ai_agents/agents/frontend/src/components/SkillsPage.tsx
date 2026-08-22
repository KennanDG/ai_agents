import { type ChangeEvent, type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import {
  CircleAlert,
  FilePlus2,
  LoaderCircle,
  RefreshCcw,
  Save,
  Sparkles,
  Trash2,
  Upload,
  Wrench,
} from "lucide-react";
import {
  deleteSkill,
  draftSkill,
  fetchSkills,
  fetchTools,
  quarantineTool,
  saveSkill,
  type AgentKind,
  type SkillSummary,
  type ToolSummary,
} from "../lib/adminApi";

type SkillsPageProps = {
  apiBaseUrl: string;
  apiKey: string;
};

const newSkillTemplate = (name = "custom_skill") => `# Skill: ${name.replaceAll("_", " ")}

Purpose: Describe the narrow purpose of this skill.

Use when:
- Describe when the router should select this skill.

Allowed tools:

Steps:
1. Inspect the relevant context.
2. Follow the repository's existing patterns.
3. Make the smallest safe change.
4. Validate the result.

Rules:
- Do not expose secrets.
- Avoid unrelated changes.
`;


const validateCanonicalSkill = (value: string): string | null => {
  const normalized = value.replace(/\r\n/g, "\n").trim();
  if (!/^#\s+Skill:\s+.+/im.test(normalized)) return "Skill must start with '# Skill: <name>'.";
  if (!/^Purpose:\s+\S+/im.test(normalized)) return "Skill must include a non-empty Purpose line.";

  const required = ["Use when", "Allowed tools", "Steps", "Rules"];
  for (const section of required) {
    const matcher = new RegExp(`^(?:#{1,6}\\s+)?${section}:?\\s*$`, "im");
    if (!matcher.test(normalized)) return `Skill is missing the '${section}' section.`;
  }
  return null;
};

const FieldLabel = ({ children }: { children: ReactNode }) => (
  <label className="mb-1 block text-xs font-semibold uppercase tracking-wider text-muted">
    {children}
  </label>
);

export const SkillsPage = ({ apiBaseUrl, apiKey }: SkillsPageProps) => {
  const [agent, setAgent] = useState<AgentKind>("coding");
  const [skills, setSkills] = useState<SkillSummary[]>([]);
  const [tools, setTools] = useState<ToolSummary[]>([]);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [name, setName] = useState("custom_skill");
  const [content, setContent] = useState(newSkillTemplate());
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [draftPrompt, setDraftPrompt] = useState("");
  const [drafting, setDrafting] = useState(false);
  const [toolPurpose, setToolPurpose] = useState("");
  const [toolName, setToolName] = useState("");
  const [toolSource, setToolSource] = useState("");
  const skillFileRef = useRef<HTMLInputElement | null>(null);
  const toolFileRef = useRef<HTMLInputElement | null>(null);

  const selectedSkill = useMemo(
    () => skills.find((item) => item.name === selectedName) ?? null,
    [selectedName, skills],
  );

  const load = async (targetAgent = agent) => {
    setLoading(true);
    setError(null);
    try {
      const [skillResults, toolResults] = await Promise.all([
        fetchSkills({ apiBaseUrl, apiKey, agent: targetAgent }),
        fetchTools({ apiBaseUrl, apiKey, agent: targetAgent }),
      ]);
      setSkills(skillResults);
      setTools(toolResults);
      const nextSelected = skillResults.find((item) => item.name === selectedName) ?? skillResults[0] ?? null;
      if (nextSelected) {
        setSelectedName(nextSelected.name);
        setName(nextSelected.name);
        setContent(nextSelected.content);
      } else {
        setSelectedName(null);
        setName("custom_skill");
        setContent(newSkillTemplate());
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Failed to load skills.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load(agent);
    // selectedName intentionally omitted so changing selection does not reload the page.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agent, apiBaseUrl, apiKey]);

  const selectSkill = (skill: SkillSummary) => {
    setSelectedName(skill.name);
    setName(skill.name);
    setContent(skill.content);
    setError(null);
    setMessage(null);
  };

  const startNewSkill = () => {
    setSelectedName(null);
    setName("custom_skill");
    setContent(newSkillTemplate());
    setMessage(null);
    setError(null);
  };

  const readTextFile = async (event: ChangeEvent<HTMLInputElement>, kind: "skill" | "tool") => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    const text = await file.text();
    const baseName = file.name.replace(/\.(md|py)$/i, "").replace(/[^a-zA-Z0-9_-]+/g, "_");
    if (kind === "skill") {
      const normalizedName = (baseName || "custom_skill").toLowerCase();
      const suggestedName = normalizedName.startsWith("custom_") ? normalizedName : `custom_${normalizedName}`;
      setSelectedName(null);
      setDrafting(true);
      setError(null);
      setMessage(null);
      try {
        const draft = await draftSkill({
          apiBaseUrl,
          apiKey,
          agent,
          suggestedName,
          sourceMarkdown: text,
          prompt: (
            "Translate this imported Markdown into the canonical skill format while " +
            "preserving its intent, workflow, constraints, and safe executable tool dependencies."
          ),
        });
        setName(draft.name);
        setContent(draft.content);
        setMessage(
          draft.warnings.length
            ? `Imported and normalized '${file.name}'. ${draft.warnings.join(" ")}`
            : `Imported and normalized '${file.name}' into the canonical skill format.`,
        );
      } catch (reason) {
        setName(suggestedName);
        setContent(text);
        setError(
          reason instanceof Error
            ? `Could not normalize the imported skill: ${reason.message}`
            : "Could not normalize the imported skill.",
        );
      } finally {
        setDrafting(false);
      }
    } else {
      setToolName(baseName || "custom_tool");
      setToolSource(text);
    }
  };

  const persistSkill = async () => {
    if (selectedSkill && !selectedSkill.custom) {
      setError("Built-in skills are read-only in the UI. Create a custom_ skill instead.");
      return;
    }
    const formatError = validateCanonicalSkill(content);

    if (formatError) {
      setError(formatError);
      return;
    }
    
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      const saved = await saveSkill({
        apiBaseUrl,
        apiKey,
        agent,
        name: name.trim(),
        content,
        overwrite: Boolean(selectedSkill),
      });
      await load(agent);
      setSelectedName(saved.name);
      setName(saved.name);
      setContent(saved.content);
      setMessage(`Saved ${agent} skill '${saved.name}'. New runs can load it immediately.`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Failed to save skill.");
    } finally {
      setSaving(false);
    }
  };

  const generateFromPrompt = async () => {
    const prompt = draftPrompt.trim();
    if (!prompt) return;

    setDrafting(true);
    setError(null);
    setMessage(null);
    try {
      const draft = await draftSkill({
        apiBaseUrl,
        apiKey,
        agent,
        prompt,
        suggestedName: name.trim() || "custom_skill",
      });
      setSelectedName(null);
      setName(draft.name);
      setContent(draft.content);
      setMessage(
        draft.warnings.length
          ? `Generated '${draft.name}'. ${draft.warnings.join(" ")}`
          : `Generated '${draft.name}'. Review it, then save when ready.`,
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Failed to generate skill.");
    } finally {
      setDrafting(false);
    }
  };

  const removeSkill = async () => {
    if (!selectedSkill?.custom) return;
    setSaving(true);
    setError(null);
    try {
      await deleteSkill({ apiBaseUrl, apiKey, agent, name: selectedSkill.name });
      setMessage(`Deleted custom skill '${selectedSkill.name}'.`);
      await load(agent);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Failed to delete skill.");
    } finally {
      setSaving(false);
    }
  };

  const submitToolForReview = async () => {
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      const result = await quarantineTool({
        apiBaseUrl,
        apiKey,
        agent,
        name: toolName.trim(),
        purpose: toolPurpose.trim(),
        source: toolSource,
      });
      setTools((current) => [result, ...current.filter((item) => item.name !== result.name)]);
      setToolName("");
      setToolPurpose("");
      setToolSource("");
      setMessage(
        `Uploaded '${result.name}' to the review queue. It is not executable until a developer promotes and wires it into the agent graph.`,
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Failed to upload tool.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="flex min-h-0 min-w-0 flex-1 flex-col bg-canvas">
      <header className="flex h-12 shrink-0 items-center justify-between border-b border-line px-5">
        <div className="flex items-center gap-2">
          <Sparkles size={16} className="text-accent-light" />
          <h1 className="text-sm font-semibold text-ink">Skills and tools</h1>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={agent}
            onChange={(event) => setAgent(event.target.value as AgentKind)}
            className="rounded-md border border-line bg-surface px-3 py-1.5 text-xs text-ink outline-none focus:border-accent/70"
          >
            <option value="coding">Coding agent</option>
            <option value="voice">Voice agent</option>
          </select>
          <button type="button" className="secondary-button" onClick={() => void load()} disabled={loading}>
            {loading ? <LoaderCircle size={12} className="animate-spin" /> : <RefreshCcw size={12} />}
            Refresh
          </button>
        </div>
      </header>

      <div className="grid min-h-0 flex-1 lg:grid-cols-[260px_minmax(0,1fr)_340px]">
        <aside className="min-h-0 overflow-auto border-r border-line bg-panel-soft p-3">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-[10px] font-semibold uppercase tracking-wider text-muted">
              {agent} skills · {skills.length}
            </h2>
            <button type="button" className="icon-button" title="New skill" onClick={startNewSkill}>
              <FilePlus2 size={14} />
            </button>
          </div>

          <input
            ref={skillFileRef}
            type="file"
            accept=".md"
            className="hidden"
            onChange={(event) => void readTextFile(event, "skill")}
          />
          <button
            type="button"
            className="secondary-button mb-3 w-full justify-center"
            onClick={() => skillFileRef.current?.click()}
            disabled={drafting}
          >
            {drafting ? <LoaderCircle size={12} className="animate-spin" /> : <Upload size={12} />}
            Import Markdown
          </button>

          <div className="space-y-1">
            {skills.map((skill) => (
              <button
                key={skill.name}
                type="button"
                onClick={() => selectSkill(skill)}
                className={`w-full rounded-md border px-3 py-2 text-left ${
                  selectedName === skill.name
                    ? "border-accent/50 bg-selected"
                    : "border-transparent hover:border-line hover:bg-hover"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className={`truncate font-mono text-ink-soft ${selectedName === skill.name ? "text-xs" : "text-[11px]"}`}>{skill.name}</span>
                  {skill.custom ? (
                    <span className="rounded bg-accent/10 px-1.5 py-0.5 text-[8px] uppercase text-accent-light">
                      custom
                    </span>
                  ) : null}
                </div>
                <p className={`mt-1 line-clamp-2 leading-4 text-faint ${selectedName === skill.name ? "text-xs" : "text-[9px]"}`}>{skill.purpose}</p>
              </button>
            ))}
          </div>
        </aside>

        <main className="min-h-0 overflow-auto p-4">
          <div className="mx-auto max-w-3xl">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold text-ink">Skill playbook</h2>
                <p className="mt-1 text-[10px] leading-4 text-muted w-fit">
                  Coding skills are routed by the coding registry on every run. Voice skills are injected into the voice intake context by the companion backend patch.
                </p>
              </div>
              <div className="flex gap-2">
                {selectedSkill?.custom ? (
                  <button type="button" className="secondary-button" onClick={removeSkill} disabled={saving}>
                    <Trash2 size={12} />
                    Delete
                  </button>
                ) : null}
                <button
                  type="button"
                  className="primary-button size-auto w-28"
                  onClick={persistSkill}
                  disabled={saving || drafting || Boolean(selectedSkill && !selectedSkill.custom)}
                  title={selectedSkill && !selectedSkill.custom ? "Built-in skills are read-only" : "Save skill"}
                >
                  {saving ? <LoaderCircle size={12} className="animate-spin" /> : <Save size={12} />}
                  Save skill
                </button>
              </div>
            </div>

            <div className="mt-4 rounded-md border border-line bg-panel-soft p-3">
              <div className="flex items-center gap-2">
                <Sparkles size={13} className="text-accent-light" />
                <div>
                  <h3 className="text-xs font-semibold text-ink">Generate skill with AI</h3>
                  <p className="mt-0.5 text-[9px] leading-4 text-muted">
                    The backend generates canonical Markdown using only executable tools registered for the selected agent.
                  </p>
                </div>
              </div>
              <textarea
                value={draftPrompt}
                onChange={(event) => setDraftPrompt(event.target.value)}
                rows={4}
                placeholder="Example: Create a skill for reviewing FastAPI endpoint changes, checking schemas, auth, error handling, and targeted tests."
                className="mt-3 w-full resize-y rounded-md border border-line bg-surface px-3 py-2 text-xs leading-5 text-ink outline-none focus:border-accent/70"
              />
              <button
                type="button"
                className="secondary-button mt-2 justify-center"
                disabled={drafting || !draftPrompt.trim()}
                onClick={generateFromPrompt}
              >
                {drafting ? <LoaderCircle size={12} className="animate-spin" /> : <Sparkles size={12} />}
                Generate draft
              </button>
            </div>

            <div className="mt-4">
              <FieldLabel>Registry name</FieldLabel>
              <input
                value={name}
                disabled={Boolean(selectedSkill)}
                onChange={(event) => setName(event.target.value)}
                className="w-full rounded-md border border-line bg-surface px-3 py-2 font-mono text-xs text-ink outline-none focus:border-accent/70 disabled:opacity-60"
                placeholder="custom_skill"
              />
            </div>
            <div className="mt-3">
              <FieldLabel>Markdown instructions</FieldLabel>
              <p className="mb-2 text-[9px] leading-4 text-muted">
                Required format: # Skill, Purpose, Use when, Allowed tools, Steps, and Rules. Unknown or pending-review tools are rejected on save.
              </p>
              <textarea
                value={content}
                onChange={(event) => setContent(event.target.value)}
                disabled={Boolean(selectedSkill && !selectedSkill.custom)}
                spellCheck={false}
                className="min-h-140 w-full resize-y rounded-md border border-line bg-panel px-3 py-3 font-mono text-[11px] leading-5 text-ink-soft outline-none focus:border-accent/70 disabled:cursor-not-allowed disabled:opacity-65"
              />
            </div>

            {message ? (
              <div className="mt-3 rounded-md border border-emerald-500/20 bg-emerald-500/8 p-3 text-[11px] leading-5 text-emerald-300">
                {message}
              </div>
            ) : null}
            {error ? (
              <div className="mt-3 flex items-start gap-2 rounded-md border border-rose-500/20 bg-rose-500/8 p-3 text-[11px] leading-5 text-rose-300">
                <CircleAlert size={14} className="mt-0.5 shrink-0" />
                {error}
              </div>
            ) : null}
          </div>
        </main>

        <aside className="min-h-0 overflow-auto border-l border-line bg-panel-soft p-4">
          <div className="flex items-center gap-2">
            <Wrench size={14} className="text-accent-light" />
            <div>
              <h2 className="text-xs font-semibold text-ink">Tool catalog</h2>
              <p className="mt-0.5 text-[9px] text-muted">Built-ins plus quarantined uploads</p>
            </div>
          </div>

          <div className="mt-3 max-h-56 space-y-1 overflow-auto">
            {tools.map((tool) => (
              <div key={`${tool.status}:${tool.module}:${tool.name}`} className="rounded-md border border-line bg-panel p-2.5">
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate font-mono text-xs text-ink-soft">{tool.name}</span>
                  <span className={`text-[8px] uppercase ${tool.status === "builtin" ? "text-emerald-300" : "text-amber-300"}`}>
                    {tool.status.replace("_", " ")}
                  </span>
                </div>
                <p className="mt-1 text-xs leading-5 text-faint">{tool.purpose || tool.module}</p>
              </div>
            ))}
          </div>

          <div className="mt-5 border-t border-line pt-4">
            <h3 className="text-xs font-semibold text-ink">Upload tool for review</h3>
            <p className="mt-1 text-[9px] leading-4 text-muted">
              Arbitrary Python is not imported into the API process. Uploads remain quarantined until code review and explicit graph integration.
            </p>

            <input
              ref={toolFileRef}
              type="file"
              accept=".py"
              className="hidden"
              onChange={(event) => void readTextFile(event, "tool")}
            />
            <button
              type="button"
              className="secondary-button mt-3 w-full justify-center"
              onClick={() => toolFileRef.current?.click()}
            >
              <Upload size={12} />
              Select Python file
            </button>

            <div className="mt-3 space-y-3">
              <div>
                <FieldLabel>Tool name</FieldLabel>
                <input
                  value={toolName}
                  onChange={(event) => setToolName(event.target.value)}
                  className="w-full rounded-md border border-line bg-surface px-3 py-2 font-mono text-xs text-ink outline-none focus:border-accent/70"
                />
              </div>
              <div>
                <FieldLabel>Purpose</FieldLabel>
                <input
                  value={toolPurpose}
                  onChange={(event) => setToolPurpose(event.target.value)}
                  className="w-full rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-accent/70"
                />
              </div>
              <div>
                <FieldLabel>Python source</FieldLabel>
                <textarea
                  value={toolSource}
                  onChange={(event) => setToolSource(event.target.value)}
                  rows={10}
                  spellCheck={false}
                  className="w-full resize-y rounded-md border border-line bg-panel px-3 py-2 font-mono text-[10px] leading-4 text-ink-soft outline-none focus:border-accent/70"
                />
              </div>
              <button
                type="button"
                className="secondary-button h-8 w-full justify-center"
                disabled={saving || !toolName.trim() || !toolPurpose.trim() || !toolSource.trim()}
                onClick={submitToolForReview}
              >
                {saving ? <LoaderCircle size={12} className="animate-spin" /> : <Upload size={12} />}
                Submit for review
              </button>
            </div>
          </div>
        </aside>
      </div>
    </section>
  );
}
