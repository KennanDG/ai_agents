from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SKILL_NAME = "implement_change"
MAX_SELECTED_SKILLS = 3
_CUSTOM_SKILL_PREFIX = "custom_"
_SECTION_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


@dataclass(frozen=True)
class Skill:
    name: str
    purpose: str
    instructions: str
    path: Path
    allowed_tools: tuple[str, ...] = ()
    custom: bool = False


def _extract_purpose(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.lower().startswith("purpose:"):
            return line.split(":", 1)[1].strip()
    return "No purpose declared."


def extract_allowed_tools(markdown: str) -> tuple[str, ...]:
    """Read a Markdown ``Allowed tools`` list without executing the skill file."""

    lines = markdown.splitlines()
    in_section = False
    tools: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        heading = _SECTION_RE.match(line)
        normalized = line.lower().rstrip(":")

        if heading:
            in_section = heading.group(1).strip().lower().rstrip(":") == "allowed tools"
            continue

        if normalized == "allowed tools":
            in_section = True
            continue

        if in_section and line and not line.startswith(("-", "*")):
            # A new prose block starts; stop rather than accidentally treating steps
            # or rules as tool declarations.
            break

        if in_section and line.startswith(("-", "*")):
            name = line[1:].strip().strip("`")
            if name and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}", name):
                tools.append(name)

    return tuple(dict.fromkeys(tools))


def _normalize_available_skills(available_skills: Iterable[str] | None) -> set[str]:
    return {name.strip() for name in available_skills or [] if name.strip()}




def _choose_existing_skill(
    candidate: str,
    *,
    available_skills: set[str],
    default_skill: str = DEFAULT_SKILL_NAME,
) -> str | None:
    
    if not available_skills or candidate in available_skills:
        return candidate

    if default_skill in available_skills:
        return default_skill

    return next(iter(sorted(available_skills)), None)




def route_skills(
    user_request: str,
    available_skills: Iterable[str] | None = None,
    *,
    default_skill: str = DEFAULT_SKILL_NAME,
    max_skills: int = MAX_SELECTED_SKILLS,
) -> list[str]:
    """Deterministic multi-skill fallback router.

    The deterministic router intentionally stays conservative. It can combine
    complementary built-ins for obvious mixed requests and honors an explicitly
    named custom skill. The LLM router remains responsible for semantic matching
    of arbitrary custom skills.
    """

    text = user_request.lower()
    available = _normalize_available_skills(available_skills)
    selected: list[str] = []

    # If the user explicitly names a skill, prefer it first. Match both the registry
    # name and a space-separated form so ``custom_api_review`` can be requested as
    # "custom api review".
    for skill_name in sorted(available):
        if skill_name.lower() in text or skill_name.lower().replace("_", " ") in text:
            selected.append(skill_name)

    keyword_routes: tuple[tuple[tuple[str, ...], str], ...] = (
        (("traceback", "stack trace", "error", "exception", "failing", "bug", "fix"), "debug"),
        (("test", "pytest", "unit test", "regression", "coverage"), "tests"),
        (("react", "tsx", "component", "frontend component"), "frontend_component"),
        (("tailwind", "css", "styling", "responsive", "layout", "frontend styling"), "frontend_styling"),
        (("repository", "repo structure", "inspect repo", "find files", "navigate repo"), "repo"),
        (("search the web", "google", "bing", "online search", "look up online"), "web_search"),
        (("gmail", "gmail access", "email", "send email", "email draft", "gmail api"), "gmail_access"),
    )

    for terms, skill_name in keyword_routes:
        if skill_name in available and any(term in text for term in terms):
            selected.append(skill_name)

    selected = list(dict.fromkeys(selected))
    if selected:
        return selected[:max(1, max_skills)]

    fallback = _choose_existing_skill(
        default_skill,
        available_skills=available,
        default_skill=default_skill,
    )
    return [fallback or default_skill]


def route_skill(
    user_request: str,
    available_skills: Iterable[str] | None = None,
    *,
    default_skill: str = DEFAULT_SKILL_NAME,
) -> str:
    """Backward-compatible single-skill fallback router."""

    return route_skills(
        user_request,
        available_skills,
        default_skill=default_skill,
        max_skills=1,
    )[0]


class SkillRegistry:
    """Load Markdown skill playbooks from one agent-specific directory.

    The registry reads from disk on every ``load`` call, which means skills created
    through the admin endpoint become available to the next agent run without a
    process restart. Skill files are treated as data only and are never executed.
    """

    def __init__(self, skills_dir: Path | None = None) -> None:
        self.skills_dir = skills_dir or Path(__file__).parent / "skills"
        self._skills: dict[str, Skill] = {}

    def load(self) -> SkillRegistry:
        self._skills.clear()

        if not self.skills_dir.exists():
            return self

        for path in sorted(self.skills_dir.glob("*.md")):
            try:
                instructions = path.read_text(encoding="utf-8")
            except OSError:
                continue

            self._skills[path.stem] = Skill(
                name=path.stem,
                purpose=_extract_purpose(instructions),
                instructions=instructions,
                path=path,
                allowed_tools=extract_allowed_tools(instructions),
                custom=path.stem.startswith(_CUSTOM_SKILL_PREFIX),
            )

        return self

    def get(self, name: str) -> Skill:
        if not self._skills:
            self.load()

        try:
            return self._skills[name]
        except KeyError as exc:
            available = ", ".join(self.list_names()) or "none"
            raise KeyError(f"Unknown skill '{name}'. Available skills: {available}") from exc

    def get_many(self, names: Iterable[str]) -> list[Skill]:
        if not self._skills:
            self.load()

        output: list[Skill] = []
        for name in dict.fromkeys(item.strip() for item in names if item.strip()):
            if name in self._skills:
                output.append(self._skills[name])
        return output

    def has(self, name: str) -> bool:
        if not self._skills:
            self.load()
        return name in self._skills

    def default_skill_name(self) -> str:
        if not self._skills:
            self.load()

        if DEFAULT_SKILL_NAME in self._skills:
            return DEFAULT_SKILL_NAME

        names = self.list_names()
        if names:
            return names[0]

        raise ValueError(f"No skills found in {self.skills_dir}")

    def router_catalog(self) -> str:
        if not self._skills:
            self.load()
        return "\n".join(
            f"- {skill.name}: {skill.purpose}"
            + (f" [tools: {', '.join(skill.allowed_tools)}]" if skill.allowed_tools else "")
            for skill in self.list()
        )

    def combined_instructions(
        self,
        names: Iterable[str],
        *,
        max_chars: int = 18_000,
    ) -> str:
        """Combine selected playbooks in priority order with a hard prompt budget."""

        skills = self.get_many(names)
        blocks: list[str] = []
        used = 0
        for index, skill in enumerate(skills):
            role = "primary" if index == 0 else "supplemental"
            block = f"## Skill {index + 1} ({role}): {skill.name}\n{skill.instructions.strip()}"
            remaining = max_chars - used
            if remaining <= 0:
                break
            if len(block) > remaining:
                block = block[:remaining] + "\n...[skill instructions truncated]"
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)

    def allowed_tools_for(self, names: Iterable[str]) -> list[str]:
        return list(
            dict.fromkeys(
                tool
                for skill in self.get_many(names)
                for tool in skill.allowed_tools
            )
        )


    def rank_for_request(
        self,
        user_request: str,
        *,
        max_skills: int = MAX_SELECTED_SKILLS,
    ) -> list[str]:
        """Rank available skills using deterministic routes plus purpose/name overlap."""

        if not self._skills:
            self.load()

        base = route_skills(
            user_request,
            self.list_names(),
            default_skill=self.default_skill_name(),
            max_skills=max_skills,
        )
        request_tokens = {token.lower() for token in _TOKEN_RE.findall(user_request)}
        stopwords = {
            "the", "and", "for", "with", "this", "that", "from", "into", "use",
            "using", "want", "please", "help", "agent", "coding", "skill", "skills",
        }
        request_tokens -= stopwords

        scored: list[tuple[float, str]] = []
        for skill in self.list():
            skill_tokens = {
                token.lower()
                for token in _TOKEN_RE.findall(
                    f"{skill.name.replace('_', ' ')} {skill.purpose}"
                )
            } - stopwords
            overlap = request_tokens & skill_tokens
            score = float(len(overlap))
            if skill.custom and overlap:
                score += 0.25
            if skill.name.lower() in user_request.lower():
                score += 10.0
            if score > 0:
                scored.append((score, skill.name))

        ranked = [name for _, name in sorted(scored, key=lambda item: (-item[0], item[1]))]
        combined = list(dict.fromkeys([*base, *ranked]))

        # Do not let the generic default crowd out more specific matches. Keep it
        # only when it is the sole route or was explicitly named.
        default = self.default_skill_name()
        if len(combined) > 1 and default in combined and default.lower() not in user_request.lower():
            combined = [name for name in combined if name != default]

        return (combined or [default])[:max(1, max_skills)]

    def prompt_context(self, *, max_skills: int = 12, max_chars: int = 18_000) -> str:
        """Return bounded playbook text suitable for a voice/intake prompt."""

        if not self._skills:
            self.load()

        blocks: list[str] = []
        used = 0
        # Custom skills first so user-authored behavior is not displaced by built-ins.
        ordered = sorted(self.list(), key=lambda skill: (not skill.custom, skill.name))
        for skill in ordered[:max_skills]:
            block = f"## {skill.name}\n{skill.instructions.strip()}"
            remaining = max_chars - used
            if remaining <= 0:
                break
            if len(block) > remaining:
                block = block[:remaining] + "\n...[skill context truncated]"
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)

    def list_names(self) -> list[str]:
        if not self._skills:
            self.load()
        return sorted(self._skills)

    def list(self) -> list[Skill]:
        if not self._skills:
            self.load()
        return [self._skills[name] for name in self.list_names()]
