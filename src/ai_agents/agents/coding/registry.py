from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SKILL_NAME = "implement_change"
_CUSTOM_SKILL_PREFIX = "custom_"
_SECTION_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")


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


def _extract_allowed_tools(markdown: str) -> tuple[str, ...]:
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




def route_skill(
    user_request: str,
    available_skills: Iterable[str] | None = None,
    *,
    default_skill: str = DEFAULT_SKILL_NAME,
) -> str:
    """Deterministic fallback router used when LLM routing is unavailable."""

    text = user_request.lower()
    available = _normalize_available_skills(available_skills)

    keyword_routes: tuple[tuple[tuple[str, ...], str], ...] = (
        (("traceback", "stack trace", "error", "exception", "failing", "bug", "fix"), "debug"),
        (("test", "pytest", "unit test", "regression", "coverage"), "tests"),
        (("search the web", "google", "bing", "online search", "look up online"), "web_search"),
        (("gmail", "gmail access", "email", "send email", "email draft", "gmail api"), "gmail_access"),
    )

    for terms, skill_name in keyword_routes:
        if any(term in text for term in terms):
            selected = _choose_existing_skill(
                skill_name,
                available_skills=available,
                default_skill=default_skill,
            )
            if selected:
                return selected

    return (
        _choose_existing_skill(
            default_skill,
            available_skills=available,
            default_skill=default_skill,
        )
        or default_skill
    )


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
                allowed_tools=_extract_allowed_tools(instructions),
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
        return "\n".join(f"- {skill.name}: {skill.purpose}" for skill in self.list())

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
