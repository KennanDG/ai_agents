from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SKILL_NAME = "implement_change"


@dataclass(frozen=True)
class Skill:
    name: str
    purpose: str
    instructions: str
    path: Path




def _extract_purpose(markdown: str) -> str:

    for line in markdown.splitlines():
        if line.lower().startswith("purpose:"):
            return line.split(":", 1)[1].strip()
        
    return "No purpose declared."



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
    """Deterministic fallback router used when LLM routing is unavailable.

    The main routing node can use an LLM for nuanced decisions, but this fallback
    must stay cheap, predictable, and safe.
    """

    text = user_request.lower()
    available = _normalize_available_skills(available_skills)

    keyword_routes: tuple[tuple[tuple[str, ...], str], ...] = (
        (
            ("traceback", "stack trace", "error", "exception", "failing", "bug", "fix"),
            "debug",
        ),
        (
            ("test", "pytest", "unit test", "regression", "coverage"),
            "tests",
        ),
        (
            ("search the web", "google", "bing", "online search", "look up online"),
            "web_search",
        ),
        (
            ("gmail", "gmail access", "email", "send email", "email draft", "gmail api"),
            "gmail_access",
        ),
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

    selected = _choose_existing_skill(
        default_skill,
        available_skills=available,
        default_skill=default_skill,
    )

    if selected:
        return selected

    return default_skill





class SkillRegistry:
    """Loads skill playbooks from markdown files."""

    def __init__(self, skills_dir: Path | None = None) -> None:
        self.skills_dir = skills_dir or Path(__file__).parent / "skills"
        self._skills: dict[str, Skill] = {}

    def load(self) -> SkillRegistry:
        self._skills.clear()

        if not self.skills_dir.exists():
            return self

        for path in sorted(self.skills_dir.glob("*.md")):
            instructions = path.read_text(encoding="utf-8")

            self._skills[path.stem] = Skill(
                name=path.stem,
                purpose=_extract_purpose(instructions),
                instructions=instructions,
                path=path,
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
        """Return compact skill metadata for the skill-routing prompt."""

        if not self._skills:
            self.load()

        lines = [
            f"- {skill.name}: {skill.purpose}"
            for skill in self.list()
        ]
        return "\n".join(lines)

    def list_names(self) -> list[str]:
        if not self._skills:
            self.load()

        return sorted(self._skills)

    def list(self) -> list[Skill]:
        if not self._skills:
            self.load()

        return [self._skills[name] for name in self.list_names()]
