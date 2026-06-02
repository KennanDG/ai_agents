from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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




def route_skill(user_request: str) -> str:
    """Simple deterministic router. Upgrade to LLM routing later."""
    text = user_request.lower()

    if any(term in text for term in ["traceback", "stack trace", "error", "exception", "failing"]):
        return "debug"
    if any(term in text for term in ["test", "pytest", "unit test", "regression"]):
        return "tests"
    # if any(term in text for term in ["langgraph", "graph", "node", "edge", "state"]):
    #     return "add_langgraph_node"
    
    if any(term in text for term in ["web search", "search the web", "google", "bing", "online search"]):
        return "web_search"
    if any(term in text for term in ["gmail", "gmail access", "email", "send email", "email draft", "gmail api"]):
        return "gmail_access"

    return "implement_change"






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

    
    
    def list_names(self) -> list[str]:
        if not self._skills:
            self.load()

        return sorted(self._skills)

    
    
    def list(self) -> list[Skill]:
        if not self._skills:
            self.load()

        return [self._skills[name] for name in self.list_names()]
