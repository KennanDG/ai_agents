from __future__ import annotations

from pydantic import BaseModel, Field


class PlanDecision(BaseModel):
    plan: list[str] = Field(
        default_factory=list,
        description="Short implementation plan steps.",
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="Repository search terms to gather context.",
    )
    validation_commands: list[str] = Field(
        default_factory=list,
        description="Safe validation commands to run after edits.",
    )


class FileToInspect(BaseModel):
    path: str
    reason: str = ""


class ContextDecision(BaseModel):
    files_to_inspect: list[FileToInspect] = Field(default_factory=list)


class FileEdit(BaseModel):
    path: str = Field(description="Repository-relative path to edit.")
    old: str = Field(description="Exact existing text to replace.")
    new: str = Field(description="Replacement text.")
    reason: str = Field(default="", description="Why this edit is needed.")


class PatchDecision(BaseModel):
    summary: str = ""
    edits: list[FileEdit] = Field(default_factory=list)
    validation_commands: list[str] = Field(default_factory=list)


class ReportDecision(BaseModel):
    report: str
