from __future__ import annotations

from typing import Literal

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
    operation: Literal["replace", "create"] = Field(
        default="replace",
        description=(
            "Patch operation type. Use 'replace' for exact text replacement in an "
            "existing file. Use 'create' only for creating a brand-new file."
        ),
    )

    path: str = Field(description="Repository-relative path to edit.")

    old: str = Field(
        default="",
        description=(
            "Exact existing text to replace. Required for replace operations. "
            "Must be empty for create operations."
        ),
    )

    new: str = Field(
        description=(
            "Replacement text for replace operations, or full file contents for "
            "create operations."
        )
    )
    
    reason: str = Field(default="", description="Why this edit is needed.")


class PatchDecision(BaseModel):
    summary: str = ""
    edits: list[FileEdit] = Field(default_factory=list)
    validation_commands: list[str] = Field(default_factory=list)


class ReportDecision(BaseModel):
    report: str
