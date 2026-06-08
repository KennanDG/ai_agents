from pydantic import BaseModel, Field


class PlanDecision(BaseModel):
    plan: list[str] = Field(description="Ordered list of high-level research steps")
    search_queries: list[str] = Field(description="Exact search queries to execute")


class SynthesizeDecision(BaseModel):
    research_summary: str = Field(description="Concise summary of the gathered information")
    report: str = Field(description="Final research report synthesising all findings")
