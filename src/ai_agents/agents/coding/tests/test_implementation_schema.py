from ai_agents.agents.coding.coding_agent_schemas import (
    ImplementationUnitDecision,
    PatchDecision,
    PlanDecision,
)


def test_plan_supports_more_units_than_worker_concurrency() -> None:
    units = [
        ImplementationUnitDecision(
            id=f"unit-{index}",
            objective=f"Implement unit {index}",
        )
        for index in range(12)
    ]

    decision = PlanDecision(
        task_mode="parallel",
        plan=["Implement in independent units."],
        implementation_units=units,
    )

    assert len(decision.implementation_units) == 12


def test_patch_decision_can_explicitly_complete_without_edits() -> None:
    decision = PatchDecision(
        summary="The requested behavior is already implemented.",
        no_change_needed=True,
    )

    assert decision.no_change_needed is True
    assert decision.edits == []
