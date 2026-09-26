from backend.agent.context.builder import ReactContextBuilder
from backend.agent.schemas import RuntimeBudget


def test_context_builder_rebuilds_bounded_typed_sections():
    result = ReactContextBuilder().build(
        current_query="what changed?",
        budget=RuntimeBudget(profile="chat", input_token_limit=3000),
        runtime_policy={"iteration": 1},
        working_memory={"schema_version": "wm.v2", "user_goal": "answer"},
        memory_context={"recent_messages": [{"role": "user", "content": "x" * 1000}]},
        evidence_ledger={"entries": {}, "slot_coverage": {}, "conflicts": [], "next_index": 1},
        observations=[],
        tool_specs=[],
    )
    assert result["token_usage"]["used"] <= result["token_usage"]["budget"]
    assert "current_query" in result["loaded_sections"]
    assert '<CONTEXT_SECTION name="working_memory"' in result["text"]
