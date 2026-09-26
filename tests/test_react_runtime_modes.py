from backend.agent.runtime import build_runtime_budget, get_runtime_mode


def test_runtime_mode_defaults_safely_for_invalid_value():
    assert get_runtime_mode("not-a-mode") == "legacy"
    assert get_runtime_mode("react") == "react"
    assert get_runtime_mode("react_shadow") == "react_shadow"


def test_runtime_budget_is_bounded_and_explicit():
    budget = build_runtime_budget()
    assert budget.max_iterations >= 1
    assert budget.max_tool_calls >= budget.max_kb_calls
    assert budget.input_token_limit >= 2048
