"""Information-boundary and controller tests; fake CLI is used ONLY in tests."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from option_agent_lab.agent_protocol import (
    build_prompt, call_plan, make_agent_view, validate_proposal,
)


DOMAIN = {"m": [0.5, 1.5], "T": [1 / 365, 1.0], "sigma": [0.05, 0.8]}


def proposal(n=128):
    return {"bounds": deepcopy(DOMAIN), "n": n, "reason": "Choose this region."}


def make_status(rounds=3):
    history = []
    for index in range(rounds):
        history.append({
            "round": index + 1, "proposal": proposal(),
            "sampler": "DO_NOT_FORWARD", "sampler_seed": 99000 + index,
            "evaluation_seconds": 777777,
            "summary": {"n": 128, "max_error_normalized": 10001.123 + index,
                        "failure_regions": index + 1, "hidden_debug": "DO_NOT_FORWARD",
                        "top_points": [{"m": 1.0, "T": 0.01, "sigma": 0.1,
                                        "error_normalized": 20001.123 + index,
                                        "extra_payload": "DO_NOT_FORWARD"}]},
        })
    return {"budget": 1024, "used": rounds * 128, "remaining": 1024 - rounds * 128,
            "round": rounds, "threshold_normalized": 0.001, "domain": deepcopy(DOMAIN),
            "history": history, "summary": {"payload": "DO_NOT_FORWARD"},
            "extra_metadata": {"hidden_price": "DO_NOT_FORWARD"}}


def test_control_never_contains_later_outcomes_or_backend_payloads():
    status = make_status()
    before = deepcopy(status)
    view = make_agent_view(status, "no_feedback")
    serialized = json.dumps(view)
    assert "10001.123" in serialized and "20001.123" in serialized
    for hidden in ("10002.123", "20002.123", "10003.123", "20003.123",
                   "DO_NOT_FORWARD", "777777", "99000", "99001", "99002"):
        assert hidden not in serialized
    assert view["history"][1]["summary"] is None
    assert view["history"][2]["summary"] is None
    assert all(record["proposal"] == proposal() for record in view["history"])
    assert "summary" not in view
    assert status == before


def test_initial_information_and_prompt_are_exactly_identical():
    status = make_status(1)
    feedback = make_agent_view(status, "feedback")
    control = make_agent_view(status, "no_feedback")
    assert feedback == control
    assert build_prompt(feedback, 128) == build_prompt(control, 128)


def test_only_summary_visibility_changes_between_conditions():
    status = make_status()
    feedback = make_agent_view(status, "feedback")
    control = make_agent_view(status, "no_feedback")
    assert feedback.keys() == control.keys()
    for live, masked in zip(feedback["history"], control["history"]):
        assert live.keys() == masked.keys()
        assert live["proposal"] == masked["proposal"]
        assert live["round"] == masked["round"]
    assert feedback["history"][2]["summary"]["max_error_normalized"] == 10003.123
    assert "DO_NOT_FORWARD" not in json.dumps(feedback)
    changed = deepcopy(status)
    changed["summary"] = {"max_error_normalized": 9e99}
    changed["history"][2]["summary"] = {"arbitrary": ["secret", {"deep": 987654321}]}
    assert make_agent_view(changed, "no_feedback") == control
    control["domain"]["m"][0] = 0.7
    assert status["domain"]["m"][0] == 0.5


def test_call_plan_has_exactly_seven_llm_decisions():
    plan = call_plan()
    assert plan == [128] * 8
    assert sum(plan) == 1024 and len(plan[1:]) == 7
    assert call_plan(300, 3) == [128, 128, 44]
    assert call_plan(64, 1) == [64]
    with pytest.raises(ValueError):
        call_plan(1024, 7)


@pytest.mark.parametrize("n", [1, 64, 127, 129, True, 128.0])
def test_llm_cannot_change_requested_batch_size(n):
    with pytest.raises(ValueError):
        validate_proposal(proposal(n), 128, DOMAIN)


@pytest.mark.parametrize("pair", [[0.0, 1.0], [1.0, 1.0], [1.5, 0.5],
                                  [float("nan"), 1.0], [False, 1.0], [1.0]])
def test_invalid_region_rejected_before_backend(pair):
    value = proposal()
    value["bounds"]["m"] = pair
    with pytest.raises(ValueError):
        validate_proposal(value, 128, DOMAIN)


def test_valid_proposal_and_invalid_condition():
    validate_proposal(proposal(), 128, DOMAIN)
    with pytest.raises(ValueError):
        make_agent_view(make_status(), "unrecognized")


@pytest.fixture
def bridge():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_codex_agent.py"
    spec = importlib.util.spec_from_file_location("codex_bridge_test_only", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_controller_environment(bridge, monkeypatch, tmp_path, returned_count=128):
    """Test double: no model evaluation, real Codex, or claimed research result."""
    run = tmp_path / "test-run"
    run.mkdir()
    (run / "model.pt").write_bytes(b"TEST DOUBLE ONLY")
    state, calls, evaluations = make_status(0), [], []

    def fake_backend(args, action, *extra):
        if action == "session-step":
            value = json.loads(Path(extra[1]).read_text())
            evaluations.append(value)
            index = state["round"]
            record = make_status(index + 1)["history"][-1]
            record["proposal"] = value
            state["history"].append(record)
            state["used"] += value["n"]
            state["remaining"] -= value["n"]
            state["round"] += 1
        return json.dumps(state)

    def fake_cli(command, **kwargs):
        if command[1] == "--version":
            return subprocess.CompletedProcess(command, 0, "codex TEST DOUBLE", "")
        calls.append(kwargs["input"])
        Path(command[command.index("-o") + 1]).write_text(json.dumps(proposal(returned_count)))
        stdout = json.dumps({"type": "turn.completed", "usage": {"input_tokens": 12, "output_tokens": 13}})
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(bridge.shutil, "which", lambda _: "/test-only/codex")
    monkeypatch.setattr(bridge, "backend", fake_backend)
    monkeypatch.setattr(bridge.subprocess, "run", fake_cli)
    return run, calls, evaluations


@pytest.mark.parametrize("condition", ["feedback", "no_feedback"])
def test_controller_equal_calls_and_complete_audit_logs(bridge, monkeypatch, tmp_path, condition):
    run, calls, evaluations = fake_controller_environment(bridge, monkeypatch, tmp_path)
    assert bridge.main(["--run", str(run), "--name", "unit-test", "--condition", condition,
                        "--model", "test-only-model"]) == 0
    assert len(calls) == 7
    assert len(evaluations) == 8 and sum(x["n"] for x in evaluations) == 1024
    assert evaluations[0]["reason"] == "Controller-fixed common full-domain initial batch."
    if condition == "no_feedback":
        assert all("10002.123" not in prompt and "20002.123" not in prompt for prompt in calls)
    else:
        assert "10002.123" in calls[-1]
    logs = run / "codex-agent-logs" / "unit-test"
    metadata = json.loads((logs / "metadata.json").read_text())
    assert metadata["condition"] == condition and metadata["llm_calls_attempted"] == 7
    assert metadata["status"] == "completed" and metadata["model_selection"] == "explicit"
    for index in range(1, 8):
        record = json.loads((logs / f"round-{index:02d}-call.json").read_text())
        assert record["status"] == "valid_proposal"
        assert record["token_usage_status"] == "reported_by_cli"
        assert record["wall_seconds"] >= 0 and record["prompt_sha256"]
        assert (logs / f"round-{index:02d}-visible-state.json").is_file()


def test_controller_stops_after_unequal_batch_no_fallback(bridge, monkeypatch, tmp_path):
    run, calls, evaluations = fake_controller_environment(bridge, monkeypatch, tmp_path, 64)
    assert bridge.main(["--run", str(run), "--name", "invalid-batch"]) == 1
    assert len(calls) == 1 and len(evaluations) == 1
    logs = run / "codex-agent-logs" / "invalid-batch"
    failure = json.loads((logs / "failure.json").read_text())
    assert failure["no_fallback_used"] and "exactly 128" in failure["message"]
    assert json.loads((logs / "metadata.json").read_text())["status"] == "failed"
    assert (logs / "round-01-raw-proposal.txt").is_file()


def test_missing_cli_does_not_create_session(bridge, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(bridge.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit) as error:
        bridge.main(["--run", str(tmp_path / "missing-cli")])
    assert error.value.code == 2
    assert "No agent experiment ran" in capsys.readouterr().err
    assert not (tmp_path / "missing-cli").exists()


def test_unknown_usage_is_not_estimated_and_tool_events_are_detected(bridge):
    info = bridge.cli_observations("not JSON\n" + json.dumps({"type": "item.started", "item": {"type": "command_execution"}}))
    assert info["token_usage_status"] == "unknown_not_reported"
    assert info["token_usage_records"] is None
    assert info["detected_tool_events"]
