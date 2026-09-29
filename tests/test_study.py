"""Protocol and missing-result integrity checks; no neural training or real LLM calls.

Toy session files below exist only to exercise validation, not to supply study
results. Real reported agent scores must come from the authenticated CLI bridge.
"""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from option_agent_lab import study
from option_agent_lab.agent_protocol import call_plan
from option_agent_lab.data import labels, sample_points
from option_agent_lab.evaluation import evaluate


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return json.loads((REPO / "configs/erdos_v2.json").read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_default_matrix_has_nine_pairs_eighteen_sessions_and_126_calls(config):
    study.validate_config(config)
    rows = study.build_matrix(config)
    assert rows == study.build_matrix(config)
    assert len(rows) == 18
    pairs = Counter((r["model_seed"], r["search_seed"]) for r in rows)
    assert len(pairs) == 9 and set(pairs.values()) == {2}
    assert len({(r["run"], r["session"]) for r in rows}) == 18
    for model, seed in pairs:
        assert {r["condition"] for r in rows
                if (r["model_seed"], r["search_seed"]) == (model, seed)} == {"feedback", "no_feedback"}
    counts = call_plan(config["evaluation_budget"], max_rounds=8)
    assert counts == [128] * 8
    assert len(rows) * (len(counts) - 1) == 126
    changed = deepcopy(config)
    changed["execution_order_seed"] += 1
    assert study.build_matrix(changed) != rows
    assert sorted(study.build_matrix(changed), key=lambda r: (r["run"], r["session"])) == sorted(
        rows, key=lambda r: (r["run"], r["session"]))


@pytest.mark.parametrize("key,value", [
    ("evaluation_budget", 512), ("initial_points", 64), ("batch_size", 64),
    ("model_seeds", []), ("model_seeds", [17, 17, 43]),
    ("search_seeds", [True, 702, 703]), ("search_seeds", [-1, 702, 703]),
    ("data_seeds", [10, 11, 20]), ("data_seeds", [10, 20]),
    ("conditions", ["feedback", "no_feedback"]), ("de_populations", [128]),
    ("price_failure_threshold", .2), ("epochs", 0), ("audit_n", 1.5),
    ("execution_order_seed", -1), ("execution_order_seed", True),
    ("delta_diagnostic_threshold", -1), ("delta_diagnostic_threshold", float("nan")),
    ("model_seeds", None),
])
def test_invalid_configs_are_rejected(config, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        study.validate_config(config)


@pytest.fixture
def locked_study(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    core = repo / "core.py"
    core.write_text("# frozen source\n")
    root = tmp_path / "study"
    root.mkdir()
    source = {"core.py": study.digest(core)}
    save(root / "protocol.json", {"source_sha256": source, "config": {}})
    save(root / "matrix.json", [])
    save(root / "artifact_lock.json", {
        "source_sha256": source,
        "artifact_sha256": {name: study.digest(root / name)
                            for name in ("protocol.json", "matrix.json")},
    })
    monkeypatch.setattr(study, "REPO", repo)
    return root, repo


def test_unchanged_lock_verifies(locked_study):
    root, _ = locked_study
    assert "source_sha256" in study.verify_lock(root)


def test_changed_evaluator_source_is_rejected(locked_study):
    root, repo = locked_study
    (repo / "core.py").write_text("# altered evaluator\n")
    with pytest.raises(ValueError, match="source changed"):
        study.verify_lock(root)


@pytest.mark.parametrize("name", ["protocol.json", "matrix.json"])
def test_changed_protocol_or_matrix_is_rejected(locked_study, name):
    root, _ = locked_study
    (root / name).write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        study.verify_lock(root)


@pytest.fixture
def matrix_study(tmp_path, config):
    save(tmp_path / "matrix.json", study.build_matrix(config))
    return tmp_path


def test_missing_agent_results_remain_pending_without_scores(matrix_study):
    rows = study.collect_agent_results(matrix_study)
    assert len(rows) == 18
    assert all(r["status"] == "pending" for r in rows)
    assert all("max_error_normalized" not in r and "n" not in r for r in rows)


def toy_session(root, *, used=1024, observations=1024, final=True, failed=False):
    """Write one explicitly artificial result to test completion verification."""
    spec = study.read_json(root / "matrix.json")[0]
    run = root / spec["run"]
    folder = run / "sessions" / spec["session"]
    logs = run / "codex-agent-logs" / spec["session"]
    folder.mkdir(parents=True)
    logs.mkdir(parents=True)
    (run / "model.pt").write_bytes(b"unit-test-checkpoint-only")
    checkpoint = study.checkpoint_hash(run)
    rounds = (observations + 127) // 128
    save(folder / "state.json", {
        "used": used, "budget": 1024, "round": rounds, "seed": spec["search_seed"],
        "checkpoint_sha256": checkpoint, "history": [],
    })
    save(logs / "metadata.json", {
        "condition": spec["condition"], "seed": spec["search_seed"], "budget": 1024,
        "model_requested": "unit-test-model", "checkpoint_sha256": checkpoint,
        "status": "completed", "llm_calls_attempted": 7,
    })
    save(root / "agent_execution.json", {"model": "unit-test-model", "timeout_seconds": 240})
    remaining = observations
    for i in range(1, rounds + 1):
        n = min(128, remaining)
        x = sample_points(n, spec["search_seed"] + i - 1, "sobol")
        result = evaluate(x, lambda z: labels(z) + .01)
        np.savez_compressed(folder / f"round_{i:03d}.npz", **result)
        remaining -= n
    if final:
        save(logs / "final-status.json", {
            "used": used, "budget": 1024, "remaining": 1024 - used,
            "round": rounds, "summary": {"n": observations},
        })
    if failed:
        save(logs / "failure.json", {"type": "UnitTestFailure", "message": "artificial fixture"})
    return spec, folder, logs


def test_incomplete_session_is_not_scored(matrix_study):
    toy_session(matrix_study, used=128, observations=128, final=False)
    rows = study.collect_agent_results(matrix_study)
    assert rows[0]["status"] == "incomplete"
    assert "max_error_normalized" not in rows[0]
    assert sum(r["status"] == "pending" for r in rows) == 17


def test_failure_marker_never_presents_a_completed_score(matrix_study):
    toy_session(matrix_study, failed=True)
    row = study.collect_agent_results(matrix_study)[0]
    assert row["status"] == "failed"
    assert "max_error_normalized" not in row and "n" not in row


def test_missing_final_status_is_not_completed(matrix_study):
    toy_session(matrix_study, final=False)
    row = study.collect_agent_results(matrix_study)[0]
    assert row["status"] == "incomplete"
    assert "max_error_normalized" not in row


def test_forged_budget_count_cannot_pass_with_fewer_observations(matrix_study):
    toy_session(matrix_study, used=1024, observations=512)
    row = study.collect_agent_results(matrix_study)[0]
    assert row["status"] == "invalid"
    assert "max_error_normalized" not in row


def test_condition_mismatch_is_rejected(matrix_study):
    spec, _, logs = toy_session(matrix_study)
    metadata = study.read_json(logs / "metadata.json")
    metadata["condition"] = "feedback" if spec["condition"] == "no_feedback" else "no_feedback"
    save(logs / "metadata.json", metadata)
    row = study.collect_agent_results(matrix_study)[0]
    assert row["status"] == "invalid"
    assert "max_error_normalized" not in row


def test_complete_toy_session_is_scored_from_saved_observations(matrix_study):
    toy_session(matrix_study)
    row = study.collect_agent_results(matrix_study)[0]
    assert row["status"] == "completed"
    assert row["n"] == 1024
    assert row["max_error_normalized"] == pytest.approx(.01)


@pytest.mark.parametrize("field,value", [
    ("seed", 0), ("model_requested", "different-model"), ("budget", 512),
    ("status", "failed"), ("llm_calls_attempted", 6),
])
def test_changed_metadata_invalidates_session(matrix_study, field, value):
    _, _, logs = toy_session(matrix_study)
    meta = study.read_json(logs / "metadata.json")
    meta[field] = value
    save(logs / "metadata.json", meta)
    row = study.collect_agent_results(matrix_study)[0]
    assert row["status"] == "invalid"
    assert "max_error_normalized" not in row


def test_changed_checkpoint_invalidates_completed_session(matrix_study):
    spec, _, _ = toy_session(matrix_study)
    (matrix_study / spec["run"] / "model.pt").write_bytes(b"changed-unit-test-weights")
    row = study.collect_agent_results(matrix_study)[0]
    assert row["status"] == "invalid"
    assert "max_error_normalized" not in row


def test_changed_initial_points_invalidates_shared_start(matrix_study):
    _, folder, _ = toy_session(matrix_study)
    data = evaluate(sample_points(128, 19, "sobol"), lambda z: labels(z) + .01)
    np.savez_compressed(folder / "round_001.npz", **data)
    row = study.collect_agent_results(matrix_study)[0]
    assert row["status"] == "invalid"
    assert "max_error_normalized" not in row


def test_controller_exit_zero_without_results_does_not_mark_complete(locked_study, monkeypatch):
    root, _ = locked_study
    spec = {"model_seed": 17, "search_seed": 701, "condition": "feedback",
            "run": "models/model_17", "session": "test_missing_output"}
    save(root / "matrix.json", [spec])
    lock = study.read_json(root / "artifact_lock.json")
    lock["artifact_sha256"]["matrix.json"] = study.digest(root / "matrix.json")
    save(root / "artifact_lock.json", lock)
    monkeypatch.setattr(study.shutil, "which", lambda binary: "/unit-test/codex")
    monkeypatch.setattr(study.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="unit-test", stderr=""))
    rows = study.run_agents(root, model="unit-test-model")
    assert rows[0]["status"] not in ("completed", "already_completed")


def test_unavailable_codex_leaves_pending_results(locked_study, monkeypatch):
    root, _ = locked_study
    monkeypatch.setattr(study.shutil, "which", lambda binary: None)
    with pytest.raises(ValueError, match="pending"):
        study.run_agents(root, model="unit-test-model")
    assert not (root / "agent_execution.json").exists()
