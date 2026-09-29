"""Budget and fairness tests using explicit toy predictors, never reported data."""

import json

import numpy as np
import pytest

from option_agent_lab import benchmarks
from option_agent_lab.data import BOUNDS, labels, sample_points


@pytest.fixture
def toy_run(tmp_path, monkeypatch):
    """Fake weights and a synthetic error surface isolate search accounting."""
    (tmp_path / "model.pt").write_bytes(b"benchmark-unit-test-only")
    calls = []

    def toy_predictor(x):
        calls.append(len(x))
        return labels(x) + .01 * np.exp(-3 * (x[:, 0] - 1) ** 2) / (1 + x[:, 1])

    monkeypatch.setattr(benchmarks, "load_predictor", lambda path: toy_predictor)
    return tmp_path, calls


@pytest.mark.parametrize("population", [16, 32, 64])
def test_all_evaluations_count_and_initial_information_matches(toy_run, population):
    run, calls = toy_run
    before = (run / "model.pt").read_bytes()
    summary = benchmarks.run_searches(run, budget=256, seeds=(17,), de_population=population)
    assert sum(calls) == 4 * 256
    assert len(summary["runs"]) == 4
    for row in summary["runs"]:
        assert row["n"] == 256
        with np.load(run / "benchmarks_v2" / f"{row['method']}_17.npz") as data:
            assert data["X"].shape == (256, 3)
            np.testing.assert_array_equal(data["X"][:128], sample_points(128, 17, "sobol"))
        if row["method"] == "de":
            details = row["details"]
            assert details["actual_population"] == population
            assert details["population_reevaluations_charged"] == population
            assert (128 + details["objective_point_evaluations"]
                    + details["early_stop_fill_points"]) == 256
            assert len(set(details["population_source_indices"])) == population
    assert (run / "model.pt").read_bytes() == before
    saved = json.loads((run / "benchmarks_v2/summary.json").read_text())
    assert saved["checkpoint_sha256"] == summary["checkpoint_sha256"]


def test_default_stress_schedule_is_fixed_and_covers_all_seven_regions(toy_run, monkeypatch):
    run, _ = toy_run
    first = benchmarks.run_searches(run, seeds=(4,), output_name="first")
    # Changing the predictor must not change this baseline's parameter choices.
    monkeypatch.setattr(benchmarks, "load_predictor", lambda path: lambda x: labels(x) - .25)
    second = benchmarks.run_searches(run, seeds=(4,), output_name="second")
    for summary in (first, second):
        stress = next(row for row in summary["runs"] if row["method"] == "financial_stress")
        assert stress["n"] == 1024
        assert not stress["details"]["uses_feedback"]
        assert len(stress["details"]["schedule"]) == 7
        for batch in stress["details"]["schedule"]:
            assert batch["n"] == 128
    with np.load(run / "first/financial_stress_4.npz") as a, np.load(run / "second/financial_stress_4.npz") as b:
        np.testing.assert_array_equal(a["X"], b["X"])
        for i, region in enumerate(benchmarks.FINANCIAL_STRESS_REGIONS):
            bounds = np.array(region["bounds"])
            points = a["X"][128 * (i + 1):128 * (i + 2)]
            assert np.all(points >= bounds[:, 0]) and np.all(points <= bounds[:, 1])


def test_stress_schedule_cycles_and_truncates_without_using_feedback():
    # Public benchmark budgets are powers of two. The internal schedule also
    # handles partial batches, making its counting policy explicit.
    points, batches = benchmarks._stress_points(128 + 7 * 128 + 13, seed=10)
    assert len(points) == 1037
    assert len(batches) == 8
    assert batches[-1]["name"] == batches[0]["name"]
    assert batches[-1]["n"] == 13 and batches[-1]["sampler"] == "random"
    assert np.all(points >= BOUNDS[:, 0]) and np.all(points <= BOUNDS[:, 1])


def test_flat_surface_early_stop_fill_is_charged(toy_run, monkeypatch):
    run, calls = toy_run
    calls.clear()

    def exact(x):
        calls.append(len(x))
        return labels(x)

    monkeypatch.setattr(benchmarks, "load_predictor", lambda path: exact)
    summary = benchmarks.run_searches(run, seeds=(7,), budget=1024)
    details = next(row["details"] for row in summary["runs"] if row["method"] == "de")
    assert details["early_stop_fill_points"] > 0
    assert sum(calls) == 4 * 1024
    assert details["objective_point_evaluations"] + details["early_stop_fill_points"] == 896


def test_existing_output_rejected_without_evaluation(toy_run):
    run, calls = toy_run
    folder = run / "benchmarks_v2"
    folder.mkdir()
    marker = folder / "preserved.txt"
    marker.write_text("preserve prior experiment")
    with pytest.raises(ValueError, match="already exists"):
        benchmarks.run_searches(run)
    assert not calls and marker.read_text() == "preserve prior experiment"


def test_checkpoint_mutation_rejects_report(toy_run, monkeypatch):
    run, _ = toy_run

    def mutating_predictor(x):
        (run / "model.pt").write_bytes(b"changed-unit-test-weights")
        return labels(x)

    monkeypatch.setattr(benchmarks, "load_predictor", lambda path: mutating_predictor)
    with pytest.raises(ValueError, match="checkpoint changed"):
        benchmarks.run_searches(run, budget=256, seeds=(1,))
    assert not (run / "benchmarks_v2/summary.json").exists()
    assert not list((run / "benchmarks_v2").glob("*.npz"))


@pytest.mark.parametrize("settings", [
    {"budget": 1023}, {"budget": True}, {"budget": 128},
    {"de_population": 128}, {"de_population": True},
    {"seeds": ()}, {"seeds": (1, 1)}, {"seeds": (-1,)},
    {"output_name": "../outside"},
])
def test_invalid_protocol_rejected_before_evaluation(toy_run, settings):
    run, calls = toy_run
    with pytest.raises(ValueError):
        benchmarks.run_searches(run, **settings)
    assert not calls
