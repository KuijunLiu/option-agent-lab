"""Numerical oracles and experiment-integrity checks.

The dummy predictor used by session unit tests is never a trained model or a
reported experiment result. The published CSV is independent of our formula.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from option_agent_lab import search
from option_agent_lab.data import BOUNDS, NAMES, generate_data, labels, sample_points
from option_agent_lab.evaluation import evaluate, summarize
from option_agent_lab.pricing import bs_call


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "reference" / "black_scholes_reference.csv"
with REFERENCE.open(newline="") as stream:
    PUBLISHED_CASES = list(csv.DictReader(stream))


@pytest.mark.parametrize("case", PUBLISHED_CASES,
                         ids=[f"QuantLib-line-{c['source_row']}" for c in PUBLISHED_CASES])
def test_published_quantlib_call_prices(case):
    price = bs_call(*[float(case[key]) for key in
                     ("spot", "strike", "maturity_years", "volatility", "rate", "dividend_yield")])
    assert abs(float(price) - float(case["expected_price"])) <= float(case["absolute_tolerance"])


def test_reference_integrity_and_provenance():
    provenance = json.loads((REFERENCE.parent / "provenance.json").read_text())
    assert len(PUBLISHED_CASES) == provenance["row_count"] == 22
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == provenance["csv_sha256"]
    assert len(provenance["source_commit"]) == 40


def test_expiry_and_zero_volatility_limits():
    # At expiry, rates, dividends, and volatility cannot change the payoff.
    np.testing.assert_allclose(bs_call([80, 100, 120], 100, 0, .8, -.01, .1),
                               [0, 0, 20], atol=1e-14)
    # Without volatility, the discounted deterministic payoff is known directly.
    spots = np.array([80., 100., 120.])
    expected = np.maximum(spots * np.exp(-.03 * 2) - 100 * np.exp(-.05 * 2), 0)
    np.testing.assert_allclose(bs_call(spots, 100, 2, 0, .05, .03), expected, atol=1e-14)


@pytest.mark.parametrize("index,value", [(0, 0), (1, -1), (2, -.1),
                                         (3, -.1), (4, np.inf), (5, np.nan)])
def test_pricer_rejects_invalid_parameters(index, value):
    parameters = [100, 100, 1, .2, .02, 0]
    parameters[index] = value
    with pytest.raises(ValueError):
        bs_call(*parameters)


def test_independent_quantlib_oracle_randomized():
    ql = pytest.importorskip("QuantLib", reason="Install the optional oracle extra for this independent check")
    rng = np.random.default_rng(19473)
    # Compare against an independently maintained numerical library, including
    # negative rates and nonzero dividends outside the training configuration.
    spot = rng.uniform(10, 300, 100)
    strike = rng.uniform(10, 300, 100)
    maturity = rng.uniform(1 / 365, 5, 100)
    volatility = rng.uniform(.01, 1.5, 100)
    rate = rng.uniform(-.03, .15, 100)
    dividend = rng.uniform(0, .12, 100)
    expected = np.array([
        ql.blackFormula(ql.Option.Call, float(k), float(s * np.exp((r - q) * t)),
                        float(v * np.sqrt(t)), float(np.exp(-r * t)))
        for s, k, t, v, r, q in zip(spot, strike, maturity, volatility, rate, dividend)
    ])
    actual = bs_call(spot, strike, maturity, volatility, rate, dividend)
    assert np.max(np.abs(actual - expected)) < 1e-10


def test_generated_splits_are_reproducible_separate_and_correctly_labeled(tmp_path):
    sizes = {"train": 64, "validation": 16, "test": 32, "stress": 32}
    first, second = tmp_path / "first", tmp_path / "second"
    metadata = generate_data(first, seed=500, train=64, validation=16, test=32)
    again = generate_data(second, seed=500, train=64, validation=16, test=32)
    assert metadata == again
    seen = set()
    for offset, (name, n) in enumerate(sizes.items()):
        with np.load(first / f"{name}.npz") as data, np.load(second / f"{name}.npz") as repeat:
            x, y = data["X"], data["y"]
            assert x.shape == (n, 3) and y.shape == (n,)
            np.testing.assert_array_equal(x, repeat["X"])
            np.testing.assert_array_equal(y, repeat["y"])
            assert np.isfinite(x).all() and np.isfinite(y).all()
            assert np.all((x >= BOUNDS[:, 0]) & (x <= BOUNDS[:, 1]))
            np.testing.assert_allclose(y, bs_call(x[:, 0] * 100, 100, x[:, 1], x[:, 2], .02) / 100)
            rows = set(map(tuple, x))
            assert not rows & seen
            seen.update(rows)
            assert metadata["splits"][name]["seed"] == 500 + offset
            assert metadata["splits"][name]["n"] == n
            csv_data = np.loadtxt(first / f"{name}.csv.gz", delimiter=",", skiprows=1)
            np.testing.assert_allclose(csv_data, np.column_stack((x, y)), rtol=1e-10)


def test_failure_regions_count_regions_not_duplicate_points():
    points = np.array([[1., .1, .2], [1., .1, .2], [1.2, .5, .6]])
    result = evaluate(points, lambda x: labels(x) + .002)
    summary = summarize(result)
    assert summary["n"] == summary["failures"] == 3
    assert summary["failure_regions"] == 2
    assert summary["max_error_normalized"] == pytest.approx(.002)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_predictions_are_rejected(bad):
    with pytest.raises(ValueError, match="invalid prices"):
        evaluate(np.array([[1., .1, .2]]), lambda x: np.full(len(x), bad))


@pytest.fixture
def session_run(tmp_path, monkeypatch):
    # This fake checkpoint and exact-price predictor isolate bookkeeping only.
    # They are not substituted into the actual numerical feasibility run.
    (tmp_path / "model.pt").write_bytes(b"session-unit-test-fixture")
    monkeypatch.setattr(search, "load_predictor", lambda checkpoint: labels)
    search.init_session(tmp_path, "test", budget=8, seed=10)
    return tmp_path


def proposal(n=4):
    return {"bounds": dict(zip(NAMES, BOUNDS.tolist())), "n": n,
            "reason": "Unit test: cover the declared parameter domain."}


def test_agent_budget_is_persistent_and_cannot_be_exceeded(session_run):
    first = search.step_session(session_run, "test", proposal())
    assert (first["used"], first["remaining"], first["round"]) == (4, 4, 1)
    second = search.step_session(session_run, "test", proposal())
    assert (second["used"], second["remaining"], second["round"]) == (8, 0, 2)
    assert second["summary"]["n"] == 8
    before = (session_run / "sessions/test/state.json").read_bytes()
    with pytest.raises(ValueError, match="remaining budget"):
        search.step_session(session_run, "test", proposal(1))
    assert (session_run / "sessions/test/state.json").read_bytes() == before
    assert len(list((session_run / "sessions/test").glob("round_*.npz"))) == 2


@pytest.mark.parametrize("invalid", [
    None,
    {"bounds": [], "n": 4, "reason": "Malformed bounds should fail closed."},
    *[{**proposal(), "bounds": {**proposal()["bounds"], "m": bounds}}
      for bounds in [[.4, 1.5], [1.1, .9], [np.nan, 1.5]]],
])
def test_invalid_proposal_does_not_charge_or_write_results(session_run, invalid):
    before = (session_run / "sessions/test/state.json").read_bytes()
    with pytest.raises(ValueError):
        search.step_session(session_run, "test", invalid)
    assert (session_run / "sessions/test/state.json").read_bytes() == before
    assert not list((session_run / "sessions/test").glob("round_*.npz"))


def test_changed_checkpoint_and_session_reset_are_rejected(session_run):
    before = (session_run / "sessions/test/state.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        search.init_session(session_run, "test", budget=1000)
    (session_run / "model.pt").write_bytes(b"different-checkpoint")
    with pytest.raises(ValueError, match="checkpoint changed"):
        search.step_session(session_run, "test", proposal())
    assert (session_run / "sessions/test/state.json").read_bytes() == before


def test_initial_agent_batch_matches_baseline_initial_points(session_run):
    # Agent/baseline starting information must agree before feedback can matter.
    search.init_session(session_run, "common_start", budget=128, seed=123)
    search.step_session(session_run, "common_start", proposal(128))
    with np.load(session_run / "sessions/common_start/round_001.npz") as batch:
        np.testing.assert_array_equal(batch["X"], sample_points(128, 123, "sobol"))
