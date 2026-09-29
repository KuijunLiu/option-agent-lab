"""Independent oracle and chain-rule checks for price/delta diagnostics."""
import hashlib
import json

import numpy as np
import pytest
import torch

import option_agent_lab.greeks as greeks
from option_agent_lab.data import BOUNDS, sample_points
from option_agent_lab.model import Pricer, load_predictor
from option_agent_lab.pricing import bs_call


def save_checkpoint(path, net=None, bounds=BOUNDS):
    torch.manual_seed(194)
    net = Pricer() if net is None else net
    torch.save({"state_dict": net.state_dict(),
                "metadata": {"bounds": np.asarray(bounds).tolist(),
                             "strike": 100.0, "risk_free_rate": 0.02}}, path)
    return net


def linear_pricer(slope=0.25, intercept=0.3):
    """The actual SiLU architecture can represent a linear function exactly:
    SiLU(z) - SiLU(-z) = z, applied through each hidden layer.
    """
    net = Pricer()
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
        net.network[0].weight[0, 0] = 1
        net.network[0].weight[1, 0] = -1
        net.network[2].weight[0, 0] = 1
        net.network[2].weight[0, 1] = -1
        net.network[2].weight[1, 0] = -1
        net.network[2].weight[1, 1] = 1
        net.network[4].weight[0, 0] = slope
        net.network[4].weight[0, 1] = -slope
        net.network[4].bias[0] = intercept
    return net


def test_analytic_delta_matches_price_finite_difference_and_broadcasts():
    rng = np.random.default_rng(828)
    spot = rng.uniform(60, 140, 100)
    strike = 100.0
    maturity = rng.uniform(0.05, 2, 100)
    vol = rng.uniform(0.1, 0.7, 100)
    rate = rng.uniform(-0.02, 0.1, 100)
    dividend = rng.uniform(0, 0.08, 100)
    step = 1e-4
    numerical = (bs_call(spot + step, strike, maturity, vol, rate, dividend)
                 - bs_call(spot - step, strike, maturity, vol, rate, dividend)) / (2 * step)
    analytic = greeks.bs_delta(spot, strike, maturity, vol, rate, dividend)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-6, atol=2e-10)
    assert greeks.bs_delta(100, 100, 1, 0.2, 0.02).shape == ()


def test_analytic_delta_matches_independent_quantlib():
    ql = pytest.importorskip("QuantLib")
    rng = np.random.default_rng(933)
    params = zip(rng.uniform(50, 150, 50), rng.uniform(80, 120, 50),
                 rng.uniform(0.003, 1.5, 50), rng.uniform(0.05, 0.8, 50),
                 rng.uniform(-0.02, 0.08, 50), rng.uniform(0.0, 0.1, 50))
    for spot, strike, maturity, vol, rate, dividend in params:
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, float(strike))
        forward = spot * np.exp((rate - dividend) * maturity)
        calculator = ql.BlackCalculator(payoff, float(forward),
                                        float(vol * np.sqrt(maturity)),
                                        float(np.exp(-rate * maturity)))
        assert float(greeks.bs_delta(spot, strike, maturity, vol, rate, dividend)) == pytest.approx(
            calculator.delta(float(spot)), abs=1e-12)


@pytest.mark.parametrize("index,value", [(0, 0), (1, -1), (2, 0), (3, 0),
                                        (0, np.nan), (4, np.inf), (5, np.nan)])
def test_delta_rejects_degenerate_or_nonfinite_parameters(index, value):
    args = [100, 100, 0.5, 0.2, 0.02, 0.01]
    args[index] = value
    with pytest.raises(ValueError):
        greeks.bs_delta(*args)


def test_autograd_uses_saved_scale_without_strike_factor(tmp_path, monkeypatch):
    bounds = BOUNDS.copy()
    bounds[0] = [0.6, 1.4]  # dc/dm = 0.25 * 2/0.8 = 0.625, not 62.5.
    net = linear_pricer()
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, net, bounds)
    model_bytes = checkpoint.read_bytes()
    original = {name: value.clone() for name, value in net.state_dict().items()}
    # Capture the loaded instance to inspect frozen parameters and .grad buffers.
    monkeypatch.setattr(greeks, "Pricer", lambda: net)
    predict = greeks.load_price_delta(checkpoint, batch_size=2)
    x = np.array([[0.6, 0.1, 0.2], [0.85, 0.2, 0.3],
                  [1.0, 0.5, 0.4], [1.4, 0.9, 0.7]])
    expected_price = 0.25 * (2 * (x[:, 0] - 0.6) / 0.8 - 1) + 0.3
    with torch.no_grad():
        price, delta = predict(x)
    np.testing.assert_allclose(price, expected_price, atol=1e-7)
    np.testing.assert_allclose(delta, 0.625, atol=2e-7)
    # Repeated audits cannot accumulate gradients or mutate the checkpoint.
    price2, delta2 = predict(x)
    np.testing.assert_array_equal(price, price2)
    np.testing.assert_array_equal(delta, delta2)
    for parameter in net.parameters():
        assert parameter.grad is None
        assert not parameter.requires_grad
    for name, value in net.state_dict().items():
        assert torch.equal(value, original[name])
    assert checkpoint.read_bytes() == model_bytes


def test_random_network_delta_matches_finite_difference_and_existing_predictor(tmp_path):
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint)
    x = sample_points(29, 939)
    predict = greeks.load_price_delta(checkpoint, batch_size=7)
    price, delta = predict(x)
    standard = load_predictor(checkpoint)
    np.testing.assert_allclose(price, standard(x), atol=3e-8)
    plus, minus = x.copy(), x.copy()
    plus[:, 0] += 0.002
    minus[:, 0] -= 0.002
    numerical = (standard(plus) - standard(minus)) / 0.004
    np.testing.assert_allclose(delta, numerical, atol=2e-5, rtol=1e-3)
    assert np.isfinite(price).all() and np.isfinite(delta).all()
    empty_price, empty_delta = predict(np.empty((0, 3)))
    assert empty_price.shape == empty_delta.shape == (0,)


@pytest.mark.parametrize("x", [np.ones((2, 2)), [[1, 0, 0.2]],
                               [[1, 0.2, -0.1]], [[np.nan, 0.2, 0.2]]])
def test_neural_delta_rejects_invalid_input(tmp_path, x):
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint)
    with pytest.raises(ValueError):
        greeks.load_price_delta(checkpoint)(x)


def test_neural_delta_rejects_nonfinite_prediction(tmp_path):
    checkpoint = tmp_path / "model.pt"
    net = Pricer()
    with torch.no_grad():
        net.network[-1].bias.fill_(float("nan"))
    save_checkpoint(checkpoint, net)
    with pytest.raises(ValueError, match="nonfinite"):
        greeks.load_price_delta(checkpoint)([[1, 0.5, 0.2]])


def test_audit_outputs_real_measurements_and_preserves_frozen_checkpoint(tmp_path):
    run, data, output = tmp_path / "run", tmp_path / "data", tmp_path / "audit"
    run.mkdir()
    data.mkdir()
    save_checkpoint(run / "model.pt", linear_pricer())
    expected_hash = hashlib.sha256((run / "model.pt").read_bytes()).hexdigest()
    for i, split in enumerate(("test", "stress")):
        # y is deliberately absent: reference prices must come from the oracle.
        np.savez(data / f"{split}.npz", X=sample_points(15 + i, 200 + i))
    report = greeks.audit_greeks(data, run, output)
    assert report["checkpoint_sha256"] == expected_hash
    assert hashlib.sha256((run / "model.pt").read_bytes()).hexdigest() == expected_hash
    assert json.loads((output / "audit.json").read_text()) == report
    for i, split in enumerate(("test", "stress")):
        with np.load(output / f"{split}.npz") as result:
            assert set(result.files) == {"X", "pred_price", "ref_price", "pred_delta", "ref_delta",
                                         "price_error", "delta_error"}
            metrics = report["splits"][split]
            assert metrics["n"] == 15 + i
            assert metrics["price"]["mae"] == pytest.approx(result["price_error"].mean())
            assert metrics["delta"]["p99_abs_error"] == pytest.approx(np.quantile(result["delta_error"], 0.99))
            assert metrics["price_accurate_subset"]["n"] == int((result["price_error"] <= 0.001).sum())
            np.testing.assert_allclose(result["pred_delta"], 0.5, atol=1e-7)
    with pytest.raises(ValueError, match="exists"):
        greeks.audit_greeks(data, run, output)


def test_empty_price_accurate_subset_serializes_as_null():
    assert all(value is None for value in greeks._error_metrics(np.array([])).values())
