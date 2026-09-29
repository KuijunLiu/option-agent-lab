"""Independent price/delta diagnostics for the frozen, normalized neural pricer.

The model predicts c(m, T, sigma) = C / K with m = S / K. Consequently
delta = dC/dS = dc/dm: no factor of K belongs in this derivative. Autograd
below differentiates scaled inputs and explicitly applies their chain rule.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.special import ndtr

from .data import RATE, STRIKE, THRESHOLD
from .model import Pricer, _normalize
from .pricing import bs_call


def bs_delta(spot, strike, maturity_years, volatility, rate, dividend_yield=0.0):
    """Vectorized European call delta, exp(-qT) N(d1).

    Only strictly positive spot, strike, maturity, and volatility are accepted.
    In particular this routine assigns no arbitrary derivative to the payoff
    kink at expiry or the analogous zero-volatility exercise boundary.
    """
    s, k, t, v, r, q = np.broadcast_arrays(*[
        np.asarray(a, dtype=np.float64)
        for a in (spot, strike, maturity_years, volatility, rate, dividend_yield)
    ])
    if not all(np.isfinite(a).all() for a in (s, k, t, v, r, q)):
        raise ValueError("All parameters must be finite")
    if np.any(s <= 0) or np.any(k <= 0) or np.any(t <= 0) or np.any(v <= 0):
        raise ValueError("Delta requires spot,strike,maturity,volatility > 0")
    d1 = (np.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * np.sqrt(t))
    return np.exp(-q * t) * ndtr(d1)


def load_price_delta(checkpoint: Path, batch_size: int = 4096
                     ) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load once and return (C/K, delta) for physical [S/K, T, sigma] rows.

    Batches bound the autograd graph's memory use. Parameters are frozen;
    autograd.grad computes input derivatives without populating parameter .grad
    buffers. Predictions and derivatives are not clipped. No checkpoint or
    model parameter is updated. Saved normalization bounds define the scale.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    batch_size = int(batch_size)
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    bounds = np.asarray(saved["metadata"]["bounds"], dtype=np.float64)
    if (bounds.shape != (3, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 1] <= bounds[:, 0])):
        raise ValueError("Checkpoint must contain finite, increasing (3,2) bounds")
    net = Pricer()
    net.load_state_dict(saved["state_dict"])
    net.eval()
    net.requires_grad_(False)
    m_scaling_derivative = 2.0 / (bounds[0, 1] - bounds[0, 0])

    def predict(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        # _normalize also checks dimensions and finiteness, including empty X.
        normalized = _normalize(x, bounds)
        if np.any(x <= 0):
            raise ValueError("Require positive moneyness, maturity and volatility")
        if len(normalized) == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        prices, deltas = [], []
        # This intentionally enables input differentiation, even if a caller
        # has placed the audit inside torch.no_grad(). Each batch graph is freed.
        with torch.inference_mode(False), torch.enable_grad():
            for values in normalized.split(batch_size):
                inputs = values.detach().clone().requires_grad_(True)
                prediction = net(inputs)
                gradient = torch.autograd.grad(prediction.sum(), inputs,
                                               create_graph=False)[0]
                prices.append(prediction.detach().numpy().astype(np.float64))
                deltas.append(gradient[:, 0].detach().numpy().astype(np.float64)
                              * m_scaling_derivative)
        price, delta = np.concatenate(prices), np.concatenate(deltas)
        if not np.isfinite(price).all() or not np.isfinite(delta).all():
            raise ValueError("Neural model returned nonfinite price or delta")
        return price, delta

    return predict


def _error_metrics(error: np.ndarray) -> dict:
    error = np.asarray(error, dtype=np.float64)
    if len(error) == 0:
        return {name: None for name in ("mae", "rmse", "max_abs_error",
                                        "median_abs_error", "p90_abs_error",
                                        "p99_abs_error")}
    return {"mae": float(error.mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "max_abs_error": float(error.max()),
            "median_abs_error": float(np.median(error)),
            "p90_abs_error": float(np.quantile(error, 0.90)),
            "p99_abs_error": float(np.quantile(error, 0.99))}


def audit_greeks(data_dir: Path, run_dir: Path, output_dir: Path | None = None) -> dict:
    """Write held-out test/stress price and delta diagnostics, never search feedback.

    Output contains audit.json plus test.npz and stress.npz with individual
    observations. Price errors are in C/K units; delta errors are dimensionless.
    Existing output files are refused to preserve prior experiment artifacts.
    The delta [0,1] check applies to this repo's non-dividend-paying call setup.
    """
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir / "greeks"
    targets = [output_dir / name for name in ("audit.json", "test.npz", "stress.npz")]
    if any(path.exists() for path in targets):
        raise ValueError("Greeks audit output exists; choose a new output directory")
    checkpoint = run_dir / "model.pt"
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)["metadata"]
    if metadata.get("strike", STRIKE) != STRIKE or metadata.get("risk_free_rate", RATE) != RATE:
        raise ValueError("Audit requires the repo's fixed strike and risk-free rate")
    if metadata.get("dividend_yield", 0.0) != 0.0:
        raise ValueError("The [0,1] delta audit requires zero dividend yield")
    predict = load_price_delta(checkpoint)
    report = {"checkpoint_sha256": checkpoint_hash,
              "purpose": "Independent held-out diagnostic; never agent feedback",
              "delta_definition": "d(C/K)/d(S/K) = dC/dS",
              "price_error_units": "C/K (multiply by strike for price units)",
              "delta_error_units": "dimensionless",
              "strike": STRIKE, "rate": RATE, "dividend_yield": 0.0,
              "price_threshold_normalized": THRESHOLD,
              "delta_bound_tolerance": 1e-7, "splits": {}}
    arrays = {}
    for split in ("test", "stress"):
        with np.load(data_dir / f"{split}.npz", allow_pickle=False) as data:
            x = np.asarray(data["X"], dtype=np.float64)
        if len(x) == 0:
            raise ValueError("Audit sets must contain observations")
        pred_price, pred_delta = predict(x)
        ref_price = bs_call(x[:, 0] * STRIKE, STRIKE, x[:, 1], x[:, 2], RATE) / STRIKE
        ref_delta = bs_delta(x[:, 0] * STRIKE, STRIKE, x[:, 1], x[:, 2], RATE)
        price_error, delta_error = abs(pred_price - ref_price), abs(pred_delta - ref_delta)
        small_price_error = price_error <= THRESHOLD
        report["splits"][split] = {
            "n": len(x), "input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            "price": _error_metrics(price_error), "delta": _error_metrics(delta_error),
            "delta_outside_unit_interval_count": int(((pred_delta < -1e-7) | (pred_delta > 1 + 1e-7)).sum()),
            "price_accurate_subset": {
                "n": int(small_price_error.sum()),
                "fraction": float(small_price_error.mean()),
                "condition": "absolute price error in C/K <= price_threshold_normalized",
                "delta": _error_metrics(delta_error[small_price_error])}}
        arrays[split] = {"X": x, "pred_price": pred_price, "ref_price": ref_price,
                         "pred_delta": pred_delta, "ref_delta": ref_delta,
                         "price_error": price_error, "delta_error": delta_error}
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
        raise ValueError("Checkpoint changed during the Greeks audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, result in arrays.items():
        temporary = output_dir / f".{split}.tmp.npz"
        np.savez_compressed(temporary, **result)
        temporary.replace(output_dir / f"{split}.npz")
    temporary = output_dir / ".audit.tmp.json"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output_dir / "audit.json")
    return report
