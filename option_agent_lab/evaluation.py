"""Deterministic evaluation: LLM text never supplies prices or scores."""

import numpy as np

from .data import BOUNDS, RATE, REGION_EDGES, THRESHOLD, labels


def evaluate(x, predictor):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) == 0:
        raise ValueError("Expected a nonempty (n,3) array")
    if not np.isfinite(x).all() or np.any(x < BOUNDS[:,0]) or np.any(x > BOUNDS[:,1]):
        raise ValueError("Test points must lie in the declared domain")
    ref, pred = labels(x), np.asarray(predictor(x), dtype=float)
    if pred.shape != ref.shape or not np.isfinite(pred).all():
        raise ValueError("Predictor returned invalid prices")
    error = np.abs(pred - ref)
    lower = np.maximum(x[:,0] - np.exp(-RATE*x[:,1]), 0)
    violation = (pred < lower - 1e-7) | (pred > x[:,0] + 1e-7)
    bins = np.column_stack([np.clip(np.searchsorted(edges, x[:,i], side="right")-1, 0, len(edges)-2) for i,edges in enumerate(REGION_EDGES)])
    return {"X": x, "reference": ref, "prediction": pred, "error": error,
            "violation": violation, "bins": bins}


def summarize(result):
    err = result["error"]
    failed = err > THRESHOLD
    top = np.argsort(err)[-5:][::-1]
    return {"n": len(err), "mae_normalized": float(err.mean()),
            "rmse_normalized": float(np.sqrt(np.mean(err**2))),
            "max_error_normalized": float(err.max()), "p99_error_normalized": float(np.quantile(err,.99)),
            "failures": int(failed.sum()), "failure_fraction": float(failed.mean()),
            "failure_regions": len(set(map(tuple, result["bins"][failed]))),
            "bound_violations": int(result["violation"].sum()),
            "top_points": [{"m": float(result["X"][i,0]), "T": float(result["X"][i,1]),
                            "sigma": float(result["X"][i,2]), "error_normalized": float(err[i]),
                            "reference_normalized": float(result["reference"][i]),
                            "prediction_normalized": float(result["prediction"][i])} for i in top]}


def concatenate(results):
    return {key: np.concatenate([r[key] for r in results]) for key in results[0]}
