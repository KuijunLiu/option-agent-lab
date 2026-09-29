"""Budget-accounted v2 baselines for a frozen neural pricing model.

All methods pay for the same initial 128 Sobol points. The financial stress
schedule is fixed in advance; DE alone uses measured initial errors to select
its smaller population. Population re-evaluations are charged, not reused for
free. These searches optimize price error; delta diagnostics are separate.
"""

import re
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from .data import BOUNDS, THRESHOLD, sample_points
from .evaluation import concatenate, evaluate, summarize
from .model import load_predictor
from .search import checkpoint_hash, write_json


COMMON_INITIAL_POINTS = 128
METHODS = ("random", "sobol", "financial_stress", "de")

# Financial priors fixed before the new comparisons; this schedule never adapts
# to a compared run's observations. The existing public demo is development data.
# At the default budget, every region receives one 128-point batch. Larger
# budgets cycle through this order; smaller budgets use its initial prefix.
FINANCIAL_STRESS_REGIONS = (
    {"name": "short_expiry_atm",
     "reason": "Near-expiry payoffs change rapidly around the strike.",
     "bounds": [[.95, 1.05], [1 / 365, 7 / 365], [.05, .8]]},
    {"name": "short_expiry_otm",
     "reason": "Probe short-expiry out-of-the-money prices toward the domain edge.",
     "bounds": [[.5, .95], [1 / 365, 7 / 365], [.05, .8]]},
    {"name": "short_expiry_itm",
     "reason": "Probe short-expiry in-the-money prices toward the domain edge.",
     "bounds": [[1.05, 1.5], [1 / 365, 7 / 365], [.05, .8]]},
    {"name": "low_volatility_atm",
     "reason": "Low volatility sharpens the transition near the strike.",
     "bounds": [[.9, 1.1], [1 / 365, 1.], [.05, .15]]},
    {"name": "high_volatility_atm",
     "reason": "Cover the high-volatility edge and a broad near-strike region.",
     "bounds": [[.8, 1.2], [1 / 365, 1.], [.65, .8]]},
    {"name": "long_maturity_otm",
     "reason": "Cover long maturities jointly with low moneyness.",
     "bounds": [[.5, .8], [.75, 1.], [.05, .8]]},
    {"name": "long_maturity_itm",
     "reason": "Cover long maturities jointly with high moneyness.",
     "bounds": [[1.2, 1.5], [.75, 1.], [.05, .8]]},
)


def _stress_points(budget, seed):
    """Return a deterministic, outcome-independent stress schedule."""
    points = [sample_points(COMMON_INITIAL_POINTS, seed, "sobol")]
    batches = []
    remaining = budget - COMMON_INITIAL_POINTS
    index = 0
    while remaining:
        region = FINANCIAL_STRESS_REGIONS[index % len(FINANCIAL_STRESS_REGIONS)]
        n = min(COMMON_INITIAL_POINTS, remaining)
        sampler = "sobol" if not n & (n - 1) else "random"
        batch_seed = seed + index + 1
        points.append(sample_points(n, batch_seed, sampler, region["bounds"]))
        batches.append({"name": region["name"], "reason": region["reason"],
                        "bounds": region["bounds"], "n": n,
                        "sampler": sampler, "sampler_seed": batch_seed})
        remaining -= n
        index += 1
    return np.vstack(points), batches


def _de_population_indices(initial, population):
    """Select high-error points plus deterministic farthest-point coverage."""
    points, errors = initial["X"], initial["error"]
    selected = list(np.argsort(-errors, kind="stable")[:population // 2])
    normalized = (points - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
    nearest = np.min(np.sum((normalized[:, None, :] - normalized[selected]) ** 2,
                           axis=2), axis=1)
    nearest[selected] = -np.inf
    while len(selected) < population:
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(nearest, np.sum((normalized - normalized[index]) ** 2, axis=1))
        nearest[selected] = -np.inf
    return np.asarray(selected, dtype=int)


def _run_de(initial, predictor, budget, seed, population):
    observations = [initial]
    used = COMMON_INITIAL_POINTS
    indices = _de_population_indices(initial, population)
    population_points = initial["X"][indices]
    if len(np.unique(population_points, axis=0)) != population:
        raise RuntimeError("DE initial population must contain distinct points")

    def objective(transposed_x):
        nonlocal used
        x = np.asarray(transposed_x).T
        if used + len(x) > budget:
            raise RuntimeError("DE attempted to exceed the evaluation budget")
        result = evaluate(x, predictor)
        observations.append(result)
        used += len(x)
        return -result["error"]

    # SciPy charges one evaluation of the supplied initial population, then
    # one population per generation; maxiter excludes that first evaluation.
    maxiter = (budget - COMMON_INITIAL_POINTS) // population - 1
    optimization = differential_evolution(
        objective, BOUNDS, init=population_points, strategy="best1bin",
        maxiter=maxiter, popsize=population, polish=False, tol=0, atol=0,
        seed=seed, vectorized=True, updating="deferred", mutation=(.5, 1.),
        recombination=.7,
    )
    fill = budget - used
    if fill:
        observations.append(evaluate(sample_points(fill, seed + 20000), predictor))
    details = {
        "actual_population": len(population_points),
        "population_source_indices": indices.tolist(),
        "population_selection": "best half by initial error; half geometric farthest points",
        "population_reevaluations_charged": population,
        "maxiter": maxiter, "iterations_completed": int(optimization.nit),
        "objective_point_evaluations": used - COMMON_INITIAL_POINTS,
        "objective_batch_calls": int(optimization.nfev),
        "early_stop_fill_points": fill,
        "optimizer_message": str(optimization.message),
        "strategy": "best1bin", "mutation": [.5, 1.], "recombination": .7,
        "polish": False,
    }
    return concatenate(observations), details


def run_searches(run, budget=1024, seeds=(123, 124, 125), de_population=32,
                 output_name="benchmarks_v2"):
    """Execute four methods with equal point budgets and preserve observations.

    ``output_name`` must be new. A failure leaves the partial directory for
    inspection; subsequent runs must choose a fresh name. All evaluations,
    including repeat points and DE initialization, consume the budget. Timings
    exclude predictor loading and result-file writes, include reference prices,
    and are not comparable to LLM end-to-end latency without separately counting
    LLM waiting time.
    """
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
        raise ValueError("Budget must be a power of two >=256")
    budget = int(budget)
    if budget < 256 or budget & (budget - 1):
        raise ValueError("Budget must be a power of two >=256")
    if isinstance(de_population, bool) or de_population not in (16, 32, 64):
        raise ValueError("DE population must be 16, 32, or 64")
    de_population = int(de_population)
    seeds = tuple(seeds)
    if (not seeds or any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 0
                         for s in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("Seeds must be distinct nonnegative integers")
    seeds = tuple(int(s) for s in seeds)
    if not isinstance(output_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", output_name):
        raise ValueError("Output name must contain only letters, numbers, _ and -")
    run = Path(run)
    destination = run / output_name
    if destination.exists():
        raise ValueError("Output directory already exists; choose a new output_name")
    frozen_hash = checkpoint_hash(run)
    predictor = load_predictor(run / "model.pt")
    destination.mkdir()
    rows = []
    for seed in seeds:
        for method in METHODS:
            if checkpoint_hash(run) != frozen_hash:
                raise ValueError("Model checkpoint changed during benchmark")
            start = time.perf_counter()
            details = {}
            if method == "random":
                points = np.vstack([sample_points(COMMON_INITIAL_POINTS, seed, "sobol"),
                                    sample_points(budget - COMMON_INITIAL_POINTS, seed + 10000)])
                result = evaluate(points, predictor)
            elif method == "sobol":
                result = evaluate(sample_points(budget, seed, "sobol"), predictor)
            elif method == "financial_stress":
                points, batches = _stress_points(budget, seed)
                result = evaluate(points, predictor)
                details = {"schedule": batches, "uses_feedback": False}
            else:
                initial = evaluate(sample_points(COMMON_INITIAL_POINTS, seed, "sobol"), predictor)
                result, details = _run_de(initial, predictor, budget, seed, de_population)
            seconds = time.perf_counter() - start
            if len(result["X"]) != budget:
                raise RuntimeError("Search did not use exactly its declared point budget")
            if checkpoint_hash(run) != frozen_hash:
                raise ValueError("Model checkpoint changed during benchmark")
            row = {"method": method, "seed": seed, "seconds": seconds,
                   "details": details, **summarize(result)}
            np.savez_compressed(destination / f"{method}_{seed}.npz", **result)
            rows.append(row)
    summary = {"protocol": "frozen_pricer_search_v2", "budget": budget,
               "seeds": list(seeds), "checkpoint_sha256": frozen_hash,
               "threshold_normalized": THRESHOLD,
               "common_initial_points": COMMON_INITIAL_POINTS,
               "de_population": de_population,
               "financial_stress_regions": FINANCIAL_STRESS_REGIONS,
               "timing": "excludes predictor loading and file writes; includes numerical oracle",
               "runs": rows}
    write_json(destination / "summary.json", summary)
    return summary
