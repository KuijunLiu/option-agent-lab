"""Budget-matched black-box searches and a file-based agent interface."""

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from .data import BOUNDS, NAMES, THRESHOLD, sample_points
from .evaluation import concatenate, evaluate, summarize
from .model import load_predictor


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def checkpoint_hash(run):
    return hashlib.sha256((Path(run)/"model.pt").read_bytes()).hexdigest()


def run_baselines(run, budget=1024, seeds=(123,124,125,126,127)):
    if budget < 256 or budget & (budget-1):
        raise ValueError("Budget must be a power of two >=256")
    run = Path(run)
    predictor = load_predictor(run/"model.pt")
    destination = run/"baselines"
    destination.mkdir(exist_ok=True)
    rows = []
    for seed in seeds:
        initial = sample_points(128, seed, "sobol")
        for method in ["random", "sobol", "differential_evolution"]:
            start = time.perf_counter()
            if method == "random":
                x = np.vstack([initial, sample_points(budget-128, seed+10000)])
                result = evaluate(x, predictor)
            elif method == "sobol":
                result = evaluate(sample_points(budget, seed, "sobol"), predictor)
            else:
                observations = []
                def objective(transposed_x):
                    measured = evaluate(np.asarray(transposed_x).T, predictor)
                    observations.append(measured)
                    return -measured["error"]
                differential_evolution(objective, BOUNDS, init=initial,
                    maxiter=budget//128-1, polish=False, tol=0, atol=0,
                    seed=seed, vectorized=True, updating="deferred")
                used = sum(len(o["X"]) for o in observations)
                if used < budget:  # Count any fallback evaluations too.
                    observations.append(evaluate(sample_points(budget-used,seed+20000),predictor))
                result = concatenate(observations)
            if len(result["X"]) != budget:
                raise RuntimeError("Search exceeded its declared budget")
            row = {"method": method, "seed": seed, "seconds": time.perf_counter()-start,
                   **summarize(result)}
            np.savez_compressed(destination/f"{method}_{seed}.npz", **result)
            rows.append(row)
    write_json(destination/"summary.json", {"budget":budget,"checkpoint_sha256":checkpoint_hash(run),
        "threshold_normalized":THRESHOLD,"common_initial_points":128,"runs":rows})
    return rows


def session_path(run, name):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Session name may contain letters, numbers, _ and -")
    return Path(run)/"sessions"/name


def init_session(run, name, budget=1024, seed=123):
    if budget < 1:
        raise ValueError("Budget must be positive")
    folder = session_path(run, name)
    if folder.exists():
        raise ValueError("Session already exists; choose a new name (no silent budget reset)")
    state = {"budget":budget,"seed":seed,"used":0,"round":0,"history":[],
             "checkpoint_sha256":checkpoint_hash(run),"threshold_normalized":THRESHOLD}
    write_json(folder/"state.json",state)
    return session_status(run,name)


def session_status(run, name):
    folder = session_path(run,name)
    state = json.loads((folder/"state.json").read_text())
    result = {k:state[k] for k in ["budget","used","round","history","threshold_normalized"]}
    result.update(remaining=state["budget"]-state["used"],domain=dict(zip(NAMES,BOUNDS.tolist())))
    if state["round"]:
        observations = [dict(np.load(folder/f"round_{i:03d}.npz")) for i in range(1,state["round"]+1)]
        result["summary"] = summarize(concatenate(observations))
    else:
        result["summary"] = None
    return result


def step_session(run, name, proposal):
    folder = session_path(run,name)
    state = json.loads((folder/"state.json").read_text())
    if checkpoint_hash(run) != state["checkpoint_sha256"]:
        raise ValueError("Model checkpoint changed during session")
    if not isinstance(proposal, dict) or set(proposal) != {"bounds","n","reason"}:
        raise ValueError("Proposal needs exactly bounds, n, reason")
    n = proposal["n"]
    if isinstance(n,bool) or not isinstance(n,int) or not 1 <= n <= 128:
        raise ValueError("n must be an integer between 1 and 128")
    if n > state["budget"]-state["used"]:
        raise ValueError("Requested evaluation exceeds remaining budget")
    if not isinstance(proposal["reason"],str) or not proposal["reason"].strip():
        raise ValueError("Provide a brief reason for this region")
    if not isinstance(proposal["bounds"], dict) or set(proposal["bounds"]) != set(NAMES):
        raise ValueError("bounds must contain m, T, sigma")
    bounds = np.asarray([proposal["bounds"][key] for key in NAMES],dtype=float)
    if bounds.shape != (3,2) or not np.isfinite(bounds).all():
        raise ValueError("Invalid bounds shape or nonfinite value")
    if np.any(bounds[:,0] >= bounds[:,1]) or np.any(bounds[:,0]<BOUNDS[:,0]) or np.any(bounds[:,1]>BOUNDS[:,1]):
        raise ValueError("Region must be a nonempty subset of declared domain")
    # Each round gets a deterministic new scramble; the first full-domain batch
    # exactly matches the baselines' common initial 128 points when n=128.
    sampler = "sobol" if not (n & (n-1)) else "random"
    start = time.perf_counter()
    x = sample_points(n,state["seed"]+state["round"],sampler,bounds)
    result = evaluate(x,load_predictor(Path(run)/"model.pt"))
    index = state["round"]+1
    np.savez_compressed(folder/f"round_{index:03d}.npz",**result)
    state["used"] += n
    state["round"] = index
    state["history"].append({"round":index,"proposal":proposal,"sampler":sampler,
        "sampler_seed":state["seed"]+index-1,"evaluation_seconds":time.perf_counter()-start,
        "summary":summarize(result)})
    write_json(folder/"state.json",state)
    return session_status(run,name)
