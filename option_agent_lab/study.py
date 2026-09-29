"""A small locked experiment matrix; missing LLM runs remain missing."""

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import THRESHOLD, generate_data
from .model import train_model
from .search import checkpoint_hash, write_json

REPO = Path(__file__).resolve().parents[1]
CORE_FILES = [
    "option_agent_lab/data.py", "option_agent_lab/pricing.py",
    "option_agent_lab/model.py", "option_agent_lab/evaluation.py",
    "option_agent_lab/search.py", "option_agent_lab/benchmarks.py",
    "option_agent_lab/greeks.py", "option_agent_lab/agent_protocol.py",
    "option_agent_lab/study.py", "scripts/run_codex_agent.py",
    "scripts/run_study.py", "option_agent_lab/study_report.py",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_config(config):
    """Keep this version's comparison small and its call budgets unambiguous."""
    required = {"protocol_version", "development_run", "model_seeds", "data_seeds", "search_seeds",
                "development_search_seeds", "de_populations", "epochs", "train_n", "validation_n",
                "audit_n", "evaluation_budget", "initial_points", "batch_size", "conditions",
                "execution_order_seed", "primary_objective", "price_failure_threshold",
                "delta_diagnostic_threshold", "delta_role"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError("Configuration is missing required fields")
    for key in ["protocol_version", "development_run", "primary_objective", "delta_role"]:
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if config["evaluation_budget"] != 1024 or config["initial_points"] != 128 or config["batch_size"] != 128:
        raise ValueError("v2 protocol requires 1024 total points, 128 initial and 128 per batch")
    for key in ["model_seeds", "data_seeds", "search_seeds", "development_search_seeds"]:
        values = config[key]
        if not isinstance(values, list) or not values or any(type(v) is not int or v < 0 for v in values) or len(set(values)) != len(values):
            raise ValueError(f"{key} must contain distinct nonnegative integers")
    if len(config["model_seeds"]) != len(config["data_seeds"]):
        raise ValueError("Each model needs its own data seed")
    # Data generation uses four consecutive seeds for train/validation/test/stress.
    expanded = [v + i for v in config["data_seeds"] for i in range(4)]
    if len(set(expanded)) != len(expanded):
        raise ValueError("Data split seeds overlap between models")
    if config["conditions"] != ["no_feedback", "feedback"]:
        raise ValueError("Expected the two paired conditions in canonical order")
    if config["de_populations"] != [16, 32, 64]:
        raise ValueError("Development-only DE candidates must be [16,32,64]")
    if config["price_failure_threshold"] != THRESHOLD:
        raise ValueError("Configured price threshold disagrees with evaluator")
    if type(config["execution_order_seed"]) is not int or config["execution_order_seed"] < 0:
        raise ValueError("execution_order_seed must be a nonnegative integer")
    if type(config["delta_diagnostic_threshold"]) not in (int, float) or not math.isfinite(config["delta_diagnostic_threshold"]) or config["delta_diagnostic_threshold"] <= 0:
        raise ValueError("delta_diagnostic_threshold must be positive and finite")
    for key in ["epochs", "train_n", "validation_n", "audit_n"]:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")


def build_matrix(config):
    """Pair conditions within each network/seed, randomizing execution order."""
    rng = np.random.default_rng(config["execution_order_seed"])
    pairs = [(m, s) for m in config["model_seeds"] for s in config["search_seeds"]]
    rng.shuffle(pairs)
    rows = []
    for m, seed in pairs:
        for condition in rng.permutation(config["conditions"]):
            rows.append({"model_seed": m, "search_seed": seed, "condition": str(condition),
                         "run": f"models/model_{m}",
                         "session": f"v2_{condition}_s{seed}"})
    return rows


def prepare_study(study, config_path):
    from .benchmarks import run_searches
    from .greeks import audit_greeks
    import torch

    study, config_path = Path(study).resolve(), Path(config_path).resolve()
    config = read_json(config_path)
    validate_config(config)
    if study.exists():
        raise ValueError("Study directory exists; use a new --study to preserve results")
    development = (REPO / config["development_run"]).resolve()
    if not (development / "model.pt").is_file():
        raise FileNotFoundError("Development checkpoint is missing")
    torch.set_num_threads(2)
    study.mkdir(parents=True)
    original_sources = {name: digest(REPO / name) for name in CORE_FILES}
    write_json(study / "protocol.json", {"created_utc": utc_now(), "config": config,
        "role": "Local pre-specified protocol; not an externally registered study",
        "development_checkpoint_sha256": checkpoint_hash(development),
        "source_sha256": original_sources})
    write_json(study / "matrix.json", build_matrix(config))
    write_json(study / "environment.json", {"python": platform.python_version(),
        "platform": platform.platform(), "packages": {p: importlib.metadata.version(p)
        for p in ["numpy", "scipy", "torch", "matplotlib"]}, "training_device": "CPU"})

    dev = study / "development"
    dev.mkdir()
    shutil.copy2(development / "model.pt", dev / "model.pt")
    candidates = []
    for population in config["de_populations"]:
        outcome = run_searches(dev, budget=config["evaluation_budget"],
            seeds=config["development_search_seeds"], de_population=population,
            output_name=f"population_{population}")
        scores = [r["max_error_normalized"] for r in outcome["runs"] if r["method"] == "de"]
        if not scores:
            raise RuntimeError("No DE development scores returned")
        candidates.append({"population": population, "mean_max_error_normalized": float(np.mean(scores))})
    selected = sorted(candidates, key=lambda r: (-r["mean_max_error_normalized"], r["population"]))[0]["population"]
    write_json(study / "development_selection.json", {"candidates": candidates,
        "selected_population": selected,
        "rule": "Highest mean maximum pricing error on the old development model; tie -> smaller population",
        "selected_utc": utc_now()})
    print(f"Development-only DE selection: population {selected}", flush=True)

    for model_seed, data_seed in zip(config["model_seeds"], config["data_seeds"]):
        run = study / "models" / f"model_{model_seed}"
        data = study / "data" / f"model_{model_seed}"
        generate_data(data, seed=data_seed, train=config["train_n"],
                      validation=config["validation_n"], test=config["audit_n"])
        train_model(data, run, epochs=config["epochs"], seed=model_seed)
        run_searches(run, budget=config["evaluation_budget"], seeds=config["search_seeds"],
                     de_population=selected)
        audit_greeks(data, run)
        print(f"Completed numerical evaluation for model seed {model_seed}", flush=True)

    # This file locks the same evaluators, checkpoints and schedules for later agent runs.
    current_sources = {name: digest(REPO / name) for name in CORE_FILES}
    if current_sources != original_sources:
        raise ValueError("Experiment source changed during preparation; use a fresh study")
    locked = sorted(p for p in study.rglob("*") if p.is_file())
    write_json(study / "artifact_lock.json", {
        "created_utc": utc_now(),
        "source_sha256": original_sources,
        "artifact_sha256": {str(p.relative_to(study)): digest(p) for p in locked}})
    return {"study": str(study), "numerical_status": "completed", "agent_status": "pending",
            "planned_llm_calls": len(build_matrix(config)) * 7}


def verify_lock(study):
    study = Path(study).resolve()
    lock = read_json(study / "artifact_lock.json")
    for name, expected in lock["source_sha256"].items():
        if digest(REPO / name) != expected:
            raise ValueError(f"Experiment source changed: {name}; use a new study")
    for name, expected in lock["artifact_sha256"].items():
        if digest(study / name) != expected:
            raise ValueError(f"Experiment artifact changed: {name}")
    return lock


def run_agents(study, model, timeout=240):
    study = Path(study).resolve()
    verify_lock(study)
    if not model or not model.strip():
        raise ValueError("An explicit --model is required for the paired LLM study")
    if shutil.which("codex") is None:
        raise ValueError("Codex CLI is unavailable; numerical results remain valid, LLM comparisons remain pending")
    config_file = study / "agent_execution.json"
    execution = {"model": model, "timeout_seconds": timeout}
    if config_file.exists():
        previous = read_json(config_file)
        if any(previous[k] != v for k, v in execution.items()):
            raise ValueError("LLM model/timeout changed within study; use a new study")
    else:
        write_json(config_file, {**execution, "started_utc": utc_now()})
    records = []
    for row in read_json(study / "matrix.json"):
        run = study / row["run"]
        state_path = run / "sessions" / row["session"] / "state.json"
        log_dir = run / "codex-agent-logs" / row["session"]
        if state_path.exists() or log_dir.exists():
            checked = inspect_agent_session(study, row)
            records.append({**row, "status": "already_completed" if checked["status"] == "completed" else "partial_or_failed_not_retried",
                            "inspection_status": checked["status"]})
            continue
        command = [sys.executable, str(REPO / "scripts/run_codex_agent.py"),
            "--run", str(run), "--name", row["session"], "--condition", row["condition"],
            "--budget", "1024", "--seed", str(row["search_seed"]),
            "--max-rounds", "8", "--model", model, "--timeout", str(timeout)]
        # Each process logs its own visible prompts, actual outputs and failures.
        result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
        out = study / "controller_logs"
        out.mkdir(exist_ok=True)
        stem = f"model{row['model_seed']}_{row['session']}"
        (out / f"{stem}.stdout.txt").write_text(result.stdout)
        (out / f"{stem}.stderr.txt").write_text(result.stderr)
        checked = inspect_agent_session(study, row)
        status = "completed" if result.returncode == 0 and checked["status"] == "completed" else "failed" if result.returncode else "invalid"
        records.append({**row, "status": status, "inspection_status": checked["status"],
                        "returncode": result.returncode})
        write_json(study / "agent_execution_status.json", records)
        if status != "completed":
            # Avoid consuming many calls on an authentication/quota/systemic error.
            print(f"Stopped on {stem}; inspect preserved logs. No silent retries.", flush=True)
            break
    write_json(study / "agent_execution_status.json", records)
    return records


def inspect_agent_session(study, spec):
    """Validate logged experiment completion; failed/invalid runs have no scores."""
    from .evaluation import concatenate, summarize
    from .data import sample_points
    study = Path(study)
    run = study / spec["run"]
    folder = run / "sessions" / spec["session"]
    logs = run / "codex-agent-logs" / spec["session"]
    row = {**spec, "status": "pending"}
    if (logs / "failure.json").exists():
        return {**row, "status": "failed"}
    try:
        if not (folder / "state.json").exists():
            return {**row, "status": "incomplete" if logs.exists() else "pending"}
        else:
            state = read_json(folder / "state.json")
            row.update(used=state["used"], status="incomplete")
            if not (logs / "final-status.json").exists():
                return row
            meta = read_json(logs / "metadata.json")
            final = read_json(logs / "final-status.json")
            model = read_json(study / "agent_execution.json")["model"]
            checks = [state["used"] == state["budget"] == 1024, state["round"] == 8,
                state["seed"] == spec["search_seed"],
                state["checkpoint_sha256"] == checkpoint_hash(run),
                meta.get("condition") == spec["condition"], meta.get("seed") == spec["search_seed"],
                meta.get("budget") == 1024, meta.get("model_requested") == model,
                meta.get("status") == "completed", meta.get("llm_calls_attempted") == 7,
                final.get("remaining") == 0, final.get("used") == final.get("budget") == 1024]
            if not all(checks):
                raise ValueError("Completion metadata disagrees with the locked matrix")
            batches = [dict(np.load(folder / f"round_{i:03d}.npz")) for i in range(1, 9)]
            if any(len(r["X"]) != 128 or len(r["error"]) != 128 or not np.isfinite(r["error"]).all() for r in batches):
                raise ValueError("Invalid or incomplete numerical observations")
            initial = sample_points(128, spec["search_seed"], "sobol")
            if not np.allclose(batches[0]["X"], initial, rtol=0, atol=1e-15):
                raise ValueError("Initial observations do not match common starting points")
            measured = concatenate(batches)
            row.update(status="completed", **summarize(measured))
            return row
    except (ValueError, KeyError, OSError, TypeError) as exc:
        return {**spec, "status": "invalid", "reason": str(exc)}


def collect_agent_results(study):
    """A pair is scoreable only after both logged CLI sessions complete."""
    return [inspect_agent_session(study, spec) for spec in read_json(Path(study) / "matrix.json")]


def save_rows(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
