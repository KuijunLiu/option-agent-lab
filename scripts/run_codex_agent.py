#!/usr/bin/env python3
"""Run a call-matched feedback/no-feedback experiment through local Codex CLI."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from option_agent_lab.agent_protocol import (  # noqa: E402
    CONDITIONS, PROMPT_TEMPLATE, PROMPT_VERSION, PROPOSAL_SCHEMA,
    build_prompt, call_plan, make_agent_view, validate_proposal,
)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def backend(args, action, *extra):
    command = [sys.executable, "-m", "option_agent_lab", action,
               "--run", str(args.run), "--name", args.name, *extra]
    result = subprocess.run(command, cwd=REPO, text=True, capture_output=True,
                            timeout=300, check=False)
    if result.returncode:
        raise RuntimeError(f"Backend {action} failed:\n{result.stderr}\n{result.stdout}")
    return result.stdout


def cli_observations(stdout):
    """Read documented JSONL completion usage if available; do not estimate it."""
    usage_records, tool_events, models = [], [], []
    forbidden_items = {"command_execution", "mcp_tool_call", "web_search", "file_change",
                       "collab_agent_tool_call"}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            usage_records.append(event["usage"])
        if isinstance(event.get("model"), str):
            models.append(event["model"])
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") in forbidden_items:
            tool_events.append({"event_type": event.get("type"), "item_type": item["type"]})
    return {
        "token_usage_status": "reported_by_cli" if usage_records else "unknown_not_reported",
        "token_usage_records": usage_records or None,
        "model_observed_in_events": sorted(set(models)) or None,
        "detected_tool_events": tool_events,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/demo"))
    parser.add_argument("--name", default="codex-" + dt.datetime.now(dt.timezone.utc)
                        .strftime("%Y%m%dT%H%M%S"))
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-rounds", type=int, default=8,
                        help="Maximum rounds INCLUDING the controller-fixed initial batch.")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--model", help="Optional locally; formal comparisons require an explicit model.")
    parser.add_argument("--condition", choices=CONDITIONS, default="feedback")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("--name must contain only letters, digits, _ or -")
    try:
        plan = call_plan(args.budget, args.max_rounds)
    except ValueError as error:
        parser.error(str(error))
    if args.timeout < 1:
        parser.error("timeout must be positive")
    codex = shutil.which("codex")
    if codex is None:
        parser.error("Codex CLI not found. Install it and sign in first; see docs/CODEX.md. No agent experiment ran.")
    args.run = args.run.resolve()
    if not (args.run / "model.pt").is_file():
        parser.error("No frozen model.pt in --run; train a model first.")
    logs = args.run / "codex-agent-logs" / args.name
    try:
        logs.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("This log directory already exists; choose a new --name.")
    metadata = {
        "status": "running", "started_at_utc": utc_now(),
        "condition": args.condition, "codex_version": None,
        "model_requested": args.model,
        "model_selection": "explicit" if args.model else "local_default_unpinned",
        "budget": args.budget, "seed": args.seed,
        "max_rounds": args.max_rounds, "batch_counts_including_initial": plan,
        "llm_calls_planned": len(plan) - 1, "llm_calls_attempted": 0,
        "timeout_seconds": args.timeout, "prompt_version": PROMPT_VERSION,
        "prompt_template_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest(),
        "token_usage_status": "per_call_logs_or_unknown_not_reported",
        "information_boundary": "filtered_prompt_only_not_filesystem_security_isolation",
    }
    save(logs / "metadata.json", metadata)
    try:
        version = subprocess.run([codex, "--version"], text=True,
                                 capture_output=True, timeout=15, check=True)
        metadata["codex_version"] = version.stdout.strip()
        save(logs / "metadata.json", metadata)
        backend(args, "session-init", "--budget", str(args.budget), "--seed", str(args.seed))
        for index, count in enumerate(plan):
            status = json.loads(backend(args, "session-status"))
            # Full operator audit log; never interpolate it into an LLM prompt.
            save(logs / f"round-{index:02d}-backend-status.json", status)
            if status["remaining"] < count:
                raise RuntimeError("Backend remaining budget disagrees with the fixed call plan.")
            if index == 0:
                initial = logs / "round-00-initial-proposal.json"
                save(initial, {"bounds": status["domain"], "n": count,
                               "reason": "Controller-fixed common full-domain initial batch."})
                backend(args, "session-step", "--proposal", str(initial))
                print("Completed common initial batch; no LLM decision used.", flush=True)
                continue
            view = make_agent_view(status, args.condition)
            save(logs / f"round-{index:02d}-visible-state.json", view)
            prompt = build_prompt(view, count)
            (logs / f"round-{index:02d}-prompt.txt").write_text(prompt)
            call_record = {
                "condition": args.condition, "decision": index, "requested_count": count,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "visible_state_sha256": hashlib.sha256(json.dumps(view, sort_keys=True).encode()).hexdigest(),
                "status": "running", "started_at_utc": utc_now(),
                "token_usage_status": "unknown_not_reported", "token_usage_records": None,
            }
            record_path = logs / f"round-{index:02d}-call.json"
            save(record_path, call_record)
            with tempfile.TemporaryDirectory(prefix="option-agent-context-") as temporary:
                work = Path(temporary)
                schema, proposal = work / "schema.json", work / "proposal.json"
                save(schema, PROPOSAL_SCHEMA)
                command = [codex, "exec", "--sandbox", "read-only", "--json",
                           "--skip-git-repo-check", "--output-schema", str(schema),
                           "-o", str(proposal)]
                if args.model:
                    command += ["--model", args.model]
                command += ["-"]
                save(logs / f"round-{index:02d}-command.json", command)
                metadata["llm_calls_attempted"] += 1
                save(logs / "metadata.json", metadata)
                started = time.perf_counter()
                stdout = stderr = ""
                try:
                    result = subprocess.run(command, input=prompt, cwd=work, text=True,
                                            capture_output=True, timeout=args.timeout, check=False)
                    stdout, stderr = result.stdout, result.stderr
                    call_record["returncode"] = result.returncode
                    observations = cli_observations(stdout)
                    call_record.update(observations)
                    if result.returncode:
                        raise RuntimeError(f"Codex exited {result.returncode}; inspect round logs.")
                    if observations["detected_tool_events"]:
                        raise RuntimeError("Codex used a tool despite the no-tools protocol; exclude this run.")
                    raw_proposal = proposal.read_text()
                    (logs / f"round-{index:02d}-raw-proposal.txt").write_text(raw_proposal)
                    value = json.loads(raw_proposal)
                    validate_proposal(value, count, status["domain"])
                    target = logs / f"round-{index:02d}-proposal.json"
                    save(target, value)
                    call_record["status"] = "valid_proposal"
                except subprocess.TimeoutExpired as error:
                    stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout or ""
                    stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr or ""
                    call_record.update(cli_observations(stdout))
                    call_record["status"] = "timeout"
                    raise
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                    call_record["status"] = "failed"
                    call_record["failure_type"] = type(error).__name__
                    call_record["failure_message"] = str(error)
                    raise
                finally:
                    call_record["wall_seconds"] = time.perf_counter() - started
                    call_record["finished_at_utc"] = utc_now()
                    save(record_path, call_record)
                    (logs / f"round-{index:02d}-stdout.txt").write_text(stdout)
                    (logs / f"round-{index:02d}-stderr.txt").write_text(stderr)
            backend(args, "session-step", "--proposal", str(target))
            print(f"Completed LLM decision {index}/{len(plan) - 1}; logs: {target}", flush=True)
        final = json.loads(backend(args, "session-status"))
        save(logs / "final-status.json", final)
        if final["remaining"]:
            raise RuntimeError("Round limit reached with unused budget; no automatic fallback.")
        metadata.update(status="completed", finished_at_utc=utc_now(), evaluated_points=final["used"])
        save(logs / "metadata.json", metadata)
        print(json.dumps(final, indent=2, ensure_ascii=False))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        failure = {"type": type(error).__name__, "message": str(error), "at_utc": utc_now(),
                   "condition": args.condition, "llm_calls_attempted": metadata["llm_calls_attempted"],
                   "no_fallback_used": True}
        save(logs / "failure.json", failure)
        metadata.update(status="failed", finished_at_utc=utc_now(), failure=failure)
        save(logs / "metadata.json", metadata)
        print(f"Agent run stopped: {error}\nLogs: {logs}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
