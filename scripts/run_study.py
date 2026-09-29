#!/usr/bin/env python3
"""Prepare the locked study, run the real Codex comparisons, or refresh reports."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from option_agent_lab.study import prepare_study, run_agents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "agents", "report"], required=True)
    parser.add_argument("--study", type=Path, default=REPO / "studies/erdos_v2")
    parser.add_argument("--config", type=Path, default=REPO / "configs/erdos_v2.json")
    parser.add_argument("--model", help="Explicit account-available Codex model identifier")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    try:
        if args.phase == "prepare":
            result = prepare_study(args.study, args.config)
        elif args.phase == "agents":
            if args.timeout <= 0:
                raise ValueError("timeout must be positive")
            result = run_agents(args.study, args.model, args.timeout)
        else:
            from option_agent_lab.study_report import make_study_report
            result = make_study_report(args.study)
        print(json.dumps(result, indent=2))
        if args.phase == "agents" and any(r["status"] not in ["completed", "already_completed"] for r in result):
            return 1
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
