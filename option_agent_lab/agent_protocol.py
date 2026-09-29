"""Explicit information boundary for the paired LLM feedback experiment.

The controller may retain full numerical state. Only ``make_agent_view`` output
may enter the LLM prompt; in the control condition later outcomes are absent.
"""

from copy import deepcopy
import json
import math


CONDITIONS = ("feedback", "no_feedback")
PROMPT_VERSION = "price-error-feedback-v2"
PROPOSAL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["bounds", "n", "reason"],
    "properties": {
        "bounds": {
            "type": "object", "additionalProperties": False,
            "required": ["m", "T", "sigma"],
            "properties": {
                key: {"type": "array", "items": {"type": "number"},
                      "minItems": 2, "maxItems": 2}
                for key in ("m", "T", "sigma")
            },
        },
        "n": {"type": "integer", "minimum": 1, "maximum": 128},
        "reason": {"type": "string", "minLength": 1},
    },
}

PROMPT_TEMPLATE = """You select rectangular test regions for a frozen neural European call pricer.
Your sole optimization objective is to discover the largest absolute pricing error
normalized by strike: abs(neural price - Black-Scholes reference price) / K.
The reference is a non-dividend-paying European call with K=100 and r=0.02.
m is S/K, T is remaining maturity in years, and sigma is annualized volatility.
Coverage, delta errors, and bound violations are diagnostics, not optimization objectives.
Use only the observations supplied below. A null summary means that outcome is
unavailable to you; do not invent its value. Previous proposals remain visible.
Return only one JSON proposal matching the schema. Do not use tools, read other
files, execute tests, access the internet, or change the evaluator. Python will
separately sample the proposed region using scrambled Sobol points when n is a
power of two, and uniform random points otherwise. Every lower bound must be
strictly below its upper bound and both must stay within the declared domain.
Request exactly {count} points. Briefly justify your choice in reason.
VISIBLE EXPERIMENT STATE:
{state}
"""

# Whitelists prevent future evaluator additions from silently entering a prompt.
# Delta diagnostics stay in the evaluator's report, outside both agent prompts.
_SUMMARY_FIELDS = (
    "n", "mae_normalized", "rmse_normalized", "max_error_normalized",
    "p99_error_normalized", "failures", "failure_fraction", "failure_regions",
    "bound_violations",
)
_POINT_FIELDS = (
    "m", "T", "sigma", "error_normalized", "reference_normalized",
    "prediction_normalized",
)


def _summary_view(summary):
    if summary is None:
        return None
    result = {key: deepcopy(summary[key]) for key in _SUMMARY_FIELDS if key in summary}
    result["top_points"] = [
        {key: deepcopy(point[key]) for key in _POINT_FIELDS if key in point}
        for point in summary.get("top_points", [])
    ]
    return result


def make_agent_view(status, condition):
    """Return a new, allowlisted prompt payload without mutating backend state.

    Both conditions have identical keys/history structure. The first fixed-batch
    summary is visible to both; all later summaries are null in ``no_feedback``.
    Cumulative backend summaries, timings, sampler internals and extra metadata
    are never forwarded. Proposal reasons are the agent's own earlier text.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    view = {key: deepcopy(status[key]) for key in (
        "budget", "used", "remaining", "round", "threshold_normalized",
    )}
    view["domain"] = {key: deepcopy(status["domain"][key]) for key in ("m", "T", "sigma")}
    view["history"] = []
    for index, record in enumerate(status["history"]):
        proposal = record["proposal"]
        visible_proposal = {
            "bounds": {key: deepcopy(proposal["bounds"][key]) for key in ("m", "T", "sigma")},
            "n": proposal["n"], "reason": proposal["reason"],
        }
        view["history"].append({
            "round": record["round"], "proposal": visible_proposal,
            "summary": _summary_view(record["summary"])
                if condition == "feedback" or index == 0 else None,
        })
    return view


def build_prompt(view, count):
    """Use exactly the same instructions for the two information conditions."""
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 128:
        raise ValueError("count must be an integer between 1 and 128")
    return PROMPT_TEMPLATE.format(count=count, state=json.dumps(view, ensure_ascii=False, allow_nan=False))


def call_plan(budget=1024, max_rounds=8):
    """Point counts including initialization; default leaves exactly seven LLM calls."""
    for name, value in (("budget", budget), ("max_rounds", max_rounds)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if budget > 128 * max_rounds:
        raise ValueError("budget exceeds 128 * max-rounds")
    return [min(128, budget - start) for start in range(0, budget, 128)]


def validate_proposal(proposal, count, domain):
    """Fail before evaluation if an LLM changes the matched batch size or bounds."""
    if not isinstance(proposal, dict) or set(proposal) != {"bounds", "n", "reason"}:
        raise ValueError("Proposal needs exactly bounds, n, reason")
    n = proposal["n"]
    if isinstance(n, bool) or not isinstance(n, int) or n != count:
        raise ValueError(f"Proposal must request exactly {count} points")
    if not 1 <= count <= 128:
        raise ValueError("Requested count must be between 1 and 128")
    if not isinstance(proposal["reason"], str) or not proposal["reason"].strip():
        raise ValueError("Proposal must give a nonempty reason")
    bounds = proposal["bounds"]
    if not isinstance(bounds, dict) or set(bounds) != {"m", "T", "sigma"}:
        raise ValueError("bounds must contain exactly m, T, sigma")
    for key in ("m", "T", "sigma"):
        pair = bounds[key]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Invalid bounds for {key}")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in pair):
            raise ValueError(f"Nonfinite or nonnumeric bounds for {key}")
        lower, upper = pair
        if not domain[key][0] <= lower < upper <= domain[key][1]:
            raise ValueError(f"Region for {key} is outside the declared domain or empty")
