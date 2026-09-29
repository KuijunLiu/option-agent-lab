# Working on this project

This is an educational neural-pricer validation experiment, not a market-pricing service.

- Read README.md and docs/DATA_SOURCES.md before changing the experiment.
- Keep theoretical reference prices separate from neural predictions. Never replace measured results with invented or mocked agent outputs.
- Preserve the QuantLib reference cases, source attribution and third-party license.
- Use a new run/session name for new experiments; do not overwrite the delivered demo.
- Freeze the neural checkpoint during a search. Count all evaluated parameter points, including initialization, against the declared budget.
- Do not feed audit sets or competing search results into an agent run. Saved demo data is public; formal studies need fresh held-out networks.
- After numerical or session-control changes run `python -m pytest -q`.
- The local Codex bridge is optional and requires an authenticated Codex CLI. Do not substitute fabricated results when it is unavailable.
