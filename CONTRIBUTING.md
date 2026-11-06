# Contributing to ExpressEval

Thanks for your interest. This is an early research prototype, so small, focused changes are easiest to review.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` elsewhere
pip install -e ".[dev]"       # or: pip install -r requirements.txt
pytest -q
```

## Pull requests

- Open an issue first for anything larger than a bug fix or doc change.
- Keep each PR to one topic and include or update tests.
- Be honest in docs: say whether a metric is a real model or a heuristic placeholder.
- Do not commit audio you do not have the right to share, or any credentials.

## Reporting security problems

See [SECURITY.md](SECURITY.md).
