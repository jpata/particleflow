# Contributing to the documentation

The documentation uses MyST Markdown and Jupyter Book 2. Keep pages readable as ordinary Markdown so they remain useful in GitHub reviews.

## Build locally

From the repository root, enter the documentation directory while keeping the repository's uv project active:

```bash
cd docs
uv run --project .. jupyter-book build --html --strict
```

For a live preview:

```bash
cd docs
uv run --project .. jupyter-book start
```

Generated output is written below `docs/_build/` and is not committed.

## Writing style

- Start from the user's goal, then introduce the responsible command or code.
- Use short sentences and define detector or machine-learning terms at first use.
- Give every runnable example its prerequisites, inputs, outputs, and success condition.
- Use canonical command-line option names, even when `argparse` accepts abbreviations.
- Separate a smoke test from physics validation.
- Do not present a paper's result as a guarantee for the current branch.

## Physics claims

Link physics and performance claims directly to a paper or official public result. State the detector, sample, and comparison in the same paragraph. Prefer a plain-language summary to an unexplained metric.

If a number depends on a particular code or dataset release, link the archived artifact. The [publications page](../science/publications.md) is the central map from each study to its paper, code, and data.

## Commands and configuration

Use these sources in order:

1. `mlpf/conf.py` for types and base defaults;
2. `particleflow_spec.yaml` for detector and model recipes;
3. executable `--help` output for the current command spelling;
4. maintained scripts for complete examples.

Site-specific scripts are useful examples, but do not describe their paths or scheduler settings as universal defaults.

## Updating status pages

Date changes to [Current capabilities](../science/capabilities.md) and the [Roadmap](../science/roadmap.md). A feature should move to **Supported** only when there is a maintained configuration, a tested user path, and enough documentation to run it.
