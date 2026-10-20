#!/usr/bin/env python3
"""Build the calorimeter-clustering comparison table (markdown) from a YAML spec.

    python scripts/cluster_eval/cluster_table.py --workers 12
    python scripts/cluster_eval/cluster_table.py --transpose                       # datasets as rows

The spec has `columns` (what to evaluate: a dataset + a clustering; sources and result caching
in columns.py) and `rows` (which metrics to show, how to format them; `best: max|min` bolds the
best value among columns evaluated on the same dataset). Each column's aggregated metrics are
stored in `results_dir` (shared with cluster_plots.py), so editing `rows`, relabelling or
reordering only re-renders; adding or changing a column, or changing a preset, a default or a
metric definition (bump cluster_metrics.METRICS_VERSION), recomputes just the affected columns
(`--recompute` forces everything). A row whose key a source cannot provide shows "–".

Below the table the output carries a "Rows" glossary (per-row `expl` strings in the spec), a
"Data and versions" section (datasets, source paths, event ranges) built from the columns, and
a "Caveats" section (the spec's top-level `caveats` list).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import yaml

import mlpf.data.colliderml.cluster_eval as ce
from columns import column_result, dataset_id

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- table


def _missing(v):
    return v is None or (isinstance(v, float) and not np.isfinite(v))


def _fmt(v, fmt):
    if _missing(v):
        return "–"
    if fmt == "count":  # 2924, 17.3k, 129.5k
        return f"{v / 1000:.1f}k" if abs(v) >= 10000 else f"{v:.0f}"
    return fmt.format(v)


def render(spec, results, transpose=False):
    cols = spec["columns"]
    head = [f"**{c['label']}**" if c.get("bold") else c["label"] for c in cols]
    groups = {}
    for ci, c in enumerate(cols):
        groups.setdefault(dataset_id(c), []).append(ci)
    groups = [g for g in groups.values() if len(g) >= 2]  # only these have a "best" column
    body = []
    for row in spec["rows"]:
        keys = row["keys"] if isinstance(row["keys"], list) else [row["keys"]]
        fmt = row.get("fmt", "{:.3f}")
        m = re.search(r"\.(\d+)f", fmt)  # compare at DISPLAY precision: values that render equal tie
        ndig = int(m.group(1)) if m else None
        # bold mask: rows declaring `best: max|min` bold the best-valued column(s) within each
        # same-dataset group of >= 2 columns (ties share the bold; missing values are skipped)
        mask = [set() for _ in cols]
        if row.get("best") in ("max", "min"):
            pick = max if row["best"] == "max" else min
            for j, k in enumerate(keys):
                vals = {ci: r["metrics"].get(k) for ci, r in enumerate(results) if not _missing(r["metrics"].get(k))}
                vals = {ci: round(float(v), ndig) if ndig is not None else v for ci, v in vals.items()}
                for gcols in groups:
                    gvals = [vals[ci] for ci in gcols if ci in vals]
                    for ci in gcols:
                        if gvals and vals.get(ci) == pick(gvals):
                            mask[ci].add(j)
        pct = row.get("pct", [])
        pct = pct if isinstance(pct, list) else [pct]
        cells = []
        for ci, r in enumerate(results):
            vals = []
            for j, k in enumerate(keys):
                s = _fmt(r["metrics"].get(k), fmt)
                if j < len(pct) and s != "–" and r["metrics"].get(pct[j]) is not None:
                    s += f" ({_fmt(100 * r['metrics'][pct[j]], row.get('pct_fmt', '{:.2f}'))}%)"
                # best value: bold + green inline HTML (preview renders both; renders stripping
                # HTML, e.g. stricter sanitizers, degrade to bold)
                vals.append(f'<span style="color:green">**{s}**</span>' if j in mask[ci] else s)
            cells.append("–" if all(v == "–" for v in vals) else " / ".join(vals))
        body.append((row["label"], cells))
    if transpose:
        header = ["", *[lab for lab, _ in body]]
        lines = [header] + [[h, *[cells[j] for _, cells in body]] for j, h in enumerate(head)]
    else:
        lines = [["", *head]] + [[lab, *cells] for lab, cells in body]
    out = ["| " + " | ".join(lines[0]) + " |", "|" + "---|" * len(lines[0])]
    out += ["| " + " | ".join(line) + " |" for line in lines[1:]]
    return "\n".join(out)


def glossary(spec):
    """Row explanations: per-row `expl` strings from the spec (verbatim metric definitions:
    the header of mlpf/data/cluster_metrics.py)."""
    lines = [f"- **{row['label']}** — {row['expl']}" for row in spec["rows"] if row.get("expl")]
    return ["**Rows**", ""] + lines if lines else []


def dataset_notes(columns):
    """One bullet per distinct dataset behind the columns (config/display keys ignored)."""
    lines, seen = [], set()
    for col in columns:
        if dataset_id(col) in seen:
            continue
        seen.add(dataset_id(col))
        first, n = col.get("events", [None, None])
        span = f"events {first}–{first + n - 1}" if first is not None else "all events"
        if col["source"] == "colliderml":
            lines.append(
                f"- **ColliderML `{col['sample']}`** — our clusterers: Release 1 (HF `CERN/ColliderML-Release-1`,"
                f" revision `64c3d2f1`) `{ce.V1_SAMPLE[col['sample']]}` calo hits and tracks; Pandora columns: the"
                f" v20 pilot (NERSC m4958) re-reconstruction of the same events (DDCaloDigi cells, ACTS tracks,"
                f" Pandora clusters); {span}. Loaded from `{ce.V1_ROOT}` and `{ce.V20_ROOT}`."
            )
        elif col["source"] == "key4hep":
            lines.append(f"- **key4hep `{col['detector']}`** — EDM4hep ROOT `{col['file']}` (HF `jpata/particleflow`); {span}.")
        elif col["source"] == "cms_ntuple":
            lines.append(f"- **CMS PF ntuple** — `{col['file']}` (jpata.web.cern.ch, `scripts/fetch_test_data_cms.sh`); {span}.")
        elif col["source"] == "tfds":
            lines.append(f"- **TFDS** — `{col['data_dir']}`, split `{col['split']}`; {span}.")
    return ["**Data and versions**", ""] + lines if lines else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=str(HERE / "cluster_table.yaml"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--recompute", action="store_true", help="ignore stored column results")
    ap.add_argument("--transpose", action="store_true", help="one row per column entry (datasets as rows)")
    ap.add_argument("--out", default=None, help="also write the markdown table here")
    args = ap.parse_args()
    spec = yaml.safe_load(Path(args.spec).read_text())
    defaults = spec.get("defaults", {})
    results = [column_result(c, defaults, args.workers, args.recompute) for c in spec["columns"]]
    table = render(spec, results, args.transpose or spec.get("transpose", False))
    notes = [f"- {c['label']}: {r['n_events']} events" for c, r in zip(spec["columns"], results)]
    caveats = ["**Caveats**", ""] + [f"- {c}" for c in spec["caveats"]] if spec.get("caveats") else []
    sections = ["\n".join(notes)] + ["\n".join(s) for s in (glossary(spec), dataset_notes(spec["columns"]), caveats) if s]
    text = table + "\n\n" + "\n\n".join(sections) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
