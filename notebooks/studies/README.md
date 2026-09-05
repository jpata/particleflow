# Reproducible notebook studies

Use this directory for dated, reproducible analyses and presentations. A study
should remain renderable after its source experiment directory has been moved or
deleted.

## Directory layout

Name a study `YYYYMMDD_short_description` and use this layout:

```text
YYYYMMDD_short_description/
├── README.md
├── study_name.ipynb
├── render.sh
├── inputs/
│   └── <descriptive run or data name>/
│       ├── history/
│       └── tensorboard/
│           ├── train/
│           └── valid/
└── output/
    ├── study_name.executed.ipynb
    ├── study_name.slides.html
    ├── study_name.slides.pdf
    └── generated figures
```

Use descriptive names such as `elementwise`, `set`, or `pf_baseline` inside
`inputs/`; do not retain timestamp-heavy experiment directory names as the only
description of a run.

## Archive the inputs

Copy the inputs needed to reproduce the notebook into the study directory. Do
not use symlinks or make the notebook read mutable paths under `experiments/`,
scratch storage, or a user's home directory.

For a training comparison, the useful archive normally includes:

- history JSON files used for curves and tables;
- the rank-zero training log used for timing, memory, or loss calibration;
- TensorBoard event files from both the training and validation writers, even
  when the current notebook reads only one of them;
- the training configuration, hyperparameters, scenario manifest, and resolved
  `particleflow_spec.yaml`;
- the final validation plots or compact source data used in the presentation.

Copy files without modifying their contents. Renaming a rank-zero log to
`train.log` is fine when documented by the directory structure. Preserve the
complete history and final plot directory rather than selecting only individual
points or panels; this leaves enough context for later follow-up plots.

Keep training and validation TensorBoard event files in separate directories,
preferably `tensorboard/train/` and `tensorboard/valid/`. Do not combine the two
writers in one directory: they can use overlapping tag names, and TensorBoard or
`EventAccumulator` may otherwise merge them into a misleading scalar history.

Avoid copying large derived artifacts that are not required by the notebook,
especially checkpoints, per-checkpoint prediction parquet files, and full local
dataset caches. If one of these is essential, include only the necessary subset
and explain the choice and size in the study README.

A study may reuse a baseline already archived by an earlier study instead of
duplicating it. Point only to that dated study, and document the dependency in
the new README. New campaign inputs must still be archived under the new study.

## Make the notebook portable

Resolve paths from the study directory and write generated files only under
`output/`. A typical setup cell is:

```python
from pathlib import Path

try:
    STUDY_DIR = Path.cwd()
    if not (STUDY_DIR / "README.md").is_file():
        STUDY_DIR = Path("notebooks/studies/YYYYMMDD_short_description").resolve()
except NameError:
    STUDY_DIR = Path.cwd()

INPUT_DIR = STUDY_DIR / "inputs"
OUTPUT_DIR = STUDY_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
```

Do not embed absolute paths. Keep exploratory code out of the final slide export
with notebook tags such as `hide-input`, while retaining it in the source and
executed notebooks.

## Document provenance and limitations

The study README should record:

- the question being studied and the compared runs;
- exact run identifiers, timestamps, random seeds, dataset names and versions;
- hardware, batch size, training length, and the selected validation step;
- what is present under `inputs/` and what was intentionally excluded;
- dependencies on earlier archived studies;
- commands needed to render the notebook or regenerate validation inputs;
- known comparison limitations, incomplete jobs, or differences in sample size.

Keep important caveats visible in the slides as well as in the README.

## Render and verify

Provide an executable `render.sh` that can be called from the repository root.
It should execute the source notebook, export HTML slides, and export a PDF when
Chrome or Chromium is available.

Before considering the study complete:

1. Confirm that the source notebook is valid JSON.
2. Search the notebook for references to `experiments/`, absolute paths, and
   temporary locations.
3. Execute `render.sh` using only the archived input paths.
4. Check that every expected figure, the executed notebook, HTML, and PDF were
   regenerated successfully.
5. Open or render the PDF and inspect the first, representative middle, and last
   pages for clipping, blank pages, or missing images.
6. Review the final archive size and verify that no checkpoint, dataset cache, or
   unintended prediction dump was included.
7. Confirm that every archived training run has nonempty, separately stored
   TensorBoard event files for both training and validation.

## Upload to the private study bucket

The off-machine archive is the private Hugging Face bucket
[`jpata/particleflow-studies`](https://huggingface.co/buckets/jpata/particleflow-studies).
Access requires a Hugging Face account authorized for the bucket and an
authenticated `hf` CLI. Keep each dated directory at the same top-level name in
the bucket, and upload this README along with the studies.

From the repository root, prepare and review a non-deleting plan for the dated
study directory, apply it, and copy this README separately:

```bash
.venv/bin/hf auth whoami
.venv/bin/hf buckets sync \
  notebooks/studies/YYYYMMDD_short_description \
  hf://buckets/jpata/particleflow-studies/YYYYMMDD_short_description \
  --no-delete \
  --plan /tmp/YYYYMMDD_short_description-sync.jsonl
.venv/bin/hf buckets sync \
  --apply /tmp/YYYYMMDD_short_description-sync.jsonl
.venv/bin/hf buckets cp \
  notebooks/studies/README.md \
  hf://buckets/jpata/particleflow-studies/README.md
```

After uploading, run the same command with a new `--plan` path. A complete sync
should report zero uploads, downloads, and deletes. Sync dated directories
individually so sibling transfer archives, such as `.zip` or `.tar.gz` files, are
not uploaded accidentally. Do not use `--delete` unless remote removal is
intentional and the complete plan has been reviewed.

For a standalone transfer archive, run from the repository root:

```bash
tar -czf YYYYMMDD_short_description.tar.gz \
  notebooks/studies/YYYYMMDD_short_description
```
