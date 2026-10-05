# Hit versus track/cluster CLD comparison

This directory is a self-contained snapshot of the 19 August 2026 comparison between:

- hit inputs: `pyg-cld-hits-v1_cld_20260815_011516_887054` (8 H100s, 20k steps);
- track/cluster inputs: `pyg-cld-v1_cld_20260815_135053_268848` (8 H100s, gradient accumulation 256, 40k steps).

The `inputs/` directory contains the saved history JSON, TensorBoard training events, training logs, configuration, hyperparameters, and one CLD `ttbar` EDM4hep ROOT file needed by the notebook. The TensorBoard events provide the detailed 100-step loss curves; the logs provide timestamped walltime and eight-GPU memory snapshots. Checkpoints are intentionally excluded.

From the repository root, render and execute the slides with:

```bash
notebooks/studies/20260819_hit_vs_pf_comparison/render.sh
```

This produces an executed notebook, PNG/PDF figures, `output/hit_vs_pf_comparison.slides.html`, and `output/hit_vs_pf_comparison.slides.pdf`. Notebook code is omitted from both slide exports; only the setup, plots, numerical results, and conclusions are shown. The HTML presentation uses Reveal.js from a public CDN when viewed. PDF rendering requires Google Chrome or Chromium.

To archive the complete study after rendering:

```bash
tar -czf 20260819_hit_vs_pf_comparison.tar.gz \
  notebooks/studies/20260819_hit_vs_pf_comparison
```

## Physics validation

The `physics_validation/` directory contains predictions and standard CLD physics plots for 1,000 `ttbar` events from both checkpoints. They can be regenerated from the repository root with:

```bash
scripts/local/validation.sh
```

The default validation expects both the hit and PF TFDS under `data/tfds_validation_cld/tensorflow_datasets/cld`. Both use split 1, version 3.2.0, whose test sets contain 10,000 events each. Paths, checkpoints, event count, versions, and splits can all be overridden through the environment variables defined at the top of the script.

## CLD event displays

The reusable `scripts/visualize_cld.py` Matplotlib utility reads an EDM4hep ROOT file and overlays reconstructed tracks, Pandora clusters, raw tracker/calorimeter/muon hits, and stable generator particles. Its implementation is adapted from [erwulff/particlemind's CLD visualization notebook](https://github.com/erwulff/particlemind/blob/main/notebooks/cld-visualize.ipynb). The input file was downloaded from [jpata/particleflow on Hugging Face](https://huggingface.co/datasets/jpata/particleflow/tree/main/root/cld/ttbar).

The slide renderer generates perspective 3D PNG displays for events 4, 10, and 25. They can also be generated directly with:

```bash
uv run python scripts/visualize_cld.py \
  notebooks/studies/20260819_hit_vs_pf_comparison/inputs/reco_p8_ee_ttbar_ecm365_300000.root \
  --events 4 10 25 \
  --output-dir notebooks/studies/20260819_hit_vs_pf_comparison/output/event_displays
```

Pass `--debug` to additionally produce two 3×3 association checks per event: fitted tracks against their associated tracker hits in the x–z plane, and Pandora cluster centroids against their associated calorimeter hits in the x–y plane. The panels select the tracks with the most associated hits and the highest-energy clusters.
