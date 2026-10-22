# CLD hit-output comparison

This study compares the historical CLD hit baseline with two 3.2.1 campaigns:

- the previous elementwise model on CLD hits TFDS 3.2.0, archived in
  `notebooks/studies/20260819_hit_vs_pf_comparison`;
- the 6 September 2026 set-output model without particle-origin query loss;
- the 7 September 2026 elementwise model trained for 40,000 steps;
- the 7 September 2026 set-output model with particle-origin query loss,
  evaluated at its last complete archived checkpoint at step 20,000.

The 6 September elementwise run remains archived under `inputs/elementwise/`,
but the refreshed slides replace it with the longer 7 September elementwise
run. The 6 September set run remains in the plots because it provides the
closest available no-origin comparison.

The recorded jet-metric plots also retain the 40,000-step track/cluster PF
training on TFDS 3.2.0 as the learned PF-input baseline from the earlier study.
The physics validation slides include its archived PF-input plots as a fourth
panel. A constant heuristic PF (`ycand`) reference is included for all three
samples (ttbar, WW to 4q, qq), each evaluated on 1,000 events.

The `_hits` TFDS used by every trained run in this study (elementwise and set,
both campaigns) has an empty `ycand` field for every event and every sample:
the hit-based input pipeline carries no heuristic reconstruction to compare
against. The heuristic reference instead comes from `eval_pf_baseline.py` in
this directory, which evaluates the non-hits CLD 3.2.1 track/cluster TFDS
(`cld_edm_{ttbar,ww_fullhad,qq}_pf`, splits 1-10) directly, independent of any
trained MLPF model: it clusters jets from the dataset's own `ycand` and
`ytarget` particle collections with the standard CLD jet definition
(`ee_genkt_algorithm`, R=0.4, p=-1, pt>5 GeV) and matches them within
$\Delta R<0.1$, mirroring the clustering in `mlpf/model/inference.py`. Run it
with:

```bash
uv run python notebooks/studies/20260905_hit_output_comparison/eval_pf_baseline.py
```

This reads the non-hits `_pf` TFDS from
`/mnt/work/mlpf/cld/v1.2.5_key4hep_2025-05-29/tfds`, which is not archived by
this study (only the hits-based inputs are); regenerating the heuristic
numbers requires that dataset, or an equivalent local copy, to be present at
that path. The recorded output is reproduced in the script's own docstring.

The earlier 3.2.1 inputs were produced under
`experiments/cld_hits_output_comparison` as
`elementwise_seed12345_20260906_012757_908052` and
`set_seed12345_20260906_012757_938629`. They use 8 H200 GPUs, seed 12345, a
global event batch of 512, and a 20,000-step cosine schedule. Both have a
complete step-20,000 checkpoint.

The new production campaign is archived under `inputs/campaign_20260907/`:

- `elementwise_40k/` comes from
  `elementwise_seed12345_20260907_090315_620002`. It completed 40,000 steps on
  8 H100 GPUs. The selected physics plots and endpoint use step 40,000.
- `set_query_origin_20k/` comes from
  `set_seed12345_20260907_090321_871639`. It was configured for 40,000 steps
  on 8 H100 GPUs but produced complete archived history and physics plots only
  through step 20,000. Its copied rank-zero log continues through step 20,400
  and contains no completion record.
- `set_query_origin_smoke_2k/` comes from
  `set_seed12345_20260907_132200_683525`. This completed 2,000 steps on one
  GeForce RTX 5060 Ti using only split 10, a global event batch of 8, and 100
  validation and test events. It verifies the code path but is excluded from
  the production comparison.

The 7 September set configurations add `query_origin_loss_weight: 0.02` and
`query_origin_temperature: 0.1`, corresponding to the particle-origin query
alignment change committed as `52f6846f`. The experiment manifests do not
record a Git hash, so this association comes from the archived configuration
and repository history rather than an embedded revision identifier.

Each new campaign directory contains the complete available history JSON,
rank-zero `train.log`, separate `tensorboard/train/` and
`tensorboard/valid/` event files, the training configuration,
hyperparameters, scenario manifest, resolved particle-flow specification, and
the complete selected checkpoint plot directory. Checkpoints, model pickle
files, batch and prediction parquet files, and plots from unselected
checkpoints are intentionally excluded. The previous elementwise and PF
baselines continue to come from the archived
`20260819_hit_vs_pf_comparison` study.

From the repository root, render and execute the slides with:

```bash
notebooks/studies/20260905_hit_output_comparison/render.sh
```

This produces an executed notebook, PNG/PDF figures,
`output/hit_output_comparison.slides.html`, and
`output/hit_output_comparison.slides.pdf`. Notebook code is omitted from both
slide exports. The HTML presentation uses Reveal.js from a public CDN when
viewed. PDF rendering requires Google Chrome or Chromium.

The slide-ready elementwise-versus-set schematic is authored in CeTZ as
`elementwise_set_output_modes.typ`. Regenerate its vector and raster outputs
with Typst 0.15 or newer:

```bash
typst compile \
  notebooks/studies/20260905_hit_output_comparison/elementwise_set_output_modes.typ \
  notebooks/studies/20260905_hit_output_comparison/output/elementwise_set_output_modes.svg
typst compile \
  notebooks/studies/20260905_hit_output_comparison/elementwise_set_output_modes.typ \
  notebooks/studies/20260905_hit_output_comparison/output/elementwise_set_output_modes.pdf
typst compile --ppi 180 \
  notebooks/studies/20260905_hit_output_comparison/elementwise_set_output_modes.typ \
  notebooks/studies/20260905_hit_output_comparison/output/elementwise_set_output_modes.png
```

The recorded jet metrics are shown as one slide per sample (ttbar, WW to 4q,
qq) rather than a single combined grid, with the median response, response
IQR, and match fraction panels sharing the same y-axis range across all three
samples so the panels are directly comparable at a glance.

The comparison keeps these limitations visible in the slides:

- the old training used the full validation collection and a global batch of
  384, while the new campaign used 512 validation events and a global batch of
  512;
- the displayed old physics plots contain 1,000 events, while the new plots
  contain 512 events;
- the no-origin set run uses a 20,000-step cosine schedule, while step 20,000
  is only the midpoint of the new 40,000-step schedule, so the difference does
  not isolate the query-origin loss;
- the production comparison contains one seed, and the new set campaign does
  not provide a complete 40,000-step endpoint.
