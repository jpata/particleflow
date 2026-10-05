# CLD hit backbone comparison: attention versus HEPTv2

This study compares two hit-based MLPF encoder backbones under an otherwise
identical elementwise-output training recipe:

- the dense flash multi-head self-attention encoder (`attention`);
- the locality-sensitive-hashing sparse-attention encoder (`heptv2`).

Both runs were produced under `experiments/cld_hits_backbone_comparison`
(`attention_seed12345_20260906_023152_860154` and
`heptv2_seed12345_20260906_023209_587826`), use CLD hits TFDS 3.2.1 splits
1-10 (ttbar, WW to 4q, qqbar), the same seed (12345), the same six-layer
shared hit backbone slot, a global event batch of 512, LAMB/BF16, the
`interleaved-shards` sampler, 512 validation and test events per checkpoint,
and a 20,000-step cosine-decay schedule on 8 H200 GPUs. Both reach a complete
checkpoint at step 20,000. Only the backbone convolution type and its
internal configuration (attention heads/dimensions versus HEPTv2
embedding/width/hashing) differ; the elementwise task heads, hit feature
engineering, and loss weighting scheme are identical.

Because both runs share the same elementwise output mode, their raw loss
components and physics metrics are directly comparable, unlike an
elementwise-versus-set comparison, which needs to account for different
target axes and normalizations.

At 20,000 steps HEPTv2 (2.78M parameters) reaches a lower validation loss and
better particle- and jet-level metrics than attention (3.14M parameters), at
the cost of about 34% more wall-clock time (7.93 h vs 5.94 h) and roughly
double the GPU memory (108.4 vs 52.8 GiB median, 138.4 vs 61.9 GiB peak). Full
figures and the metric-by-metric breakdown are in the slides.

The `inputs/attention/` and `inputs/heptv2/` directories archive the runs'
history JSON, rank-zero training log, TensorBoard training and validation
events, training configuration, hyperparameters, scenario manifest, resolved
particle-flow specification, and complete step-20,000 standard physics plots.
For each run, the training event file is in `tensorboard/` and the separately
stored validation event file is in `tensorboard/valid/`. The rank-zero log
archived as `train.log` is the full per-rank log (source file `train.log.0`
in the experiment directory); the terse `train.log` written at the top level
of each experiment directory is not archived here. The notebook reads these
local copies. Checkpoints, prediction parquet dumps, and per-checkpoint
`preds_step_*`/`plots_step_*` directories other than the final step-20,000
plots are intentionally excluded.

From the repository root, render and execute the slides with:

```bash
notebooks/studies/20260907_backbone_comparison/render.sh
```

This produces an executed notebook, PNG/PDF/SVG figures,
`output/backbone_comparison.slides.html`, and
`output/backbone_comparison.slides.pdf`. Notebook code is omitted from both
slide exports. The HTML presentation uses Reveal.js from a public CDN when
viewed. PDF rendering requires Google Chrome or Chromium.

The comparison keeps one limitation visible in the slides: the campaign
contains a single seed per backbone, so the ranking is not yet confirmed
across seeds.
