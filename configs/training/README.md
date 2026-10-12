# Reusable training scenarios

Scientific comparisons live under `scenarios/`; machine-dependent paths and runtime
tuning live under `platforms/`. Run a scenario locally with:

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/cld_hits_output_comparison.yaml \
  --platform configs/training/platforms/local.yaml \
  --global-batch-size 8 \
  --dry-run
```

The scenario declares a global batch size. The runner derives
`gpu_batch_multiplier` from the number of GPUs and the dataset batch size, and rejects
non-integral combinations. It also resolves every variant through `MLPFConfig` and
checks that variants differ only in the fields listed by
`allowed_variant_differences`.

The local picker discovers the same scenario files and applies the local platform
profile and short-run defaults:

```bash
scripts/local/train_scenario.sh --list
scripts/local/train_scenario.sh cld_hits_output_comparison --dry-run
scripts/local/train_scenario.sh cld_hits_output_comparison --seed 2468
```

Additional arguments are forwarded to the generic scenario runner, such as
`--variant set`, `--global-batch-size 4`, or `--set num_steps=100`.

With multiple variants or seeds, jobs are ordered by seed and then by variant. A
Slurm array can select one job using `--task-index $SLURM_ARRAY_TASK_ID`.
`--seed N` replaces the scenario seed list, including when a task index is used.
The site shell launchers expose this as the `SEED` environment variable. Without an
override, seeds come from the scenario file and are recorded in both the resolved
configuration and run manifest.

List the available scenarios and accelerators, then submit using the Flatiron
picker:

```bash
scripts/flatiron/train_scenario.sh --list
scripts/flatiron/train_scenario.sh cld_hits_output_comparison h100 --dry-run
scripts/flatiron/train_scenario.sh cld_hits_output_comparison h100
```

The picker reads Slurm resources from the selected platform profile and derives
the array size from the scenario's variants and seeds. Use `--seed N` to submit
one comparison pair with an explicit seed.

Tallinn and LUMI use the same interface with site-specific profiles and workers:

```bash
scripts/tallinn/train_scenario.sh --list
scripts/tallinn/train_scenario.sh cld_hits_output_comparison l40 --dry-run
scripts/tallinn/train_scenario.sh cld_hits_output_comparison l40

scripts/lumi/train_scenario.sh --list
scripts/lumi/train_scenario.sh cld_hits_output_comparison mi250x --dry-run
scripts/lumi/train_scenario.sh cld_hits_output_comparison mi250x
```

For the three CLD hit-set references (Flash backbone with global Flash decoder,
HEPTv2 backbone with global Flash decoder, and HEPTv2 backbone with sectorized
Flash decoder), preview and submit the comparison campaign:

```bash
scripts/lumi/train_scenario.sh cld_set_hits_heptv2_sector_comparison mi250x --dry-run
scripts/lumi/train_scenario.sh cld_set_hits_heptv2_sector_comparison mi250x
```

The Tallinn worker runs the repository's `uv` environment directly. The LUMI
submitter uses `particleflow-env` (override its interpreter with
`PYTHON_EXECUTABLE`) and the worker executes that environment in the standard
PyTorch ROCm container (override the image with `IMG`).

Use repeated `--set KEY=VALUE` options only for explicit one-off overrides. Every
resolved run writes `scenario-manifest.json` containing the scenario, platform,
seed, final configuration, command, and git revision. Runs are grouped as
`<experiments_dir>/<scenario_name>/<variant>_seed<seed>_<timestamp>/`.

## Grouped HEPTv2 tracking follow-up (50k)

`scenarios/cld_pf_heptv2_grouped_tracking_50k.yaml` defines five runs, all with
seed 12345, global batch 64, learning rate 0.001, 50,000 steps, and reconstruction
diagnostics on four validation batches per rank. Every variant changes one
mechanism from the completed grouped HEPTv2 model:

| Variant suffix | Change from `hit_heptv2_grouped_baseline` |
|---|---|
| `tracker_features` | Enable existing tracker-neighborhood features |
| `balanced` | Average tracker and calorimeter grouping losses with equal detector weight |
| `tracker_readout` | Add predicted-group geometric summaries and reliable circle-fit momentum anchors |
| `event_consistency` | Add visible event momentum/energy consistency, weight 0.1 |

Detector balancing preserves the background weight of 0.1 within each detector,
normalizes over the local training batch, and averages only detector populations
present in that batch. It does not rebalance individual particles.

The tracker readout uses predicted groups only. It fits transverse circles with
three Huber-reweighted least-squares iterations, in double precision, and passes
nine summaries to a zero-initialized learned slot projection: hit and surface
counts, radial and longitudinal spans, transverse spatial scale, fitted log-pT,
relative fit residual, conditioning, and a validity flag. A valid fit requires
at least four hits, three distinct (system, side, layer) surfaces, 100 mm radial
span, a Gram-matrix eigenvalue ratio above 1e-7, a radial RMS residual below 2%
of that span, and fitted pT in (0.1, 10,000) GeV. The conversion assumes unit
charge magnitude and the CLD 2 T solenoid, set explicitly by
`tracker_group_magnetic_field`. It is a geometric prior rather than a full
material-aware track fit.

Reliable fits supply log-pT and a massless energy estimate from the initial
slot eta to the decoder's learned residual scale head. Existing calorimeter
seed anchors are retained. Unreliable groups keep their learned scales; their
geometric summaries can still inform the decoder. Validation reports
`metrics/set_slot/tracker_fit_fraction` (over all slots) and
`tracker_fits_per_event` on diagnostic batches to show how often anchors apply.

Event consistency uses the final slots' soft presence probabilities to sum
visible px, py, pz and energy. Momentum residuals are normalized by the target
scalar momentum sum; energy residuals by target total energy, both floored at
1 GeV. Smooth-L1 momentum and energy terms are added with the configured weight,
outside automatic task weighting. The target visible momentum need not vanish.
This is a weak auxiliary objective; compare jet widths, vector closure, class
momentum errors, efficiencies, and count bias rather than total training loss.

Resolve all production jobs without launching training:

```bash
uv run python3 scripts/training/run_scenario.py \
  --scenario configs/training/scenarios/cld_pf_heptv2_grouped_tracking_50k.yaml \
  --platform configs/training/platforms/lumi_mi250x.yaml --dry-run
```

Submit on LUMI using the same scenario through `scripts/lumi/train_scenario.sh`.
For local smoke runs, `scripts/local/train_scenario.sh
cld_pf_heptv2_grouped_tracking_50k` uses global batch 1 by default to accommodate
local GPU memory; it is not the production batch-64 comparison.
