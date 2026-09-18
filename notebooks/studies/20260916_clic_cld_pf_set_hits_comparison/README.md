# CLIC and CLD PF-input versus set-output hits comparison

This study summarizes the completed 50,000-step trainings under
`experiments/clic_cld_pf_set_hits_comparison/`:

- `cld_pf_seed12345_20260911_090253_631130`
- `cld_set_hits_seed12345_20260911_090255_959044`
- `clic_pf_seed12345_20260911_090257_092959`
- `clic_set_hits_seed12345_20260911_121623_644456`

All runs use seed 12345, a global batch of 512, 8 NVIDIA H100 PCIe GPUs, a
50,000-step schedule, approximately 512 validation events, and 512 test events
per process.
The PF-input models use six shared backbone layers. The hit-set models use a
12-layer split/shared backbone (4 tracker, 4 calorimeter, 4 common) and an
eight-layer, 256-slot set decoder. The earlier CLD hit-set study used four
decoder layers.

The introductory recap reuses the archived particle-matching, ttbar jet, and
event-closure figures from `20260911_cld_pf_hits_comparison` without
regenerating them. It summarizes why the hit-based output moved from a fixed
per-hit elementwise association to an unordered, Hungarian-matched particle
set. Immediately after the recap, the deck directly includes
`../20260905_hit_output_comparison/elementwise_set_output_modes.typ` as a
full-slide schematic of the two output modes.

The training and validation sample slide is derived from the archived
`scenario-manifest.json` files and rank-zero logs. All four runs use TFDS
3.2.1, logical train/valid/test splits, and split configs 1–10 for ttbar,
fully hadronic WW, and qq. The logs report about 900.5–900.9k training events
per process for CLD (2.702M total) and 450.7–450.9k per process for CLIC
(1.352M total). Validation exposes 51 events from each of ten configs, or 510
raw events per process, with distributed batching padding to approximately
512; test evaluation is capped at 512 events per process.

The factor-of-two training-pool difference is part of the dataset release
lineage, not a log aggregation error. The CLD builder records version 3.1.1 as
the expansion to 1M events, while the CLIC builder records version 3.1.0 as a
500k-event production; version 3.2.1 updates target definitions and energy
accounting without equalizing those pools. The TFDS metadata confirms
90k/10k train/test events per CLD config and 45k/5k per CLIC config. Since
both models run 50,000 steps at global batch 512, both receive 25.6M event
presentations, corresponding to approximately 9.5 passes over the CLD pool
and 18.9 over the CLIC pool.

## Event displays

The early detector-input and target-particle slides use side-by-side CLD and
CLIC transverse-plane projections produced by
`scripts/visualize_key4hep.py`. The visualizer supports both CLD and CLIC and
overlays reconstructed tracks, Pandora clusters, tracker hits, ECAL hits,
HCAL hits, and muon hits. All panels use the same camera orientation: the beam
axis is perpendicular to the screen, +x points left, +y points up, and +z
points out of the screen. The archived slide images use representative ttbar
event 5 from the matched Pythia-seed files with suffix `300000`, omit
generator-particle guide lines, and subsample at most 600 hits per collection.
For multi-file comparisons the visualizer rejects mismatched trailing numerical
suffixes:

```bash
uv run python scripts/visualize_key4hep.py \
  /mnt/work/mlpf/root/cld/ttbar/reco_p8_ee_ttbar_ecm365_300000.root \
  /mnt/work/mlpf/root/clic/ttbar/reco_p8_ee_ttbar_ecm380_300000.root \
  --events 5 --max-hits 600 --no-particles \
  --output-dir notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/inputs/event_displays
```

The following target-only slide uses visible generator-status-1 particles as
a proxy for the target population, omit neutrinos, and map the remaining
particles to the five training PID categories. Charged particles are
propagated as helices in the nominal solenoidal field (2 T for CLD and 4 T for
CLIC), while photons and neutral hadrons follow straight trajectories:

```bash
uv run python scripts/visualize_key4hep.py \
  /mnt/work/mlpf/root/cld/ttbar/reco_p8_ee_ttbar_ecm365_300000.root \
  /mnt/work/mlpf/root/clic/ttbar/reco_p8_ee_ttbar_ecm380_300000.root \
  --events 5 --target-only \
  --output-dir notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/inputs/event_displays
```

For small print-ready event icons, the compact SVG mode writes each detector
as separate, exact 1 x 1 cm, 2 x 2 cm, and 5 x 5 cm vector images. It includes
only hits, reconstructed tracks, and calorimeter clusters: particle guides,
titles, legends, axes, and orientation annotations are omitted. Marker areas
and track widths scale with the physical output size.

```bash
uv run python scripts/visualize_key4hep.py \
  /mnt/work/mlpf/root/cld/ttbar/reco_p8_ee_ttbar_ecm365_300000.root \
  /mnt/work/mlpf/root/clic/ttbar/reco_p8_ee_ttbar_ecm380_300000.root \
  --events 5 --max-hits 600 --compact-svg \
  --output-dir notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/inputs/event_displays
```

Adding `--target-only` produces matching compact SVGs containing only the
visible status-1 target-particle trajectories and their energy-scaled endpoint
markers. One representative high-energy trace per particle type receives a
subtle endpoint label. Charged and neutral hadron classes use the shorthand
`pi` and `K0`; compact target exports also include neutrinos, and photon traces
use `gamma`.

```bash
uv run python scripts/visualize_key4hep.py \
  /mnt/work/mlpf/root/cld/ttbar/reco_p8_ee_ttbar_ecm365_300000.root \
  /mnt/work/mlpf/root/clic/ttbar/reco_p8_ee_ttbar_ecm380_300000.root \
  --events 5 --target-only --compact-svg \
  --output-dir notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/inputs/event_displays
```

## Decoder-depth diagnostic

The layer diagnostic from `20260911_set_hits_diagnostics` was rerun on the new
step-50,000 CLD and CLIC checkpoints. The comparison plot includes the full
four-layer trajectory from the earlier study alongside both new eight-layer
trajectories. Each run uses 1,000 ttbar test events
from split 1, presence threshold 0.5, and the same particle matching code as
training validation. The GPU cache was generated with:

```bash
uv run python notebooks/studies/20260911_set_hits_diagnostics/cache_set_slots.py \
  --config <run>/train-config.yaml \
  --checkpoint <run>/checkpoints/checkpoint-50000.pth \
  --data-dir /mnt/work/mlpf/tensorflow_datasets/{cld,clic} \
  --sample {cld,clic}_edm_ttbar_hits --num-events 1000 \
  --output /tmp/{cld,clic}_set_hits_8layer_raw_slots.parquet

uv run python notebooks/studies/20260911_set_hits_diagnostics/analyze_set_slots.py \
  --cache /tmp/<detector>_set_hits_8layer_raw_slots.parquet \
  --output-dir notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/output/diagnostics/<detector>
```

The regenerable raw slot caches remain under `/tmp` and are not archived.
The compact CSV tables, figures, and diagnostic README files are retained
under `output/diagnostics/`.

The CPU-only existence-versus-regression counterfactual uses the same caches
and an angular-only Hungarian association, so pT does not determine whether a
slot is treated as an existing particle. It compares the default read-out
with regression-only, existence-only, and combined target-aware oracles and
decomposes the missing-momentum residual into matched regression, missed
targets, and unmatched predictions:

```bash
uv run python notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/diagnose_existence_vs_regression.py \
  /tmp/cld_set_hits_8layer_raw_slots.parquet \
  /tmp/clic_set_hits_8layer_raw_slots.parquet \
  --output notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/output/diagnostics/existence_vs_regression.json
```

On the 1,000-event ttbar caches, default angular recall is 0.594/0.632 and
all-slot angular coverage is 0.839/0.861 for CLD/CLIC. Conditional pT response
has median 1.034/1.025 but IQR 0.546/0.419. The combined oracle reduces the
missing-three-momentum vector error from 18.8/15.9 GeV to 8.3/8.0 GeV;
correcting only regression or only existence makes closure worse because the
current error components partially cancel.

## Frozen-slot probes

To test whether the regression and existence deficits can be repaired by the
final read-out alone, the step-50,000 models were rerun with the final
normalized 256-dimensional decoder-slot embeddings cached. Linear and
two-layer MLP probes were trained on an event-level 700/150/150 train,
validation, and test split. Presence labels use angular-only Hungarian
matching with delta-R < 0.1; regression uses only those angularly matched
slots and predicts target log-pT and log-energy. The encoder, backbone, and
set decoder remain frozen throughout.

```bash
uv run python notebooks/studies/20260911_set_hits_diagnostics/cache_set_slots.py \
  --config <run>/train-config.yaml \
  --checkpoint <run>/checkpoints/checkpoint-50000.pth \
  --data-dir /mnt/work/mlpf/tensorflow_datasets/{cld,clic} \
  --sample {cld,clic}_edm_ttbar_hits --num-events 1000 \
  --cache-embeddings \
  --output /tmp/{cld,clic}_set_hits_8layer_embeddings.parquet

uv run python notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/train_frozen_slot_probes.py \
  --cache /tmp/<detector>_set_hits_8layer_embeddings.parquet \
  --detector <detector> \
  --output-dir notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/output/probes \
  --device cuda --epochs 40 --batch-size 4096
```

On held-out events, CLD presence F1 changes from 0.607 to 0.637/0.641 for
the linear/MLP probes, while CLIC changes from 0.657 to 0.679/0.679. This is
well below the all-slot angular coverage of 0.826/0.848. CLD conditional pT
IQR changes from 0.663 to 0.654/0.624; CLIC changes from 0.450 to
0.483/0.465. Thus a more expressive final head does not recover narrow
particle regression, and the presence gain is modest. The limiting
information is already missing or entangled in the frozen slot
representation. Detailed JSON metrics and the small trained heads are stored
under `output/probes/`.

## Heuristic PF baseline

Fixed heuristic PF (`ycand`) is evaluated independently for both detectors
with `eval_heuristic_pf_baseline.py`. The archived
`inputs/heuristic_{cld,clic}.json` files use 512 test events from TFDS split 1
for each of ttbar, WW to 4q, and qq. Jet metrics therefore use the same 512
events/process scope as the trained models. Particle metrics pool all 1,536
test events. The final trained-model comparison uses the same three
512-event-per-process test populations and equal-weight process macros;
logged checkpoint curves use the separate validation populations.

Regenerate the references with:

```bash
uv run python notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/eval_heuristic_pf_baseline.py \
  --detector cld --data-dir /mnt/work/mlpf/tensorflow_datasets/cld \
  --splits 1 --num-samples-per-split 512

uv run python notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/eval_heuristic_pf_baseline.py \
  --detector clic --data-dir /mnt/work/mlpf/tensorflow_datasets/clic \
  --splits 1 --num-samples-per-split 512
```

## Archived inputs

The four campaign runs are archived under descriptive directories in
`inputs/{cld_pf,cld_set_hits,clic_pf,clic_set_hits}/`. Each archive contains
the complete 5,000-step history, rank-zero log as `train.log`, separate
nonempty `tensorboard/train/` and `tensorboard/valid/` writers,
`train-config.yaml`, `hyperparameters.json`, `scenario-manifest.json`, the
resolved `particleflow_spec.yaml`, and the complete `plots_step_50000/`
directory. The heuristic JSON summaries are also archived. Checkpoints,
prediction parquet files, and intermediate plot
directories are intentionally excluded. The analysis and Typst source read
only these archived inputs, not the mutable experiment directories.

`inputs/particle_evolution_by_process.csv` is a compact derived archive of
particle matching, per-class efficiency, multiplicity, and event-level
closure on the 512-event ttbar, WW to 4q, and qq test samples at all ten
checkpoints. It supplies the equal-weight process macro-averages and
process-minimum-to-maximum bands on the particle and event-level slides,
including the step-50,000 summaries. The
event observables are the energy response `sum(E_pred) / sum(E_target)`, the
transverse vector error `|sum(pT_pred) - sum(pT_target)|`, and the
three-dimensional vector error `|sum(p_pred) - sum(p_target)|`; the errors are
evaluated per event and then averaged within each process. Regenerate it while
the campaign prediction dumps are available with:

```bash
uv run python notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/compute_sample_particle_evolution.py
```

## Main result

The learned PF-input models finish near particle F1 0.75 for both detectors.
The hit-set models finish at F1 0.524 for CLD and 0.575 for CLIC. Their decoder
layers continue to improve through layer 8, but the final models remain worse
than the earlier four-layer CLD hit-set model (F1 0.617 on the same 1,000-event
diagnostic definition). The new kinematic oracle ceilings also fall from
0.892 to 0.801 (CLD) and 0.837 (CLIC), so the regression is not explained by
the presence threshold alone.

## Rendering and output

The presentation is authored in Typst as
`clic_cld_pf_set_hits_comparison.typ`. From the repository root, regenerate
the analysis figures and compile the deck with:

```bash
notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/render.sh
```

This writes `output/clic_cld_pf_set_hits_comparison.pdf`, slide PNGs under
`output/rendered/`, compact SVG/PNG figures under `output/figures/`, and
`output/summary.json`.

## Slide narrative

The Typst deck follows the structure and visual cadence of the preceding CLD
comparison deck:

1. question and formulations; inputs and targets; the earlier-results recap;
   the detailed elementwise-versus-set schematic; model architecture; loss
   functions; training setup; sample composition; and metric definitions;
2. Part 1, training behaviour: walltime and sampled GPU memory/utilization,
   particle-F1 convergence, the
   checkpoint-by-checkpoint evolution of jet F1 and response width, and the
   event-level missing-three-momentum closure error;
3. Part 2, particle-level results: final three-sample macro metrics and
   per-class efficiency;
4. Part 3, jet-level results: validation scope, recorded metrics, and ttbar
   response plots;
5. a concise conclusions slide with the three decision-level takeaways;
6. backup slides containing the rerun eight-layer diagnostic, per-layer
   read-out, kinematic oracles, and frozen-slot presence/regression probes.

## Limitations

- Each model has one seed, so the comparison has no repeated-seed uncertainty.
- PF and hit-set models differ in both input representation and output head.
- The current hit-set model changes backbone depth and decoder depth together;
  this campaign cannot attribute the regression to one change.
- The per-layer diagnostic uses ttbar only. Logged training curves use the
  approximately 512-event-per-process validation populations; final particle
  and jet summaries use 512 test events per process.
- Heuristic particle metrics pool 1,536 split-1 test events across the three
  processes. The trained-model final summaries use the corresponding
  512-event-per-process test scope and equal-weight process macros.
