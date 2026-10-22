# Reconstruction diagnostics during validation

Set `validation_diagnostics_batches: 4` (or another positive number) in training
parameters to enable these checks on the first N validation batches **per rank**.
Zero disables them. The 50k grouping and momentum scenarios already set 4.
These diagnostics are observational: they do not change predictions, losses,
matching, or gradients. They use the validation population, not the test dumps.
When `validation_particle_metrics` is true, ordinary particle metrics cover
the complete validation pass. Setting it false also disables the detailed
particle-momentum metrics, but not the set-slot or hit-grouping diagnostics.

Metrics are stored in each `history/step_*.json` under `valid` and sent through
the existing TensorBoard/Comet logging. TensorBoard strips the `metrics/` prefix
and prepends `validation/`. All values are sums/counts reduced across ranks;
ratios are computed after reduction. Names ending in `samples` report total
support in the diagnostic subset, including zero support. Other zero-denominator
metrics are omitted, so absence is not evidence of perfect reconstruction.
Residual RMS metrics are square roots of globally averaged squared residuals.

## Momentum reconstruction

History prefix: `metrics/particle/momentum/`.

Slices are `all`, `class_1` through the last physical class, and disjoint absolute
eta bins `abs_eta_lt1`, `abs_eta_1to2`, `abs_eta_ge2`. CLD classes 1–5 are charged
hadrons, neutral hadrons, photons, electrons, and muons. Boundaries belong to the
upper bin. No extra matching is done: the diagnostic reuses the standard
Hungarian geometry/log-pT assignment. The current validation implementation
caps each assignment cost at 2.0; historical October study results use the
archived uncapped estimator. Do not mix their absolute values without rescoring
all models with the same definition.

Each slice reports:

- `target_samples` and `efficiency`: accepted matches divided by truth count.
- `assigned/`: all Hungarian pairs, before delta-R and pT acceptance cuts.
- `accepted/`: pairs with delta-R < 0.1 and absolute relative pT error < 0.5.
- Both matched populations report `samples`, relative pT bias/MAE/RMS, relative
  energy MAE, mean delta-R, longitudinal momentum bias/MAE/RMS, and fractions
  with relative pT error > 0.5 or absolute pz error > 10 GeV.
- `event/`: signed errors of summed px, py, pz and energy; pz MAE/RMS; norms of
  the 3-momentum and transverse-momentum residual vectors; and event support.

Residuals are **prediction minus target**. Momentum and energy errors use GeV;
relative errors are dimensionless. Matched slices use the **target's** class and
eta, with no PID requirement. Assigned errors retain large failures but still
exclude unpaired objects. Accepted errors are conditioned on the acceptance
cuts and must be read alongside efficiency and support.

Event sums include all selected predictions and all physical targets, including
unmatched objects. Each collection uses its own class/eta for event slicing;
PID errors and migration therefore affect slice closure. Signed component
errors add across the class slices or across eta slices; norms and absolute
errors do not. These are errors relative to the true visible-particle sums,
not constraints that the event momentum must vanish. Event means include events
with no objects in a slice (zero contribution unless a prediction migrates in).
The same metrics apply to elementwise MLPF and set-output models.

Examples:

- `metrics/particle/momentum/class_1/accepted/pt_relative_abs_mean`
- `metrics/particle/momentum/abs_eta_ge2/assigned/pz_abs_gt10_fraction`
- `metrics/particle/momentum/all/event/p3_error_mean`

## Predicted hit ownership

History prefix: `metrics/hit_grouping/`. Only models with the explicit grouping
head produce these metrics. The assignment is the grouping head's actual hard
argmax over all slots plus background, **before presence selection and decoder
refinement**. Scores are reconstructed in chunks using `hit_grouping_chunk_size`;
no full hit-by-slot matrix is retained or exported. The training Hungarian
assignment maps slot IDs to target particle numbers from the same forward pass.
This additional assignment calculation is restricted to diagnostic batches.

For sparse pooling, these remain hard-argmax ownership diagnostics; they do not
measure the fractional memberships used by the readout.

Tracker (element type 1) and calorimeter (type 2) are evaluated separately:

- `purity`: correctly assigned hits / all hits assigned to foreground slots.
- `completeness`: correctly assigned hits / all hits owned by a target particle.
- `background_recall`: background hits assigned to background / background hits.
- `background_assignment_fraction`: fraction of all hits assigned to background.
- `hit_samples` and `owned_hit_samples`: total and truth-owned hit counts.
- Calorimeter `energy_purity` and `energy_completeness` use nonnegative deposited
  hit energy in place of hit counts. These describe the hard particle-number
  labels in the dataset, not fractional contributions from simulation.
- `merge_fraction`: fraction of occupied foreground slots containing significant
  contributions from at least two truth particles.

An owned hit has a positive particle number appearing among that event's target
particles. All other hits count as background. A target without a Hungarian
slot still contributes to completeness's denominator, with zero correct hits.
Padding is excluded. Background is not counted as a constituent truth particle
for merge detection, but it contaminates purity.

Object averages appear under `tracker/{slice}/` and `calorimeter/{slice}/`:

- `object_purity` / `object_completeness`: average the mapped slot's purity and
  completeness over truth objects with at least one hit in that detector.
  Missing or empty mapped slots contribute zero.
- `best_slot_completeness`: average largest foreground-slot overlap / truth hit
  count, independent of the Hungarian map. A high value with low mapped-slot
  completeness can indicate a slot-to-particle mismatch rather than splitting.
- `split_fraction`: truth objects contributing significantly to at least two
  foreground slots.
- `merged_slot_fraction`: truth objects whose mapped slot is a merge.
- Calorimeter `energy_completeness`: mean energy completeness over truth objects
  with positive deposited energy.
- `object_samples`: number of truth objects supporting the slice.

A significant split fragment has at least **two hits and 10% of the truth
object's detector hits**. A significant merged constituent has at least **two
hits and 10% of the predicted slot's detector hits**. Background assignments do
not count as split fragments; their loss appears in completeness. These are
explicit diagnostic conventions, not universal tracking-efficiency definitions.

Object slices include `all` and these independent partitions (not a joint grid):

| Variable | Bins | Definition |
|---|---|---|
| Truth pT | <1, 1–10, >=10 GeV | Target particle momentum |
| Absolute truth eta | <1, 1–2, >=2 | Target particle direction |
| Detector surface coverage | 1–3, 4–7, >=8 | Unique (system, side, layer) tuples among owned hits |
| Local target density | 0, 1–2, >=3 neighbors | Other target particles within delta-R < 0.1, with periodic phi |

These truth-defined bins prevent incorrect assignments from changing the
population on which grouping is judged. Read calorimeter count-based and
energy-based completeness together, and compare tracker completeness with the
charged-particle momentum slices before attributing an error to association.

## Linear calorimeter continuation probes

Set `validation_linear_probe: true` to fit and evaluate frozen raw/embedding
continuation probes on disjoint events from the configured diagnostic batches.
`validation_linear_probe_max_events: 16` caps collected events per rank. Scalars
appear in TensorBoard under `validation/linear_probe/calorimeter/`, in history
JSON, and in `VALIDATION_PROBE` log lines. See
[calo_continuation_probe.md](calo_continuation_probe.md#during-training-validation)
for support, rank-local reduction semantics, costs and minimum event counts.

The same switch logs held-out instance ARI for **tracker and calorimeter** under
`validation/linear_probe/{tracker,calorimeter}/{raw,embedding}/ari`, plus the
embedding-minus-raw gain, oracle-graph ARI and support. These instance probes use
disjoint fit/calibration/evaluation events and the original within-detector
neighbor graph. See the ARI subsection of the continuation-probe documentation.
