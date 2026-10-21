# Set-output (hits) model diagnostics

Follow-up to `20260911_cld_pf_hits_comparison`: a set of local, retraining-free
studies to understand *why* the set (hits) model reaches particle F1 0.625 and
MET error 7.7 GeV, where it loses and invents particles, and how much of the
gap to learned track/cluster PF (F1 0.745) is recoverable at inference time.

All studies use the step-40,000 checkpoint of
`experiments/cld_pf_hits_comparison/set_hits_seed12345_20260908_164625_682170`
(seed 12345, 40k steps, batch 512, 8xH100) and, as a reference on the same
events, the matching elementwise (hits) run
`elementwise_hits_seed12345_20260908_164625_578183`. The matching code is the
same as in training validation (`mlpf/model/validation_metrics.py`): a
per-event Hungarian assignment with cost (dR/0.1)^2 + (dlog pT / ln 2)^2 and
acceptance at dR < 0.1 and |dpT|/pT < 0.5.

Everything is run from the repository root with `uv run`.

## Study A: prediction-level anatomy (CPU, seconds)

Question: which targets does the set model miss, how well does it measure the
ones it finds, and what are its non-accepted predictions (duplicates vs
fakes)? Uses the campaign's own step-40,000 prediction dumps, so it needs no
GPU and no dataset.

```bash
uv run python notebooks/studies/20260911_set_hits_diagnostics/slim_predictions.py
uv run python notebooks/studies/20260911_set_hits_diagnostics/analyze_predictions.py --sample cld_edm_ttbar_hits
```

`slim_predictions.py` reads `experiments/.../preds_step_40000/<sample>/*.parquet`
(one row per input hit, hundreds of MB) and archives only the physical
particles, jets, and jet matching under
`inputs/predictions_step_40000/{set_hits,elementwise_hits}/<sample>.parquet`
(1-2 MB per sample, all three processes). `analyze_predictions.py` runs from
that archive alone and writes to `output/predictions/<sample>/`:

| Output | What it answers |
|---|---|
| `efficiency_vs_pt.png`, `efficiency_vs_eta_and_track.png`, `efficiency.csv` | Efficiency per class vs true pT, per class vs abs(eta), and split by target track association (`gp_to_track`) |
| `resolution_vs_pt.png`, `resolution.csv` | Median and 16-84% relative pT error and median dR of accepted matches vs true pT, per class |
| `pid_confusion.png`, `pid_confusion.csv` | Row-normalised PID confusion on accepted matches |
| `unaccepted_predictions.png`, `unaccepted_predictions.csv` | Duplicates (close to a target but not the accepted match) vs fakes: dR to the nearest accepted prediction, pT ratio to it, class and pT composition |
| `jet_response.png`, `jet_response.csv` | Jet pT response median and IQR vs jet pT and abs(eta) from the archived jets and target-to-prediction matching |
| `README.md` | Headline numbers |

First result (512 events per process): the set model finds 0.715 of track-associated
targets but only 0.498 of calorimeter-only targets, the reverse of elementwise
(0.539 / 0.630). Its deficit is concentrated in photons and neutral hadrons
below 2 GeV; duplicates are 0.20 of its predictions and typically carry
10-30% of the accepted prediction's pT, i.e. split clusters rather than exact
copies.

## Study B: cache every decoder slot (GPU, about 16 events/s)

Question: what does the decoder produce *before* the presence threshold? The
prediction dumps keep only selected slots. This script runs the checkpoint on
the ttbar hits test split and stores, per event, all 256 slots with presence
probability, PID probabilities, kinematics, the query type each slot was
seeded from (tracker hit, calorimeter hit, fallback), the seed hit's
eta/phi/type/energy, and every decoder layer's auxiliary output, plus the
targets with `gp_to_track` / `gp_to_cluster`.

```bash
uv run python notebooks/studies/20260911_set_hits_diagnostics/cache_set_slots.py --num-events 1000
```

Defaults: config from
`notebooks/studies/20260911_cld_pf_hits_comparison/inputs/set_hits/train-config.yaml`,
checkpoint `experiments/.../set_hits_.../checkpoints/checkpoint-40000.pth`,
TFDS `/mnt/work/mlpf/cld/v1.2.5_key4hep_2025-05-29/tfds`, sample
`cld_edm_ttbar_hits`, split `1`, version `3.2.1`, batch size 1, `--device cuda`.
Output: `output/slot_cache/raw_slots.parquet`. The seeding order is recovered
by recording the decoder's `_take_topk` calls (tracker, then calorimeter,
then fallback), so the cache does not depend on any change to `mlpf/`.

`--device cpu` switches the backbone to the math attention kernel (flash has
no CPU backend) and is only meant for smoke tests with a handful of events.

## Study C: slot-level diagnostics (CPU, minutes)

```bash
uv run python notebooks/studies/20260911_set_hits_diagnostics/analyze_set_slots.py
```

Reads the Study B cache and writes to `output/slots/`:

| Output | What it answers |
|---|---|
| `presence_calibration.*` | Is the presence probability calibrated against "this slot is an accepted match" (all-slot Hungarian)? Overall and per query type; ECE in the README |
| `selection_policies.*` | The kinematics-limited oracle (select exactly the slots the all-slot Hungarian accepts) vs presence thresholds 0.1-0.9, soft-cardinality top-k, and threshold + NMS. If the oracle F1 is far above 0.625 the selection rule is the bottleneck; if not, the slot kinematics are |
| `query_types.*` | Which query type finds each target class; for missed targets, the dR to the nearest slot seed hit (was there a query near it at all?) |
| `reference_refinement.*` | How far the four decoder layers move a slot from its seed hit, and whether that brings it closer to its matched target |
| `per_layer.*` | Efficiency, purity, F1, duplicates, count bias, MET from each decoder layer's own output: does depth help, and where does it saturate? |
| `slot_budget.*` | Active tracker-/calorimeter-seeded slots vs track-associated / calorimeter-only targets; whether 256 slots or the 60/40 tracker/calo split is ever binding |
| `README.md` | Headline numbers |

## How to read the results together

- Study A `efficiency_vs_eta_and_track` and Study C `query_types` answer
  whether the calorimeter-only deficit comes from missing seeds (fix: raise
  the calorimeter query fraction or seed differently) or from seeded slots
  that are suppressed or mis-measured (fix: presence head / regression).
- Study C `selection_policies` bounds what threshold tuning, top-k, or NMS can
  recover without retraining; `presence_calibration` says whether a
  calibrated threshold should differ between tracker- and calo-seeded slots.
- Study A `unaccepted_predictions` and Study C `slot_budget` together show
  whether duplicates are neighbouring slots splitting one target (NMS
  territory) or spare slots firing on nothing.
- Study C `per_layer` shows whether the auxiliary-loss layers are already as
  good as the final one (depth is not the constraint) or still improving.

## Archive contents and exclusions

`inputs/predictions_step_40000/` (about 8 MB) is the only archived input; it
is a derived, compact view of the campaign's prediction dumps and is what
Study A needs. The raw prediction dumps, the 84 MB checkpoint, and the TFDS
are not archived; Studies B and C need the local experiment directory and
`/mnt/work` dataset. The Study B cache (`output/slot_cache/`) is regenerable
and sized about 32 KB per event; it is kept under `output/` so the sync to the
studies bucket can exclude it if it grows large.

## Results (11 September 2026)

Study A was run for all three processes (512 events each); Studies B and C
were run on 1,000 ttbar test events from split 1 (16 events/s on an RTX 5060
Ti; 32 MB cache). Default presence threshold 0.5 on the 1,000 events: F1
0.617, efficiency 0.605, purity 0.630, duplicates 0.205, count bias -3.8,
MET error 8.2 GeV, consistent with the campaign's validation numbers.

**1. Selection is the larger bottleneck, kinematics the smaller one.** The
kinematics-limited oracle (select exactly the slots the all-slot Hungarian
accepts) reaches F1 0.892 with efficiency 0.805: the decoder already places a
slot within dR<0.1 and 50% pT of 80% of targets, but the presence head and
threshold turn that into 0.605 efficiency at 0.630 purity. The remaining 20%
of targets have no acceptable slot at all.

**2. The presence probability is globally over-confident, identically for
tracker- and calorimeter-seeded slots.** A slot at presence 0.5 is an
accepted match only about 32% of the time; 0.9 corresponds to about 75%.
There is no case for query-type-specific thresholds.

**3. No single threshold improves on 0.5 across the board.** F1 has two
maxima, 0.648 at threshold 0.20-0.25 (count bias +20 to +27, MET 10-11 GeV)
and 0.706 at 0.75 (purity 0.84, count bias -25, MET 10 GeV); 0.5 sits in a
local F1 minimum but gives the best MET (8.2 GeV) and near-zero count bias.
Mid-presence slots (0.35-0.5) are dominantly duplicates (duplicate fraction
peaks at 0.22 there). Threshold 0.5 + NMS dR<0.05 halves duplicates
(0.121) and lifts F1 to 0.647 but degrades MET to 12.3 GeV: the duplicates
carry real energy (Study A: 10-30% of the accepted prediction's pT), so
removing them hurts closure. Soft-cardinality top-k is equivalent to a 0.45
threshold.

**4. Calorimeter-only targets are lost at seeding, not only at selection.**
Photons: 26% found by tracker-seeded slots, 26% by calorimeter-seeded, 48%
missed; neutral hadrons: 64% missed. Only 51% of missed targets have any slot
seed hit within dR<0.1, so half the misses never had a nearby query. Seeds
are chosen as the top-k hits by log(1+pT)+log(1+E) within the 60/40
tracker/calorimeter split, so low-energy photon showers are never seeded and
several seeds can land on one bright shower. Per event there are 102
calorimeter-seeded slots of which 31 fire, against 47.6 calorimeter-only
targets; 154 tracker-seeded slots of which 59 fire, against 47.0
track-associated targets (over-firing, i.e. duplicates on tracks).

**5. Reference refinement works.** The median distance from a slot's seed hit
to its Hungarian-matched target is 0.245; after the four decoder layers the
slot sits 0.029 from it, and 95% of matched slots end closer than their seed.
The decoder is not limited by the 0.4 local attention radius.

**6. Depth is still paying off.** F1 from each decoder layer's own output is
0.216, 0.331, 0.472, 0.617 and PID accuracy 0.71, 0.79, 0.87, 0.91; neither
has saturated at layer 4.

**Study A across processes** (set vs elementwise): efficiency on
track-associated targets 0.715 / 0.718 / 0.740 (ttbar / WW / qq) against
0.498 / 0.481 / 0.484 on calorimeter-only targets, the reverse of elementwise
(0.539 / 0.502 / 0.481 vs 0.630 / 0.584 / 0.570). The set deficit is photons
and neutral hadrons below about 2 GeV. Set electron and muon PID errors are
almost entirely "charged hadron" (0.52 each), not photon confusion.

### Implications for the next training

- Presence calibration or a loss change (focal / class-balanced presence,
  or a duplicate-aware target such as one-to-one matching with explicit
  no-object penalty on near-duplicates) is the cheapest lever: the oracle
  shows 0.2 of efficiency and 0.37 of purity available with current
  kinematics.
- Seed diversity for calorimeter objects: seed on clusters or apply NMS to
  seed hits so one shower yields one query; raise the calorimeter query
  fraction or budget, since calorimeter slots under-fire while tracker slots
  over-fire.
- More decoder layers or iterative refinement: per-layer metrics have not
  saturated.
- Do not add NMS at inference: it trades MET for F1.
- Lepton PID needs class weighting; electrons and muons collapse into charged
  hadrons.

## Status

Studies A, B, and C have been run and their outputs are under `output/`.
A notebook and slides summarising them should be added following
`notebooks/studies/README.md`.
