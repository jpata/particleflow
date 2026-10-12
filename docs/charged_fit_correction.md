# Charged pT from a tracker fit with a learned correction

Enable these set-decoder settings on a CLD/CLIC hit model:

```yaml
query_init: input-conditioned
hit_grouping: true
tracker_group_readout: true
tracker_fit_correction: true
tracker_fit_max_log_correction: 0.4054651081081644  # log(1.5)
```

This is an opt-in branch inside the existing particle-set model, not an
external tracking stage. Defaults and the legacy tracker-anchor mode are
unchanged. The new branch adds a zero-initialized linear correction head and
requires training (or fine-tuning with explicit handling of its new weights).
Existing checkpoints remain compatible with their original configuration.

## Prediction

The existing group readout fits tracker coordinates with a centered, scaled
transverse-circle estimator in FP64 and three Huber reweighting iterations.
It converts the radius to pT using the configured magnetic field (2 T by
default). The fit requires at least four finite hits, three detector surfaces,
a radial span of 0.1 m, sufficient conditioning, relative residual below 0.02,
and pT between 0.1 and 10,000 GeV. These gates reject underconstrained or
inconsistent groups; they do not guarantee that a mixed group belongs to one
particle. Nonfinite coordinates are excluded.

For an eligible tracker slot predicted as charged hadron, electron or muon:

```
log(pT) = log(pT_fit) + max_log_correction * tanh(correction_head(slot))
```

The default bounds pT / pT_fit between 2/3 and 1.5. At initialization, the
correction is zero and the output equals the fit. The correction head and
final addition use FP32 even under BF16 autocast. The head receives the final
normalized slot, which includes detector embeddings and fit summaries.

Physical PID selection matches inference (argmax over classes 1–5, excluding
the empty class). Neutral predictions, invalid fits and slots with existing
calorimeter anchors retain the ordinary scale readout. Energy and direction
are always learned by the ordinary heads in this mode; this first version
does not fit a longitudinal slope, derive energy from a mass hypothesis, or
model material effects. A bounded correction cannot rescue a grossly wrong
fit that passes the gates. Charge/PID mistakes can route a slot to the wrong
branch; monitor species efficiencies as well as momentum residuals.

The same head is used for intermediate auxiliary predictions. The geometry
fit and hard membership remain detached; the correction and ordinary heads
train jointly with the existing losses, and hit ownership learns through its
supervised grouping objective. This is not a differentiable track fitter.
`tracker_fit_fraction` and `tracker_fits_per_event` count eligible geometrical
fits before the charged-PID gate, not the number of final charged predictions
that use them.

## Comparison

`configs/training/scenarios/cld_pf_heptv2_fit_correction_50k.yaml` defines three
50k-step runs: ordinary grouped regression, legacy tracker anchors, and the
new bounded charged-pT correction. Use the normal scenario runner. The
correction limit is an initial hypothesis, not a tuned resolution guarantee.

Evaluate fit coverage, charged-particle efficiency, uncensored momentum tails,
and central residual widths versus truth pT and eta. Compare with truth-owned
hit fits before interpreting limitations as detector resolution. No training
result is implied by the synthetic fit and forward/backward tests.
