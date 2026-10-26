# Paired CLD event displays across granularity scans

Each grid shows all six pitches for one generator seed and event index. The
display uses `scripts/visualize_key4hep.py` with a fixed transverse camera
orientation and view range within each grid. All hits are included; no
per-collection subsampling was needed. The generator-status-1 PDG sequence
was checked to agree across the six points in each grid.

## ECAL: Z→ττ, seed 700000, event 6

This event contains two direct π⁰→γγ pairs with truth opening angles 0.0192
and 0.0155 rad. Under the analysis matching definition, both pairs have two
matched Pandora photons at 5.1 mm, and neither does at 1.5 mm. The zoom uses
the same display-coordinate region at every pitch: center `(x,y)=(410,2230)`
mm and ±180 mm half-width. It includes both pairs' ECAL showers. The full
view uses ±2700 mm.

- [Full event overlay](ecal_seed700000_event6/ecal_combined_grid.png)
- [Full event hits](ecal_seed700000_event6/ecal_hits_grid.png)
- [ECAL shower zoom](ecal_seed700000_event6/ecal_zoom_grid.png)

## HCAL: ttbar, seed 300000, event 0

The full view uses ±4200 mm so HCAL deposits remain visible.

- [Full event overlay](hcal_seed300000_event0/hcal_combined_grid.png)
- [Full event hits](hcal_seed300000_event0/hcal_hits_grid.png)

These displays illustrate individual Geant4 showers. Their stochastic
differences should be interpreted alongside the 10,000-event scan metrics in
`../performance/README.md`.

Reproduce with:

```bash
uv run --no-sync python scripts/cld_ecal_readout/plot_granularity_event_grid.py ecal \
  --seed 700000 --event 6 --zoom-xy 410 2230 --zoom-limit 180
uv run --no-sync python scripts/cld_ecal_readout/plot_granularity_event_grid.py hcal \
  --seed 300000 --event 0
```
