# CLICdet datasets

CLICdet datasets use Key4HEP/EDM4hep production at 380 GeV. The configured
samples are `ttbar`, fully hadronic `WW`, and inclusive `qq`.

| Representation | Dataset names | Model recipe |
|---|---|---|
| Tracks and clusters | `clic_edm_{ttbar,ww_fullhad,qq}_pf` | `pyg-clic-v1` |
| Detector hits | `clic_edm_{ttbar,ww_fullhad,qq}_hits` | `pyg-clic-hits-v1` |

Both representations use dataset version 3.2.1 and configuration partitions
1--10. The full corpus contains about one million events per sample, or three
million events in total.

## Event view

Tracks and calorimeter clusters are shown beside the target particles for a
representative `ttbar` event in the transverse plane.

| Tracks and clusters | Target particles |
|:---:|:---:|
| ![CLIC reconstructed tracks and calorimeter clusters.](../../notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/inputs/event_displays/clic_event_5_pf_5x5cm.svg) | ![CLIC target particles.](../../notebooks/studies/20260916_clic_cld_pf_set_hits_comparison/inputs/event_displays/clic_event_5_targets_5x5cm.svg) |

Target labels use $e$, $\mu$, $\tau$, $\nu$, $\gamma$, $\pi$, and $K^0$.

## Production

Use the [shared Key4HEP workflow](key4hep.md) with `PROD=clic`. Build the
track/cluster representation with `pixi run tfds`, or follow [Hit-based
datasets](hit-based.md) for detector-hit inputs.
