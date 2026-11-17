# Detector event views

The [reusable asset gallery](https://huggingface.co/jpata/particleflow/blob/main/images/event_displays/README.md) provides hits,
tracks, clusters, particles and overlays for one ttbar event per detector family:
CLD, CLIC, IDEA, MAIA and ColliderML. Each family has white PNG, transparent PNG
and SVG exports, plus an extracted scene archive for regeneration.

![CLD reconstruction flow with track/cluster and raw-hit MLPF routes](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/pf_event.png)

All CLD panels show event 0 from the same 365 GeV ttbar simulation. The particle
panel shows Pandora reference candidates. Blue and orange routes identify the
inputs and output roles of the two MLPF model families.

## Particle sources

| Family | Particle view | Track view |
| --- | --- | --- |
| CLD, CLIC, MAIA | Reconstructed Pandora PFOs | Reconstructed tracks |
| IDEA | Visible stable truth proxy | Truth-seeded tracks |
| ColliderML | Visible primary-leaf truth proxy | ACTS tracks |

The gallery records sampling, scale, source collections and trajectory conventions.
Use the [dataset catalog](catalog.md) for model-compatible datasets and the
[raw-data comparison guide](../validation/detector-comparison.md) for source provenance.

## Hosting and regeneration

Visual assets, extracted scene archives and the
[provenance manifest](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/manifest.json)
are hosted in `jpata/particleflow` on Hugging Face. The documentation embeds those
files directly. Generated assets stay under the ignored `local_test_data/` directory.

To regenerate from locally available detector data:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/pf-mpl .venv/bin/python scripts/create_reconstruction_event_assets.py
```

To reproduce the published events without the original detector files, download
the extracted scenes and render them locally:

```bash
hf download jpata/particleflow \
  --include "images/event_displays/*/event_0_scene.json.gz" \
  --include "images/event_displays/manifest.json" \
  --local-dir local_test_data/event_assets

PYTHONPATH=. MPLCONFIGDIR=/tmp/pf-mpl .venv/bin/python scripts/create_reconstruction_event_assets.py \
  --from-scenes --output-dir local_test_data/event_assets/images/event_displays
```

Use `--revision <commit>` with `hf download` to pin an asset version. The manifest
records scene SHA-256 hashes, source collections, event identifiers and display settings.
Use `--source cld=/path/to/file.root` to override a raw source, `--event` to select
an event and `--detectors` to choose families. ColliderML also accepts
`--colliderml-parquet` for saved clusters.
