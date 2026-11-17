# Tracks and clusters

These datasets use reconstructed tracks and calorimeter clusters as input elements. They contain fewer elements per event than [hit datasets](hit-based.md) and provide the default training path for CMS, CLD, and CLIC. ColliderML, MAIA, and IDEA have different reconstruction and validation boundaries, listed below.

![Tracks and calorimeter clusters from CLD ttbar event 0.](https://huggingface.co/jpata/particleflow/resolve/main/images/event_displays/cld/event_0_tracks_clusters.png)

This example shows [CLD](cld.md) event 0 in the transverse plane.
The detector pages use matching input and particle views; the
[event gallery](event-displays.md) provides overlays and reusable assets.

## Choose a detector and physics sample

| Detector or dataset family | Physics samples | Workflow |
|---|---|---|
| [CMS Run 3](cms.md) | Top pairs, QCD multijets, hadronic tau pairs; with and without pileup | Supported CMSSW track/cluster path |
| [CLD, 365 GeV](cld.md) | Top pairs, fully hadronic WW, inclusive qq, ZZ | Supported Key4HEP track/cluster path |
| [CLICdet, 380 GeV](clic.md) | Top pairs, fully hadronic WW, inclusive qq | Supported Key4HEP track/cluster path |
| [ColliderML Release 1](colliderml.md) | Top pairs without pileup and with approximately 200 pileup interactions | Research workflow with spatial calorimeter clustering |
| [MAIA](maia.md) | Top pairs | Partial workflow with data-validation limitations |
| [IDEA](idea.md) | Top pairs, fully hadronic WW, inclusive qq | Pipeline validation with truth-seeded proxy tracks |

Open the detector page for exact names, schema versions, recipes, and sample-specific limits. Public availability and configuration partitions are recorded in the [catalog](catalog.md).

## Use the data

For a first run, [download CLD configuration 1](download.md) and follow [Train a model](../training/train.md). To change the relative training exposure of physics samples, use [Dataset mixing](../training/data-mixing.md).

Custom production follows [Generate a dataset](generate.md). CMS has its own postprocessing format; CLD and CLIC share the [Key4HEP workflow](key4hep.md). The `*_pf` suffix identifies this representation, but feature definitions and target conventions remain detector-specific.
