from mlpf.heptfds.colliderml_pf.ttbar import CollidermlTtbarNopuPf

_DESCRIPTION = """
ColliderML Release 1 (di-Higgs (pp -> HH), with ~200 pileup interactions) converted for MLPF training.
Clustered view: reconstructed ACTS tracks + CLUE clusters derived from raw calo hits.
Pileup membership is carried per target/gen particle in the `ispu` feature (pileup share of the target energy).
"""


class CollidermlDihiggsPu200Pf(CollidermlTtbarNopuPf):
    """dihiggs_pu200 variant: identical schema and split logic to CollidermlTtbarNopuPf.

    Only the dataset name (derived from the class name) and the description differ; the
    manual_dir content (converted parquet under clue/dihiggs_pu200/) drives the data.

    Upstream ships only 295 of the 1000 tracker_hits shards at the pinned revision; the converter
    skips shards missing a table, so this dataset has ~29.5k events (vs 100k for the other pu200
    samples).
    """

    DESCRIPTION = _DESCRIPTION
    RELEASE_NOTES = {
        "1.5.0": "Initial release, converted like ttbar 1.5.0 (CLUE --clue-preset pu200).",
    }
