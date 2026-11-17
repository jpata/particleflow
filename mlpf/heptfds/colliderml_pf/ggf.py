from mlpf.heptfds.colliderml_pf.ttbar import CollidermlTtbarNopuPf

_DESCRIPTION = """
ColliderML Release 1 (gluon-fusion Higgs (ggF), no pileup) converted for MLPF training.
Clustered view: reconstructed ACTS tracks + CLUE clusters derived from raw calo hits.
"""


class CollidermlGgfNopuPf(CollidermlTtbarNopuPf):
    """ggf_pu0 variant: identical schema and split logic to CollidermlTtbarNopuPf.

    Only the dataset name (derived from the class name) and the description differ; the
    manual_dir content (converted parquet under clue/ggf_pu0/) drives the data.
    """

    DESCRIPTION = _DESCRIPTION
    RELEASE_NOTES = {
        "1.5.0": "Initial release, converted like ttbar 1.5.0 (CLUE --clue-preset pu0).",
    }
