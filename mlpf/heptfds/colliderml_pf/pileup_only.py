from mlpf.heptfds.colliderml_pf.ttbar import CollidermlTtbarNopuPf

_DESCRIPTION = """
ColliderML Release 1 (pileup-only (minimum bias), no pileup) converted for MLPF training.
Clustered view: reconstructed ACTS tracks + CLUE clusters derived from raw calo hits.
Every particle is counted as pileup (ispu = pufrac = 1; the hard-scatter genmet/genjets are empty, the
*_withpu gen reference holds everything): the converter overrides the vertex_primary rule for this sample.
"""


class CollidermlPileupOnlyNopuPf(CollidermlTtbarNopuPf):
    """pileup_only_pu0 variant: identical schema and split logic to CollidermlTtbarNopuPf.

    Only the dataset name (derived from the class name) and the description differ; the
    manual_dir content (converted parquet under clue/pileup_only_pu0/) drives the data.
    """

    DESCRIPTION = _DESCRIPTION
    RELEASE_NOTES = {
        "1.5.0": "Initial release, converted like ttbar 1.5.0 (CLUE --clue-preset pu0); "
        "every particle counted as pileup (ispu = pufrac = 1, empty hard-scatter genmet/genjets).",
    }
