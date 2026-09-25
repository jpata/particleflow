from mlpf.heptfds.colliderml_pf.ttbar import CollidermlTtbarNopuPf

_DESCRIPTION = """
ColliderML Release 1 (ttbar with ~200 pileup interactions) converted for MLPF training.
Clustered view: reconstructed ACTS tracks + spatial clusters derived from raw calo hits.
Pileup membership is carried per target/gen particle in the `ispu` feature.
"""


class CollidermlTtbarPu200Pf(CollidermlTtbarNopuPf):
    """ttbar_pu200 variant: identical schema and split logic to CollidermlTtbarNopuPf.

    Only the dataset name (derived from the class name) and the description differ; the
    manual_dir content (converted parquet under clustered/ttbar_pu200/) drives the data.
    """

    DESCRIPTION = _DESCRIPTION
