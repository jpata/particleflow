from mlpf.heptfds.colliderml_hits.ttbar import CollidermlTtbarHits

_DESCRIPTION = """
ColliderML Release 1 (ttbar with ~200 pileup interactions) converted for MLPF training.
Hits view: raw tracker + calorimeter hits carry the input elements; ytarget rows live on the
hits. Pileup membership is carried per target particle in the `ispu` feature.
"""


class CollidermlTtbarPu200Hits(CollidermlTtbarHits):
    """ttbar_pu200 hits variant: identical schema and split logic to CollidermlTtbarHits."""

    DESCRIPTION = _DESCRIPTION
