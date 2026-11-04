from mlpf.heptfds.colliderml_pf.ttbar import CollidermlTtbarNopuPf

_DESCRIPTION = """
ColliderML Release 1 (ttbar with ~200 pileup interactions) converted for MLPF training.
Clustered view: reconstructed ACTS tracks + CLUE clusters derived from raw calo hits.
Pileup membership is carried per target/gen particle in the `ispu` feature.
"""


class CollidermlTtbarPu200Pf(CollidermlTtbarNopuPf):
    """ttbar_pu200 variant: identical schema and split logic to CollidermlTtbarNopuPf.

    Only the dataset name (derived from the class name) and the description differ; the
    manual_dir content (converted parquet under clue/ttbar_pu200/) drives the data.
    """

    DESCRIPTION = _DESCRIPTION
    RELEASE_NOTES = {
        "1.3.0": "Bipartite track/cluster assignment majority_energy: majority-owned tracks only, then a cluster matching with an energy score.",
        "1.0.0": "Prepared 2026-10-06 from parquet converted with the tuned pu200 clustering "
        "(bfs, merge-frac 0, radii ECAL 16.25 mm / HCAL 36 mm).",
        "1.1.0": "CLUE clustering (converter --algorithm clue --clue-preset pu200).",
        "1.2.0": "Surrogate ancestors in the truth (key4hep parity): generator-decayed particles that "
        "interacted before decaying become targets, recovering their deposits. Clustering as in 1.1.0.",
        "1.4.0": "Pileup truth: ispu is the pileup share of the target energy (fractional for targets merging "
        "hard-scatter and pileup particles); genmet/genjets restricted to the hard scatter, the previous ones kept as "
        "genmet_withpu/genjets_withpu; new features event_id and pufrac (pileup fraction of each element).",
        "1.5.0": "Out-of-acceptance truth: a particle heading beyond |eta| > 3.5 never enters a calorimeter "
        "(endcap bore edge |eta| 3.4) and is replaced as a target by its Geant4 children that deposit (beam-pipe "
        "interactions), instead of carrying its full truth 4-vector onto a forward cluster.",
    }
