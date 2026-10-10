# Deterministic, truth-blind spatial clusterer for ColliderML release 1: the cluster_event
# entry point and the properties every algorithm enforces. The algorithms themselves live in
# radius_graph.py (union_find / bfs / bfs_merge) and clue.py (CLUE); shared helpers in
# cluster_common.py.
#
# Region codes are the six OpenDataDetector calorimeter subsystems stored in the
# `detector` field of the calo_hits table:
#   9  ECAL endcap -      (z in -5440..-3200 mm, r in ~300..1500)
#   10 ECAL barrel        (|z| <= 3200,      r in ~1300..1500)
#   11 ECAL endcap +      (z in 3200..5440, r in ~300..1500)
#   12 HCAL endcap -
#   13 HCAL barrel
#   14 HCAL endcap +
#
# Algorithms, selected by `algorithm=` on `cluster_event` (default bfs_merge): the
# radius-graph clusterers (radius_graph.py; defaults DEFAULT_REGION_RADII_MM /
# DEFAULT_MERGE_FRAC, re-exported here), plus CLUE (`algorithm="clue"`), which lives in
# clue.py (presets and defaults: clue.CLUE_PRESETS / DEFAULT_CLUE_PARAMS /
# DEFAULT_CLUE_OPTIONS).
#
# Properties enforced by all clusterers (CLUE included):
#   * deterministic per-event: identical inputs always give the same output
#   * truth-blind: contrib_particle_ids/energies/times never read
#   * region-crossing hits allowed (radius-graph clusterers: a single tree over all hits +
#     radius max per pair, so ECAL cells can join HCAL cells when a shower crosses the
#     detector boundary; CLUE: post-hoc ECAL->HCAL cluster linking via cross_region_mm)
#   * energy-conserving: every retained hit belongs to exactly one cluster (only CLUE's
#     optional drop_E / drop_1hit leave hits unclustered, with cluster -1 and no
#     hit_to_cluster entry)
from typing import Any, Dict, Tuple

import numpy as np

from mlpf.data.colliderml.clue import _cluster_event_clue, resolve_clue_config
from mlpf.data.colliderml.radius_graph import DEFAULT_MERGE_FRAC, DEFAULT_REGION_RADII_MM, _cluster_event_bfs, _cluster_event_union_find
from mlpf.data.target_building import SparseMatrixCOO


def cluster_event(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    E: np.ndarray,
    detector: np.ndarray,
    radii_mm: Dict[int, float] | None = None,
    merge_frac: float | None = None,
    algorithm: str = "bfs_merge",
    clue_params: Dict[int, Dict[str, float]] | None = None,
    clue_options: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, SparseMatrixCOO, np.ndarray]:
    """Cluster one event's calorimeter hits.

    Four algorithms, selected by `algorithm`:

    - **bfs_merge** (default): seeded multi-source BFS watershed (strict local maxima claim
      hits by hop count) + a Pandora-style fragment merge over the seed clusters that touch
      via a link. The merge rule is cluster-level: a fragment is absorbed into a bigger
      neighbouring cluster when its total energy E_frag <= merge_frac × E_big. Default
      merge_frac is 0.25 (tuned on 5 ttbar events); turning it off (`merge_frac=0`) recovers
      the raw BFS output (**bfs**).
    - **union_find**: connected components of the whole radius graph (with an optional edge
      gate based on hit energy asymmetry if merge_frac > 0). No seeding. Higher R2 on the
      5-event sweep before fragment merging, but melts separate showers when they sit close.
    - **bfs**: raw seeded BFS on the radius graph, no merging.
    - **clue**: CLUE density-peak clustering per region (clue.py); radii_mm
      and merge_frac are ignored, clue_params/clue_options configure it.

    Args:
      x, y, z:     float32/64 positions in mm, length N
      E:           reconstructed hit energy (calibrated GeV), same length
      detector:    uint8 region code 9..14 (ECAL 9-11, HCAL 12-14)
      radii_mm:    override of the per-region link radius in mm
      merge_frac:  None (per-algorithm defaults), 0.0 (disable merging), or a fraction.
                  For bfs_merge this is the fragment-absorption threshold; for union_find
                  this is the link-refusal gate.
      algorithm:   "bfs_merge" | "union_find" | "bfs" | "clue"
      clue_params: per-region overrides of DEFAULT_CLUE_PARAMS for "clue", e.g.
                   {9: {"dc": 8.0}}; keys dc, rhoc, dm, seed_dc, alpha (see clue.py's header)
      clue_options: overrides of DEFAULT_CLUE_OPTIONS for "clue" (keys in clue.py's header;
                   clue.CLUE_PRESETS holds the named per-pileup settings)
    """

    # ColliderML parquet columns are option-typed, so ak.to_numpy hands us numpy.ma
    # MaskedArrays whose per-element getitem is far slower than plain ndarrays. Nothing is
    # ever masked in these inputs — assert that (a masked entry would corrupt energy
    # comparisons silently) and strip the mask.
    def _plain(a):
        if isinstance(a, np.ma.MaskedArray):
            m = np.ma.getmask(a)
            if np.any(m):
                raise ValueError("cluster_event received masked entries")
            return np.asarray(a.data)
        return a

    x, y, z, E, detector = _plain(x), _plain(y), _plain(z), _plain(E), _plain(detector)

    if merge_frac is None:
        merge_frac = DEFAULT_MERGE_FRAC if algorithm == "bfs_merge" else 0.0
    if radii_mm is None:
        radii_mm = DEFAULT_REGION_RADII_MM

    if algorithm == "union_find":
        return _cluster_event_union_find(x, y, z, E, detector, radii_mm, merge_frac)
    elif algorithm == "bfs":
        return _cluster_event_bfs(x, y, z, E, detector, radii_mm, merge_frac=0.0)
    elif algorithm == "bfs_merge":
        return _cluster_event_bfs(x, y, z, E, detector, radii_mm, merge_frac=merge_frac)
    elif algorithm == "clue":
        params, options = resolve_clue_config(params=clue_params, options=clue_options)
        return _cluster_event_clue(x, y, z, E, detector, params, **options)
    else:
        raise ValueError(f"algorithm must be 'bfs_merge', 'union_find', 'bfs' or 'clue', got {algorithm!r}")
