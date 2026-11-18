"""COLLIDERML ttbar_pu200 reconstructed tracks and calorimeter hits."""

from mlpf.heptfds.track_calo_utils.builder import TrackCaloBuilder


class CollidermlTrackCaloTtbarPu200(TrackCaloBuilder):
    DETECTOR = "colliderml"
    PROCESS = "ttbar_pu200"
