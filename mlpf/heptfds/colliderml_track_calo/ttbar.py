"""COLLIDERML ttbar reconstructed tracks and calorimeter hits."""

from mlpf.heptfds.track_calo_utils.builder import TrackCaloBuilder


class CollidermlTrackCaloTtbar(TrackCaloBuilder):
    DETECTOR = "colliderml"
