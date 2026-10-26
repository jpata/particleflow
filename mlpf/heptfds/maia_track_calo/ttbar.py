"""MAIA ttbar reconstructed tracks and calorimeter hits."""

from mlpf.heptfds.track_calo_utils.builder import TrackCaloBuilder


class MaiaTrackCaloTtbar(TrackCaloBuilder):
    DETECTOR = "maia"
