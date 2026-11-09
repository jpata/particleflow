"""CLIC qq reconstructed tracks and calorimeter hits."""

from mlpf.heptfds.track_calo_utils.builder import TrackCaloBuilder


class ClicTrackCaloQq(TrackCaloBuilder):
    DETECTOR = "clic"
    PROCESS = "qq"
