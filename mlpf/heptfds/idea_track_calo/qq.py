"""IDEA qq reconstructed tracks and calorimeter hits."""

from mlpf.heptfds.track_calo_utils.builder import TrackCaloBuilder


class IdeaTrackCaloQq(TrackCaloBuilder):
    DETECTOR = "idea"
    PROCESS = "qq"
