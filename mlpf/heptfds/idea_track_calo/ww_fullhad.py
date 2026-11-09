"""IDEA ww_fullhad reconstructed tracks and calorimeter hits."""

from mlpf.heptfds.track_calo_utils.builder import TrackCaloBuilder


class IdeaTrackCaloWwFullhad(TrackCaloBuilder):
    DETECTOR = "idea"
    PROCESS = "ww_fullhad"
