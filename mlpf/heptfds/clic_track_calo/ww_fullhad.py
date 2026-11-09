"""CLIC ww_fullhad reconstructed tracks and calorimeter hits."""

from mlpf.heptfds.track_calo_utils.builder import TrackCaloBuilder


class ClicTrackCaloWwFullhad(TrackCaloBuilder):
    DETECTOR = "clic"
    PROCESS = "ww_fullhad"
