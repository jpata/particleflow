"""CLD ww_fullhad reconstructed tracks and calorimeter hits."""

from mlpf.heptfds.track_calo_utils.builder import TrackCaloBuilder


class CldTrackCaloWwFullhad(TrackCaloBuilder):
    DETECTOR = "cld"
    PROCESS = "ww_fullhad"
