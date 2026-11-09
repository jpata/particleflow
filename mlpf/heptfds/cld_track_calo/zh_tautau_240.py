"""CLD zh_tautau_240 reconstructed tracks and calorimeter hits."""

from mlpf.heptfds.track_calo_utils.builder import TrackCaloBuilder


class CldTrackCaloZhTautau240(TrackCaloBuilder):
    DETECTOR = "cld"
    PROCESS = "zh_tautau_240"
