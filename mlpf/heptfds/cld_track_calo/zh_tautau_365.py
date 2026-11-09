"""CLD zh_tautau_365 reconstructed tracks and calorimeter hits."""

from mlpf.heptfds.track_calo_utils.builder import TrackCaloBuilder


class CldTrackCaloZhTautau365(TrackCaloBuilder):
    DETECTOR = "cld"
    PROCESS = "zh_tautau_365"
