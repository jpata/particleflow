"""CLD zz_tautau_365 reconstructed tracks and calorimeter hits."""

from mlpf.heptfds.track_calo_utils.builder import TrackCaloBuilder


class CldTrackCaloZzTautau365(TrackCaloBuilder):
    DETECTOR = "cld"
    PROCESS = "zz_tautau_365"
