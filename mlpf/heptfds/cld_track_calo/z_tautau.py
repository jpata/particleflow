"""CLD z_tautau reconstructed tracks and calorimeter hits."""

from mlpf.heptfds.track_calo_utils.builder import TrackCaloBuilder


class CldTrackCaloZTautau(TrackCaloBuilder):
    DETECTOR = "cld"
    PROCESS = "z_tautau"
