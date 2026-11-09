"""CLD zz reconstructed tracks and calorimeter hits."""

from mlpf.heptfds.track_calo_utils.builder import TrackCaloBuilder


class CldTrackCaloZz(TrackCaloBuilder):
    DETECTOR = "cld"
    PROCESS = "zz"
