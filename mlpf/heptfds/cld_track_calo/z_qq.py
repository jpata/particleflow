"""CLD z_qq reconstructed tracks and calorimeter hits."""

from mlpf.heptfds.track_calo_utils.builder import TrackCaloBuilder


class CldTrackCaloZQq(TrackCaloBuilder):
    DETECTOR = "cld"
    PROCESS = "z_qq"
