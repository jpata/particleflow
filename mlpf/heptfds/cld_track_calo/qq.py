"""CLD qq reconstructed tracks and calorimeter hits."""

from mlpf.heptfds.track_calo_utils.builder import TrackCaloBuilder


class CldTrackCaloQq(TrackCaloBuilder):
    DETECTOR = "cld"
    PROCESS = "qq"
