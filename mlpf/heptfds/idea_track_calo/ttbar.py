"""IDEA truth-seeded tracks and separate Cherenkov/scintillation hit signals."""

from mlpf.heptfds.track_calo_utils.builder import TrackCaloBuilder


class IdeaTrackCaloTtbar(TrackCaloBuilder):
    DETECTOR = "idea"
