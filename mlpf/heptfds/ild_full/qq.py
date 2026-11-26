"""ILD qq: tracks, clusters, tracker hits and calorimeter hits with truth labels."""

from mlpf.heptfds.full_utils.builder import FullBuilder


class IldEdmQqFull(FullBuilder):
    DETECTOR = "ild"
    PROCESS = "qq"
