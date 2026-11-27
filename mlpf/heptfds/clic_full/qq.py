"""CLIC qq: tracks, clusters, tracker hits and calorimeter hits with truth labels."""

from mlpf.heptfds.full_utils.builder import FullBuilder


class ClicEdmQqFull(FullBuilder):
    DETECTOR = "clic"
    PROCESS = "qq"
