"""CLIC z_qq: tracks, clusters, tracker hits and calorimeter hits with truth labels."""

from mlpf.heptfds.full_utils.builder import FullBuilder


class ClicEdmZQqFull(FullBuilder):
    DETECTOR = "clic"
    PROCESS = "z_qq"
