"""CLIC ttbar: tracks, clusters, tracker hits and calorimeter hits with truth labels."""

from mlpf.heptfds.full_utils.builder import FullBuilder


class ClicEdmTtbarFull(FullBuilder):
    DETECTOR = "clic"
    PROCESS = "ttbar"
