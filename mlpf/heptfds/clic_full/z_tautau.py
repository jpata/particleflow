"""CLIC z_tautau: tracks, clusters, tracker hits and calorimeter hits with truth labels."""

from mlpf.heptfds.full_utils.builder import FullBuilder


class ClicEdmZTautauFull(FullBuilder):
    DETECTOR = "clic"
    PROCESS = "z_tautau"
