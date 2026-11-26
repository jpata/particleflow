"""IDEA z_tautau: tracks, clusters, tracker hits and calorimeter hits with truth labels."""

from mlpf.heptfds.full_utils.builder import FullBuilder


class IdeaEdmZTautauFull(FullBuilder):
    DETECTOR = "idea"
    PROCESS = "z_tautau"
