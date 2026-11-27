"""IDEA z_qq: tracks, clusters, tracker hits and calorimeter hits with truth labels."""

from mlpf.heptfds.full_utils.builder import FullBuilder


class IdeaEdmZQqFull(FullBuilder):
    DETECTOR = "idea"
    PROCESS = "z_qq"
