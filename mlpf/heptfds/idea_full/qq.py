"""IDEA qq: tracks, clusters, tracker hits and calorimeter hits with truth labels."""

from mlpf.heptfds.full_utils.builder import FullBuilder


class IdeaEdmQqFull(FullBuilder):
    DETECTOR = "idea"
    PROCESS = "qq"
