"""IDEA ww_fullhad: tracks, clusters, tracker hits and calorimeter hits with truth labels."""

from mlpf.heptfds.full_utils.builder import FullBuilder


class IdeaEdmWwFullhadFull(FullBuilder):
    DETECTOR = "idea"
    PROCESS = "ww_fullhad"
