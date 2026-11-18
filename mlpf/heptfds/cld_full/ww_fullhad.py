"""CLD ww_fullhad: tracks, clusters, tracker hits and calorimeter hits with truth labels."""

from mlpf.heptfds.full_utils.builder import FullBuilder


class CldEdmWwFullhadFull(FullBuilder):
    DETECTOR = "cld"
    PROCESS = "ww_fullhad"
