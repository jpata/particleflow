"""CLD zh_tautau_240: tracks, clusters, tracker hits and calorimeter hits with truth labels."""

from mlpf.heptfds.full_utils.builder import FullBuilder


class CldEdmZhTautau240Full(FullBuilder):
    DETECTOR = "cld"
    PROCESS = "zh_tautau_240"
