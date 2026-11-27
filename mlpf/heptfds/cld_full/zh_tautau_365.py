"""CLD zh_tautau_365: tracks, clusters, tracker hits and calorimeter hits with truth labels."""

from mlpf.heptfds.full_utils.builder import FullBuilder


class CldEdmZhTautau365Full(FullBuilder):
    DETECTOR = "cld"
    PROCESS = "zh_tautau_365"
