"""CLD zz_tautau_365: tracks, clusters, tracker hits and calorimeter hits with truth labels."""

from mlpf.heptfds.full_utils.builder import FullBuilder


class CldEdmZzTautau365Full(FullBuilder):
    DETECTOR = "cld"
    PROCESS = "zz_tautau_365"
