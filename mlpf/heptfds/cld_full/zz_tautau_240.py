"""CLD zz_tautau_240: tracks, clusters, tracker hits and calorimeter hits with truth labels."""

from mlpf.heptfds.full_utils.builder import FullBuilder


class CldEdmZzTautau240Full(FullBuilder):
    DETECTOR = "cld"
    PROCESS = "zz_tautau_240"
