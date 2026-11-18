"""CLD zz: tracks, clusters, tracker hits and calorimeter hits with truth labels."""

from mlpf.heptfds.full_utils.builder import FullBuilder


class CldEdmZzFull(FullBuilder):
    DETECTOR = "cld"
    PROCESS = "zz"
