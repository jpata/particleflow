"""CLD z_qq: tracks, clusters, tracker hits and calorimeter hits with truth labels."""

from mlpf.heptfds.full_utils.builder import FullBuilder


class CldEdmZQqFull(FullBuilder):
    DETECTOR = "cld"
    PROCESS = "z_qq"
