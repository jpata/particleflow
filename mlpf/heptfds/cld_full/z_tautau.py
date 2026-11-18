"""CLD z_tautau: tracks, clusters, tracker hits and calorimeter hits with truth labels."""

from mlpf.heptfds.full_utils.builder import FullBuilder


class CldEdmZTautauFull(FullBuilder):
    DETECTOR = "cld"
    PROCESS = "z_tautau"
