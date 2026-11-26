"""CLD ttbar: tracks, clusters, tracker hits and calorimeter hits with truth labels."""

from mlpf.heptfds.full_utils.builder import FullBuilder


class CldEdmTtbarFull(FullBuilder):
    DETECTOR = "cld"
    PROCESS = "ttbar"
