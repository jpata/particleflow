"""ColliderML ttbar, no pileup: tracks, clusters, tracker hits and calorimeter hits with truth labels."""

from mlpf.heptfds.full_utils.builder import FullBuilder


class CollidermlTtbarFull(FullBuilder):
    DETECTOR = "colliderml"
    PROCESS = "ttbar"
