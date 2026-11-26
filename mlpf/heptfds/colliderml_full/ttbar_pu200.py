"""ColliderML ttbar, pileup 200: tracks, clusters, tracker hits and calorimeter hits with truth labels."""

from mlpf.heptfds.full_utils.builder import FullBuilder


class CollidermlTtbarPu200Full(FullBuilder):
    DETECTOR = "colliderml"
    PROCESS = "ttbar_pu200"
