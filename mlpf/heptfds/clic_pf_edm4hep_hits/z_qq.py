"""CLIC EDM4hep Z -> qq at 91 GeV with hit-level inputs."""

from .qq import ClicEdmQqHits


class ClicEdmZQqHits(ClicEdmQqHits):
    DESCRIPTION = "CLIC EDM4hep Z -> qq at 91 GeV with hit-level inputs."
    PROCESS_NAME = "p8_ee_Z_qq_ecm91"
