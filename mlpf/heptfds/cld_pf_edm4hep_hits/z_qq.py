"""CLD EDM4hep Z -> qq at 91 GeV with hit-level inputs."""

from .qq import CldEdmQqHits


class CldEdmZQqHits(CldEdmQqHits):
    DESCRIPTION = "CLD EDM4hep Z -> qq at 91 GeV with hit-level inputs."
    PROCESS_NAME = "p8_ee_Z_qq_ecm91"
