"""CLIC EDM4hep Z -> tautau at 91 GeV with hit-level inputs."""

from .qq import ClicEdmQqHits


class ClicEdmZTautauHits(ClicEdmQqHits):
    DESCRIPTION = "CLIC EDM4hep Z -> tautau at 91 GeV with hit-level inputs."
    PROCESS_NAME = "p8_ee_Z_tautau_ecm91"
