"""CLD EDM4hep Z -> qq at 91 GeV with reconstructed tracks and clusters."""

from .qq import CldEdmQqPf


class CldEdmZQqPf(CldEdmQqPf):
    DESCRIPTION = "CLD EDM4hep Z -> qq at 91 GeV with reconstructed tracks and clusters."
    PROCESS_NAME = "p8_ee_Z_qq_ecm91"
