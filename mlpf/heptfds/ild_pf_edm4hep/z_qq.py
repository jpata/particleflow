"""ILD EDM4hep Z -> qq at 91 GeV with reconstructed tracks and clusters."""

from .qq import IldEdmQqPf, describe


class IldEdmZQqPf(IldEdmQqPf):
    DESCRIPTION = describe("Z -> qq at 91 GeV")
    PROCESS_NAME = "p8_ee_Z_qq_ecm91"
