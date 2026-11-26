"""ILD EDM4hep Z -> tautau at 91 GeV with reconstructed tracks and clusters."""

from .qq import IldEdmQqPf, describe


class IldEdmZTautauPf(IldEdmQqPf):
    DESCRIPTION = describe("Z -> tautau at 91 GeV")
    PROCESS_NAME = "p8_ee_Z_tautau_ecm91"
