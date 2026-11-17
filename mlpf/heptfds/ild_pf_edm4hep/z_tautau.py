"""ILD EDM4hep Z -> tautau at 91 GeV with reconstructed tracks and clusters."""

from .qq import IldEdmQqPf


class IldEdmZTautauPf(IldEdmQqPf):
    DESCRIPTION = "ILD EDM4hep Z -> tautau at 91 GeV with reconstructed tracks and clusters."
    PROCESS_NAME = "p8_ee_Z_tautau_ecm91"
