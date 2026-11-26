"""ILD EDM4hep WW_fullhad at 365 GeV with reconstructed tracks and clusters."""

from .qq import IldEdmQqPf, describe


class IldEdmWwFullhadPf(IldEdmQqPf):
    DESCRIPTION = describe("WW_fullhad at 365 GeV")
    PROCESS_NAME = "p8_ee_WW_fullhad_ecm365"
