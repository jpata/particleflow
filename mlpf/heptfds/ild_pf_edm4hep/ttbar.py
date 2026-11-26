"""ILD EDM4hep ttbar at 365 GeV with reconstructed tracks and clusters."""

from .qq import IldEdmQqPf, describe


class IldEdmTtbarPf(IldEdmQqPf):
    DESCRIPTION = describe("ttbar at 365 GeV")
    PROCESS_NAME = "p8_ee_ttbar_ecm365"
