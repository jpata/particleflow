"""Keep physical drops separate from numerical energy residuals in the study."""

import numpy as np
from scripts.studies.compare_colliderml_matching import merge_accounting


def test_no_drops_with_float_residual():
    result = merge_accounting(np.array([10.0, 2.0]), np.array([[-1, 0], [-1, -1]]), [1], 12.000001)
    assert result["dropped_particles"] == 0
    assert result["dropped_energy"] == 0
    assert abs(result["energy_balance_residual"] - 1e-6) < 1e-9


def test_drops_are_original_particle_energies():
    result = merge_accounting(np.array([10.0, 2.0, 3.0]), np.array([[-1, 0], [-1, -1], [-1, -1]]), [1], 12.0)
    assert result == dict(dropped_particles=1, dropped_energy=3.0, energy_balance_residual=0.0)
