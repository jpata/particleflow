# Unit tests for the shared detector-independent visibility mask (#463).
import numpy as np

from mlpf.data.target_building import visibility_mask, visible_energy_deposit, visible_energy_fraction


def test_constants_match_key4hep_lineage():
    assert visible_energy_fraction == 0.10
    assert visible_energy_deposit == 0.5


def test_fraction_term():
    # 15% passes, 9% fails — and the comparison is strict (10% exactly does not pass)
    einh = np.array([1.5, 0.9, 1.0])
    e = np.array([10.0, 10.0, 10.0])
    ch = np.array([False, False, False])
    assert visibility_mask(einh, e, ch).tolist() == [True, False, False]


def test_absolute_mip_term_is_charged_only():
    # deposit 0.6 GeV at 100 GeV truth energy: fraction term fails for all; the absolute term
    # admits only the charged particle. 0.4 GeV is below the absolute threshold for either charge.
    einh = np.array([0.6, 0.6, 0.4, 0.4])
    e = np.array([100.0, 100.0, 100.0, 100.0])
    ch = np.array([True, False, True, False])
    assert visibility_mask(einh, e, ch).tolist() == [True, False, False, False]


def test_tracker_term_is_optional_or():
    einh = np.array([0.0, 0.0])
    e = np.array([10.0, 10.0])
    ch = np.array([False, False])
    assert visibility_mask(einh, e, ch, np.array([True, False])).tolist() == [True, False]
    assert visibility_mask(einh, e, ch).tolist() == [False, False]


def test_zero_energy_guard():
    # E=0 with no deposit -> not visible; E=0 with a real deposit -> fraction blows up -> visible
    einh = np.array([0.0, 1.0])
    e = np.array([0.0, 0.0])
    ch = np.array([False, False])
    assert visibility_mask(einh, e, ch).tolist() == [False, True]
