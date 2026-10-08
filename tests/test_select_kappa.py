"""Unit tests for the frozen kappa selection rule (preregistration v1.2 section 5.5).

The rule itself is pre-registered; these tests pin the deterministic
implementation including tie-breaks, independent of any measured data.
"""

import pytest

from scripts.select_kappa import interval_distance, kappa_column, select_kappa


def test_kappa_column_names():
    assert kappa_column(1.0) == "k_g_kappa_1_0"
    assert kappa_column(1.2) == "k_g_kappa_1_2"
    assert kappa_column(1.5) == "k_g_kappa_1_5"


def test_interval_distance():
    assert interval_distance(0.30) == 0.0
    assert interval_distance(0.20) == 0.0
    assert interval_distance(0.50) == 0.0
    assert interval_distance(0.10) == pytest.approx(0.10)
    assert interval_distance(0.65) == pytest.approx(0.15)


def test_single_candidate_in_range():
    sel, _ = select_kappa({1.0: 0.60, 1.2: 0.35, 1.5: 0.10})
    assert sel == 1.2
    sel, _ = select_kappa({1.0: 0.45, 1.2: 0.55, 1.5: 0.10})
    assert sel == 1.0


def test_multiple_in_range_closest_to_target():
    sel, _ = select_kappa({1.0: 0.45, 1.2: 0.30, 1.5: 0.22})
    assert sel == 1.2
    # 1.2 out of range; 1.0 (dist 0.2) beats 1.5 (dist 0.3)
    sel, _ = select_kappa({1.0: 0.45, 1.2: 0.55, 1.5: 0.22})
    assert sel == 1.0


def test_none_in_range_closest_to_interval():
    sel, reason = select_kappa({1.0: 0.60, 1.2: 0.12, 1.5: 0.02})
    # distances: 0.10 / 0.08 / 0.18 -> 1.2
    assert sel == 1.2
    assert "no candidate in range" in reason
    sel, _ = select_kappa({1.0: 0.55, 1.2: 0.15, 1.5: 0.02})
    # distances: 0.05 / 0.05 / 0.18 -> tie -> closest to 1.2
    assert sel == 1.2


def test_monotone_shares_typical_shape():
    # binding share should fall as kappa rises; rule handles that shape
    sel, _ = select_kappa({1.0: 0.70, 1.2: 0.40, 1.5: 0.15})
    assert sel == 1.2
