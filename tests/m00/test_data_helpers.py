"""Provided dataset helpers (identical in workshop/ and solutions/), so tested in m00."""
import numpy as np
import pytest

from workshop.data import enzyme_kinetics


@pytest.mark.parametrize("n", [8, 24, 32])
def test_enzyme_kinetics_returns_exactly_n_points(n):
    s, v, *_ = enzyme_kinetics(n=n)
    assert s.shape == v.shape == (n,)
    assert len(np.unique(s)) == 8


@pytest.mark.parametrize("n", [-8, 0, 5, 20])
def test_enzyme_kinetics_rejects_n_not_a_multiple_of_8(n):
    with pytest.raises(ValueError):
        enzyme_kinetics(n=n)
