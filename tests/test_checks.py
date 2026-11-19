import numpy as np
import pytest

from ml_math import assert_shape, finite_difference_gradient, relative_error


def test_assert_shape() -> None:
    assert_shape(np.zeros((2, 3)), (2, 3))
    with pytest.raises(AssertionError):
        assert_shape(np.zeros((2, 3)), (3, 2))


def test_finite_difference_gradient() -> None:
    x = np.array([1.0, -2.0, 3.0])
    numerical = finite_difference_gradient(lambda z: float(z @ z), x)
    np.testing.assert_allclose(numerical, 2 * x, rtol=1e-5, atol=1e-7)


def test_relative_error() -> None:
    assert relative_error([1, 2], [1, 2]) == 0.0
    assert relative_error([1, 2], [1, 2.1]) > 0.0

