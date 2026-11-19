"""Các hàm kiểm tra shape và đạo hàm số."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def assert_shape(array: ArrayLike, expected: tuple[int, ...]) -> None:
    """Báo lỗi rõ ràng nếu mảng không có shape mong đợi."""
    actual = np.asarray(array).shape
    if actual != expected:
        raise AssertionError(f"Expected shape {expected}, received {actual}")


def finite_difference_gradient(
    function: Callable[[NDArray[np.float64]], float],
    x: ArrayLike,
    step: float = 1e-5,
) -> NDArray[np.float64]:
    """Xấp xỉ gradient của hàm scalar bằng central difference."""
    if step <= 0:
        raise ValueError("step must be positive")
    point = np.asarray(x, dtype=float).copy()
    gradient = np.zeros_like(point)
    for index in np.ndindex(point.shape):
        original = point[index]
        point[index] = original + step
        right = float(function(point))
        point[index] = original - step
        left = float(function(point))
        point[index] = original
        gradient[index] = (right - left) / (2 * step)
    return gradient


def relative_error(actual: ArrayLike, expected: ArrayLike, epsilon: float = 1e-12) -> float:
    """Sai số tương đối ổn định giữa hai mảng."""
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    denominator = max(float(np.linalg.norm(a) + np.linalg.norm(e)), epsilon)
    return float(np.linalg.norm(a - e) / denominator)

