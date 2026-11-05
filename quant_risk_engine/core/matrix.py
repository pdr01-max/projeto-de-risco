"""Symmetric matrix validation and nearest PSD projections."""
import numpy as np
from numpy.typing import ArrayLike, NDArray


def repair_psd(matrix: ArrayLike, *, floor: float = 1e-10, correlation: bool = False) -> NDArray[np.float64]:
    """Project A onto PSD cone: Q diag(max(lambda_i, floor)) Q'; preserve covariance scale."""
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError("finite square matrix required")
    if floor <= 0:
        raise ValueError("floor must be positive")
    a = (a + a.T) / 2
    if np.any(np.diag(a) <= 0):
        raise ValueError("diagonal must be positive")
    scale = np.sqrt(np.diag(a))
    corr = a / np.outer(scale, scale)
    eig, vec = np.linalg.eigh(corr)
    projected = (vec * np.maximum(eig, floor)) @ vec.T
    d = np.sqrt(np.diag(projected))
    projected = projected / np.outer(d, d)
    return (projected + projected.T) / 2 if correlation else projected * np.outer(scale, scale)


def validate_correlation(matrix: ArrayLike) -> NDArray[np.float64]:
    """Check correlation bounds and return a PSD unit-diagonal correlation matrix."""
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError("finite square matrix required")
    if not np.allclose(a, a.T, atol=1e-8) or not np.allclose(np.diag(a), 1, atol=1e-8) or np.max(np.abs(a)) > 1 + 1e-8:
        raise ValueError("invalid correlation matrix")
    return repair_psd(a, correlation=True)
