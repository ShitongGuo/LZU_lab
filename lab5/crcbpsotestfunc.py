# crcbpsotestfunc.py
import numpy as np
from typing import Dict, Tuple, Union

Array = np.ndarray

def crcbchkstdsrchrng(X: Array) -> Array:
    """Check points lie in [0,1]^D (row-wise)."""
    return np.all((X >= 0.0) & (X <= 1.0), axis=1)

def s2rv(X: Array, params: Dict[str, Array]) -> Array:
    """Standardized -> real coordinates: X*(rmax-rmin)+rmin."""
    rmin = np.asarray(params["rmin"])
    rmax = np.asarray(params["rmax"])
    return X * (rmax - rmin) + rmin

def r2sv(R: Array, params: Dict[str, Array]) -> Array:
    """Real -> standardized coordinates: (R-rmin)/(rmax-rmin)."""
    rmin = np.asarray(params["rmin"])
    rmax = np.asarray(params["rmax"])
    return (R - rmin) / (rmax - rmin)

def crcbpsotestfunc(
    X: Array,
    params: Dict[str, Array],
    return_coords: bool = False
) -> Union[Array, Tuple[Array, Array, Array]]:
    """
    Generalized Rastrigin fitness for each ROW of X (standardized coords).
    X.shape = (n_points, n_dim); params['rmin'], params['rmax'] are (n_dim,).
    Returns F (n_points,). If return_coords=True, also returns (R, Xp).
    """
    X = np.asarray(X, dtype=float)
    nrows, _ = X.shape

    # Validity mask in standardized cube
    valid = crcbchkstdsrchrng(X)

    # Real coordinates (only convert valid rows)
    R = X.copy()
    if np.any(valid):
        R[valid, :] = s2rv(X[valid, :], params)

    # Fitness vector
    F = np.full(nrows, np.inf, dtype=float)
    if np.any(valid):
        x = R[valid, :]
        # Rastrigin: sum(x^2 - 10 cos(2πx) + 10)
        F[valid] = np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0, axis=1)

    if return_coords:
        Xp = r2sv(R, params)  # standardized coords (for特殊边界时可用)
        return F, R, Xp
    return F

