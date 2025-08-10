# ackley_fitness.py
import numpy as np
from typing import Dict, Tuple, Union

Array = np.ndarray

def s2rv(X: Array, params: Dict[str, Array]) -> Array:
    rmin = np.asarray(params["rmin"]); rmax = np.asarray(params["rmax"])
    return X * (rmax - rmin) + rmin

def chk_unit_cube(X: Array) -> Array:
    return np.all((X >= 0.0) & (X <= 1.0), axis=1)

def crcbpso_ackley(X: Array, params: Dict[str, Array], return_coords: bool = False):
    """
    Ackley fitness for each ROW of standardized X in [0,1]^D.
    params: {"rmin": (D,), "rmax": (D,)}
    Returns F (n,), optionally (F, R, Xp) 以兼容“特殊边界条件”用法。
    """
    X = np.asarray(X, dtype=float)
    nrows, dim = X.shape
    valid = chk_unit_cube(X)

    # real coords
    R = X.copy()
    if np.any(valid):
        R[valid, :] = s2rv(X[valid, :], params)

    # fitness
    F = np.full(nrows, np.inf, dtype=float)
    if np.any(valid):
        x = R[valid, :]
        s1 = np.sqrt(np.mean(x**2, axis=1))
        s2 = np.mean(np.cos(2*np.pi*x), axis=1)
        F[valid] = -20.0*np.exp(-0.2*s1) - np.exp(s2) + 20.0 + np.e

    if return_coords:
        # 标准化回写（通常不用，但接口兼容）
        rmin = np.asarray(params["rmin"]); rmax = np.asarray(params["rmax"])
        Xp = (R - rmin) / (rmax - rmin)
        return F, R, Xp
    return F

