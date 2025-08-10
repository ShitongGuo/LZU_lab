# glrtqcsig4pso.py
import numpy as np
from glrtcalc import innerprodpsd   # 用同一套内积/PSD定义，避免尺度不一致

def glrtqcsig4pso(
    X,
    P,
    return_R: bool = False,
    return_S: bool = False,
    maximize: bool = False,
):
    """
    Fitness for PSO: GLRT (colored noise, amplitude maximized).
    默认返回 fitness = -GLRT（便于最小化）。若 maximize=True，则直接返回 GLRT。

    Parameters
    ----------
    X : (n, 3) ndarray
        标准化坐标（每列 ∈ [0,1]），每行对应一组 [x1, x2, x3]。
    P : dict
        必需字段：
          - 'dataY'      : (N,) 数据 y
          - 'dataX'      : (N,) 时间轴 t
          - 'fs'         : 采样频率
          - 'psdPosFreq' : (N/2+1,) one-sided PSD（与 rfft 网格对齐）
          - 'rmin'       : (3,) 真实参数下界 [a1_min, a2_min, a3_min]
          - 'rmax'       : (3,) 真实参数上界 [a1_max, a2_max, a3_max]
        可选加速字段：
          - 'dataXSq'    : (N,) 预存 t**2
          - 'dataXCb'    : (N,) 预存 t**3

    Returns
    -------
    F : (n,) ndarray
        fitness 数组（默认 -GLRT；maximize=True 时为 GLRT）
    以及可选：
    R : (n,3) ndarray
        每行对应的真实参数 [a1, a2, a3]（越界行为 NaN）
    S : (n,N) ndarray
        对应各行参数生成的 **未归一化** QC 信号（越界行填 NaN）
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    # ---- 解包参数 ----
    y    = np.asarray(P["dataY"], dtype=float)
    t    = np.asarray(P["dataX"], dtype=float)
    fs   = float(P["fs"])
    Spsd = np.asarray(P["psdPosFreq"], dtype=float)
    rmin = np.asarray(P["rmin"], dtype=float)
    rmax = np.asarray(P["rmax"], dtype=float)

    # 可选缓存
    t2 = P.get("dataXSq");  t2 = t2 if t2 is not None else t**2
    t3 = P.get("dataXCb");  t3 = t3 if t3 is not None else t**3

    # ---- ★ 边界检查：对越界行直接设 inf，且不参与后续计算 ----
    valid = np.all((X >= 0.0) & (X <= 1.0), axis=1)

    # 预分配
    F = np.full(n, np.inf, dtype=float)                 # ★ 越界默认 inf
    R = None
    S_out = None
    if return_R:
        R = np.full_like(X, np.nan, dtype=float)        # ★ 越界行填 NaN
    if return_S:
        S_out = np.full((n, t.size), np.nan, dtype=float)

    if not np.any(valid):
        # 全部越界：直接返回（形状与 MATLAB 行为一致）
        if return_R and return_S:
            return F, R, S_out
        if return_R:
            return F, R
        if return_S:
            return F, S_out
        return F

    # ---- 标准化 -> 真实参数（仅对合法行） ----
    real_valid = rmin + X[valid] * (rmax - rmin)
    if return_R:
        R[valid] = real_valid

    # ---- 逐行计算 GLRT（PSD 范数单位化） ----
    valid_idx = np.where(valid)[0]
    for row, (a1, a2, a3) in zip(valid_idx, real_valid):
        # 生成未归一化模板
        phase = a1*t + a2*t2 + a3*t3
        s = np.sin(2*np.pi*phase)

        # ★ 在 PSD 内积下单位化：q = s / ||s||_PSD
        nrm2 = innerprodpsd(s, s, fs, Spsd)
        if not np.isfinite(nrm2) or nrm2 <= 0:
            F[row] = np.inf
            continue
        q = s / np.sqrt(nrm2)

        # GLRT（未知幅度）= <y, q>^2
        llr = innerprodpsd(y, q, fs, Spsd)
        glrt = llr * llr

        # fitness：默认最小化 -GLRT；若 maximize=True 则直接返回 GLRT
        F[row] = glrt if maximize else -glrt

        if return_S:
            S_out[row] = s

    # ---- 组织返回 ----
    if return_R and return_S:
        return F, R, S_out
    if return_R:
        return F, R
    if return_S:
        return F, S_out
    return F


