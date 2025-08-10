# crcbqcfitfunc.py
import numpy as np
from typing import Tuple, Union, List, Any, Dict

def crcbchkstdsrchrng(xVec: np.ndarray) -> np.ndarray:
    """
    检查标准化坐标是否在 [0,1] 内。
    输入: xVec: (nVecs, nDim)
    返回: validPts: bool array
    """
    return np.all((xVec >= 0) & (xVec <= 1), axis=1)


def s2rv(xVec: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    标准化坐标 -> 实际搜索坐标。
    X(:,j) -> X(:,j)*(rmax(j)-rmin(j)) + rmin(j)
    """
    rmin = np.asarray(params["rmin"], dtype=float)
    rmax = np.asarray(params["rmax"], dtype=float)
    return xVec * (rmax - rmin) + rmin


def ssrqc(x: np.ndarray, params: Dict[str, Any]) -> float:
    """
    Sum of squared residuals after maximizing over amplitude parameter.
    MATLAB 中是 -(dataY * qc')^2。
    """
    phaseVec = x[0] * params["dataX"] + x[1] * params["dataXSq"] + x[2] * params["dataXCb"]
    qc = np.sin(2 * np.pi * phaseVec)
    qc = qc / np.linalg.norm(qc)  # normalized quadratic chirp
    ssrVal = - (np.dot(params["dataY"], qc) ** 2)
    return ssrVal


def crcbqcfitfunc(
    xVec: np.ndarray,
    params: Dict[str, Any]
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fitness function for quadratic chirp regression.
    完全按 MATLAB 版实现。

    Parameters
    ----------
    xVec : ndarray
        标准化坐标数组 (nVecs, nDim) 或 (nDim,)。
    params : dict
        包含 rmin, rmax, dataY, dataX, dataXSq, dataXCb 等字段。

    Returns
    -------
    fitVal : ndarray
        (nVecs,) fitness 值。
    qcCoefs (optional) : ndarray
        对应的真实 QC 系数 (nVecs, nDim)。
    sigs (optional) : ndarray
        对应的 QC 信号 (nVecs, nSamples)。
    """
    # 保证是二维
    xVec = np.atleast_2d(xVec)
    nVecs, _ = xVec.shape

    fitVal = np.zeros(nVecs)
    validPts = crcbchkstdsrchrng(xVec)
    fitVal[~validPts] = np.inf

    # 转换有效点到真实坐标
    realCoords = np.copy(xVec)
    realCoords[validPts, :] = s2rv(realCoords[validPts, :], params)

    # 对有效点计算 fitness
    for i in range(nVecs):
        if validPts[i]:
            fitVal[i] = ssrqc(realCoords[i, :], params)

    # 多返回值情况
    # nargout > 1: 返回真实坐标
    # nargout > 2: 返回 QC 信号
    if hasattr(crcbqcfitfunc, "_return_qc_signal") and crcbqcfitfunc._return_qc_signal:
        # 这个是个技巧：你可以在外部设置这个标志来要求返回 QC 信号
        sigs = []
        for i in range(nVecs):
            phaseVec = realCoords[i, 0] * params["dataX"] \
                     + realCoords[i, 1] * params["dataXSq"] \
                     + realCoords[i, 2] * params["dataXCb"]
            qc = np.sin(2 * np.pi * phaseVec)
            qc = qc / np.linalg.norm(qc)
            sigs.append(qc)
        sigs = np.array(sigs)
        return fitVal, realCoords, sigs
    else:
        return fitVal, realCoords

