# statgaussnoisegen.py
import numpy as np
from scipy.signal import firwin2, lfilter

def statgaussnoisegen(nSamples: int,
                      psdVals: np.ndarray,
                      fltrOrdr: int,
                      sampFreq: float) -> np.ndarray:
    """
    生成具有给定 one-sided PSD 的平稳高斯噪声（与 MATLAB statgaussnoisegen.m 等价）。

    Parameters
    ----------
    nSamples : int
        输出噪声样本数。
    psdVals : array_like, shape (M, 2)
        第一列为频率 (Hz)，从 0 到 Fs/2；第二列为对应的 PSD 值。
    fltrOrdr : int
        FIR 滤波器阶数（MATLAB 中 fir2 的 O）。
    sampFreq : float
        采样频率 Fs。

    Returns
    -------
    outNoise : np.ndarray, shape (nSamples,)
        生成的噪声实现。
    """
    psdVals = np.asarray(psdVals, float)
    freqVec = psdVals[:, 0]
    sqrtPSD = np.sqrt(psdVals[:, 1])

    nyq = sampFreq / 2.0
    # firwin2 频率需要归一化到 [0,1]
    f_norm = np.clip(freqVec / nyq, 0.0, 1.0)

    # MATLAB: b = fir2(fltrOrdr, freqVec/(Fs/2), sqrtPSD)
    b = firwin2(fltrOrdr + 1, f_norm, sqrtPSD)

    # 生成白噪声并通过滤波器（MATLAB 用 fftfilt；这里用等价的 FIR lfilter）
    inNoise = np.random.randn(nSamples)
    outNoise = lfilter(b, [1.0], inNoise)

    # MATLAB 中有 outNoise = sqrt(Fs) * fftfilt(...)，保留同样的缩放
    outNoise = np.sqrt(sampFreq) * outNoise

    return outNoise

def wgn_from_psd(N, psd_posfreq, fs, rng=None):
    """
    从 one-sided PSD 直接采样生成平稳高斯噪声。
    psd_posfreq: 长度 N//2+1，与 rfft 频点对齐；单位与 innerprodpsd 一致
    返回: 时域长度 N 的噪声，实现与 innerprodpsd 完全匹配
    """
    if rng is None:
        rng = np.random.default_rng()

    psd = np.asarray(psd_posfreq, float)
    df = fs / N
    K = len(psd)
    X = np.zeros(K, dtype=complex)

    # interior bins: 复高斯，方差 = S*df/2（实部、虚部分别）
    if K > 2:
        sigma = np.sqrt(psd[1:-1] * df / 2.0)
        X[1:-1] = rng.normal(0, sigma) + 1j * rng.normal(0, sigma)

    # DC
    X[0] = rng.normal(0, np.sqrt(psd[0] * df))
    # Nyquist（偶长）
    if N % 2 == 0:
        X[-1] = rng.normal(0, np.sqrt(psd[-1] * df))

    n = np.fft.irfft(X, n=N)
    return n
