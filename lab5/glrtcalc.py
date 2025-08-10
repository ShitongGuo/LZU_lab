# glrtcalc.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, firwin2, lfilter

# ===== Helper functions (与 MATLAB 同名/同义) =====

def crcbgenqcsig(time_vec: np.ndarray, snr: float, a1a2a3) -> np.ndarray:
    """Quadratic chirp: sin(2π(a1 t + a2 t^2 + a3 t^3)), 再按 L2 范数归一到给定 snr。"""
    a1, a2, a3 = a1a2a3
    phase = a1*time_vec + a2*time_vec**2 + a3*time_vec**3
    s = np.sin(2*np.pi*phase)
    nrm = np.linalg.norm(s)
    return s if nrm == 0 else snr * s / nrm

def innerprodpsd(x: np.ndarray, y: np.ndarray, fs: float, psd_posfreq: np.ndarray) -> float:
    """PSD 加权内积（与 one-sided PSD 匹配）。"""
    N = len(x)
    df = fs / N
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)
    S = np.asarray(psd_posfreq, float)
    w = np.full_like(S, 4.0)  # 内部频点权重 4
    w[0] = 2.0                # DC
    if N % 2 == 0:
        w[-1] = 2.0           # Nyquist（偶长时存在）
    return float(np.sum(w * (X * np.conj(Y)).real / S) * df)

def normsig4psd(sig: np.ndarray, fs: float, psd_posfreq: np.ndarray, A: float = 1.0):
    """返回单位范数模板并按 A 缩放（匹配滤波 SNR）。"""
    nrm2 = innerprodpsd(sig, sig, fs, psd_posfreq)
    nrm = np.sqrt(max(nrm2, 1e-18))
    template_unit = sig / nrm
    return A * template_unit, (A / nrm)

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

# ====== 按 GLRTcalc.m 的主流程 ======
if __name__ == "__main__":
    # --- Parameters for data realization ---
    nSamples = 2048
    sampFreq = 1024.0
    timeVec = np.arange(nSamples) / sampFreq

    # --- Supply PSD values ---
    # noisePSD(f) = 1 + 三角凸起（100~300 Hz）
    def noisePSD(f):
        f = np.asarray(f, float)
        tri = ((f >= 100) & (f <= 300)) * (f - 100) * (300 - f) / 10000.0
        return tri + 1.0

    dataLen = nSamples / sampFreq
    kNyq = nSamples // 2 + 1
    posFreq = np.arange(kNyq) * (1.0 / dataLen)
    psdPosFreq = noisePSD(posFreq)

    # --- Generate data realization ---
    a1, a2, a3 = 9.5, 2.8, 3.2
    A = 2.0  # SNR
    sig4data = crcbgenqcsig(timeVec, 1.0, (a1, a2, a3))
    sig4data, _ = normsig4psd(sig4data, sampFreq, psdPosFreq, A)

    freq_psd_pairs = np.column_stack([posFreq, psdPosFreq])
    noiseVec = statgaussnoisegen(nSamples, freq_psd_pairs, 1000, fs=sampFreq)
    dataVec = noiseVec + sig4data

    # --- Plots: time series ---
    plt.figure()
    plt.plot(timeVec, dataVec)
    plt.plot(timeVec, sig4data)
    plt.xlabel("Time (sec)")
    plt.ylabel("Data")
    plt.title("Data realization for calculation of LR")

    # --- Plots: periodograms (magnitude of FFT) ---
    kNyq = nSamples // 2 + 1
    dataLen = nSamples / sampFreq
    posFreq = np.arange(kNyq) * (1.0 / dataLen)
    datFFT = np.abs(np.fft.fft(dataVec))
    sigFFT = np.abs(np.fft.fft(sig4data))
    plt.figure()
    plt.plot(posFreq, datFFT[:kNyq])
    plt.plot(posFreq, sigFFT[:kNyq])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Periodogram of data")

    # --- Plots: spectrogram ---
    f, t, S = spectrogram(dataVec, fs=sampFreq, nperseg=64, noverlap=60, mode="complex")
    plt.figure()
    plt.imshow(np.abs(S), aspect="auto", origin="lower",
               extent=[t.min(), t.max(), f.min(), f.max()])
    plt.xlabel("Time (sec)")
    plt.ylabel("Frequency (Hz)")

    # --- Compute GLRT ---
    sigVec = crcbgenqcsig(timeVec, 1.0, (a1, a2, a3))
    templateVec, _ = normsig4psd(sigVec, sampFreq, psdPosFreq, 1.0)
    llr = innerprodpsd(dataVec, templateVec, sampFreq, psdPosFreq)
    llr = llr**2
    print(llr)

    # --- Estimate GLRT distributions under H0 and H1 ---
    nRlz = 500
    glrtH0 = np.zeros(nRlz)
    glrtH1 = np.zeros(nRlz)

    rng = np.random.default_rng(0)

    # H0: pure noise
    for lpr in range(nRlz):
        noiseVec = statgaussnoisegen(nSamples, freq_psd_pairs,
                                     100, fs=sampFreq)
        llr0 = innerprodpsd(noiseVec, templateVec, sampFreq, psdPosFreq)
        glrtH0[lpr] = llr0**2

    # H1: noise + signal
    for lpr in range(nRlz):
        noiseVec = statgaussnoisegen(nSamples, freq_psd_pairs,
                                     100, fs=sampFreq)
        dataVec = noiseVec + sig4data
        llr1 = innerprodpsd(dataVec, templateVec, sampFreq, psdPosFreq)
        glrtH1[lpr] = llr1**2

    # --- Histograms ---
    plt.figure()
    plt.hist(glrtH0, bins=30, density=True, alpha=0.6, label="H0")
    plt.hist(glrtH1, bins=30, density=True, alpha=0.6, label="H1")
    plt.legend()
    plt.xlabel("GLRT")

    plt.show()

