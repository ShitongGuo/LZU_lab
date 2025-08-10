# glrtqcsig.py
import numpy as np
from glrtcalc import normsig4psd  # 或你存放 normsig4psd 的位置
# crcbgenqcsig 只用于生成注入信号
from glrtcalc import crcbgenqcsig
from glrtcalc import innerprodpsd
# 需要用到你已有的这两个工具（与 glrtqcsig 同一实现）

def glrtqcsig(data_vec: np.ndarray,
              time_vec: np.ndarray,
              fs: float,
              psd_posfreq: np.ndarray,
              a1: float, a2: float, a3: float) -> float:
    """
    GLRT for a quadratic chirp with unknown amplitude.
    Inputs:
      data_vec     : y (N,)
      time_vec     : t (N,)
      fs           : sampling frequency
      psd_posfreq  : one-sided PSD for positive DFT freqs, length N//2+1
      a1, a2, a3   : quadratic chirp parameters
    Output:
      scalar GLRT value
    """
    # Generate template s(t; a1,a2,a3) (amplitude arbitrary; will be normalized)
    sig_vec = crcbgenqcsig(time_vec, 1.0, (a1, a2, a3))

    # Unit-norm template under given PSD
    template_vec, _ = normsig4psd(sig_vec, fs, psd_posfreq, 1.0)

    # GLRT (unknown amplitude): ( <y, q(theta)> )^2
    llr = innerprodpsd(data_vec, template_vec, fs, psd_posfreq)
    return float(llr**2)

