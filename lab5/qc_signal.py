import numpy as np
from dataclasses import dataclass

@dataclass
class QCParams:
    a1: float
    a2: float
    a3: float

def crcbgenqcsig(dataX: np.ndarray, snr: float, P: QCParams) -> np.ndarray:
    """
    Quadratic chirp signal (struct-style params).
    dataX : time stamps (seconds)
    snr   : scale so that ||s|| = snr (与原 MATLAB 实现一致)
    P     : QCParams(a1,a2,a3)
    """
    phase = P.a1*dataX + P.a2*dataX**2 + P.a3*dataX**3   # cycles
    sig = np.sin(2*np.pi*phase)
    norm = np.linalg.norm(sig)
    if norm == 0:
        return sig
    return snr * sig / norm

def crcbgenqcsig1(dataX: np.ndarray, snr: float, coefs: np.ndarray) -> np.ndarray:
    """
    Generate quadratic chirp signal.
    dataX : time stamps (seconds)
    snr   : scale so that ||s|| = snr
    coefs : [a1, a2, a3]
    """
    a1, a2, a3 = coefs
    phase = a1 * dataX + a2 * dataX**2 + a3 * dataX**3
    sig = np.sin(2 * np.pi * phase)
    norm = np.linalg.norm(sig)
    if norm == 0:
        return sig
    return snr * sig / norm

