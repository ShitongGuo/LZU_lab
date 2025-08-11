import numpy as np

def generate_quadratic_chirp(t, A, a1, a2, a3):
    phi = a1 * t +a2 * t**2 +a3 * t**3
    return A * np.sin(2 * np.pi * phi)

def generate_sinusoidal(t, A, f0, phi0):
    return A * np.sin(2 * np.pi * f0 * t + phi0)

def generate_linear_chirp(t, A, f0, f1, phi0):
    phi = f0 * t + f1 * t**2
    return A * np.sin(2 * np.pi * phi +phi0)

def generate_sine_gaussian(t, A, t0, sigma, f0, phi0):
    exp_term = np.exp(-((t - t0)**2)/(2 * sigma**2))
    return A * exp_term * np.sin(2 * np.pi * f0 * t +phi0)

def generate_frequency_modulated_sinusid(t, A, b, f0, f1):
    cos = np.cos(2 * np.pi * f1 * t)
    return A * np.sin(2 * np.pi * f0 * t + b * cos)

def generate_amplitude_modulated_sinusoid(t, A, f0, f1, phi0):
    return A * np.cos(2 * np.pi * f1 * t) * np.sin(f0 * t + phi0)

def generate_AMFM_sinusoid(t, A, b, f0, f1):
    phi = 2 * np.pi * f1 * t
    phi1 = 2 * np.pi * f0 * t
    return A * np.cos(phi) * np.sin(phi1 + b * cos(phi))

def find_pow_2(N):
    # Find the smallest power of 2 greater than or equal to N
    pow_2 = 1
    while pow_2 < N:
        pow_2 *= 2
    return pow_2

def Nyquist_frequency_of_quadratic_chirp(a1, a2, a3, T):
    # The Nyquist frequency is half the sampling rate
    # For a quadratic chirp, the maximum frequency is determined by the coefficients
    # Assuming a1, a2, a3 are such that the maximum frequency is f_max
    # Sampling frequency should be at least twice the maximum frequency
    f_max = a1 + 2 * a2 * T + 3 * a3 * T**2
    return f_max

def innerprodpsd(xVec: np.ndarray, yVec: np.ndarray, sampFreq: float, psdVals: np.ndarray) -> float:
    """
    Calculates the PSD-weighted inner product of two real-valued vectors xVec and yVec,
    analogous to the MATLAB innerprodpsd function.

    Parameters
    ----------
    xVec : ndarray
        Time-domain signal (length N).
    yVec : ndarray
        Time-domain signal (length N).
    sampFreq : float
        Sampling frequency Fs.
    psdVals : ndarray
        1D array of length floor(N/2)+1 containing the one-sided PSD values
        at positive DFT frequencies [0 .. Fs/2].

    Returns
    -------
    innProd : float
        The real-valued inner product: (1/(Fs*N)) * fft(x)/Sn * conj(fft(y)) summed over all freqs.
    """
    xVec = np.asarray(xVec)
    yVec = np.asarray(yVec)
    N = xVec.size

    if yVec.size != N:
        raise ValueError("xVec and yVec must have the same length")

    kNyq = N//2 + 1
    if psdVals.size != kNyq:
        raise ValueError("psdVals must have length floor(N/2)+1")

    # Compute full PSD array (two-sided) by mirroring positive side
    # For even N, omit the Nyquist duplicate; for odd N, include all.
    negFStart = 1 - (N % 2)
    # psdVals: [0 ... Nyquist]; we mirror indices kNyq-negFStart down to 2
    psd_full = np.concatenate([
        psdVals,
        psdVals[kNyq-negFStart : 1 : -1]
    ])

    # FFT of both signals
    FFTx = np.fft.fft(xVec)
    FFTy = np.fft.fft(yVec)

    # Compute weighted inner product
    dataLen = sampFreq * N


    prod = (1.0 / dataLen) * np.vdot(FFTx / psd_full, FFTy)
    # np.vdot computes conj(first) * second and sums
    return float(np.real(prod))

def statgaussnoisegen(nSamples, psdVals, fltrOrdr, sampFreq):
    """
    Generate a realization of stationary Gaussian noise with a given two-sided PSD.
    
    Parameters
    ----------
    nSamples : int
        Number of time-domain samples to generate.
    psdVals : array_like, shape (M,2)
        Two-column array: first column is frequencies [0 ... Fs/2], 
        second column is the desired two-sided PSD at those freqs.
    fltrOrdr : int
        Order of the FIR filter to approximate sqrt(PSD).
    sampFreq : float
        Sampling frequency Fs.
    
    Returns
    -------
    outNoise : ndarray, shape (nSamples,)
        Time-domain noise realization with the target PSD.
    """
    # Extract frequency grid and PSD values
    freqVec = psdVals[:, 0]            # Frequencies from 0 to Fs/2
    psd_vals = psdVals[:, 1]           # Desired two-sided PSD at freqVec

    # Desired amplitude response is sqrt(PSD)
    sqrtPSD = np.sqrt(psd_vals)

    # Design FIR filter: numtaps = filter order + 1
    numtaps = fltrOrdr + 1
    # firwin2 takes frequency points normalized to [0, fs/2], so:
    #   freqVec must be monotonic in [0, sampFreq/2]
    #   gain points = sqrtPSD
    # Use fs parameter introduced in scipy ≥1.2.0
    b = firwin2(numtaps,
                freqVec,
                sqrtPSD,
                fs=sampFreq)

    # Generate white Gaussian noise (zero mean, unit variance)
    inNoise = np.random.randn(nSamples)

    # Filter via FFT-based convolution (equivalent to fftfilt)
    # Scale by sqrt(sampFreq) to match MATLAB's sqrt(Fs)*fftfilt(...)
    out = fftconvolve(inNoise, b, mode='same') * np.sqrt(sampFreq)

    return out


def glrtqcsig(y, t, fs, psdPosFreq, params):
    a1, a2, a3 = params
    s = generate_quadratic_chirp(t, 1.0, a1, a2, a3)
    norm_s_sq = innerprodpsd(s, s, fs, psdPosFreq)
    template = s / np.sqrt(norm_s_sq)
    llr = innerprodpsd(y, template, fs, psdPosFreq)
    return llr**2