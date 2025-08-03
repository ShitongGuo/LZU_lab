import numpy as np

def gen_quadra_chirp_signal(dataX, snr, qcCoefs):
    """
    Generate a signal of the form:
        s(t) = A * sin(2πΦ(t)),
        where Φ(t) = a1*t + a2*t^2 + a3*t^3

    Parameters:
        dataX (np.ndarray): Time vector
        snr (float): Signal-to-noise ratio (used to scale amplitude), snr = A
        qcCoefs (list or np.ndarray): Coefficients [a1, a2, a3]

    Returns:
        sigVec (np.ndarray): Generated signal vector
    """
    # Ensure array types
    dataX = np.asarray(dataX)
    qcCoefs = np.asarray(qcCoefs)

    # Phase function: Φ(t)
    phaseVec = qcCoefs[0] * dataX + qcCoefs[1] * dataX**2 + qcCoefs[2] * dataX**3

    # Raw signal
    sigVec = np.sin(2 * np.pi * phaseVec)

    # Normalize and scale
    sigVec = snr * sigVec / np.linalg.norm(sigVec)

    return sigVec

def gen_linear_chirp_signal(dataX, snr, lcCoefs):
    """
    Generate a signal of the form:
        s(t) = A * sin(2π(f0*t + f1*t^2) + phi0)

    Parameters:
        dataX (np.ndarray): Time vector
        snr (float): Signal-to-noise ratio (used to scale amplitude), snr = A
        lcCoefs (list or np.ndarray): Coefficients [f0, f1, phi0]

    Returns:
        sigVec (np.ndarray): Generated signal vector
    """
    dataX = np.asarray(dataX)
    lcCoefs = np.asarray(lcCoefs)

    # Phase function
    phaseVec = 2 * np.pi * (lcCoefs[0] * dataX + lcCoefs[1] * dataX**2) + lcCoefs[2]

    # Raw signal
    sigVec = np.sin(phaseVec)

    # Normalize and scale
    sigVec = snr * sigVec / np.linalg.norm(sigVec)

    return sigVec

def gen_sine_Gauss_signal(dataX, snr, sGCoefs):
    """
    Generate a signal of the form:
        s(t) = A * exp(-((t-t0)^2)/(2*sigma^2))*sin(2πf0*t+phi0)

    Parameters:
        dataX (np.ndarray): Time vector
        snr (float): Signal-to-noise ratio (used to scale amplitude), snr = A
        sGCoefs (list or np.ndarray): Coefficients [t0, sigma, f0, phi0]

    Returns:
        sigVec (np.ndarray): Generated signal vector
    """
    dataX = np.asarray(dataX)
    sGCoefs = np.asarray(sGCoefs)

    # Guassian envelope
    envelope = np.exp(-((dataX - sGCoefs[0]) ** 2) / (2 * sGCoefs[1] ** 2))

    # Phase function
    phaseVec = 2 * np.pi * sGCoefs[2] * dataX + sGCoefs[3]

    # Raw signal
    sigVec = envelope * np.sin(phaseVec)
    
    # Normalize and scale
    sigVec = snr * sigVec / np.linalg.norm(sigVec)

    return sigVec

def gen_linear_transient_chirp_signal(dataX, snr, lintracCoefs):
    """
    Generate a signal of the form:
        s(t) = 0,                                          t ∉ [ta, ta+L]
             = A * sin(2π*(f0*(t-ta)+f1*(t-ta)^2)+phi0),   t ∈ [ta, ta+L]

    Parameters:
        dataX (np.ndarray): Time vector
        snr (float): Signal-to-noise ratio (used to scale amplitude), snr = A
        lintracCoefs (list or np.ndarray): Coefficients [f0, ta, f1, phi0, L]

    Returns:
        sigVec (np.ndarray): Generated signal vector
    """
    dataX = np.asarray(dataX)
    lintracCoefs = np.asarray(lintracCoefs)

    # Create empty data
    sigVec = np.zeros_like(dataX)

    # Select range
    mask = (dataX >= lintracCoefs[1]) & (dataX <= lintracCoefs[1] + lintracCoefs[4])

    # Generate chirp in active region
    t_active = dataX[mask] - lintracCoefs[1]
    sigVec[mask] = np.sin(2 * np.pi * (lintracCoefs[0] * t_active + lintracCoefs[2] * t_active**2) + lintracCoefs[3])

    # Test and normalize
    if np.linalg.norm(sigVec) > 0:
        sigVec = snr * sigVec / np.linalg.norm(sigVec)
    else :
        print("Warning: Signal norm is zero! Check parameter settings.")

    return sigVec
    
def gen_sinu_signal(dataX, snr, sinCoefs):
    """
    Generate a signal of the form:
        s(t) = A * sin(2πf0*t + phi0)

    Parameters:
        dataX (np.ndarray): Time vector
        snr (float): Signal-to-noise ratio (used to scale amplitude), snr = A
        sinCoefs (list or np.ndarray): Coefficients [f0, phi0]

    Returns:
        sigVec (np.ndarray): Generated signal vector
    """
    dataX = np.asarray(dataX)
    sinCoefs = np.asarray(sinCoefs)

    # Phase function
    phaseVec = 2 * np.pi * sinCoefs[0] * dataX + sinCoefs[1]

    # Raw signal
    sigVec = np.sin(phaseVec)

    # Normalize and scale
    sigVec = snr * sigVec / np.linalg.norm(sigVec)

    return sigVec
    
    


    