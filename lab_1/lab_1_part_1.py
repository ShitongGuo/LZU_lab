import numpy as np

def generate_quadratic_chirp(t, A, a1, a2, a3):
    # Generate a quadratic chirp signal
    phi = a1 * t + a2 * t**2 + a3 * t**3
    return A * np.sin(2*np.pi*phi)

def generate_sinusoidal(t, A, f0, phi0):
    return A * np.sin(2 * np.pi * f0 * t + phi0)

def generate_linear_chirp(t, A, f0, f1, phi0):
    phi = f0 * t + f1 * t**2
    return A * np.sin(2 * np.pi * phi + phi0)

def generate_sine_gaussian(t, A, t0, sigma, f0, phi0):
    exp_term = np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))
    return A * exp_term * np.sin(2 * np.pi * f0 * t + phi0)

def Nyquist_frequency_of_quadratic_chirp(a1, a2, a3, T):
    # The Nyquist frequency is half the sampling rate
    # For a quadratic chirp, the maximum frequency is determined by the coefficients
    # Assuming a1, a2, a3 are such that the maximum frequency is f_max
    # Sampling frequency should be at least twice the maximum frequency
    f_max = a1 + 2 * a2 * T + 3 * a3 * T**2
    return f_max

def find_pow_2(N):
    # Find the smallest power of 2 greater than or equal to N
    pow_2 = 1
    while pow_2 < N:
        pow_2 *= 2
    return pow_2