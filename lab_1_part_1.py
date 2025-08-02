import numpy as np

def generate_quadratic_chirp(t, A, a1, a2, a3):
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