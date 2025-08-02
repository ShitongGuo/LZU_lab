import numpy as np

def generate_quadratic_chirp(t, A, a1, a2, a3):
    phi = a1 * t + a2 * t**2 + a3 * t**3
    return np.sin(2*np.pi*phi) * A