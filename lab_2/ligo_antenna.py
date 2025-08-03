import numpy as np

def f_plus(theta, phi):
    # Calculate the plus polarization of the gravitational wave
    return 0.5 * (1 + np.cos(theta)**2) * np.cos(2 * phi)

def f_cross(theta, phi):
    # Calculate the cross polarization of the gravitational wave
    return np.cos(theta) * np.sin(2 * phi)