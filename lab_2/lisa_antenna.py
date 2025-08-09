import numpy as np
from .ligo_antenna import rotate_by_psi
from lab_1.signal import generate_sinusoidal

# Earth's orbit
a = 1.496e11  # semi-major axis in meters
T = 365.25636 * 24 * 3600  # orbital period in seconds

# LISA spacecraft
L_default = 2.5e9  # distance between spacecraft in meters

c = 299792458  # speed of light in m/s

def orbit(t, L=L_default):
    """
    Compute the position of the Lisa spacecraft in a circular orbit at time t.
    
    Parameters:
    t : float or array-like
        Time in years.
    L : float, optional
        Distance between spacecraft in meters (default is 2.5e9 m).
    
    Returns:
    orbit_positions : np.ndarray
        Positions of the three spacecraft in a triangular formation (shape: (3, 3, N)).
    """
    e = L / (2 * a * 3**0.5)  # eccentricity
    alpha = 2 * np.pi * t
    beta = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])  # spacecraft phase angles
    orbit_positions = []
    for i in range(3):
        x = a * np.cos(alpha) + a * e * (np.sin(alpha)*np.cos(alpha)*np.sin(beta[i]) - (1+np.sin(alpha)**2) * np.cos(beta[i]))
        y = a * np.sin(alpha) + a * e * (np.sin(alpha)*np.cos(alpha)*np.cos(beta[i]) - (1+np.cos(alpha)**2) * np.sin(beta[i]))
        z = -3**0.5 * a * e * np.cos(alpha - beta[i])
        orbit_positions.append([x, y, z])

    orbit_positions = np.array(orbit_positions)
    return orbit_positions

def earth_orbit(t, phase=0):
    """
    Compute the position of the Earth in its orbit at time t.
    
    Parameters:
    t : float or array-like
        Time in years.
    
    Returns:
    np.ndarray
        Position of the Earth in its orbit (x, y, z).
    """
    alpha = 2 * np.pi * t + phase
    x = a * np.cos(alpha)
    y = a * np.sin(alpha)
    z = np.zeros_like(t)  # Assuming a circular orbit in the xy-plane
    return np.array([x, y, z])

def lisa_antenna_patterns_numerical(t, theta, phi, L=L_default, psi=0, eps=1e-15):
    """
    Compute the antenna pattern functions f_plus and f_cross for given theta, phi values.
    Supports only scalar inputs.
    Parameters:
    t : float or array-like
        Time in years.
    theta : float
        Polar angle in radians.
    phi : float
        Azimuthal angle in radians.
    L : float, optional
        Distance between spacecraft in meters (default is 2.5e9 m).
    psi : float, optional
        Polarization angle in radians (default is 0).
    eps : float, optional
        Small value to avoid division by zero (default is 1e-15).
    
    Returns:
    dict
        Dictionary containing the antenna pattern functions for two detectors:
        'F1_plus', 'F1_cross', 'F2_plus', 'F2_cross'.
    """
    # assert np.isscalar(theta) and np.isscalar(phi), "theta and phi must be scalars for this function."
    if not np.isscalar(theta) or not np.isscalar(phi):
        raise ValueError("theta and phi must be scalars for this function.")
    
    # Compute the positions of the spacecraft
    orbit_1, orbit_2, orbit_3 = orbit(t, L)
    arm_1 = orbit_1 - orbit_2
    arm_2 = orbit_2 - orbit_3
    arm_3 = orbit_3 - orbit_1

    # Normalize the arms
    arm_1 /= np.linalg.norm(arm_1, axis=0)
    arm_2 /= np.linalg.norm(arm_2, axis=0)
    arm_3 /= np.linalg.norm(arm_3, axis=0)

    # Compute the detector tensors for each arm
    o11 = np.einsum('it,jt->ijt', arm_1, arm_1)
    o22 = np.einsum('it,jt->ijt', arm_2, arm_2)
    o33 = np.einsum('it,jt->ijt', arm_3, arm_3)
    detector_1  = 0.5 * (o11 - o22)
    detector_2 = (1.0 / (2.0 * np.sqrt(3.0))) * (o11 + o22 - 2.0 * o33)

    # Compute the direction vector for the antenna
    direction = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])  # shape (3,)
    ref_z = np.array([0.0, 0.0, 1.0])
    ref_x = np.array([1.0, 0.0, 0.0])
    use_x = np.abs(direction[2]) > 0.9  # 接近极点时用 x 轴当参考
    ref = np.where(use_x, ref_x, ref_z)
    x_wave = np.cross(ref, direction)
    x_wave /= np.linalg.norm(x_wave)
    y_wave = np.cross(direction, x_wave)
    y_wave /= np.linalg.norm(y_wave)

    e_plus = np.outer(x_wave, x_wave) - np.outer(y_wave, y_wave)
    e_cross = np.outer(x_wave, y_wave) + np.outer(y_wave, x_wave)

    # 单台探测器：
    F1_plus  = np.einsum('ij,ijt->t', e_plus,  detector_1)
    F1_cross = np.einsum('ij,ijt->t', e_cross, detector_1)

    F2_plus  = np.einsum('ij,ijt->t', e_plus,  detector_2)
    F2_cross = np.einsum('ij,ijt->t', e_cross, detector_2)

    # 旋转到含 psi 的模式：
    # F_+^ψ = F_+ cos2ψ + F_× sin2ψ
    # F_×^ψ = -F_+ sin2ψ + F_× cos2ψ
    F1p_psi, F1x_psi = rotate_by_psi(F1_plus, F1_cross, psi)
    F2p_psi, F2x_psi = rotate_by_psi(F2_plus, F2_cross, psi)
    # 返回两个探测器的结果
    return {'F1_plus': F1p_psi, 'F1_cross': F1x_psi, 'F2_plus': F2p_psi, 'F2_cross': F2x_psi}

def lisa_strain(t, h_plus, h_cross, theta, phi, L=L_default, psi=0):
    """
    Calculate the LISA strain from the antenna patterns and gravitational wave polarizations.
    
    Parameters:
    t : float or array-like
        Time in years.
    h_plus : array-like
        Plus polarization of the gravitational wave.
    h_cross : array-like
        Cross polarization of the gravitational wave.
    theta : float or array-like
        Polar angle in radians.
    phi : float or array-like
        Azimuthal angle in radians.
    L : float, optional
        Distance between spacecraft in meters (default is 2.5e9 m).
    psi : float, optional
        Polarization angle in radians (default is 0).

    Returns:
    strain_1, strain_2 : np.ndarray
        Strain signals for the two detectors.
    """

    # Compute the antenna patterns
    patterns = lisa_antenna_patterns_numerical(t, theta, phi, L=L, psi=psi)

    strain_1 = patterns['F1_plus'] * h_plus + patterns['F1_cross'] * h_cross
    strain_2 = patterns['F2_plus'] * h_plus + patterns['F2_cross'] * h_cross

    return strain_1, strain_2

def lisa_doppler_sinusoid_strain(t, A, B, f0, phi0, theta, phi, L=L_default, psi=0):
    """
    Generate a sinusoidal strain signal for LISA with Doppler effect.
    
    Parameters:
    t : array-like
        Time in years.
    A : float
        Amplitude of the plus polarization.
    B : float
        Amplitude of the cross polarization.
    f0 : float
        Frequency in year^{-1}.
    phi0 : float
        Phase difference of the cross polarization in radians.
    theta : float
        Polar angle in radians.
    phi : float
        Azimuthal angle in radians.
    L : float, optional
        Distance between spacecraft in meters (default is 2.5e9 m).
    psi : float, optional
        Polarization angle in radians (default is 0).
    
    Returns:
    strain_1, strain_2 : np.ndarray
        Strain signals for the two detectors.
    """
    # Compute the direction vector
    direction = -np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

    lisa_centroid = earth_orbit(t, phase=0)
    t_doppler = t - np.einsum('i,it->t', direction, lisa_centroid) / c / T # Doppler shifted time in years

    h_plus = generate_sinusoidal(t_doppler, A, f0, 0)
    h_cross = generate_sinusoidal(t_doppler, B, f0, phi0)
    strain_1, strain_2 = lisa_strain(t, h_plus, h_cross, theta, phi, L=L, psi=psi)
    return strain_1, strain_2
