import numpy as np

def f_plus(theta, phi):
    # Calculate the plus polarization of the gravitational wave
    return 0.5 * (1 + np.cos(theta)**2) * np.cos(2 * phi)

def f_cross(theta, phi):
    # Calculate the cross polarization of the gravitational wave
    return np.cos(theta) * np.sin(2 * phi)

def rotate_by_psi(f_plus, f_cross, psi):
    """
    Rotate the antenna pattern functions by the polarization angle psi.
    
    Parameters:
    f_plus : float or np.ndarray
        Antenna pattern function for plus polarization.
    f_cross : float or np.ndarray
        Antenna pattern function for cross polarization.
    psi : float
        Polarization angle in radians.

    Returns:
    f_psi_plus : float or np.ndarray
        Rotated antenna pattern function for plus polarization.
    f_psi_cross : float or np.ndarray
        Rotated antenna pattern function for cross polarization.
    """
    psi = np.asarray(psi)
    if psi.ndim == 0:
        c2, s2 = np.cos(2*psi), np.sin(2*psi)
    else:
        c2, s2 = np.cos(2*psi)[None, None, :], np.sin(2*psi)[None, None, :]
    return f_plus*c2 + f_cross*s2, -f_plus*s2 + f_cross*c2

# def antenna_patterns_numerical(theta, phi):
#     direction = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
#     x_wave = np.cross(np.array([0, 0, 1]), direction)
#     x_wave /= np.linalg.norm(x_wave)
#     y_wave = np.cross(direction, x_wave)
#     y_wave /= np.linalg.norm(y_wave)
#     e_plus = np.tensordot(x_wave, x_wave, axes=0) - np.tensordot(y_wave, y_wave, axes=0)
#     e_cross = np.tensordot(x_wave, y_wave, axes=0) + np.tensordot(y_wave, x_wave, axes=0)
#     detector = np.array([[0.5, 0, 0], [0, -0.5, 0], [0, 0, 0]])
#     f_plus_value = np.tensordot(detector, e_plus, axes=2)
#     f_cross_value = np.tensordot(detector, e_cross, axes=2)
#     return f_plus_value, f_cross_value

def antenna_patterns_numerical(theta, phi, psi=0, eps=1e-15):
    """
    Compute the antenna pattern functions f_plus and f_cross for given theta, phi arrays.
    Supports scalar inputs or meshgrid arrays of any shape.
    """
    # Stack the direction vectors along a new last axis
    direction = np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ], axis=-1)  # shape (..., 3)
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), eps)

    # # --- 安全的参考向量，避免在极点处叉乘为零 ---
    # ref_z = np.broadcast_to(np.array([0.0, 0.0, 1.0]), direction.shape)
    # ref_x = np.broadcast_to(np.array([1.0, 0.0, 0.0]), direction.shape)
    # use_x = (np.abs(direction[..., 2]) > 0.9)[..., None]  # 接近极点时用 x 轴当参考
    # ref = np.where(use_x, ref_x, ref_z)

    # # Define an orthonormal wave basis (x_wave, y_wave) for each direction
    # x_wave = np.cross(ref, direction, axis=-1)
    # x_wave /= np.maximum(np.linalg.norm(x_wave, axis=-1, keepdims=True), eps)
    # y_wave = np.cross(direction, x_wave, axis=-1)
    # y_wave /= np.maximum(np.linalg.norm(y_wave, axis=-1, keepdims=True), eps)
    x_wave = np.stack([np.cos(theta)*np.cos(phi),
                    np.cos(theta)*np.sin(phi),
                    -np.sin(theta)], axis=-1)        # ˆe_theta
    y_wave = np.stack([-np.sin(phi),
                    np.cos(phi),
                    np.zeros_like(theta)], axis=-1)   # ˆe_phi

    # Construct polarization tensors e_plus and e_cross
    # e_plus[i,j] = x_wave[i] x_wave[j] - y_wave[i] y_wave[j]
    # e_cross[i,j] = x_wave[i] y_wave[j] + y_wave[i] x_wave[j]
    e_plus = x_wave[..., :, None] * x_wave[..., None, :] - y_wave[..., :, None] * y_wave[..., None, :]
    e_cross = x_wave[..., :, None] * y_wave[..., None, :] + y_wave[..., :, None] * x_wave[..., None, :]

    # Detector tensor (fixed)
    detector = np.array([[0.5, 0, 0],
                         [0, -0.5, 0],
                         [0,   0,  0]])

    # Contract detector with polarization tensors to get scalar pattern functions
    f_plus  = np.einsum('...ij,ij->...', e_plus, detector)
    f_cross = np.einsum('...ij,ij->...', e_cross, detector)

    # Apply the polarization angle psi
    f_psi_plus, f_psi_cross = rotate_by_psi(f_plus, f_cross, psi)
    # f_psi_plus = f_plus * np.cos(2 * psi) + f_cross * np.sin(2 * psi)
    # f_psi_cross = -f_plus * np.sin(2 * psi) + f_cross * np.cos(2 * psi)

    return f_psi_plus, f_psi_cross

def antenna_patterns(theta, phi, psi=0):
    """
    Compute the antenna pattern functions f_plus and f_cross for given theta, phi arrays.
    Supports scalar inputs or meshgrid arrays of any shape.
    """
    f_plus_values = f_plus(theta, phi)
    f_cross_values = f_cross(theta, phi)
    
    # Apply the polarization angle psi
    f_psi_plus, f_psi_cross = rotate_by_psi(f_plus_values, f_cross_values, psi)
    # f_psi_plus = f_plus_values * np.cos(2 * psi) + f_cross_values * np.sin(2 * psi)
    # f_psi_cross = -f_plus_values * np.sin(2 * psi) + f_cross_values * np.cos(2 * psi)
    return f_psi_plus, f_psi_cross

def ligo_strain(h_plus, h_cross, theta, phi, psi=0):
    """
    Calculate the LIGO strain from the antenna patterns and gravitational wave polarizations.
    
    Parameters:
    theta : float or array-like
        Polar angle in radians.
    phi : float or array-like
        Azimuthal angle in radians.
    h_plus : float or array-like
        Plus polarization of the gravitational wave.
    h_cross : float or array-like
        Cross polarization of the gravitational wave.
    
    Returns:
    strain : float or array-like
        The LIGO strain signal.
    """
    f_plus, f_cross = antenna_patterns(theta, phi, psi)
    return f_plus * h_plus + f_cross * h_cross
