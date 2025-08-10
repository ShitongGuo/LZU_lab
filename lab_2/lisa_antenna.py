import numpy as np
from .ligo_antenna import rotate_by_psi
from lab_1.signal import generate_sinusoidal

# Earth's orbit
a = 1.496e11  # semi-major axis in meters
T = 365.25636 * 24 * 3600  # orbital period in seconds

# LISA spacecraft
L_default = 2.5e9  # distance between spacecraft in meters

c = 299792458  # speed of light in m/s

def lisa_orbit(t, L=L_default):
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

# def lisa_antenna_patterns_numerical(t, theta, phi, L=L_default, psi=0, eps=1e-15):
#     """
#     Compute the antenna pattern functions f_plus and f_cross for given theta, phi values.
#     Supports only scalar inputs.
#     Parameters:
#     t : float or array-like
#         Time in years.
#     theta : float
#         Polar angle in radians.
#     phi : float
#         Azimuthal angle in radians.
#     L : float, optional
#         Distance between spacecraft in meters (default is 2.5e9 m).
#     psi : float, optional
#         Polarization angle in radians (default is 0).
#     eps : float, optional
#         Small value to avoid division by zero (default is 1e-15).
    
#     Returns:
#     dict
#         Dictionary containing the antenna pattern functions for two detectors:
#         'F1_plus', 'F1_cross', 'F2_plus', 'F2_cross'.
#     """
#     # assert np.isscalar(theta) and np.isscalar(phi), "theta and phi must be scalars for this function."
#     if not np.isscalar(theta) or not np.isscalar(phi):
#         raise ValueError("theta and phi must be scalars for this function.")
    
#     # Compute the positions of the spacecraft
#     orbit_1, orbit_2, orbit_3 = orbit(t, L)
#     arm_1 = orbit_1 - orbit_2
#     arm_2 = orbit_2 - orbit_3
#     arm_3 = orbit_3 - orbit_1

#     # Normalize the arms
#     arm_1 /= np.linalg.norm(arm_1, axis=0)
#     arm_2 /= np.linalg.norm(arm_2, axis=0)
#     arm_3 /= np.linalg.norm(arm_3, axis=0)

#     # Compute the detector tensors for each arm
#     o11 = np.einsum('it,jt->ijt', arm_1, arm_1)
#     o22 = np.einsum('it,jt->ijt', arm_2, arm_2)
#     o33 = np.einsum('it,jt->ijt', arm_3, arm_3)
#     detector_1  = 0.5 * (o11 - o22)
#     detector_2 = (1.0 / (2.0 * np.sqrt(3.0))) * (o11 + o22 - 2.0 * o33)

#     # Compute the direction vector for the antenna
#     direction = np.array([
#         np.sin(theta) * np.cos(phi),
#         np.sin(theta) * np.sin(phi),
#         np.cos(theta)
#     ])  # shape (3,)
#     ref_z = np.array([0.0, 0.0, 1.0])
#     ref_x = np.array([1.0, 0.0, 0.0])
#     use_x = np.abs(direction[2]) > 0.9  # 接近极点时用 x 轴当参考
#     ref = np.where(use_x, ref_x, ref_z)
#     x_wave = np.cross(ref, direction)
#     x_wave /= np.linalg.norm(x_wave)
#     y_wave = np.cross(direction, x_wave)
#     y_wave /= np.linalg.norm(y_wave)

#     e_plus = np.outer(x_wave, x_wave) - np.outer(y_wave, y_wave)
#     e_cross = np.outer(x_wave, y_wave) + np.outer(y_wave, x_wave)

#     # 单台探测器：
#     F1_plus  = np.einsum('ij,ijt->t', e_plus,  detector_1)
#     F1_cross = np.einsum('ij,ijt->t', e_cross, detector_1)

#     F2_plus  = np.einsum('ij,ijt->t', e_plus,  detector_2)
#     F2_cross = np.einsum('ij,ijt->t', e_cross, detector_2)

#     # 旋转到含 psi 的模式：
#     # F_+^ψ = F_+ cos2ψ + F_× sin2ψ
#     # F_×^ψ = -F_+ sin2ψ + F_× cos2ψ
#     F1p_psi, F1x_psi = rotate_by_psi(F1_plus, F1_cross, psi)
#     F2p_psi, F2x_psi = rotate_by_psi(F2_plus, F2_cross, psi)
#     # 返回两个探测器的结果
#     return {'F1_plus': F1p_psi, 'F1_cross': F1x_psi, 'F2_plus': F2p_psi, 'F2_cross': F2x_psi}

def lisa_antenna_patterns_numerical(orbit, theta, phi, psi=0.0, eps=1e-15):
    """
    计算全天 (theta, phi) 网格随时间的天线模式：
    输入:
        orbit: (3, T) 形状的数组，表示三个探测器的轨道位置
        theta: 极角（弧度），可以是标量或 (Nθ,) 数组
        phi: 方位角（弧度），可以是标量或 (Nφ,) 数组
        psi: 极化角（弧度），可以是标量或 (T,) 数组
        eps: 避免除零的极小值
    输出:
        F1_plus, F1_cross, F2_plus, F2_cross  形状均为 (Nθ, Nφ, T)
    """
    # 1) 轨道与臂单位向量 (3,T)
    orb1, orb2, orb3 = orbit
    arm1 = orb1 - orb2
    arm2 = orb2 - orb3
    arm3 = orb3 - orb1
    arm1 /= np.maximum(np.linalg.norm(arm1, axis=0, keepdims=True), eps)
    arm2 /= np.maximum(np.linalg.norm(arm2, axis=0, keepdims=True), eps)
    arm3 /= np.maximum(np.linalg.norm(arm3, axis=0, keepdims=True), eps)

    # 2) 两个等效探测器张量 D^I, D^II (3,3,T)
    o11 = np.einsum('it,jt->ijt', arm1, arm1)
    o22 = np.einsum('it,jt->ijt', arm2, arm2)
    o33 = np.einsum('it,jt->ijt', arm3, arm3)
    D1  = 0.5 * (o11 - o22)
    D2  = (1.0/(2.0*np.sqrt(3.0))) * (o11 + o22 - 2.0*o33)

    # 3) 广播 theta/phi 到同形状（既兼容标量也兼容网格）
    TH, PH = np.broadcast_arrays(theta, phi)

    # 4) 全天/单点方向基 (...,3)
    direction = np.stack([
        np.sin(TH) * np.cos(PH),
        np.sin(TH) * np.sin(PH),
        np.cos(TH)
    ], axis=-1)

    # # 5) 构造波参考基并规避极点奇异
    # ref_z = np.array([0.0, 0.0, 1.0])
    # ref_x = np.array([1.0, 0.0, 0.0])
    # use_x = (np.abs(direction[..., 2]) > 0.9)[..., None]          # (...,1)
    # # 自动广播到 (...,3)
    # ref = np.where(use_x, ref_x, ref_z)

    # x_wave = np.cross(ref, direction)
    # x_wave /= np.maximum(np.linalg.norm(x_wave, axis=-1, keepdims=True), eps)
    # y_wave = np.cross(direction, x_wave)
    # y_wave /= np.maximum(np.linalg.norm(y_wave, axis=-1, keepdims=True), eps)
    # 5) 直接用球面正交基，避免极点切换带来的不连续
    x_wave = np.stack([np.cos(TH)*np.cos(PH),
                    np.cos(TH)*np.sin(PH),
                    -np.sin(TH)], axis=-1)        # ˆe_theta
    y_wave = np.stack([-np.sin(PH),
                    np.cos(PH),
                    np.zeros_like(TH)], axis=-1)   # ˆe_phi

    # 6) 偏振张量 e_plus/e_cross 形状 (...,3,3)
    e_plus  = x_wave[..., :, None] * x_wave[..., None, :] - y_wave[..., :, None] * y_wave[..., None, :]
    e_cross = x_wave[..., :, None] * y_wave[..., None, :] + y_wave[..., :, None] * x_wave[..., None, :]

    # 7) 与探测器缩并 -> (...,T)
    F1p = np.einsum('...ij,ijt->...t', e_plus,  D1)
    F1x = np.einsum('...ij,ijt->...t', e_cross, D1)
    F2p = np.einsum('...ij,ijt->...t', e_plus,  D2)
    F2x = np.einsum('...ij,ijt->...t', e_cross, D2)

    # 8) 极化角旋转（psi 可为标量或 (T,)）
    F1p, F1x = rotate_by_psi(F1p, F1x, psi)
    F2p, F2x = rotate_by_psi(F2p, F2x, psi)
    return F1p, F1x, F2p, F2x

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
    lisa_positions = lisa_orbit(t, L=L)
    F1p, F1x, F2p, F2x = lisa_antenna_patterns_numerical(lisa_positions, theta, phi, psi=psi)

    strain_1 = F1p * h_plus + F1x * h_cross
    strain_2 = F2p * h_plus + F2x * h_cross

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
