import numpy as np

def f_plus(theta, phi):
    # Calculate the plus polarization of the gravitational wave
    return 0.5 * (1 + np.cos(theta)**2) * np.cos(2 * phi)

def f_cross(theta, phi):
    # Calculate the cross polarization of the gravitational wave
    return np.cos(theta) * np.sin(2 * phi)

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

def antenna_patterns_numerical(theta, phi):
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

    # Define an orthonormal wave basis (x_wave, y_wave) for each direction
    x_wave = np.cross([0, 0, 1], direction, axis=-1)
    x_wave /= np.linalg.norm(x_wave, axis=-1, keepdims=True)
    y_wave = np.cross(direction, x_wave, axis=-1)
    y_wave /= np.linalg.norm(y_wave, axis=-1, keepdims=True)

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

    return f_plus, f_cross
