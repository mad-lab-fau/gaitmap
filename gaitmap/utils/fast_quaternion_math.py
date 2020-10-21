"""A set of numba accelerated quaternion functions.

Note that we follow the same order as :class:`~scipy.spatial.transform.Rotation` (x, y, z, w).
"""
import numpy as np
from numba import njit


@njit()
def rate_of_change_from_gyro(gyro: np.ndarray, current_orientation: np.ndarray) -> np.ndarray:
    """Rate of change of quaternion from gyroscope."""
    qx, qy, qz, qw = current_orientation
    qdot = np.empty(4)
    qdot[0] = qw * gyro[0] + qy * gyro[2] - qz * gyro[1]
    qdot[1] = qw * gyro[1] - qx * gyro[2] + qz * gyro[0]
    qdot[2] = qw * gyro[2] + qx * gyro[1] - qy * gyro[0]
    qdot[3] = -qx * gyro[0] - qy * gyro[1] - qz * gyro[2]

    return 0.5 * qdot


@njit()
def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two quaternions."""
    r = np.empty(4)
    r[3] = a[3] * b[3] - np.dot(a[:3], b[:3])
    r[:3] = a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3])
    return r


@njit()
def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by a quaternion.

    Formula from chapter 3 of http://graphics.stanford.edu/courses/cs348a-17-winter/Papers/quaternion.pdf
    """
    w = q[3]
    u = q[:3]
    return 2.0 * (np.dot(u, v) * u + w * np.cross(u, v)) + (w ** 2 - np.dot(u, u)) * v


@njit()
def quat_from_rotvec(sigma: np.ndarray) -> np.ndarray:
    """Construct a quaternion from a rotation vector."""
    a_c = np.cos(np.linalg.norm(sigma) / 2)
    a_s = np.sin(np.linalg.norm(sigma) / 2) / np.linalg.norm(sigma)
    return np.append(a_s * sigma, a_c)
