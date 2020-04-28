"""A set of numba accelerated quaternion functions.

Note that we follow the same order as :class:`~scipy.spatial.transform.Rotation` (x, y, z, w).
"""
import numpy as np
from numba import njit


@njit()
def rate_of_change_from_gyro(gyro: np.ndarray, current_orientation: np.ndarray) -> np.ndarray:
    """Rate of change of quaternion from gyroscope"""
    gyro_q = np.zeros(4)
    gyro_q[:3] = gyro
    return 0.5 * _q_multiply(current_orientation, gyro_q)[0]


def q_multiply(q1, q2):
    out = _q_multiply(q1, q2)
    if q1.ndim == 1:
        return out[0]
    return out


@njit()
def _q_multiply(q1, q2) -> np.ndarray:
    """Vectorised quaternion multiplication

    Modified based on https://stackoverflow.com/questions/40915069/numpy-make-batched-version-of-quaternion
    -multiplication/40915759#40915759
    """
    q1, q2 = np.atleast_2d(q1), np.atleast_2d(q2)
    x0, y0, z0, w0 = q1.T
    x1, y1, z1, w1 = q2.T

    result = np.empty((len(q1), 4))
    result[:, 0] = x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0
    result[:, 1] = -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0
    result[:, 2] = x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
    result[:, 3] = -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0

    return result
