"""A set of numba accelerated quaternion functions.

Note that we follow the same order as :class:`~scipy.spatial.transform.Rotation` (x, y, z, w).
"""
import numpy as np
from numba import njit


@njit()
def rate_of_change_from_gyro(gyro: np.ndarray, current_orientation: np.ndarray) -> np.ndarray:
    """Rate of change of quaternion from gyroscope"""
    # TODO: Add test
    qx, qy, qz, qw = current_orientation
    qdot = np.empty(4)
    qdot[0] = qw * gyro[0] + qy * gyro[2] - qz * gyro[1]
    qdot[1] = qw * gyro[1] - qx * gyro[2] + qz * gyro[0]
    qdot[2] = qw * gyro[2] + qx * gyro[1] - qy * gyro[0]
    qdot[3] = -qx * gyro[0] - qy * gyro[1] - qz * gyro[2]

    return 0.5 * qdot
