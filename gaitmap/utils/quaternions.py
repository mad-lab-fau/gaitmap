"""A set of functions to perform arithmetics with quaternions."""
import numpy as np


def exponential_form(q: np.ndarray) -> np.ndarray:
    """Get the exponential form of a quaternion to later multiply it with a second quaternion.

    Parameters
    ----------
    q
        quaternion to be represented in exponential form.

    Notes
    -----
    The exponential form is for example used when integration the gyroscope signal as described by Sabatini et al. [1].
    This code was taken from Markus Zrenner's "IMU Running Algorithms".

    """
    # TODO:
    #  - Check if this expects [w x y z] or [x y z w],
    #  - adapt code (if necessary) and add to documentation
    #  - specify it in docstring
    quaternion = np.zeros(4)
    multiplication_factor = np.exp(q[0])
    squared_helper = np.sqrt(q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    quaternion[0] = multiplication_factor * np.cos(squared_helper)
    quaternion[1] = q[1] * multiplication_factor * np.sin(squared_helper) / squared_helper
    quaternion[2] = q[2] * multiplication_factor * np.sin(squared_helper) / squared_helper
    quaternion[3] = q[3] * multiplication_factor * np.sin(squared_helper) / squared_helper
    quaternion = quaternion / np.linalg.norm(quaternion, 2)
    return quaternion


def multiply(left_quaternion: np.ndarray, right_quaternion: np.ndarray) -> np.ndarray:
    """Multiply two quaternions.

    Parameters
    ----------
    left_quaternion
        The quaternion on the left side of the mulitplication operator

    right_quaternion
        The quaternion on the right side of the multiplication operator

    Notes
    -----
    Implemented as described in: Hanson, A. J. 2006. Visualizing quaternions. Morgan Kaufmann series in
    interactive 3D technology. Morgan Kaufmann, San Francisco, CA, Amsterdam, Boston.

    """
    # TODO: Add examples
    p0, p1, p2, p3 = left_quaternion
    q0, q1, q2, q3 = right_quaternion
    result = np.array(
        [
            p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3,
            p1 * q0 + p0 * q1 + p2 * q3 - p3 * q2,
            p2 * q0 + p0 * q2 + p3 * q1 - p1 * q3,
            p3 * q0 + p0 * q3 + p1 * q2 - p2 * q1,
        ]
    )
    return np.divide(result, np.linalg.norm(result, 2))
