import numpy as np
from scipy.spatial.transform import Rotation


def align(sensor_data: np.ndarray, rotation: Rotation) -> np.ndarray:
    """Align sensor data to gravity, so that sensor "Z-Axis" will be parallel to gravity.

    Parameters
    ----------
    sensor_data : array with shape (n, 6)
        array containing sensor data

    rotation : int
        Rotation object which defines the rotation required to align the sensor data to gravity

    Returns
    -------
    Dataset aligned to gravity

    Examples
    --------
    >>> tbd

    """
    # TODO: implement
    return None


def get_rotation_to_gravity(sensor_data: np.ndarray) -> Rotation:
    """Extract the Rotation object relative to gravity from static sequences of the sensor data.

    Parameters
    ----------
    sensor_data : array with shape (n, 6)
        array containing sensor data

    Returns
    -------
    Rotation object relative to gravity

    Examples
    --------
    >>> tbd

    """
    # TODO: this function needs to assert if no static moment aka no rotation can be found!

    # TODO: implement
    return None
