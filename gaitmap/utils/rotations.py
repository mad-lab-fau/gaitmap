"""A set of util functions that ease handling rotations.

All util functions use :class:`scipy.spatial.transform.Rotation` to represent rotations.
"""
from typing import Union, Dict, Optional

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from numpy.linalg import norm

from gaitmap.utils.consts import SF_GYR, SF_ACC
from gaitmap.utils.dataset_helper import (
    get_multi_sensor_dataset_names,
    is_single_sensor_dataset,
    is_multi_sensor_dataset,
    Dataset,
    SingleSensorDataset,
)
from gaitmap.utils.vector_math import find_orthogonal, find_unsigned_3d_angle, normalize


def rotation_from_angle(axis: np.ndarray, angle: Union[float, np.ndarray]) -> Rotation:
    """Create a rotation based on a rotation axis and a angle.

    Parameters
    ----------
    axis : array with shape (3,) or (n, 3)
        normalized rotation axis ([x, y ,z]) or array of rotation axis
    angle : float or array with shape (n,)
        rotation angle or array of angeles in rad

    Returns
    -------
    rotation(s) : Rotation object with len n

    Examples
    --------
    Single rotation: 180 deg rotation around the x-axis

    >>> rot = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(180))
    >>> rot.as_quat().round(decimals=3)
    array([1., 0., 0., 0.])
    >>> rot.apply(np.array([[0, 0, 1.], [0, 1, 0.]])).round()
    array([[ 0., -0., -1.],
           [ 0., -1.,  0.]])

    Multiple rotations: 90 and 180 deg rotation around the x-axis

    >>> rot = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad([90, 180]))
    >>> rot.as_quat().round(decimals=3)
    array([[0.707, 0.   , 0.   , 0.707],
           [1.   , 0.   , 0.   , 0.   ]])
    >>> # In case of multiple rotations, the first rotation is applied to the first vector
    >>> # and the second to the second
    >>> rot.apply(np.array([[0, 0, 1.], [0, 1, 0.]])).round()
    array([[ 0., -1.,  0.],
           [ 0., -1.,  0.]])

    """
    angle = np.atleast_2d(angle)
    axis = np.atleast_2d(axis)
    return Rotation.from_rotvec(np.squeeze(axis * angle.T))


def _rotate_sensor(
    data: SingleSensorDataset, rotation: Optional[Rotation], inplace: bool = False
) -> SingleSensorDataset:
    """Rotate the data of a single sensor with acc and gyro."""
    if inplace is False:
        data = data.copy()
    if rotation is None:
        return data
    data[SF_GYR] = rotation.apply(data[SF_GYR].to_numpy())
    data[SF_ACC] = rotation.apply(data[SF_ACC].to_numpy())
    return data


def rotate_dataset(dataset: Dataset, rotation: Union[Rotation, Dict[str, Rotation]]) -> Dataset:
    """Apply a rotation to acc and gyro data of a dataset.

    Parameters
    ----------
    dataset
        dataframe representing a single or multiple sensors.
        In case of multiple sensors a df with MultiIndex columns is expected where the first level is the sensor name
        and the second level the axis names (all sensor frame axis must be present)
    rotation
        In case a single rotation object is passed, it will be applied to all sensors of the dataset.
        If a dictionary of rotations is applied, the respective rotations will be matched to the sensors based on the
        dict keys.
        If no rotation is provided for a sensor, it will not be modified.

    Returns
    -------
    rotated dataset
        This will always be a copy. The original dataframe will not be modified.

    Examples
    --------
    This will apply the same rotation to the left and the right foot

    >>> dataset = ...  # Dataset with a left and a right foot sensor
    >>> rotate_dataset(dataset, rotation=rotation_from_angle(np.array([0, 0, 1]), np.pi))
    <copy of dataset with all axis rotated>

    This will apply different rotations to the left and the right foot

    >>> dataset = ...  # Dataset with a left and a right foot sensor (sensors called "left" and "right")
    >>> rotate_dataset(dataset, rotation={'left': rotation_from_angle(np.array([0, 0, 1]), np.pi),
    ...     'right':rotation_from_angle(np.array([0, 0, 1]), np.pi / 2))
    <copy of dataset with all axis rotated>

    """
    if is_single_sensor_dataset(dataset):
        if isinstance(rotation, dict):
            raise ValueError(
                "A Dictionary for the `rotation` parameter is only supported if a MultiIndex dataset (named sensors) is"
                " passed."
            )
        return _rotate_sensor(dataset, rotation, inplace=False)

    if not is_multi_sensor_dataset(dataset):
        raise ValueError("The data input format is not supported gaitmap")
    rotation_dict = rotation
    if not isinstance(rotation_dict, dict):
        rotation_dict = {k: rotation for k in get_multi_sensor_dataset_names(dataset)}

    # TODO: Maybe refactor to be able to handle both types of input the same
    if isinstance(dataset, dict):
        rotated_dataset = {**dataset}
        for key in rotation_dict.keys():
            rotated_dataset[key] = _rotate_sensor(dataset[key], rotation_dict[key], inplace=False)
        return rotated_dataset

    original_cols = dataset.columns.copy()

    # For some strange reason, we need to unstack and stack again to use apply here:
    rotated_dataset = (
        dataset.stack(level=0)
        .groupby(level=1)
        .apply(lambda x: _rotate_sensor(x, rotation_dict.get(x.name, None), inplace=False))
        .unstack(level=1)
        .swaplevel(axis=1)
    )
    # Restore original order
    rotated_dataset = rotated_dataset[original_cols]
    return rotated_dataset


def find_shortest_rotation(v1: np.array, v2: np.array) -> Rotation:
    """Find a quaternion that rotates v1 into v2 via the shortest way.

    Parameters
    ----------
    v1 : vector with shape (3,)
        axis ([x, y ,z])
    v2 : vector with shape (3,)
        axis ([x, y ,z])

    Returns
    -------
    rotation
        Shortest rotation that rotates v1 into v2

    Examples
    --------
    >>> goal = np.array([0, 0, 1])
    >>> start = np.array([1, 0, 0])
    >>> rot = find_shortest_rotation(start, goal)
    >>> rotated = rot.apply(start)
    >>> rotated
    array([0., 0., 1.])

    """
    if (not np.isclose(norm(v1, axis=-1), 1)) or (not np.isclose(norm(v2, axis=-1), 1)):
        raise ValueError("v1 and v2 must be normalized")
    axis = find_orthogonal(v1, v2)
    angle = find_unsigned_3d_angle(v1, v2)
    return rotation_from_angle(axis, angle)


def get_gravity_rotation(
    gravity_vector: np.ndarray, expected_gravity: Optional[np.ndarray] = np.array([0.0, 0.0, 1.0])
) -> Rotation:
    """Find the rotation matrix needed to align  z-axis with gravity.

    Parameters
    ----------
    gravity_vector : vector with shape (3,)
        axis ([x, y ,z])
    expected_gravity : vector with shape (3,)
        axis ([x, y ,z])

    Returns
    -------
    rotation
        rotation between given gravity vector and the expected gravity

    Examples
    --------
    >>> goal = np.array([0, 0, 1])
    >>> start = np.array([1, 0, 0])
    >>> rot = get_gravity_rotation(start)
    >>> rotated = rot.apply(start)
    >>> rotated
    array([0., 0., 1.])

    """
    gravity_vector = normalize(gravity_vector)
    expected_gravity = normalize(expected_gravity)
    return find_shortest_rotation(gravity_vector, expected_gravity)
