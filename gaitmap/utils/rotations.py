"""A set of util functions that ease handling rotations.

All util functions use :class:`scipy.spatial.transform.Rotation` to represent rotations.
"""
from typing import Union, Dict, Optional, List

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

from gaitmap.utils.consts import SF_GYR, SF_ACC
from gaitmap.utils.dataset_helper import (
    get_multi_sensor_dataset_names,
    is_single_sensor_dataset,
    is_multi_sensor_dataset,
    Dataset,
    SingleSensorDataset,
)
from gaitmap.utils.vector_math import find_orthogonal, normalize, row_wise_dot


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

    if isinstance(dataset, dict):
        rotated_dataset = {**dataset}
        original_cols = None
    else:
        rotated_dataset = dataset.copy()
        original_cols = dataset.columns
    for key in rotation_dict.keys():
        test = _rotate_sensor(dataset[key], rotation_dict[key], inplace=False)
        rotated_dataset[key] = test

    if isinstance(dataset, pd.DataFrame):
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


def find_rotation_around_axis(rot: Rotation, rotation_axis: Union[np.ndarray, List]) -> Rotation:
    """Calculate the rotation component of rot around the given rotation axis.

    This performs a swing-twist decomposition of the rotation quaternion [1]_.
    The returned rotation is the twist component of this decomposition.
    This is equivalent to the rotation around the rotation axis.

    Parameters
    ----------
    rot : single or multi rotation
        The rotation
    rotation_axis : (3,) or (n,3) vector
        The axis around which the rotation component should be extracted.
        In case a single rotation axis and multiple rotations are provided, the angle is extracted around the same
        axis for all rotations.
        In case n axis are provided for n rotations, the angle for each rotation is extracted around the respective
        axis.

    Examples
    --------
    >>> # Create composite rotation around y and z axis
    >>> rot = Rotation.from_rotvec([0, 0, np.pi / 2]) * Rotation.from_rotvec([0, np.pi / 4, 0 ])
    >>> find_rotation_around_axis(rot, [0, 0, 1]).as_rotvec()  # Extract part around z
    array([0.        , 0.        , 1.57079633])
    >>> find_rotation_around_axis(rot, [0, 1, 0]).as_rotvec()  # Extract part around y
    array([0.        , 0.78539816, 0.        ])

    Notes
    -----
    .. [1] https://www.euclideanspace.com/maths/geometry/rotations/for/decomposition/

    """
    # Get the rotation axis from the initial quaternion
    rotation_axis = np.atleast_2d(rotation_axis)
    quaternions = np.atleast_2d(rot.as_quat())
    if rotation_axis.shape[0] != quaternions.shape[0]:
        rotation_axis_equal_d = np.repeat(rotation_axis, quaternions.shape[0], axis=0)
    else:
        rotation_axis_equal_d = rotation_axis
    original_rot_axis = quaternions[:, :3]
    # Get projection of the axis onto the quaternion
    projection = (
        rotation_axis_equal_d * row_wise_dot(original_rot_axis, normalize(rotation_axis), squeeze=False)[:, None]
    )
    angle_component = np.atleast_2d(quaternions[:, -1]).T
    twist = Rotation.from_quat(np.squeeze(np.hstack((projection, angle_component))))
    return twist


def find_angle_between_orientations(
    ori: Rotation, ref: Rotation, rotation_axis: Optional[Union[np.ndarray, List]]
) -> Union[float, np.ndarray]:
    """Get the required rotation angle between two orientations.

    This will return the angle around the rotation axis that transforms ori into ref in this dimension.

    Parameters
    ----------
    ori : Single or rotation object with n rotations
        The initial orientation
    ref : Single or rotation object with n rotations
        The reference orientation
    rotation_axis : (3,) or (n, 3) vector
        The axis of rotation around which the angle is calculated.
        If None the shortest possible rotation angle between the two quaternions is returned.

    Returns
    -------
    angle
        The angle around the given axis in rad between -np.pi and np.pi.
        The sign is defined by the right-hand-rule around the provided axis ond the order or ori and ref.
        If no axis is provided, the angle will always be positive.

    Notes
    -----
    This function works for multiple possible combinations of input-dimensions:

    - ori: 1, ref: 1, rotation_axis: (3,) / (1,3) -> angle: float
    - ori: n, ref: 1, rotation_axis: (3,) / (1,3) / (n,3) -> angle: (n,)
    - ori: 1, ref: n, rotation_axis: (3,) / (1,3) / (n,3) -> angle: (n,)
    - ori: n, ref: n, rotation_axis: (3,) / (1,3) / (n,3) -> angle: (n,)

    """
    ori_to_ref = ori * ref.inv()
    if rotation_axis is not None:
        ori_to_ref = find_rotation_around_axis(ori_to_ref, rotation_axis)
    rotvec = ori_to_ref.as_rotvec()
    if rotation_axis is None:
        rotation_axis = rotvec
    out = row_wise_dot(rotvec, normalize(rotation_axis))
    if rotvec.ndim == 1:
        return out[0]
    return out


def find_unsigned_3d_angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Find the angle (in rad) between two  3D vectors.

    Parameters
    ----------
    v1 : vector with shape (3,)  or array of vectors
        axis ([x, y ,z]) or array of axis
    v2 : vector with shape (3,) or array of vectors
        axis ([x, y ,z]) or array of axis

    Returns
    -------
        angle or array of angles between two vectors

    Examples
    --------
    two vectors: 1D

    >>> find_unsigned_3d_angle(np.array([-1, 0, 0]), np.array([-1, 0, 0]))
    0

    two vectors: 2D

    >>> find_unsigned_3d_angle(np.array([[-1, 0, 0],[-1, 0, 0]]), np.array([[-1, 0, 0],[-1, 0, 0]]))
    array([0,0])

    """
    v1_, v2_ = np.atleast_2d(v1, v2)
    v1_ = normalize(v1_)
    v2_ = normalize(v2_)
    out = np.arccos(row_wise_dot(v1_, v2_) / (norm(v1_, axis=-1) * norm(v2_, axis=-1)))
    if v1.ndim == 1:
        return np.squeeze(out)
    return out
