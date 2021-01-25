"""A set of util functions that ease handling rotations.

All util functions use :class:`scipy.spatial.transform.Rotation` to represent rotations.
"""
from typing import Union, Dict, Optional, List

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

from gaitmap.utils.consts import SF_GYR, SF_ACC, GRAV_VEC
from gaitmap.utils.datatype_helper import (
    get_multi_sensor_names,
    is_single_sensor_data,
    SensorData,
    SingleSensorData,
    is_sensor_data,
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


def _rotate_sensor(data: SingleSensorData, rotation: Optional[Rotation], inplace: bool = False) -> SingleSensorData:
    """Rotate the data of a single sensor with acc and gyro."""
    if inplace is False:
        data = data.copy()
    if rotation is None:
        return data
    data[SF_GYR] = rotation.apply(data[SF_GYR].to_numpy())
    data[SF_ACC] = rotation.apply(data[SF_ACC].to_numpy())
    return data


def rotate_dataset(dataset: SensorData, rotation: Union[Rotation, Dict[str, Rotation]]) -> SensorData:
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

    >>> dataset = ...  # Sensordata with a left and a right foot sensor
    >>> rotate_dataset(dataset, rotation=rotation_from_angle(np.array([0, 0, 1]), np.pi))
    <copy of dataset with all axis rotated>

    This will apply different rotations to the left and the right foot

    >>> dataset = ...  # Sensordata with a left and a right foot sensor (sensors called "left" and "right")
    >>> rotate_dataset(dataset, rotation={'left': rotation_from_angle(np.array([0, 0, 1]), np.pi),
    ...     'right':rotation_from_angle(np.array([0, 0, 1]), np.pi / 2))
    <copy of dataset with all axis rotated>

    See Also
    --------
    gaitmap.utils.rotations.rotate_dataset_series: Apply a series of rotations to a dataset

    """
    dataset_type = is_sensor_data(dataset, frame="sensor")
    if dataset_type == "single":
        if isinstance(rotation, dict):
            raise ValueError(
                "A Dictionary for the `rotation` parameter is only supported if a MultiIndex dataset (named sensors) is"
                " passed."
            )
        return _rotate_sensor(dataset, rotation, inplace=False)

    rotation_dict = rotation
    if not isinstance(rotation_dict, dict):
        rotation_dict = {k: rotation for k in get_multi_sensor_names(dataset)}

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


def rotate_dataset_series(dataset: SingleSensorData, rotations: Rotation) -> pd.DataFrame:
    """Rotate data of a single sensor using a series of rotations.

    This will apply a different rotation to each sample of the dataset.

    Parameters
    ----------
    dataset
        Data with axes names as `SF_COLS` in :mod:`~gaitmap.utils.consts`
    rotations
        Rotation object that contains as many rotations as there are datapoints

    Returns
    -------
    rotated_data
        copy of `data` rotated by `rotations`

    See Also
    --------
    gaitmap.utils.rotations.rotate_dataset: Apply a single rotation to the entire dataset

    """
    is_single_sensor_data(dataset, frame="sensor", raise_exception=True)
    if len(dataset) != len(rotations):
        raise ValueError("The number of rotations must fit the number of samples in the dataset!")

    return _rotate_sensor(dataset, rotations, inplace=False)


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


def get_gravity_rotation(gravity_vector: np.ndarray, expected_gravity: Optional[np.ndarray] = GRAV_VEC) -> Rotation:
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
    if np.any(np.all(rotation_axis == 0, axis=1)):
        raise ValueError("All rotation axis must not be [0, 0, 0].")
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
    if np.all(projection == 0):
        # In case the rotation axis is orthogonal to the original rotation return identity rotation
        angle_component = np.ones(shape=angle_component.shape)
    twist = Rotation.from_quat(np.squeeze(np.hstack((projection, angle_component))))
    return twist


def find_angle_between_orientations(
    ori: Rotation, ref: Rotation, rotation_axis: Optional[Union[np.ndarray, List]] = None
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
    # In case any of the rotation axis, was still 0, the angle will be None indicating an angle of zero
    out[np.isnan(out)] = 0.0
    if rotvec.ndim == 1:
        return out[0]
    return out


def find_unsigned_3d_angle(v1: np.ndarray, v2: np.ndarray) -> Union[np.ndarray, float]:
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


def angle_diff(a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate the real distance bewteen two signed angle values.

    This returns the shorter of the two possible angle differences on a unit circle.

    Parameters
    ----------
    a
        The first angle value in rad
    b
        The second angle value in rad

    Examples
    --------
    >>> np.round(angle_diff(0, -np.pi / 2), 2)
    1.57

    >>> np.round(angle_diff(-np.pi / 2, 0), 2)
    -1.57

    >>> np.round(angle_diff(1.5 * np.pi, 0), 2)
    -1.57

    >>> angle_diff(-np.pi, +np.pi)
    0.0

    >>> np.round(angle_diff(np.array([-np.pi / 2, np.pi / 2]), np.array([0, 0])), 2)
    array([-1.57,  1.57])

    """
    diff = a - b
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff


def find_signed_3d_angle(v1: np.ndarray, v2: np.ndarray, rotation_axis: np.ndarray) -> Union[float, np.ndarray]:
    """Find the signed angle (in rad) between two 3D vectors.

    Signed means that the angle varies between -180 and 180 deg (or rather -pi and pi)
    This implementation uses acrtan2 to calculate the angle.

    Parameters
    ----------
    v1
        2D or 3D vector or series of vectors
    v2
        2D or 3D vector or series of vectors
    rotation_axis
        Axis the rotation is performed around. The direction of this axis also indicates the sign of the angle

    """
    v1 = normalize(v1)
    v2 = normalize(v2)

    single = False
    if v1.ndim == 1:
        single = True

    v1 = np.atleast_2d(v1)
    v2 = np.broadcast_to(v2, v1.shape)
    # If on of the vectors is a 2D vector instead of a 3D vector, add a 0 as z-value
    if v1.shape[-1] == 2:
        v1 = np.pad(v1, ((0, 0), (0, 1)), mode="constant", constant_values=0)
    if v2.shape[-1] == 2:
        v2 = np.pad(v2, ((0, 0), (0, 1)), mode="constant", constant_values=0)

    rotation_axis = normalize(rotation_axis)

    angle = np.arctan2(np.sum(np.cross(v1, v2) * rotation_axis, axis=-1), np.sum(v1 * v2, axis=-1))
    if single:
        return angle[0]
    return angle
