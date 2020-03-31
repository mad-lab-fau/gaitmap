"""A set of util functions that ease handling rotations.

All util functions use :class:`scipy.spatial.transform.Rotation` to represent rotations.
"""
from typing import Union, Dict, Optional

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from gaitmap.utils.consts import SF_GYR, SF_ACC


def rotation_from_angle(axis: np.ndarray, angle: Union[float, np.ndarray]) -> Rotation:
    """Create a rotation based on a rotation axis and a angle.

    Parameters
    ----------
    axis : array with shape (3,) or (n, 3)
        normalized rotation axis
    angle : float or array with shape (n,)
        rotation angle in rad

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


def _rotate_sensor(data: pd.DataFrame, rotation: Optional[Rotation], inplace: bool = False) -> pd.DataFrame:
    """Rotate the data of a single sensor with acc and gyro."""
    # TODO: Put the default column names somewhere.
    if inplace is False:
        data = data.copy()
    if rotation is None:
        return data
    data[SF_GYR] = rotation.apply(data[SF_GYR].to_numpy())
    data[SF_ACC] = rotation.apply(data[SF_ACC].to_numpy())
    return data


def rotate_dataset(dataset: pd.DataFrame, rotation: Union[Rotation, Dict[str, Rotation]]) -> pd.DataFrame:
    """Apply a rotation to acc and gyro data of a dataset.

    TODO: Add example

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
    multi_index = dataset.columns.nlevels > 1
    if not multi_index and isinstance(rotation, dict):
        raise ValueError(
            "A Dictionary for the `rotation` parameter is only supported if a MultiIndex dataset (named sensors) is"
            " passed."
        )

    # TODO: create helper that checks if valid dataset is passed
    # TODO: Add warning if rotations are passed for sensors that don't exist
    if not multi_index:
        return _rotate_sensor(dataset, rotation, inplace=False)

    rotation_dict = rotation
    if not isinstance(rotation_dict, dict):
        rotation_dict = {k: rotation for k in dataset.columns.get_level_values(level=0)}

    original_cols = dataset.columns.copy()

    # For some strange reason, we need to unstack and stack again to use apply here:
    dataset = (
        dataset.stack(level=0)
        .groupby(level=1)
        .apply(lambda x: _rotate_sensor(x, rotation_dict.get(x.name, None), inplace=False))
        .unstack(level=1)
        .swaplevel(axis=1)
    )
    # Restore original order
    dataset = dataset[original_cols]
    return dataset
