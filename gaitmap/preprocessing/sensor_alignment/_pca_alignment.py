"""A implementation of a PCA based sensor alignment to perform coordinate system rotations."""

from typing import Dict, Optional, Sequence, TypeVar, Union

import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA

from gaitmap.base import BaseSensorAlignment
from gaitmap.utils._algo_helper import invert_result_dictionary, set_params_from_dict
from gaitmap.utils._types import _Hashable
from gaitmap.utils.consts import SF_ACC, SF_GYR
from gaitmap.utils.datatype_helper import SensorData, get_multi_sensor_names, is_sensor_data
from gaitmap.utils.rotations import rotate_dataset

Self = TypeVar("Self", bound="PcaAlignment")


# TODO: Move `right_handed_cord` to general utils package


def right_handed_cord(x: Optional[np.ndarray], y: Optional[np.ndarray], z: Optional[np.ndarray]) -> np.ndarray:
    """Create right handed coordinate system, with two axis provided."""
    if x is None:
        x = np.cross(y, z)
    elif y is None:
        y = np.cross(z, x)
    elif z is None:
        z = np.cross(x, y)
    else:
        raise ValueError("The missing axis must be set to None!")

    return np.array([x, y, z])


def align_pca_2d_single_sensor(dataset: SensorData, target_axis: str, pca_plane_axis: Sequence[str]):
    """Align dataset target axis, to the main foot rotation plane, which is usually the medio-lateral plane."""
    pca_plane_axis = list(pca_plane_axis)
    if len(pca_plane_axis) != 2 or not (set(pca_plane_axis).issubset(SF_GYR) or set(pca_plane_axis).issubset(SF_ACC)):
        raise ValueError('Invalid axis for pca plane! Valid axis would be e.g. ("gyr_x", "gyr_y").')

    # find dominant movement axis by PCA only search in 2D plane assuming that the dataset is already roughly aligned in
    # the third direction
    pca = PCA(n_components=2)
    pca = pca.fit(dataset[pca_plane_axis])

    # define new coordinate system
    # target axis will correspond to the pca component with highest explained_variance_
    pca_main_component_axis = np.array([pca.components_[0][0], pca.components_[0][1], 0])

    target_axis_helper = {"x": None, "y": None}

    if not target_axis.lower() in target_axis_helper:
        raise ValueError("Invalid target aixs! Axis must be one of {}".format(target_axis_helper.keys()))

    target_axis_helper[target_axis.lower()] = pca_main_component_axis

    rotation_matrix = right_handed_cord(**target_axis_helper, z=np.array([0, 0, 1]))

    r = Rotation.from_matrix(rotation_matrix)

    # apply rotation to dataset
    return {
        "aligned_data": rotate_dataset(dataset, r),
        "rotation": r,
        "pca": pca,
    }


class PcaAlignment(BaseSensorAlignment):
    """Align dataset target axis, to the main foot rotation plane, which is usually the medio-lateral plane.

    The Principle Component Analysis (PCA) can be used to determin the coordinate plane where the main movement
    component is located, which corresponds to the main component of the PCA after fitting to the provided data. This
    is typically intended to align one axis of the sensor frame to the foot medio-lateral axis. The ML-axis is therefore
    assumed to correspond to the principle component with the highest explained variance within the 2D projection
    ("birds eye view") of the X-Y sensor frame. To ensure a 2D problem the dataset should be aligned roughly to gravity
    beforhand so we can assume a fixed z-axis of [0,0,1] and solve the alignment as a pure heading issue.

    Parameters
    ----------
    target_axis
        axis to which the main component found by the pca analysis will be aligned e.g. "y"
    pca_plane_axis
        list of axis names which span the 2D-plane where the pca will be performed e.g. ("gyr_x","gyr_y"). Note: the
        order of the axis defining the pca plane will influence also your target axis! So best keep a x-y order.

    Attributes
    ----------
    aligned_data_
        The rotated sensor data after alignment
    rotation_
        The :class:`~scipy.spatial.transform.Rotation` object tranforming the original data to the aligned data
    pca_
        :class:`~sklearn.decomposition.PCA` object after fitting

    Other Parameters
    ----------------
    data
        The data passed to the `align` method.


    Examples
    --------
    Align dataset to medio-lateral plane, by aligning the y-axis with the dominant component in the
    gyro x-y-plane

    >>> pca_alignment = PcaAlignment(target_axis="y", pca_plane_axis=("gyr_x","gyr_y"))
    >>> pca_alignment = pca_alignment.align(data, 204.8)
    >>> pca_alignment.aligned_data_['left_sensor']
    <copy of dataset with axis aligned to the medio-lateral plane>
    ...

    Notes
    -----
    The PCA is sign invariant this means only an alignment to the medio-lateral plane will be performend! An additional
    180deg flip of the coordinate system might be still necessary after the PCA alignment!


    See Also
    --------
    sklearn.decomposition.PCA: Details on the used PCA implementation for this method.

    """

    rotation_: Union[Rotation, Dict[_Hashable, Rotation]]
    pca_: Union[PCA, Dict[_Hashable, PCA]]

    target_axis: str
    pca_plane_axis: Sequence[str]

    data: SensorData

    def __init__(self, target_axis: str = "y", pca_plane_axis: Sequence[str] = ("gyr_x", "gyr_y")):
        self.target_axis = target_axis
        self.pca_plane_axis = pca_plane_axis
        super().__init__()

    def align(self: Self, data: SensorData, **kwargs) -> Self:
        """Align sensor data."""
        self.data = data
        dataset_type = is_sensor_data(data, check_gyr=True, check_acc=True, frame="sensor")
        if dataset_type in ("single", "array"):
            results = align_pca_2d_single_sensor(data, self.target_axis, self.pca_plane_axis)
        else:
            # Multisensor
            result_dict = {
                sensor: align_pca_2d_single_sensor(data[sensor], self.target_axis, self.pca_plane_axis)
                for sensor in get_multi_sensor_names(data)
            }
            results = invert_result_dictionary(result_dict)

        set_params_from_dict(self, results, result_formatting=True)

        return self
