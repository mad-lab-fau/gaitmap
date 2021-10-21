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


def align_pca_2d_single_sensor(dataset: SensorData, pca_plane_axis: Sequence[str]):
    """Align dataset y-axis, to the main foot rotation plane, which is usually the medio-lateral plane."""
    pca_plane_axis = list(pca_plane_axis)
    if len(pca_plane_axis) != 2 or not (set(pca_plane_axis).issubset(SF_GYR) or set(pca_plane_axis).issubset(SF_ACC)):
        raise ValueError('Invalid axis for pca plane! Valid axis would be e.g. ("gyr_x", "gyr_y").')

    # find ml-axis by PCA only search in 2D plane assuming that the dataset is already roughly aligned to gravity
    pca = PCA(n_components=2)
    pca = pca.fit(dataset[pca_plane_axis])

    # define new coordinate system
    # ml-axis: this corresponds to the pca component with highest explained_variance_
    foot_ml_axis = np.array([pca.components_[0][0], pca.components_[0][1], 0])
    # si-axis: ensure rotation only in x-y plane by setting z-axis to constant [0,0,1] for [x,y,z]
    foot_si_axis = np.array([0, 0, 1])
    # pa-axis: this is the orthogonal axis to ml-si plane (calculate by cross product)
    foot_pa_axis = np.cross(foot_ml_axis, foot_si_axis)

    # build rotation matrix from new coordinate system
    rotation_matrix = np.array([foot_pa_axis, foot_ml_axis, foot_si_axis])
    r = Rotation.from_matrix(rotation_matrix)

    # apply rotation to dataset
    return {
        "aligned_data": rotate_dataset(dataset, r),
        "rotation": r,
        "pca": pca,
    }


class PcaAlignment(BaseSensorAlignment):
    """Align dataset y-axis, to the main foot rotation plane, which is usually the medio-lateral plane.

    The medio-lateral axis will be defined as the principle component with the highest explained variance within the 2D
    projection ("birds eye view") of the X-Y sensor frame. Therefore, we will apply 'sklearn.decomposition.PCA' to fix
    the sensor heading. To ensure a 2D problem the dataset should be aligned roughly to gravity beforhand so we can
    assume a fixed z-axis of [0,0,1].

    Parameters
    ----------
    pca_plane_axis
        list of axis names which span the 2D-plane where the pca will be performed e.g. ("gyr_x","gyr_y")

    Attributes
    ----------
    aligned_data_
        The rotated sensor data after alignment

    rotation_:
        The :class:`~scipy.spatial.transform.Rotation` object tranforming the original data to the aligned data

    pca_:
        :class:`~sklearn.decomposition.PCA` object after fitting

    Other Parameters
    ----------------
    data
        The data passed to the `align` method.
    sampling_rate_hz
        The sampling rate of the data

    Examples
    --------
    Align dataset to medio-lateral plane

    >>> pca_alignment = PcaAlignment(pca_plane_axis=("gyr_x","gyr_y"))
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

    def __init__(self, pca_plane_axis: Optional[Sequence[str]] = ("gyr_x", "gyr_y")):
        self.pca_plane_axis = pca_plane_axis

    def align(self: Self, data: SensorData, **kwargs) -> Self:
        """Align sensor data."""
        dataset_type = is_sensor_data(data, check_gyr=True, check_acc=True, frame="sensor")
        if dataset_type in ("single", "array"):
            results = align_pca_2d_single_sensor(data, self.pca_plane_axis)
        else:
            # Multisensor
            result_dict = {
                sensor: align_pca_2d_single_sensor(data[sensor], self.pca_plane_axis)
                for sensor in get_multi_sensor_names(data)
            }
            results = invert_result_dictionary(result_dict)

        set_params_from_dict(self, results, result_formatting=True)

        return self
