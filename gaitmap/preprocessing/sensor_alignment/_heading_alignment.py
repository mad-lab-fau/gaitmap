"""Correct for heading missalignment between sensor and foot coordinat frame."""

from typing import Dict, TypeVar, Union

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseSensorAlignment, BaseZuptDetector
from gaitmap.preprocessing.sensor_alignment._pca_alignment import PcaAlignment
from gaitmap.trajectory_reconstruction import MadgwickAHRS, PieceWiseLinearDedriftedIntegration
from gaitmap.utils._algo_helper import default, invert_result_dictionary, set_params_from_dict
from gaitmap.utils._types import _Hashable
from gaitmap.utils.consts import GRAV_VEC, SF_ACC, SF_GYR
from gaitmap.utils.datatype_helper import SensorData, get_multi_sensor_names, is_sensor_data
from gaitmap.utils.rotations import get_gravity_rotation, rotate_dataset
from gaitmap.zupt_detection import NormZuptDetector

Self = TypeVar("Self", bound="HeadingAlignment")


class HeadingAlignment(BaseSensorAlignment):
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
    aligned_data_ :
        The rotated sensor data after alignment
    rotation_ :
        The :class:`~scipy.spatial.transform.Rotation` object tranforming the original data to the aligned data
    pca_ :
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

    pca_alignment: PcaAlignment
    baseline_velocity_threshold: float
    madgwick_beta: float
    zupt_detector_madgwick_init: BaseZuptDetector
    zupt_detector_pwli: BaseZuptDetector

    def __init__(
        self,
        pca_alignment: PcaAlignment = default(PcaAlignment(target_axis="y", pca_plane_axis=("gyr_x", "gyr_y"))),
        baseline_velocity_threshold: float = 0.2,
        madgwick_beta: float = 0.1,
        zupt_detector_madgwick_init=default(
            NormZuptDetector(sensor="acc", window_length_s=0.15, inactive_signal_threshold=0.01, metric="variance")
        ),
        zupt_detector_pwli=default(
            NormZuptDetector(sensor="gyr", window_length_s=0.15, inactive_signal_threshold=15.0, metric="mean")
        ),
    ):
        self.pca_alignment = pca_alignment
        self.baseline_velocity_threshold = baseline_velocity_threshold
        self.madgwick_beta = madgwick_beta
        self.zupt_detector_madgwick_init = zupt_detector_madgwick_init
        self.zupt_detector_pwli = zupt_detector_pwli
        super().__init__()

    def align(self: Self, data: SensorData, sampling_rate_hz: float) -> Self:
        """Align sensor data."""
        dataset_type = is_sensor_data(data, check_gyr=True, check_acc=True, frame="sensor")
        if dataset_type in ("single", "array"):
            results = self._align_heading_single_sensor(data, sampling_rate_hz)
        else:
            # Multisensor
            result_dict = {
                sensor: self._align_heading_single_sensor(data[sensor], sampling_rate_hz)
                for sensor in get_multi_sensor_names(data)
            }
            results = invert_result_dictionary(result_dict)

        set_params_from_dict(self, results, result_formatting=True)

        return self

    def _align_heading_single_sensor(self, data, sampling_rate_hz):
        self.pca_alignment = self.pca_alignment.align(data)
        data = self.pca_alignment.aligned_data_
        r = self.pca_alignment.rotation_
        if self._forward_direction_is_flipped(data, sampling_rate_hz):
            # flip data by 180deg around z-axis
            rot_z_180 = Rotation.from_euler("z", 180, degrees=True)
            data = rotate_dataset(data, rot_z_180)
            r = r*rot_z_180

        return {
            "aligned_data": data,
            "rotation": r,
        }

    def _forward_direction_is_flipped(self, data: np.ndarray, sampling_rate_hz: float):  # noqa: no-self-use
        # estimate initial orientation for Madgwick filter from first static acc region
        zupts_madgwick = self.zupt_detector_madgwick_init.detect(data, sampling_rate_hz).zupts_
        # only consider the very first static moment
        start, end = zupts_madgwick.to_numpy()[0]
        first_static_acc_vec = data[SF_ACC][start:end].median().to_numpy()

        # apply Madgwick filter to rotate sensor frame into the global world frame
        mad = MadgwickAHRS(beta=self.madgwick_beta, initial_orientation=get_gravity_rotation(first_static_acc_vec, GRAV_VEC))
        mad = mad.estimate(data[start:], sampling_rate_hz)
        acc_data = mad.orientation_object_[:-1].apply(data[start:][SF_ACC])
        gyr_data = mad.orientation_object_[:-1].apply(data[start:][SF_GYR])

        data_wf = pd.DataFrame(np.column_stack([acc_data, gyr_data]), columns=SF_ACC + SF_GYR)

        # apply piecewise linear dedrifted integration to estimate the forward movement direction
        pwli = PieceWiseLinearDedriftedIntegration(self.zupt_detector_pwli, level_assumption=False, gravity=GRAV_VEC)
        pwli = pwli.estimate(data_wf, sampling_rate_hz)

        # to ignore any rotation component around the z-/ heading-axis we again apply the inverse of the original z
        # rotation on the estimated world frame velocity
        forward_vel_without_heading_corr = pd.DataFrame(
            Rotation.from_euler("z", mad.orientation_object_.as_euler("zxy")[:, 0]).inv().apply(pwli.velocity_),
            columns=pwli.velocity_.columns,
        )["vel_x"]

        forward_vel_without_baseline = forward_vel_without_heading_corr[
            (forward_vel_without_heading_corr <= -self.baseline_velocity_threshold)
            | (forward_vel_without_heading_corr >= self.baseline_velocity_threshold)
        ]
        return forward_vel_without_baseline.mean() < 0
