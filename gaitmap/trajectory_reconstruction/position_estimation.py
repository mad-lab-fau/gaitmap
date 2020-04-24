"""Estimation of velocity and position relative to first sample of passed data."""
from typing import Union, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import integrate

from gaitmap.base import BasePositionEstimation
from gaitmap.utils import dataset_helper
from gaitmap.utils.consts import SF_ACC, SF_VEL, SF_POS
from gaitmap.utils.dataset_helper import (
    Dataset,
    SingleSensorDataset,
    get_multi_sensor_dataset_names,
    SingleSensorStrideList,
    StrideList,
)


class ForwardBackwardIntegration(BasePositionEstimation):
    """Use forward(-backward) integration of acc to estimate velocity and position.

    Before integrating acceleration data, it is transformed using the passed rotations.

    For drift removal, a direct-and-reverse (DRI) or forward-backward integration is used for velocity estimation,
    because we assume zero velocity at the beginning and end of a signal. For position, drift removal via DRI
    is only used for the vertical axis (=z-axis or superior-inferior-axis, see :ref:`ff`), because we assume
    beginning and end of the motion are in one plane (zero-level assumption). Implementation based on the paper by
    Hannink et al. [1]_.

    Attributes
    ----------
    estimated_velocity_
        The velocity estimated by direct-and-reverse / forward-backward integration. See Examples for format hints.
    estimated_position_
        The position estimated by forward integration in the ground plane and by direct-and-reverse /
        forward-backward integration for the vertical axis. See Examples for format hints.

    Parameters
    ----------
    turning_point
        The point at which the sigmoid weighting function has a value of 0.5 and therefore forward and backward
        integrals are weighted 50/50. Specified as percentage of the signal length (0.0 < turning_point <= 1.0).
    steepness
        Steepness of the sigmoid function to weight forward and backward integral.
    subtract_gravity
        Subtract gravity after transforming strides into world frame coordinates.
        # TODO: link to gravity, when it has been exported to consts

    Other Parameters
    ----------------
    data
        The data passed to the :py:meth:`~.estimate` method. This class does NOT take care for transforming sensor
        data
        from sensor
        frame to world coordinates, just calculates the necessary rotations that have to be applied.
    event_list
        This list is used to set the start and end of each integration period.
    rotations
        Rotations that will be used to rotate acceleration data before estimating the position (i.e. transforming
        from inertial sensor frame to fixed world frame). Rotations may be obtained from
        :mod:`~gaitmap.trajectory_reconstruction.orientation_estimation`. Either use
        `estimated_orientations_without_initial_` or `estimated_orientations_without_final_` of
        :class:`gaitmap.base.BaseOrientationEstimation`.
    sampling_rate_hz
        The sampling rate of the data.

    Notes
    -----
    .. [1] Hannink, J., Ollenschläger, M., Kluge, F., Roth, N., Klucken, J., and Eskofier, B. M. 2017. Benchmarking Foot
       Trajectory Estimation Methods for Mobile Gait Analysis. Sensors (Basel, Switzerland) 17, 9.
       https://doi.org/10.3390/s17091940

    Examples
    --------
    >>> data_left = healthy_example_imu_data["left_sensor"]
    >>> events_left = healthy_example_stride_events["left_sensor"]
    >>> integrator = ForwardBackwardIntegration(0.5, 0.08, True)

     `rotations_left` can be obtained by using :mod:`~gaitmap.trajectory_reconstruction.orientation_estimation`

    >>> integrator.estimate(data_left, events_left, rotations_left 204.8)
    >>> integrator.estimated_velocity_.iloc[-1]
    vel_x   -0.000019
    vel_y    0.000447
    vel_z    0.000000
    Name: (34.619140625, 27.0), dtype: float64

    Estimated: position / velocity looks like this, where `s_id` is the stride id.

    >>> integrator.estimated_position_
                 pos_x     pos_y     pos_z
    s_id sample
    0    0       0.463356  0.542484  0.542484
         1       0.017705  0.037818  0.037818
         2       0.331350  0.924174  0.924174
         3       0.437381  0.578962  0.578962
         4       0.026348  0.590486  0.590486
    1    0       0.284609  0.713550  0.713550
         1       0.595096  0.475286  0.475286
    ...          ...       ...       ...

     Note: This is the case for a single sensor. For multiple sensors, it is a dictionary with keys being the sensor
     names in `self.data` and values being these kind of :py:class:`~pandas.DataFrame`.

    """

    steepness: float
    turning_point: float

    estimated_position_: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    estimated_velocity_: Union[pd.DataFrame, Dict[str, pd.DataFrame]]

    sampling_rate_hz: float
    data: Dataset
    event_list: StrideList

    def __init__(self, turning_point: float = 0.5, steepness: float = 0.08, subtract_gravity: bool = True):
        self.turning_point = turning_point
        self.steepness = steepness
        self.subtract_gravity = subtract_gravity

    def estimate(
        self,
        data: Dataset,
        event_list: StrideList,
        rotations: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        sampling_rate_hz: float,
    ):
        """Estimate velocity and position based on acceleration data."""
        # TODO: Make it clear/add check that this data is actual rotated data
        if not 0.0 <= self.turning_point <= 1.0:
            raise ValueError(
                "Bad ForwardBackwardIntegration initialization found. Turning point must be in the rage "
                "of 0.0 to 1.0"
            )
        self.sampling_rate_hz = sampling_rate_hz
        self.data = data
        self.event_list = event_list
        self.rotations = rotations

        if dataset_helper.is_single_sensor_dataset(data):
            self.estimated_velocity_, self.estimated_position_ = self._estimate_single_sensor(
                data, event_list, rotations
            )
        elif dataset_helper.is_multi_sensor_dataset(data):
            self.estimated_velocity_, self.estimated_position_ = self._estimate_multi_sensor()
        else:
            raise ValueError("Provided data set is not supported by gaitmap")
        return self

    def _estimate_single_sensor(
        self, data: SingleSensorDataset, event_list: SingleSensorStrideList, rotations: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        velocity = {}
        position = {}
        for _, i_stride in event_list.iterrows():
            i_start, i_end = (int(i_stride["start"]), int(i_stride["end"]))
            i_vel, i_pos = self._estimate_stride(data, i_start, i_end, rotations.xs(i_stride["s_id"], level="s_id"))
            velocity[i_stride["s_id"]] = i_vel
            position[i_stride["s_id"]] = i_pos
        velocity = pd.concat(velocity)
        velocity.index = velocity.index.rename(("s_id", "sample"))
        position = pd.concat(position)
        position.index = position.index.rename(("s_id", "sample"))
        return velocity, position

    def _estimate_stride(
        self, data: SingleSensorDataset, start: int, end: int, rotations
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        acc_data = self.rotate_stride(data[SF_ACC].iloc[start:end], rotations)
        if self.subtract_gravity:
            # TODO Gravity to consts or dataset_helper?
            acc_data = acc_data - [0, 0, 9.81]
        estimated_velocity_ = pd.DataFrame(
            self._forward_backward_integration(acc_data), index=acc_data.index, columns=SF_VEL
        )
        # TODO: This uses the level walking assumption. We should make this configurable.
        estimated_position_ = pd.DataFrame(
            index=acc_data.index,
            columns=SF_POS,
            data=np.hstack(
                (
                    integrate.cumtrapz(estimated_velocity_[SF_VEL[:2]], axis=0, initial=0) / self.sampling_rate_hz,
                    self._forward_backward_integration(estimated_velocity_[[SF_VEL[2]]]),
                )
            ),
        )
        return estimated_velocity_, estimated_position_

    def _get_weight_matrix(self, n_samples: int) -> np.ndarray:
        # TODO: support other weighting functions
        x = np.linspace(0, 1, n_samples)
        s = 1 / (1 + np.exp(-(x - self.turning_point) / self.steepness))
        weights = (s - s[0]) / (s[-1] - s[0])
        return weights

    def _forward_backward_integration(self, data: np.ndarray) -> np.ndarray:
        # TODO: make it possible to set initial value of integral from outside?
        # TODO: move to utils?
        # TODO: different steepness and turning point for velocity and position?
        integral_forward = integrate.cumtrapz(data, axis=0, initial=0) / self.sampling_rate_hz
        integral_backward = integrate.cumtrapz(data[::-1], axis=0, initial=0) / self.sampling_rate_hz
        weights = self._get_weight_matrix(data.shape[0])

        return (integral_forward.T * (1 - weights) + integral_backward[::-1].T * weights).T

    def _estimate_multi_sensor(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        estimated_position_ = dict()
        estimated_velocity_ = dict()
        for i_sensor in get_multi_sensor_dataset_names(self.data):
            vel, pos = self._estimate_single_sensor(
                self.data[i_sensor], self.event_list[i_sensor], self.rotations[i_sensor]
            )
            estimated_velocity_[i_sensor], estimated_position_[i_sensor] = vel, pos
        return estimated_velocity_, estimated_position_
