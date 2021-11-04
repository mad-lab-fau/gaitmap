"""Estimate the IMU position with dedrifting using Forward-Backwards integration."""

from typing import Optional, TypeVar

import numpy as np
from numpy.polynomial import polynomial
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

from gaitmap.base import BasePositionMethod
from gaitmap.utils.array_handling import bool_array_to_start_end_array
from gaitmap.utils.consts import GF_POS, GF_VEL, GRAV_VEC, SF_ACC, SF_GYR
from gaitmap.utils.datatype_helper import SingleSensorData, is_single_sensor_data
from gaitmap.utils.static_moment_detection import find_first_static_window_multi_sensor, find_static_samples

Self = TypeVar("Self", bound="PieceWiseLinearDedriftedIntegration")


class PieceWiseLinearDedriftedIntegration(BasePositionMethod):
    """Use forward(-backward) integration of acc to estimate velocity and position.

    .. warning::
       We assume that the acc signal is already converted into the global/world frame!
       Refer to the :ref:`Coordinate System Guide <coordinate_systems>` for details.

    This method uses the zero-velocity assumption (ZUPT) to perform a drift removal using a direct-and-reverse (DRI) or
    forward-backward integration.
    This means we assume no movement (zero velocity and zero acceleration except gravity) at the beginning and end of
    the signal.

    Further drift correction is applied using a level-assumption (i.e. we assume that the sensor starts and ends its
    movement at the same z-position/height).
    If this assumption is not true for your usecase, you can disable it using the `level_assumption` parameter.

    Implementation based on the paper by Hannink et al. [1]_.

    Parameters
    ----------
    turning_point
        The point at which the sigmoid weighting function has a value of 0.5 and therefore forward and backward
        integrals are weighted 50/50. Specified as percentage of the signal length (0.0 < turning_point <= 1.0).
    steepness
        Steepness of the sigmoid function to weight forward and backward integral.
    level_assumption
        If True, it is assumed that the stride starts and ends at z=0 and dedrifting in that direction is applied
        accordingly.
    gravity : Optional (3,) array, or None
        The value of gravity that will be subtracted from each Acc sample before integration.
        If this is `None`, no gravity subtraction will be performed.
        By default 9.81 m/s^2 will be subtracted from the z-Axis.

    Attributes
    ----------
    velocity_
        The velocity estimated by direct-and-reverse / forward-backward integration. See Examples for format hints.
    position_
        The position estimated by forward integration in the ground plane and by direct-and-reverse /
        forward-backward integration for the vertical axis. See Examples for format hints.

    Other Parameters
    ----------------
    data
        The data passed to the estimate method.
    sampling_rate_hz
        The sampling rate of the data.

    Notes
    -----
    .. [1] Hannink, J., Ollenschläger, M., Kluge, F., Roth, N., Klucken, J., and Eskofier, B. M. 2017. Benchmarking Foot
       Trajectory Estimation Methods for Mobile Gait Analysis. Sensors (Basel, Switzerland) 17, 9.
       https://doi.org/10.3390/s17091940

    Examples
    --------
    Your data must be a pd.DataFrame with at least columns defined by :obj:`~gaitmap.utils.consts.SF_ACC`.
    Remember, that this method does not transform your data into another coordinate frame, but just integrates it.
    If you want to use this method to calculate e.g. a stride length, make sure to cenvert your data into the global
    frame using any of the implemented orientation estimation methods.

    >>> import pandas as pd
    >>> from gaitmap.utils.consts import SF_ACC
    >>> data = pd.DataFrame(..., columns=SF_ACC)
    >>> sampling_rate_hz = 100
    >>> # Create an algorithm instance
    >>> fbi = ForwardBackwardIntegration(level_assumption=True, gravity=np.array([0, 0, 9.81]))
    >>> # Apply the algorithm
    >>> fbi = fbi.estimate(data, sampling_rate_hz=sampling_rate_hz)
    >>> # Inspect the results
    >>> fbi.position_
    <pd.Dataframe with resulting positions>
    >>> fbi.velocity_
    <pd.Dataframe with resulting velocities>

    See Also
    --------
    gaitmap.trajectory_reconstruction: Other implemented algorithms for orientation and position estimation
    gaitmap.trajectory_reconstruction.StrideLevelTrajectory: Apply the method for each stride of a stride list.

    """

    level_assumption: bool
    gravity: Optional[np.ndarray]

    data: SingleSensorData
    sampling_rate_hz: float

    zupts_: np.ndarray

    def __init__(
        self,
        level_assumption: bool = True,
        zupt_window_length_s: float = 0.15,
        zupt_window_overlap_s: float = None,
        zupt_metric: str = "mean",
        zupt_threshold_dps: float = 0.1,
        gravity: Optional[np.ndarray] = GRAV_VEC,
    ):
        self.level_assumption = level_assumption
        self.zupt_window_length_s = zupt_window_length_s
        self.zupt_window_overlap_s = zupt_window_overlap_s
        self.zupt_metric = zupt_metric
        self.zupt_threshold_dps = zupt_threshold_dps
        self.gravity = gravity

    def estimate(self: Self, data: SingleSensorData, sampling_rate_hz: float) -> Self:
        """Estimate the position of the sensor based on the provided global frame data.

        Parameters
        ----------
        data
            Continuous sensor data that includes at least a Acc with all values in the global world frame
        sampling_rate_hz
            The sampling rate of the data in Hz

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        is_single_sensor_data(self.data, check_gyr=False, frame="sensor", raise_exception=True)

        self.zupts_ = bool_array_to_start_end_array(self.find_zupts(data, self.sampling_rate_hz))

        acc_data = data[SF_ACC].to_numpy()
        if self.gravity is not None:
            acc_data -= self.gravity

        # self.velocity_ = pd.DataFrame(cumtrapz(acc_data, axis=0, initial=0) / self.sampling_rate_hz, columns=GF_VEL)
        # self.velocity_ = self.velocity_.apply(
        #     lambda x: x - self._estimate_piece_wise_linear_drift_model(x.to_numpy(), self.zupts_), axis=0
        # )
        self.velocity_ = cumtrapz(acc_data, axis=0, initial=0) / self.sampling_rate_hz
        self.velocity_ -= self._estimate_piece_wise_linear_drift_model(self.velocity_, self.zupts_)

        self.velocity_.index.name = "sample"

        position = cumtrapz(self.velocity_.to_numpy(), axis=0, initial=0) / self.sampling_rate_hz

        if self.level_assumption is True:
            position[:, -1] = position[:, -1] - self._estimate_piece_wise_linear_drift_model(
                position[:, -1], self.zupts_
            )

        self.position_ = pd.DataFrame(position, columns=GF_POS)
        self.position_.index.name = "sample"

        return self

    def find_zupts(self, data, sampling_rate_hz: float):
        """Find the ZUPT samples based on the provided data.

        By default this method uses only the gyro data.

        Parameters
        ----------
        data
            Continuous sensor data including gyro and acc values.s
        sampling_rate_hz
            sampling rate of the gyro data

        Returns
        -------
        zupt_array
            array of length gyro with True and False indicating a ZUPT.

        """
        window_length = max(2, round(sampling_rate_hz * self.zupt_window_length_s))
        if self.zupt_window_overlap_s is None:
            window_overlap = int(window_length // 2)
        else:
            window_overlap = round(sampling_rate_hz * self.zupt_window_overlap_s)
        zupts = find_static_samples(
            data[SF_GYR].to_numpy(), window_length, self.zupt_threshold_dps, self.zupt_metric, window_overlap
        )
        return zupts

    def _estimate_piece_wise_linear_drift_model(self, data, zupt_sequences):
        """Estimate a piece wise linear drift error model.

        Parameters
        ----------
        data : Sequence of n arrays with shape (k, m) or a 3D-array with shape (k, n, m)
            The signals of n senors with m axis and k samples.
        zupt_sequences
            Boolean array with same length as input data to indicate samples which correspond to a zupt region

        Returns
        -------
            One dimensional array corresponding to the drift error model which can be substracted form the input data
            for dedrifting

        Examples
        --------
        >>> sensor_1_gyro = ...
        >>> sensor_2_gyro = ...
        >>> find_first_static_window_multi_sensor([sensor_1_gyro, sensor_2_gyro], window_length=128, inactive_signal_th=5)

        """
        data = np.atleast_2d(data)
        data = data.T
        drift_model = np.full(data.shape, np.nan)
        # in case we have a linear part at the very end we need to enforce a reference for the last sample!
        drift_model[-1] = data[-1]
        drift_model[0] = data[0]

        if len(zupt_sequences) == 0:
            raise ValueError("No valid zupt regions available! Without any zupts we cannot dedrift this sequence!")
        if zupt_sequences[0, 0] == 0 and zupt_sequences[0, -1] == len(data):
            # if the complete region is a valid zupt the best we can do is a linear fit on the whole data
            poly_fn = np.poly1d(np.polyfit(np.arange(len(data)), data, 1))
            return poly_fn(np.arange(len(data)))

        # linear fit within all zupt sequence
        for start, end in zupt_sequences:
            poly_fn = polynomial.polyfit(np.arange(end - start), data[start:end], 1)
            drift_model[start:end] = polynomial.polyval(np.arange(end - start), poly_fn).T

        # fill all non zupt regions with linear drift models
        x = np.arange(len(data))
        mask_known_values = ~np.isnan(drift_model[:, 0])
        return interp1d(x[mask_known_values], drift_model[mask_known_values])(x)
