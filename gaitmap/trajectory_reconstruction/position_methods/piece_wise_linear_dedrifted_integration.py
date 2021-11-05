"""Estimate the IMU position with piece wise linear dedrifted integration using zupt updates."""

from typing import Optional, TypeVar

import numpy as np
import pandas as pd
from numpy.polynomial import polynomial
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

from gaitmap.base import BasePositionMethod
from gaitmap.utils.array_handling import bool_array_to_start_end_array
from gaitmap.utils.consts import GF_POS, GF_VEL, GRAV_VEC, SF_ACC, SF_GYR
from gaitmap.utils.datatype_helper import SingleSensorData, is_single_sensor_data
from gaitmap.utils.static_moment_detection import METRIC_FUNCTION_NAMES, find_static_samples

Self = TypeVar("Self", bound="PieceWiseLinearDedriftedIntegration")


class PieceWiseLinearDedriftedIntegration(BasePositionMethod):
    """Use a piecewise linear drift model based on zupts for integration of acc to estimate velocity and position.

    .. warning::
       We assume that the acc signal is already converted into the global/world frame!
       Refer to the :ref:`Coordinate System Guide <coordinate_systems>` for details.

    This method uses the zero-velocity assumption (ZUPT) to perform a drift removal using a piece wise linear dirft
    error model. This means we assume a linear integration error between zupt regions. The zupt update is currently only
    performed on the gyro norm.

    Further drift correction is applied using a level-assumption (i.e. we assume that the sensor starts and ends its
    movement at the same z-position/height between zupt regions).
    If this assumption is not true for your usecase, you can disable it using the `level_assumption` parameter.

    Parameters
    ----------
    zupt_window_length_s
        The windows length in seconds considered for zupt detection.
    zupt_window_overlap_s
        Window overlap in seconds for the zupt detection.
    zupt_metric
        The metric that should be calculated on the gyro norm in each window
    zupt_threshold_dps
        The threshold for zupt windows.
        If metric(norm(window, axis=-1))<=`inactive_signal_th` for the gyro signal, it is considered a valid zupt.
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
        The velocity estimated by the piece wise linear dedrifted integration.
    position_
        The position estimated by the piece wise linear dedrifted integration of the dedrifted velocity.
    zupts_
        The estimated zupt regions.

    Other Parameters
    ----------------
    data
        The data passed to the estimate method.
    sampling_rate_hz
        The sampling rate of the data.

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
    gaitmap.utils.static_moment_detection: Static moment detector used for estimating zupt updates.

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
        zupt_metric: METRIC_FUNCTION_NAMES = "mean",
        zupt_threshold_dps: float = 15,
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

        acc_data = data[SF_ACC].to_numpy()
        if self.gravity is not None:
            acc_data -= self.gravity
        # Add an implicit 0 to the beginning of the data
        # make sure we also add the padding to the gyro data to ensure that the zupt fit to the also padded acc data
        acc_data_padded = np.pad(acc_data, pad_width=((1, 0), (0, 0)), constant_values=0)
        gyr_data_padded = np.pad(data[SF_GYR].to_numpy(), pad_width=((1, 0), (0, 0)), constant_values=0)

        # find zupts for drift correction on the padded data
        self.zupts_ = bool_array_to_start_end_array(
            self.find_zupts(
                pd.DataFrame(np.column_stack([acc_data_padded, gyr_data_padded]), columns=SF_ACC + SF_GYR),
                self.sampling_rate_hz,
            )
        )

        velocity = cumtrapz(acc_data_padded, axis=0, initial=0) / self.sampling_rate_hz
        drift_model = self._estimate_piece_wise_linear_drift_model(velocity, self.zupts_)
        velocity -= drift_model
        self.velocity_ = pd.DataFrame(velocity, columns=GF_VEL)
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

    def _estimate_piece_wise_linear_drift_model(  # noqa: no-self-use
        self, data: np.ndarray, zupt_sequences: np.ndarray
    ):
        """Estimate a piece wise linear drift error model.

        Parameters
        ----------
        data : array with shape (k, m)
            The drift model will be estimated per columns in case of a multidimensional unput
        zupt_sequences
            Zupt sequences indicating the start and end indices for each zupt region

        Returns
        -------
            One dimensional array corresponding to the drift error model which can be substracted form the input data
            for dedrifting

        """
        data = np.atleast_2d(data)
        if data.shape[0] == 1:
            data = data.T
        drift_model = np.full(data.shape, np.nan)
        # for the interpolation we need to enforce a reference for the very first and very last sample!
        drift_model[-1] = data[-1]
        drift_model[0] = data[0]

        if len(zupt_sequences) == 0:
            raise ValueError("No valid zupt regions available! Without any zupts we cannot dedrift this sequence!")

        # linear fit within all zupt sequence
        for start, end in zupt_sequences:
            poly_fn = polynomial.polyfit(np.arange(end - start), data[start:end], 1)
            drift_model[start:end] = polynomial.polyval(np.arange(end - start), poly_fn).T

        # fill all non zupt regions with linear drift models
        # make sure our mask is only 1D as multidimensional masks result in single dimensional outputs in numpy
        mask_known_values = ~np.isnan(drift_model[:, 0])

        x = np.arange(len(drift_model))
        result = interp1d(x[mask_known_values], drift_model[mask_known_values], axis=0)(x)
        return np.squeeze(result)
