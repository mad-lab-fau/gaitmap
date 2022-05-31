"""Estimate the IMU position with piece wise linear dedrifted integration using zupt updates."""

from typing import Optional, TypeVar

import numpy as np
import pandas as pd
from numpy.polynomial import polynomial
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from tpcp import cf

from gaitmap.base import BasePositionMethod, BaseZuptDetector
from gaitmap.utils.array_handling import bool_array_to_start_end_array
from gaitmap.utils.consts import GF_POS, GF_VEL, GRAV_VEC, SF_ACC
from gaitmap.utils.datatype_helper import SingleSensorData, is_single_sensor_data
from gaitmap.zupt_detection import NormZuptDetector

Self = TypeVar("Self", bound="PieceWiseLinearDedriftedIntegration")


class PieceWiseLinearDedriftedIntegration(BasePositionMethod):
    """Use a piecewise linear drift model based on zupts for integration of acc to estimate velocity and position.

    .. warning::
       We assume that the acc signal is already converted into the global/world frame!
       Refer to the :ref:`Coordinate System Guide <coordinate_systems>` for details.

    This method uses the zero-velocity assumption (ZUPT) to perform a drift removal using a piece wise linear drift
    model.
    The method can be applied on complete movement sequences without the need for previous stride segmentation.
    We assume a linear integration error between all detected zupt regions.
    We further assume, that the signal starts with resting period (i.e. the starting velocity is assumed to be 0).

    The ZUPTS update is currently only performed on the gyro norm and with the default settings we expect to find
    mid-stances
    as ZUPTS during regular walking.

    Further drift correction is applied using a level-assumption (i.e. we assume that the sensor starts and ends its
    movement at the same z-position/height between zupt regions).
    If this assumption is not true for your usecase, you can disable it using the `level_assumption` parameter.

    Parameters
    ----------
    zupt_detector
        An instance of a valid Zupt detector that will be used to find ZUPTs.
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
    Your data must be a pd.DataFrame with all sensor columns defined.
    Remember, that this method does not transform your data into another coordinate frame, but just integrates it.
    If you want to use this method to calculate e.g. a stride length, make sure to convert your data into the global
    frame using any of the implemented orientation estimation methods.

    >>> import pandas as pd
    >>> from gaitmap.utils.consts import SF_COLS
    >>> from gaitmap.zupt_detection import NormZuptDetector
    >>> data = pd.DataFrame(..., columns=SF_COLS)
    >>> sampling_rate_hz = 100
    >>> # Create an algorithm instance
    >>> pwli = PieceWiseLinearDedriftedIntegration(NormZuptDetector(window_length_s=0.15,
    ...                                                             inactive_signal_threshold=15.
    ...                                            ),
    ...                                            gravity=np.array([0, 0, 9.81])
    ...        )
    >>> # Apply the algorithm
    >>> pwli = pwli.estimate(data, sampling_rate_hz=sampling_rate_hz)
    >>> # Inspect the results
    >>> pwli.position_
    <pd.Dataframe with resulting positions>
    >>> pwli.velocity_
    <pd.Dataframe with resulting velocities>

    Notes
    -----
    In detail, the dedrifting works by first identifing all regions of zero velocity.
    For each region a linear fit is applied to corresonding velocity data.
    This is done to also compensate for drift during these resting phases.

    For the remaing regions (non-ZUPT), a linear interpolation is applied, basically connecting the end of the linear
    fit applied to one region to the start of the next region with a straight line.
    For the first value in the sequence, we always assume a velocity of 0 and for the last value we take the final
    uncorrected velocity value.
    This multi linear baseline is than substracted from the velocity to correct it.

    The same process is repeated for the z-position in case `level_walking=True`.

    See Also
    --------
    gaitmap.trajectory_reconstruction: Other implemented algorithms for orientation and position estimation
    gaitmap.utils.static_moment_detection: Static moment detector used for estimating zupt updates.

    """

    level_assumption: bool
    gravity: Optional[np.ndarray]
    zupt_detector: BaseZuptDetector

    data: SingleSensorData
    sampling_rate_hz: float

    zupts_: np.ndarray

    def __init__(
        self,
        zupt_detector=cf(
            NormZuptDetector(
                sensor="gyr", window_length_s=0.15, window_overlap=0.5, metric="mean", inactive_signal_threshold=15.0
            )
        ),
        level_assumption: bool = True,
        gravity: Optional[np.ndarray] = cf(GRAV_VEC),
    ):
        self.zupt_detector = zupt_detector
        self.level_assumption = level_assumption
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

        # find zupts for drift correction
        zupts = self.zupt_detector.clone().detect(data, sampling_rate_hz).per_sample_zupts_
        self.zupts_ = bool_array_to_start_end_array(zupts)

        # Add an implicit 0 to the beginning of the data
        acc_data_padded = np.pad(acc_data, pad_width=((1, 0), (0, 0)), constant_values=0)
        # shift zupts to fit to the padded acc data!
        zupts_padded = self.zupts_ + 1

        velocity = cumtrapz(acc_data_padded, axis=0, initial=0) / self.sampling_rate_hz
        drift_model = self._estimate_piece_wise_linear_drift_model(velocity, zupts_padded)
        velocity -= drift_model

        position = cumtrapz(velocity, axis=0, initial=0) / self.sampling_rate_hz

        if self.level_assumption is True:
            position[:, -1] -= self._estimate_piece_wise_linear_drift_model(position[:, -1], zupts_padded)

        self.velocity_ = pd.DataFrame(velocity, columns=GF_VEL)
        self.velocity_.index.name = "sample"
        self.position_ = pd.DataFrame(position, columns=GF_POS)
        self.position_.index.name = "sample"

        return self

    def _estimate_piece_wise_linear_drift_model(  # noqa: no-self-use
        self, data: np.ndarray, zupt_sequences: np.ndarray
    ) -> np.ndarray:
        """Estimate a piece wise linear drift error model.

        Parameters
        ----------
        data : array with shape (k, m)
            The drift model will be estimated per column in case of a multidimensional input
        zupt_sequences
            Zupt sequences indicating the start and end indices for each zupt region

        Returns
        -------
            One dimensional array corresponding to the drift error model which can be substracted form the input data
            for dedrifting

        """
        if len(zupt_sequences) == 0:
            raise ValueError("No valid zupt regions available! Without any zupts we cannot dedrift this sequence!")

        data = np.atleast_2d(data)
        if data.shape[0] == 1:
            data = data.T
        drift_model = np.full(data.shape, np.nan)
        # for the interpolation we need to enforce a reference for the very first and very last sample!
        drift_model[-1] = data[-1]
        # data[0] should usually be 0 (i.e. the starting value of the velocity/position)
        drift_model[0] = data[0]

        # linear fit within all zupt sequence
        for start, end in zupt_sequences:
            poly_fn = polynomial.polyfit(np.arange(end - start), data[start:end], 1)
            drift_model[start:end] = polynomial.polyval(np.arange(end - start), poly_fn).T

        # fill all non zupt regions with linear drift models
        # make sure our mask is only 1D as multidimensional masks result in single dimensional outputs in numpy
        # We can do that, because we know the positions of the nans are the same in all axis, as the regions are defined
        # by the ZUPTS
        mask_known_values = ~np.isnan(drift_model[:, 0])

        # Interpolate the remaining regions linearly
        x = np.arange(len(drift_model))
        result = interp1d(x[mask_known_values], drift_model[mask_known_values], axis=0)(x)
        return np.squeeze(result)
