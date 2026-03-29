"""Estimate the IMU position with dedrifting using Forward-Backwards integration."""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from tpcp import cf
from typing_extensions import Self

from gaitmap.base import BasePositionMethod
from gaitmap.utils.consts import GF_POS, GF_VEL, GRAV_VEC, SF_ACC
from gaitmap.utils.datatype_helper import SingleSensorData, is_single_sensor_data


class ForwardBackwardIntegration(BasePositionMethod):
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

    Implementation based on the paper by Hannink et al. [1]_ and Zok et al. [2]_.

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
    .. [2] M. Zok, C. Mazz`a, and U. Della Croce, “Total body centre of mass displacement estimated using ground
       reactions during transitory motor tasks: Application to step ascent,” Medical Engineering & Physics, vol. 26,
       no. 9, pp. 791-798, Nov. 2004. [Online]. Available:
       https://linkinghub.elsevier.com/retrieve/pii/S1350453304001195

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

    steepness: float
    turning_point: float
    level_assumption: bool
    gravity: Optional[np.ndarray]

    data: SingleSensorData
    sampling_rate_hz: float

    def __init__(
        self,
        turning_point: float = 0.5,
        steepness: float = 0.08,
        level_assumption: bool = True,
        gravity: Optional[np.ndarray] = cf(GRAV_VEC),
    ) -> None:
        self.turning_point = turning_point
        self.steepness = steepness
        self.level_assumption = level_assumption
        self.gravity = gravity

    def estimate(self, data: SingleSensorData, *, sampling_rate_hz: float, **_) -> Self:
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

        if not 0.0 <= self.turning_point <= 1.0:
            raise ValueError("`turning_point` must be in the rage of 0.0 to 1.0")
        is_single_sensor_data(self.data, check_gyr=False, frame="sensor", raise_exception=True)

        acc_data = data[SF_ACC].to_numpy()
        if self.gravity is not None:
            acc_data -= self.gravity

        # Add an implicit 0 to the beginning of the acc data
        padded_acc = np.pad(acc_data, pad_width=((1, 0), (0, 0)), constant_values=0)
        velocity = self._forward_backward_integration(padded_acc)
        position_xy = cumulative_trapezoid(velocity[:, :2], axis=0, initial=0) / self.sampling_rate_hz
        if self.level_assumption is True:
            position_z = self._forward_backward_integration(velocity[:, [2]])
        else:
            position_z = cumulative_trapezoid(velocity[:, [2]], axis=0, initial=0) / self.sampling_rate_hz
        position = np.hstack((position_xy, position_z))

        self.velocity_ = pd.DataFrame(velocity, columns=GF_VEL)
        self.velocity_.index.name = "sample"
        self.position_ = pd.DataFrame(position, columns=GF_POS)
        self.position_.index.name = "sample"

        return self

    def _sigmoid_weight_function(self, n_samples: int) -> np.ndarray:
        x = np.linspace(0, 1, n_samples)
        s = 1 / (1 + np.exp(-(x - self.turning_point) / self.steepness))
        weights = (s - s[0]) / (s[-1] - s[0])
        return weights

    def _forward_backward_integration(self, data: np.ndarray) -> np.ndarray:
        # TODO: different steepness and turning point for velocity and position?
        integral_forward = cumulative_trapezoid(data, axis=0, initial=0) / self.sampling_rate_hz
        # for backward integration, we flip the signal and inverse the time by using a negative sampling rate.
        integral_backward = cumulative_trapezoid(data[::-1], axis=0, initial=0) / -self.sampling_rate_hz
        weights = self._sigmoid_weight_function(integral_forward.shape[0])
        combined = (integral_forward.T * (1 - weights) + integral_backward[::-1].T * weights).T
        return combined
