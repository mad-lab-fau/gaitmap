"""An error state kalman filter with Rauch-Tung-Striebel smoothing fo estimating trajectories."""
import warnings
from typing import Any, Dict, Optional, TypeVar, Union

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tpcp import NOTHING, cf

from gaitmap.base import BaseTrajectoryMethod, BaseZuptDetector
from gaitmap.trajectory_reconstruction.trajectory_methods._kalman_numba_funcs import (
    ForwardPassDependencies,
    SimpleZuptParameter,
    default_rts_kalman_forward_pass,
    rts_kalman_update_series,
    simple_navigation_equations,
)
from gaitmap.utils.array_handling import bool_array_to_start_end_array
from gaitmap.utils.consts import GF_POS, GF_VEL, SF_ACC, SF_GYR
from gaitmap.utils.datatype_helper import SingleSensorData, is_single_sensor_data
from gaitmap.zupt_detection import NormZuptDetector

Self = TypeVar("Self", bound="RtsKalman")


class RtsKalman(BaseTrajectoryMethod):
    """An Error-State-Kalman-Filter (ESKF) with Rauch-Tung-Striebel (RTS) smoothing for trajectory estimation.

    This algorithm employs an ESKF to estimate a trajectory and velocity profile
    from acceleration and gyroscope data.
    It does three passes over the data, where in the first one the data is integrated and the current error in position,
    velocity and orientation is tracked with an extended kalman filter.
    This filter uses a linearized transition matrix in the prediction phase and Zero-Velocity-Updates (ZUPT) in its
    update phase.
    In the second pass RTS smoothing is applied to the previously estimated error states and error covariances and in
    the thrid pass the smoothed error states are applied as correction to the integrated nominal states from the first
    pass.

    This method is expected to be run on long sections of data including multiple strides.

    The implementation of the RTS smoothing is based on the paper [1]_.
    The error state kalman filter part itself is based on the paper [2]_.

    Parameters
    ----------
    initial_orientation
        The initial orientation of the sensor that is assumed.
        It is critical that this value is close to the actual orientation.
        If you pass an array, remember that the order of elements must be x, y, z, w.
    zupt_threshold_dps
        .. warning::
            This parameter is deprecated and will be removed soon! Use `zupt_detector` instead.

        The threshold used in the default method for ZUPT detection in degree per second.
        It looks at the maximum energy in windows of 10 gyro samples and decides for a ZUPT, if this energy is
        smaller than `zupt_threshold_dps`.
        You can also override the `find_zupts` method to implement your own ZUPT detection.
    zupt_variance
        The variance of the noise of the measured velocity during a ZUPT.
        As we are typically pretty sure, that the velocity should be zero then, this should be very small.
    velocity_error_variance
        The variance of the noise present in the velocity error.
        Should be based on the sensor accelerometer noise.
    orientation_error_variance
        The variance of the noise present in the orientation error.
        Should be based on the sensor gyroscope noise.
        The orientation error is internally not represented as quaternion, but as axis-angle representation,
        which also explains the unit of rad^2 for this variance.
    level_walking
        Flag to control if the level walking assumptions should be used during ZUPTs.
        If this is True, additionally to the velocity, the z position is reset to zero during a ZUPT.
    level_walking_variance
        The variance of the noise of the measured position during a level walking update.
        Should typically be very small.
    zupt_window_length_s
        .. warning::
            This parameter is deprecated and will be removed soon! Use `zupt_detector` instead.

        Length of the window used in the default method to find ZUPTs.
        If the value is too small at least a window of 2 samples is used.
        Given in seconds.
    zupt_window_overlap_s
        .. warning::
            This parameter is deprecated and will be removed soon! Use `zupt_detector` instead.

        Length of the window overlap used in the default method to find ZUPTs.
        It is given in seconds and if not given it defaults to half the window length.
    zupt_detector
        An instance of a valid Zupt detector that will be used to find ZUPTs.

    Attributes
    ----------
    orientation_
        The rotations as a *SingleSensorOrientationList*, including the initial orientation.
        This means the there are len(data) + 1 orientations.
    orientation_object_
        The orientations as a single scipy Rotation object
    position_
        The calculated positions
    velocity_
        The calculated velocities
    covariance_
        The covariance matrices of the kalman filter after smoothing.
        They can be used as a measure of how good the filter worked and how accurate the results are.
    zupts_
        2D array indicating the start and the end samples of the detected ZUPTs for debug porpuses.

    Other Parameters
    ----------------
    data
        The data passed to the estimate method
    sampling_rate_hz
        The sampling rate of this data

    Notes
    -----
    The default values chosen for the noise parameters and the ZUPT-threshold are optimized based on NilsPod recordings
    with healthy subjects.
    If you are using a different sensor system or other cohorts, it is advisable to readjust these parameter values.
    When adjusting the variance of the three different noises for velocity error, orientation error and zupts,
    the relation between those values is much more important, than the absolute values assigned.
    As long as the quotient between them is kept about the same and `zupt_threshold_dps` is not changed,
    similar results are expected.

    This class uses *Numba* as a just-in-time-compiler to achieve fast run times.
    In result, the first execution of the algorithm will take longer as the methods need to be compiled first.

    .. [1] D. Simón Colomar, J. Nilsson and P. Händel, "Smoothing for ZUPT-aided INSs,"
    .. [2] Solà, Joan. (2015). Quaternion kinematics for the error-state KF.

    Examples
    --------
    Your data must be a pd.DataFrame with columns defined by :obj:`~gaitmap.utils.consts.SF_COLS`.

    >>> import pandas as pd
    >>> from gaitmap.utils.consts import SF_COLS
    >>> from gaitmap.trajectory_reconstruction import RtsKalman
    >>> data = pd.DataFrame(..., columns=SF_COLS)
    >>> sampling_rate_hz = 100
    >>> # Create an algorithm instance
    >>> kalman = RtsKalman(initial_orientation=np.array([0, 0, 0, 1.0]),
    ...                    zupt_variance=10e-8,
    ...                    velocity_error_variance=10e5,
    ...                    orientation_error_variance=10e-2,
    ...                    zupt_detector=NormZuptDetector(semsor="gyr",
    ...                                                   window_length_s=0.05
    ...                    )
    ...             )
    >>> # Apply the algorithm
    >>> kalman = kalman.estimate(data, sampling_rate_hz=sampling_rate_hz)
    >>> # Inspect the results
    >>> kalman.orientation_
    <pd.Dataframe with resulting quaternions>
    >>> kalman.orientation_object_
    <scipy.Rotation object>
    >>> kalman.position_
    <pd.Dataframe with resulting positions>
    >>> kalman.velocity_
    <pd.Dataframe with resulting velocities>
    >>> kalman.covariances_
    <pd.Dataframe with resulting covariances>

    See Also
    --------
    gaitmap.trajectory_reconstruction: Other implemented algorithms for orientation and position estimation
    gaitmap.trajectory_reconstruction.RegionLevelTrajectory: Apply the method to a gait sequence or regions of interest.

    """

    initial_orientation: Union[np.ndarray, Rotation]
    zupt_threshold_dps: float
    zupt_variance: float
    velocity_error_variance: float
    orientation_error_variance: float
    level_walking: bool
    level_walking_variance: float
    zupt_window_length_s: float
    zupt_window_overlap_s: Optional[float]
    zupt_detector: BaseZuptDetector

    data: SingleSensorData
    sampling_rate_hz: float

    covariance_: pd.DataFrame
    zupts_: np.ndarray

    _forward_pass = default_rts_kalman_forward_pass
    # A internal version of the Zupt detector.
    # This is only needed while we still have the deprecated arguments.
    # TODO: Delete after deprecation
    _zupt_detector: BaseZuptDetector

    _deprecated_args = {"zupt_threshold_dps", "zupt_window_length_s", "zupt_window_overlap_s"}

    def __init__(
        self,
        initial_orientation: Union[np.ndarray, Rotation] = cf(np.array([0, 0, 0, 1.0])),
        zupt_threshold_dps: float = NOTHING,
        zupt_variance: float = 10e-8,
        velocity_error_variance: float = 10e5,
        orientation_error_variance: float = 10e-2,
        level_walking: bool = True,
        level_walking_variance: float = 10e-8,
        zupt_window_length_s: float = NOTHING,
        zupt_window_overlap_s: Optional[float] = NOTHING,
        zupt_detector=cf(
            NormZuptDetector(
                sensor="gyr", window_length_s=0.05, window_overlap=0.5, metric="maximum", inactive_signal_threshold=34.0
            )
        ),
    ):
        self.initial_orientation = initial_orientation
        self.zupt_threshold_dps = zupt_threshold_dps
        self.zupt_variance = zupt_variance
        self.velocity_error_variance = velocity_error_variance
        self.orientation_error_variance = orientation_error_variance
        self.level_walking = level_walking
        self.level_walking_variance = level_walking_variance
        self.zupt_window_length_s = zupt_window_length_s
        self.zupt_window_overlap_s = zupt_window_overlap_s
        self.zupt_detector = zupt_detector

    def estimate(self: Self, data: SingleSensorData, sampling_rate_hz: float) -> Self:
        """Estimate the position, velocity and orientation of the sensor.

        Parameters
        ----------
        data
            Continuous sensor data including gyro and acc values.
            The gyro data is expected to be in deg/s!
        sampling_rate_hz
            The sampling rate of the data in Hz

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        # Handle deprecation:
        deprecated_arg_overwrite = {}
        for arg in self._deprecated_args:
            if getattr(self, arg) != NOTHING:
                deprecated_arg_overwrite[arg] = getattr(self, arg)
        self._zupt_detector = self.zupt_detector.clone()
        if len(deprecated_arg_overwrite) > 0:
            warnings.warn(
                "You specified values for the following deprecated parameters: "
                f"{list(deprecated_arg_overwrite.keys())}. "
                "They will be removed in a future version of gaitmap."
                "Use the generic `zupt_detector` parameter instead. "
                "For more information about the migration see the changelog for version 1.5.",
                DeprecationWarning,
            )
            if not isinstance(self.zupt_detector, NormZuptDetector):
                # Note this does not check for all modifications, but if someone strangly combines new and old
                # parameters, it is their own fault.
                raise ValueError(
                    "You specified one or more deprecated arguments AND modified the default value for "
                    "`zupt_detector`. "
                    "Don't do this! Only use `zupt_detector`."
                )
            self._zupt_detector = self._convert_deprecated_args(deprecated_arg_overwrite, self.zupt_detector.clone())

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        initial_orientation = self.initial_orientation

        is_single_sensor_data(self.data, frame="sensor", raise_exception=True)
        if isinstance(initial_orientation, Rotation):
            initial_orientation = Rotation.as_quat(initial_orientation)
        initial_orientation = initial_orientation.copy()

        process_noise = np.zeros((9, 9))
        process_noise[3:6, 3:6] = self.velocity_error_variance * np.eye(3)
        process_noise[6:, 6:] = self.orientation_error_variance * np.eye(3)

        # measure noise
        meas_noise = np.zeros((4, 4))
        meas_noise[0:3, 0:3] = np.eye(3) * self.zupt_variance
        if self.level_walking is True:
            meas_noise[3, 3] = self.level_walking_variance

        zupts = self.find_zupts(data, self.sampling_rate_hz)

        gyro_data = np.deg2rad(data[SF_GYR].to_numpy())
        acc_data = data[SF_ACC].to_numpy()

        parameters = SimpleZuptParameter(level_walking=self.level_walking)

        states, covariances = rts_kalman_update_series(
            acc_data,
            gyro_data,
            initial_orientation,
            sampling_rate_hz,
            meas_noise,
            process_noise,
            zupts,
            parameters=parameters,
            forward_pass_func=self._forward_pass,
            forward_pass_dependencies=self._prepare_forward_pass_dependencies(),
        )
        self.position_ = pd.DataFrame(states[0], columns=GF_POS)
        self.position_.index.name = "sample"
        self.velocity_ = pd.DataFrame(states[1], columns=GF_VEL)
        self.velocity_.index.name = "sample"
        self.orientation_object_ = Rotation.from_quat(states[2])

        covariance_cols = [
            "x_pos_cov",
            "y_pos_cov",
            "z_pos_cov",
            "x_vel_cov",
            "y_vel_cov",
            "z_vel_cov",
            "x_ori_cov",
            "y_ori_cov",
            "z_ori_cov",
        ]
        covariance_cols = pd.MultiIndex.from_product((covariance_cols, covariance_cols))
        covariances = covariances.reshape((covariances.shape[0], -1))
        self.covariance_ = pd.DataFrame(covariances, columns=covariance_cols)

        self.zupts_ = bool_array_to_start_end_array(zupts)
        return self

    def _convert_deprecated_args(  # noqa: no-self-use
        self, deprecated_args: Dict[str, Any], zupt_detector: NormZuptDetector
    ) -> NormZuptDetector:
        # TODO: Remove after full deprecation
        zupt_detector.window_length_s = deprecated_args.get("zupt_window_length_s", zupt_detector.window_length_s)
        zupt_detector.inactive_signal_threshold = deprecated_args.get(
            "zupt_threshold_dps", zupt_detector.inactive_signal_threshold
        )
        if "zupt_window_overlap_s" in deprecated_args and deprecated_args["zupt_window_overlap_s"] is not None:
            zupt_detector.window_overlap = (
                float(zupt_detector.window_length_s) / deprecated_args["zupt_window_overlap_s"]
            )
        return zupt_detector

    def find_zupts(self, data, sampling_rate_hz: float):
        """Find the ZUPT samples based on the provided data.

        By default this method uses only the gyro data, but custom ZUPT method can be implemented by subclassing the
        Kalmanfilter and overwriting this method.

        .. warning:: If you plan to implement your own Zupt Detection, use a custom ZuptDetector as input.
                     This method is deprecated and will be removed

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
        # TODO: Use normal ZUPT detector after deprecation
        z = self._zupt_detector.clone().detect(data, sampling_rate_hz)
        return z.per_sample_zupts_

    def _prepare_forward_pass_dependencies(self) -> ForwardPassDependencies:  # noqa: no-self-use
        """Create the dependencies for the numba functions.

        This should be overwritten by subclasses, that need custom internal functions/parameters.
        """
        return ForwardPassDependencies(motion_update_func=simple_navigation_equations, motion_update_func_parameters=())
