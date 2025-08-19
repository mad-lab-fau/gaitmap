"""An error state kalman filter with Rauch-Tung-Striebel smoothing fo estimating trajectories."""

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tpcp import cf
from typing_extensions import Self

from gaitmap.base import BaseTrajectoryMethod, BaseZuptDetector
from gaitmap.trajectory_reconstruction.trajectory_methods._kalman_numba_funcs import (
    ForwardPassDependencies,
    SimpleZuptParameter,
    default_rts_kalman_forward_pass,
    madgwick_mag_motion_update,
    madgwick_motion_update,
    rts_kalman_update_series,
    simple_navigation_equations,
)
from gaitmap.utils.consts import GF_POS, GF_VEL, SF_ACC, SF_GYR, SF_MAG
from gaitmap.utils.datatype_helper import SingleSensorData, SingleSensorStrideList, is_single_sensor_data
from gaitmap.zupt_detection import NormZuptDetector


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
    rotated_data_
        The rotated data after applying the estimated orientation to the data.
        The first sample of the data remain unrotated (initial orientation).

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
    >>> kalman = RtsKalman(
    ...     initial_orientation=np.array([0, 0, 0, 1.0]),
    ...     zupt_variance=10e-8,
    ...     velocity_error_variance=10e5,
    ...     orientation_error_variance=10e-2,
    ...     zupt_detector=NormZuptDetector(semsor="gyr", window_length_s=0.05),
    ... )
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
    zupts_: pd.DataFrame

    _forward_pass = default_rts_kalman_forward_pass

    def __init__(
        self,
        *,
        initial_orientation: Union[np.ndarray, Rotation] = cf(np.array([0, 0, 0, 1.0])),
        zupt_variance: float = 10e-8,
        velocity_error_variance: float = 10e5,
        orientation_error_variance: float = 10e-2,
        level_walking: bool = True,
        level_walking_variance: float = 10e-8,
        zupt_detector=cf(
            NormZuptDetector(
                sensor="gyr", window_length_s=0.05, window_overlap=0.5, metric="maximum", inactive_signal_threshold=34.0
            )
        ),
    ) -> None:
        self.initial_orientation = initial_orientation
        self.zupt_variance = zupt_variance
        self.velocity_error_variance = velocity_error_variance
        self.orientation_error_variance = orientation_error_variance
        self.level_walking = level_walking
        self.level_walking_variance = level_walking_variance
        self.zupt_detector = zupt_detector

    def estimate(
        self,
        data: SingleSensorData,
        *,
        sampling_rate_hz: float,
        stride_event_list: Optional[SingleSensorStrideList] = None,
    ) -> Self:
        """Estimate the position, velocity and orientation of the sensor.

        Parameters
        ----------
        data
            Continuous sensor data including gyro and acc values.
            The gyro data is expected to be in deg/s!
        sampling_rate_hz
            The sampling rate of the data in Hz
        stride_event_list
            Optional stride event list that will be passed to the ZUPT detector.
            If this information is actually used depends on the ZUPT detector.

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
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

        zupt_detector = self.zupt_detector.clone().detect(
            data, sampling_rate_hz=sampling_rate_hz, stride_event_list=stride_event_list
        )
        zupts = zupt_detector.per_sample_zupts_
        self.zupts_ = zupt_detector.zupts_

        acc_data, gyro_data, mag_data = self._prepare_data()

        parameters = SimpleZuptParameter(level_walking=self.level_walking)

        states, covariances = rts_kalman_update_series(
            acc_data,
            gyro_data,
            mag_data,
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

        return self

    def _prepare_forward_pass_dependencies(self) -> ForwardPassDependencies:
        """Create the dependencies for the numba functions.

        This should be overwritten by subclasses, that need custom internal functions/parameters.
        """
        return ForwardPassDependencies(motion_update_func=simple_navigation_equations, motion_update_func_parameters=())

    def _prepare_data(self) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        gyro_data = np.deg2rad(self.data[SF_GYR].to_numpy())
        acc_data = self.data[SF_ACC].to_numpy()
        return acc_data, gyro_data, None


class MadgwickRtsKalman(RtsKalman):
    """An extention of the RTS Kalman filter that uses the Madgwick filter for orientation estimation.

    This method is basically identical to the normal :class:`~gaitmap.trajectory_reconstruction.RtsKalman` filter,
    but uses the Madgwick filter for orientation estimation.
    This should provide more robust orientation updates during long regions without ZUPTs.
    The influence of the Madgwick filter can be controlled by the `madgwick_beta` parameter.

    Parameters
    ----------
    initial_orientation
        The initial orientation of the sensor that is assumed.
        It is critical that this value is close to the actual orientation.
        If you pass an array, remember that the order of elements must be x, y, z, w.
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
    zupt_detector
        An instance of a valid Zupt detector that will be used to find ZUPTs.
    madgwick_beta
        The beta parameter of the Madgwick filter.
        This parameter controls how harsh the acceleration based correction is.
        A high value performs large corrections and a small value small and gradual correction.
        A high value should only be used if the sensor is moved slowly.
        A value of 0 is identical to just the Gyro Integration (i.e. identical to the
        :class:`~gaitmap.trajectory_reconstruction.RtsKalman`).
    use_magnetometer
        Flag to control if the magnetometer should be used in the Madgwick filter.
        Note, that the rest of the algorithm does not change based on this parameter.
        The Kalman update steps and the error propagation still only consider the accelerometer and gyroscope data.
        If True, the data is expected to have the `mag_x, mag_y, mag_z` columns.

    Attributes
    ----------
    orientation_
        The rotations as a *SingleSensorOrientationList*, including the initial orientation.
        This means there are len(data) + 1 orientations.
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
        2D array indicating the start and the end samples of the detected ZUPTs for debug purposes.
    rotated_data_
        The rotated data after applying the estimated orientation to the data.
        The first sample of the data remain unrotated (initial orientation).

    Other Parameters
    ----------------
    data
        The data passed to the estimate method
    sampling_rate_hz
        The sampling rate of this data

    Notes
    -----
    For more information on the Kalman Filter, see :class:`~gaitmap.trajectory_reconstruction.RtsKalman`.
    For more information about the Madgwick orientation filter, see
    :class:`~gaitmap.trajectory_reconstruction.MadgwickAHRS`.
    """

    madgwick_beta: float
    use_magnetometer: bool

    def __init__(
        self,
        *,
        initial_orientation: Union[np.ndarray, Rotation] = cf(np.array([0, 0, 0, 1.0])),
        zupt_variance: float = 10e-8,
        velocity_error_variance: float = 10e5,
        orientation_error_variance: float = 10e-2,
        level_walking: bool = True,
        level_walking_variance: float = 10e-8,
        zupt_detector=cf(
            NormZuptDetector(
                sensor="gyr", window_length_s=0.05, window_overlap=0.5, metric="maximum", inactive_signal_threshold=34.0
            )
        ),
        madgwick_beta: float = 0.2,
        use_magnetometer: bool = False,
    ) -> None:
        self.madgwick_beta = madgwick_beta
        self.use_magnetometer = use_magnetometer
        super().__init__(
            initial_orientation=initial_orientation,
            zupt_variance=zupt_variance,
            velocity_error_variance=velocity_error_variance,
            orientation_error_variance=orientation_error_variance,
            level_walking=level_walking,
            level_walking_variance=level_walking_variance,
            zupt_detector=zupt_detector,
        )

    def _prepare_forward_pass_dependencies(self) -> ForwardPassDependencies:
        # This is kind of the wrong place for this check, but it is simpler then completely changing the structure
        return ForwardPassDependencies(
            motion_update_func=madgwick_mag_motion_update if self.use_magnetometer else madgwick_motion_update,
            motion_update_func_parameters=(self.madgwick_beta,),
        )

    def _prepare_data(self) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        gyro_data = np.deg2rad(self.data[SF_GYR].to_numpy())
        acc_data = self.data[SF_ACC].to_numpy()
        if self.use_magnetometer is False:
            return acc_data, gyro_data, None
        is_single_sensor_data(self.data, frame="sensor", raise_exception=True, check_mag=True)
        mag_data = self.data[SF_MAG].to_numpy()
        return acc_data, gyro_data, mag_data
