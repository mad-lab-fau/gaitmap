"""An error state kalman filter with Rauch-Tung-Striebel smoothing fo estimating trajectories."""
from typing import Union, TypeVar, Optional

import numpy as np
import pandas as pd
from numba import njit
from scipy.spatial.transform import Rotation

<<<<<<< HEAD
from gaitmap.base import BaseTrajectoryMethod
from gaitmap.utils.consts import GRAV_VEC
from gaitmap.utils.consts import SF_GYR, SF_ACC, GF_POS, GF_VEL
from gaitmap.utils.datatype_helper import is_single_sensor_data, SingleSensorData
from gaitmap.utils.fast_quaternion_math import quat_from_rotvec, multiply, rotate_vector
from gaitmap.utils.static_moment_detection import find_static_samples

Self = TypeVar("Self", bound="RtsKalman")
=======
from gaitmap.base import BaseTrajectoryMethod, BaseType
from gaitmap.utils.consts import GRAV_VEC
from gaitmap.utils.consts import SF_GYR, SF_ACC, GF_POS, GF_VEL
from gaitmap.utils.dataset_helper import is_single_sensor_dataset, SingleSensorDataset
from gaitmap.utils.fast_quaternion_math import (
    quat_from_rotvec,
    multiply,
    rotate_vector,
    normalize,
)
from gaitmap.utils.static_moment_detection import find_static_samples
>>>>>>> 48a8b94 (Implemented first version that uses quaternion correction)


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
        Length of the window used in the default method to find ZUPTs.
        If the value is too small at least a window of 2 samples is used.
        Given in seconds.
    zupt_window_overlap_s
        Length of the window overlap used in the default method to find ZUPTs.
        It is given in seconds and if not given it defaults to half the window length.

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
    ...                    zupt_threshold_dps=34.0,
    ...                    zupt_variance=10e-8,
    ...                    velocity_error_variance=10e5,
    ...                    orientation_error_variance=10e-2)
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
    zupt_orientation_update: bool
    zupt_orientation_error_variance: float

    data: SingleSensorData
    sampling_rate_hz: float
    covariance_: pd.DataFrame

    def __init__(
        self,
        initial_orientation: Union[np.ndarray, Rotation] = np.array([0, 0, 0, 1.0]),
        zupt_threshold_dps: float = 34.0,
        zupt_variance: float = 10e-8,
        velocity_error_variance: float = 10e5,
        orientation_error_variance: float = 10e-2,
        level_walking: bool = True,
        level_walking_variance: float = 10e-8,
        zupt_orientation_update: bool = True,
        zupt_orientation_error_variance: float = 10e-1,
        zupt_window_length_s: float = 0.05,
        zupt_window_overlap_s: Optional[float] = None,
    ):
        self.initial_orientation = initial_orientation
        self.zupt_threshold_dps = zupt_threshold_dps
        self.zupt_variance = zupt_variance
        self.velocity_error_variance = velocity_error_variance
        self.orientation_error_variance = orientation_error_variance
        self.level_walking = level_walking
        self.level_walking_variance = level_walking_variance
        self.zupt_orientation_update = zupt_orientation_update
        self.zupt_orientation_error_variance = zupt_orientation_error_variance
        self.zupt_window_length_s = zupt_window_length_s
        self.zupt_window_overlap_s = zupt_window_overlap_s

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
        covariance = np.copy(process_noise)

        # measure noise
        # TODO: Set different zupt noises for ori and vel?
        meas_noise = np.zeros((6, 6))
        meas_noise[0:3, 0:3] = np.eye(3) * self.zupt_variance
        if self.level_walking is True:
            meas_noise[3, 3] = self.level_walking_variance
        if self.zupt_orientation_update:
            meas_noise[4:6, 4:6] = np.eye(2) * self.zupt_orientation_error_variance

        gyro_data = np.deg2rad(data[SF_GYR].to_numpy())
        acc_data = data[SF_ACC].to_numpy()
        zupts = self.find_zupts(gyro_data)

        states, covariances = _rts_kalman_update_series(
            acc_data,
            gyro_data,
            initial_orientation,
            sampling_rate_hz,
            meas_noise,
            covariance,
            process_noise,
            zupts,
            self.level_walking,
            self.zupt_orientation_update,
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
        self.covariance_ = pd.concat(
            [pd.DataFrame(cov, columns=covariance_cols, index=covariance_cols) for cov in covariances],
            keys=range(len(covariances)),
        )
        return self

    def find_zupts(self, gyro):
        """Find the ZUPT samples based on the gyro measurements."""
        window_length = max(2, round(self.sampling_rate_hz * self.zupt_window_length_s))
        zupt_window_overlap_s = self.zupt_window_overlap_s
        if zupt_window_overlap_s is None:
            window_overlap = int(window_length // 2)
        else:
            window_overlap = round(self.sampling_rate_hz * zupt_window_overlap_s)
        return find_static_samples(
            gyro, window_length, self.zupt_threshold_dps * (np.pi / 180), "maximum", window_overlap
        )


@njit()
def cross_product_matrix(vec):
    """Get the matrix representation of a cross product with a vector."""
    return np.array([[0.0, -vec[2], vec[1]], [vec[2], 0.0, -vec[0]], [-vec[1], vec[0], 0.0]])


@njit()
def _rts_kalman_motion_update(acc, gyro, orientation, position, velocity, sampling_rate_hz):
    sigma = gyro / sampling_rate_hz
    r = quat_from_rotvec(sigma)
    new_orientation = multiply(orientation, r)

    rotated_acc = rotate_vector(new_orientation, acc)
    new_position = position + velocity / sampling_rate_hz + 0.5 * (rotated_acc - GRAV_VEC) / sampling_rate_hz ** 2
    new_velocity = velocity + (rotated_acc - GRAV_VEC) / sampling_rate_hz
    return new_position, new_velocity, new_orientation


@njit()
def _rts_kalman_forward_pass(
    accel,
    gyro,
    initial_orientation,
    sampling_rate_hz,
    meas_noise,
    covariance,
    process_noise,
    zupts,
    level_walking,
    orientation_correction,
):
    prior_covariances = np.empty((accel.shape[0] + 1, 9, 9))
    posterior_covariances = np.empty((accel.shape[0] + 1, 9, 9))
    prior_error_states = np.empty((accel.shape[0] + 1, 9))
    posterior_error_states = np.empty((accel.shape[0] + 1, 9))
    positions = np.empty((accel.shape[0] + 1, 3))
    velocities = np.empty((accel.shape[0] + 1, 3))
    orientations = np.empty((accel.shape[0] + 1, 4))
    state_transitions = np.empty((accel.shape[0], 9, 9))

    # initialize states
    positions[0, :] = np.zeros(3)
    velocities[0, :] = np.zeros(3)
    orientations[0, :] = np.copy(initial_orientation)
    prior_error_states[0, :] = np.zeros(9)
    posterior_error_states[0, :] = np.zeros(9)
    prior_covariances[0, :, :] = covariance
    posterior_covariances[0, :, :] = covariance
    transition_matrix = np.eye(9)
    transition_matrix[:3, 3:6] = np.eye(3) / sampling_rate_hz

    gravity = normalize(GRAV_VEC)

    # During zupt we measure 6 values:
    # 1-3: v_{x,y,z} = 0
    # 4 : p_z = 0
    # 5 : angle(global_acc, global_grav) around x-axis = 0
    # 6 : angle(global_acc, global_grav) around y-axis = 0
    #
    # The values 1-4 are directly part of the stater space.
    # This means we have a trivial measurement function h and jacobian H
    # Value 5 is the angle between the local z-axis and the global z-axis.
    # This can be represented as the second euler angle in a 3-1-3 euler angel configuration.
    # Hence we can calculate it from the quaternion components.
    # See notes on the full calculation in the ZUPT update code.
    zupt_measurement = np.zeros(6)
    # meas_jacob dh/d(\delta x) maps from measurement space into the error space
    # Because we directly observe the values from our error vector without any conversion, it just consists of 1 in the
    # right places.
    meas_jacob = np.zeros((6, 9))
    # The meas_func is used to mask only the correction values we want to have.
    meas_func = np.zeros(6)
    # Zero Velocity update
    meas_func[0:3] = 1
    meas_jacob[0:3, 3:6] = np.eye(3)
    if level_walking is True:
        # Zero elevation update
        meas_jacob[3, 2] = 1
        meas_func[3] = 1
    if orientation_correction is True:
        # Angle error update
        meas_func[4:6] = 1
        meas_jacob[4:6, 7:9] = np.eye(2)

    for i, zupt in enumerate(zupts):
        acc = np.ascontiguousarray(accel[i])
        omega = np.ascontiguousarray(gyro[i])

        # calculate the new nominal position, velocity and orientation by integrating the acc and gyro measurement
        position, velocity, orientation = _rts_kalman_motion_update(
            acc, omega, orientations[i], positions[i], velocities[i], sampling_rate_hz
        )
        positions[i + 1, :] = position
        velocities[i + 1, :] = velocity
        orientations[i + 1, :] = orientation

        # predict the new error state, based on the old error state and the new nominal orientation
        rotated_acc = rotate_vector(orientation, acc)
        transition_matrix[3:6, 6:] = -cross_product_matrix(rotated_acc) / sampling_rate_hz
        state_transitions[i, :, :] = transition_matrix

        prior_error_states[i + 1, :] = transition_matrix @ posterior_error_states[i]
        prior_covariances[i + 1, :, :] = (
            transition_matrix @ posterior_covariances[i] @ transition_matrix.T + process_noise
        )

        # correct the error state if the current sample is marked as a zupt, this is the update step
        # Note, that the error state is not used to correct the nominal state at this point.
        # We are just correcting the error state.
        # Fusing with the nominal state is performed in the correction step of the smoothing.
        if zupt:
            meas_space_error = meas_jacob @ prior_error_states[i + 1]

            # velocity error
            zupt_measurement[0:3] = velocity
            if level_walking:
                # z-position error
                zupt_measurement[3] = position[2]
            # orientation error
            if orientation_correction:
                # Find the angles between rotated acc and gravity
                norm_acc = normalize(rotated_acc)
                cross = np.cross(norm_acc, gravity)
                dot = np.dot(norm_acc, gravity)
                zupt_measurement[4] = np.arctan2(np.dot(cross, np.array([1.0, 0, 0])), dot)
                zupt_measurement[5] = np.arctan2(np.dot(cross, np.array([0, 1.0, 0])), dot)

            innovation = meas_func.copy()
            # Instead of using the calculated error, we calculate how much the error has increased since the last time
            # step.
            # This value is than used to calculate the change in covariance.
            # This is needed, because we do not apply the innovation to the actual state as we would do in a regular
            # kalman filter.
            # The error is just summed up and only applied to the state in the final smoothing pass.
            innovation *= zupt_measurement - meas_space_error
            innovation_cov = meas_jacob @ prior_covariances[i + 1] @ meas_jacob.T + meas_noise
            gain = prior_covariances[i + 1] @ meas_jacob.T @ np.linalg.pinv(innovation_cov)
            posterior_error_states[i + 1, :] = prior_error_states[i + 1] + gain @ innovation
            factor = np.eye(covariance.shape[0]) - gain @ meas_jacob
            covariance_adj = factor @ prior_covariances[i + 1] @ factor.T
            posterior_covariances[i + 1, :, :] = covariance_adj + gain @ meas_noise @ gain.T
        else:
            posterior_error_states[i + 1, :] = prior_error_states[i + 1]
            posterior_covariances[i + 1, :, :] = prior_covariances[i + 1]

    return (
        (prior_covariances, posterior_covariances, prior_error_states, posterior_error_states, state_transitions),
        (positions, velocities, orientations,),
    )


@njit()
def _rts_kalman_backward_pass(
    prior_covariances, posterior_covariances, prior_error_states, posterior_error_states, state_transitions
):
    corrected_error_states = np.empty_like(prior_error_states)
    corrected_covariances = np.empty_like(prior_covariances)
    corrected_error_states[-1, :] = posterior_error_states[-1]
    corrected_covariances[-1, :, :] = posterior_covariances[-1]

    for i in range(len(prior_error_states) - 2, -1, -1):
        transition_matrix = state_transitions[i]
        gain = posterior_covariances[i] @ transition_matrix.T @ np.linalg.inv(prior_covariances[i + 1])
        corrected_error_states[i, :] = posterior_error_states[i] + gain @ (
            corrected_error_states[i + 1] - prior_error_states[i + 1]
        )
        corrected_covariances[i, :, :] = (
            posterior_covariances[i] + gain @ (corrected_covariances[i + 1] - prior_covariances[i + 1]) @ gain.T
        )

    return corrected_error_states, corrected_covariances


@njit()
def _rts_kalman_correction_pass(positions, velocities, orientations, corrected_error_states):
    for i, state in enumerate(corrected_error_states):
        positions[i] -= state[:3]
        velocities[i] -= state[3:6]

        rot_correction = quat_from_rotvec(state[6:])
        orientations[i] = multiply(rot_correction, orientations[i])
    return positions, velocities, orientations


@njit(cache=True)
def _rts_kalman_update_series(
    acc,
    gyro,
    initial_orientation,
    sampling_rate_hz,
    meas_noise,
    covariance,
    process_noise,
    zupts,
    level_walking,
    orientation_correction,
):
    forward_eskf_results, forward_nominal_states = _rts_kalman_forward_pass(
        acc,
        gyro,
        initial_orientation,
        sampling_rate_hz,
        meas_noise,
        covariance,
        process_noise,
        zupts,
        level_walking,
        orientation_correction,
    )
    corrected_error_states, corrected_covariances = _rts_kalman_backward_pass(*forward_eskf_results)

    corrected_states = _rts_kalman_correction_pass(*forward_nominal_states, corrected_error_states)
    return corrected_states, corrected_covariances
