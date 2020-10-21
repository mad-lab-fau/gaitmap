"""An error state kalman filter with Rauch-Tung-Striebel smoothing fo estimating trajectories."""
from typing import Union

import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseTrajectoryMethod, BaseType
from gaitmap.utils.consts import SF_GYR, SF_ACC
from gaitmap.utils.dataset_helper import is_single_sensor_dataset, SingleSensorDataset
from gaitmap.utils.fast_quaternion_math import quat_from_rotvec, multiply, rotate_vector
from gaitmap.utils.static_moment_detection import find_static_samples
from gaitmap.utils.consts import GRAV_VEC


class RtsKalman(BaseTrajectoryMethod):
    """An ESKF with RTS smoothing for trajectory estimation.

    The two main papers used for implementing are:
    [1] D. Simón Colomar, J. Nilsson and P. Händel, "Smoothing for ZUPT-aided INSs,"
    [2] Solà, Joan. (2015). Quaternion kinematics for the error-state KF.
    """

    def __init__(
        self,
        initial_orientation: Union[np.ndarray, Rotation] = np.array([0, 0, 0, 1.0]),
        zupt_threshold: float = 7.0,
        zupt_noise: float = 0.00001,
        proc_noise_velocity: float = 0.1,
        proc_noise_orientation: float = 0.01,
    ):
        self.initial_orientation = initial_orientation
        self.zupt_threshold = zupt_threshold
        self.zupt_noise = zupt_noise
        self.proc_noise_velocity = proc_noise_velocity
        self.proc_noise_orientation = proc_noise_orientation
        self.data = None
        self.sampling_rate_hz = None
        self.covariance_ = None

    def estimate(self: BaseType, data: SingleSensorDataset, sampling_rate_hz: float) -> BaseType:
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        initial_orientation = self.initial_orientation

        is_single_sensor_dataset(self.data, frame="sensor", raise_exception=True)
        if isinstance(initial_orientation, Rotation):
            initial_orientation = Rotation.as_quat(initial_orientation)
        initial_orientation = initial_orientation.copy()

        process_noise = np.zeros((9, 9))
        process_noise[3:6, 3:6] = self.proc_noise_velocity * np.eye(3)
        process_noise[6:, 6:] = self.proc_noise_orientation * np.eye(3)
        covariance = np.copy(process_noise)

        # measure noise
        meas_noise = self.zupt_noise * np.eye(3)

        gyro_data = np.deg2rad(data[SF_GYR].to_numpy())
        acc_data = data[SF_ACC].to_numpy()
        zupts = self.find_zupts(gyro_data)

        states, covariances = _rts_kalman_update_series(
            acc_data, gyro_data, initial_orientation, sampling_rate_hz, meas_noise, covariance, process_noise, zupts
        )
        self.position_ = states[0]
        self.velocity_ = states[1]
        self.orientation_ = Rotation.from_quat(states[2])
        self.covariance_ = covariances
        return self

    def find_zupts(self, gyro):
        """Find the ZUPT sample based on the gyro measurements."""
        return find_static_samples(gyro, 10, self.zupt_threshold * (np.pi / 180), "maximum", 5)


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
    accel, gyro, initial_orientation, sampling_rate_hz, meas_noise, covariance, process_noise, zupts
):
    prior_covariances = np.empty((accel.shape[0], 9, 9))
    posterior_covariances = np.empty((accel.shape[0], 9, 9))
    prior_error_states = np.empty((accel.shape[0], 9))
    posterior_error_states = np.empty((accel.shape[0], 9))
    positions = np.empty((accel.shape[0], 3))
    velocities = np.empty((accel.shape[0], 3))
    orientations = np.empty((accel.shape[0], 4))
    state_transitions = np.empty((accel.shape[0], 9, 9))

    position = np.zeros(3)
    velocity = np.zeros(3)
    orientation = np.copy(initial_orientation)
    error_state = np.zeros(9)
    transition_matrix = np.eye(9)
    transition_matrix[:3, 3:6] = np.eye(3) / sampling_rate_hz

    meas_matrix = np.zeros((3, 9))
    meas_matrix[:, 3:6] = np.eye(3)

    for i, zupt in enumerate(zupts):
        acc = np.ascontiguousarray(accel[i])
        omega = np.ascontiguousarray(gyro[i])

        # predict the new nominal position, velocity and orientation by integrating the acc and gyro measurement
        position, velocity, orientation = _rts_kalman_motion_update(
            acc, omega, orientation, position, velocity, sampling_rate_hz
        )
        positions[i, :] = position
        velocities[i, :] = velocity
        orientations[i, :] = orientation

        # predict the new error state, based on the old error state and the new nominal orientation
        rotated_acc = rotate_vector(orientation, acc)
        transition_matrix[3:6, 6:] = -cross_product_matrix(rotated_acc) / sampling_rate_hz

        error_state = transition_matrix @ error_state
        covariance = transition_matrix @ covariance @ transition_matrix.T + process_noise

        state_transitions[i, :, :] = transition_matrix
        prior_error_states[i, :] = error_state
        prior_covariances[i, :, :] = covariance

        # correct the error state if the current sample is marked as a zupt
        if zupt:
            innovation = error_state[3:6] - velocity
            innovation_cov = meas_matrix @ covariance @ meas_matrix.T + meas_noise
            inv = np.linalg.inv(innovation_cov)
            gain = covariance @ meas_matrix.T @ inv
            error_state = error_state - gain @ innovation

            factor = np.eye(covariance.shape[0]) - gain @ meas_matrix
            covariance_adj = factor @ covariance @ factor.T
            covariance = covariance_adj + gain @ meas_noise @ gain.T

        posterior_error_states[i, :] = error_state
        posterior_covariances[i, :, :] = covariance

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
    acc, gyro, initial_orientation, sampling_rate_hz, meas_noise, covariance, process_noise, zupts
):
    forward_eskf_results, forward_nominal_states = _rts_kalman_forward_pass(
        acc, gyro, initial_orientation, sampling_rate_hz, meas_noise, covariance, process_noise, zupts
    )

    corrected_error_states, corrected_covariances = _rts_kalman_backward_pass(*forward_eskf_results)
    corrected_states = _rts_kalman_correction_pass(*forward_nominal_states, corrected_error_states)
    return corrected_states, corrected_covariances
