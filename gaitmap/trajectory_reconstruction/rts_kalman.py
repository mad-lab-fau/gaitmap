from typing import Union

import numpy as np
from numba import njit, typeof
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseOrientationMethod, BaseType
from gaitmap.utils.consts import SF_GYR, SF_ACC
from gaitmap.utils.dataset_helper import SingleSensorDataset, is_single_sensor_dataset
from gaitmap.utils.fast_quaternion_math import rate_of_change_from_gyro, quat_from_rotvec, multiply, rotate_vector
from gaitmap.utils.static_moment_detection import find_static_samples


class RtsKalman(BaseOrientationMethod):

    def __init__(self, initial_orientation: Union[np.ndarray, Rotation] = np.array([0, 0, 0, 1.0]),
                 zupt_threshold: float = 7.0, zupt_noise: float = 0.00001,
                 proc_noise_velocity: float = 0.1,
                 proc_noise_orientation: float = 0.01):
        self.initial_orientation = initial_orientation
        self.zupt_threshold = zupt_threshold
        self.zupt_noise = zupt_noise
        self.proc_noise_velocity = proc_noise_velocity
        self.proc_noise_orientation = proc_noise_orientation

    def estimate(self: BaseType, data: SingleSensorDataset, sampling_rate_hz: float) -> BaseType:

        if not is_single_sensor_dataset(data, frame="sensor"):
            raise ValueError("Data is not a single sensor dataset.")

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        initial_orientation = self.initial_orientation
        if isinstance(initial_orientation, Rotation):
            initial_orientation = Rotation.as_quat(initial_orientation)
        initial_orientation = initial_orientation.copy()

        process_noise = np.zeros((9, 9))
        process_noise[3:6, 3:6] = self.proc_noise_velocity * np.eye(3)
        process_noise[6:, 6:] = self.proc_noise_orientation * np.eye(3)
        covariance = np.copy(process_noise)

        # measure noise
        R = self.zupt_noise * np.eye(3)
        
        gyro_data = np.deg2rad(data[SF_GYR].to_numpy())
        acc_data = data[SF_ACC].to_numpy()
        zupts = self.find_zupts(acc_data, gyro_data)

        states, covariances = _rts_kalman_update_series(acc_data, gyro_data, initial_orientation, sampling_rate_hz, R, covariance, process_noise, zupts)
        #self.orientation_object_ = Rotation.from_quat(rots)
        #return self
        return np.concatenate(states, axis=1), np.array(covariances)

    def find_zupts(self, acc, gyro):
        return find_static_samples(gyro, 10, self.zupt_threshold*(np.pi/180), "maximum", 5)


@njit()
def cross_product_matrix(vec):
    return np.array([[0., -vec[2], vec[1]], [vec[2], 0., -vec[0]], [-vec[1], vec[0], 0.]])


@njit()
def _rts_kalman_motion_update(acc, gyro, orientation, position, velocity, sampling_rate_hz):
    g = np.array([0.0, 0.0, 9.81])
    sigma = gyro / sampling_rate_hz
    r = quat_from_rotvec(sigma)
    new_orientation = multiply(orientation, r)

    rotated_acc = rotate_vector(new_orientation, acc)
    new_position = position + velocity / sampling_rate_hz + \
        0.5 * (rotated_acc - g) / sampling_rate_hz**2
    new_velocity = velocity + (rotated_acc - g) / sampling_rate_hz
    return new_position, new_velocity, new_orientation


@njit()
def _rts_kalman_forward_pass(accel, gyro, initial_orientation, sampling_rate_hz, R, covariance, process_noise, zupts):    
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
    F = np.eye(9)
    F[:3, 3:6] = np.eye(3)/sampling_rate_hz

    H = np.zeros((3, 9))
    H[:, 3:6] = np.eye(3)
    
    #for i, (acc, omega, zupt) in enumerate(zip(accel, gyro, zupts)):
    for i, zupt in enumerate(zupts):
        acc = np.ascontiguousarray(accel[i])
        omega = np.ascontiguousarray(gyro[i])
        position, velocity, orientation = _rts_kalman_motion_update(acc, omega, orientation, position, velocity, sampling_rate_hz)
        positions[i, :] = position
        velocities[i, :] = velocity
        orientations[i, :] = orientation

        rotated_acc = rotate_vector(orientation, acc)
        F[3:6, 6:] = - cross_product_matrix(rotated_acc) / sampling_rate_hz

        error_state = F @ error_state
        covariance = F @ covariance @ F.T + process_noise

        state_transitions[i, :, :] = F
        prior_error_states[i, :] = error_state
        prior_covariances[i, :, :] = covariance

        if zupt:
            innovation = error_state[3:6] - velocity
            innovation_cov = H @ covariance @ H.T + R
            inv = np.linalg.inv(innovation_cov)
            gain = covariance @ H.T @ inv
            error_state = error_state - gain @ innovation

            factor = np.eye(covariance.shape[0]) - gain @ H
            noise = gain @ R @ gain.T
            covariance_adj = factor @ covariance @ factor.T
            covariance = covariance_adj + noise

        posterior_error_states[i, :] = error_state
        posterior_covariances[i, :, :] = covariance

    return prior_covariances, posterior_covariances, prior_error_states, posterior_error_states, state_transitions, positions, velocities, orientations


@njit()
def _rts_kalman_backward_pass(prior_covariances, posterior_covariances, prior_error_states, posterior_error_states, state_transitions):
    corrected_error_states = np.empty_like(prior_error_states)
    corrected_covariances = np.empty_like(prior_covariances)
    corrected_error_states[-1, :] = posterior_error_states[-1]
    corrected_covariances[-1, :, :] = posterior_covariances[-1]

    for i in range(len(prior_error_states)-2, -1, -1):
        F = state_transitions[i]
        A = posterior_covariances[i] @ F.T @ np.linalg.inv(prior_covariances[i+1])
        corrected_error_states[i, :] = posterior_error_states[i] + A @ (corrected_error_states[i+1] - prior_error_states[i+1])
        corrected_covariances[i, :, :] = posterior_covariances[i] + A @ (corrected_covariances[i+1] - prior_covariances[i+1]) @ A.T

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
def _rts_kalman_update_series(acc, gyro, initial_orientation, sampling_rate_hz, R, covariance, process_noise, zupts):
    forward_results = _rts_kalman_forward_pass(acc, gyro, initial_orientation, sampling_rate_hz, R, covariance, process_noise, zupts)
    forward_states = forward_results[-3:]

    corrected_error_states, corrected_covariances = _rts_kalman_backward_pass(*forward_results[:5])
    corrected_states = _rts_kalman_correction_pass(*forward_states, corrected_error_states)
    return corrected_states, corrected_covariances
