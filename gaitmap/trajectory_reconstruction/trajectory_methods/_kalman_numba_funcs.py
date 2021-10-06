"""Helper functions for the RTS Kalman filter."""

from typing import Callable, NamedTuple

import numpy as np
from numba import njit

from gaitmap.utils.consts import GRAV_VEC
from gaitmap.utils.fast_quaternion_math import multiply, quat_from_rotvec, rotate_vector


class _RtsParameter:
    pass


class SimpleZuptParameter(_RtsParameter, NamedTuple):
    """Parameter for a simple Zupt aided RtsKalman filter."""

    level_walking: bool


class ForwardPassDependencies(NamedTuple):
    """Required dependency functions for the forward pass."""

    motion_update_func: Callable


def rts_kalman_update_series(
    acc,
    gyro,
    initial_orientation,
    sampling_rate_hz,
    meas_noise,
    process_noise,
    zupts,
    parameters: _RtsParameter,
    forward_pass_func: Callable,
    forward_pass_dependencies: ForwardPassDependencies,
):
    """Perform a forward and backwards kalman pass with smoothing over the entire series."""
    return _rts_kalman_update_series(
        acc,
        gyro,
        initial_orientation,
        sampling_rate_hz,
        meas_noise,
        process_noise,
        zupts,
        parameters=parameters,
        forward_pass_func=forward_pass_func,
        forward_pass_dependencies=forward_pass_dependencies,
    )


@njit()
def cross_product_matrix(vec):
    """Get the matrix representation of a cross product with a vector."""
    return np.array([[0.0, -vec[2], vec[1]], [vec[2], 0.0, -vec[0]], [-vec[1], vec[0], 0.0]])


@njit()
def simple_navigation_equations(acc, gyro, orientation, position, velocity, sampling_rate_hz):
    """Calculate the next state using simple navigation equations."""
    sigma = gyro / sampling_rate_hz
    r = quat_from_rotvec(sigma)
    new_orientation = multiply(orientation, r)

    rotated_acc = rotate_vector(new_orientation, acc)
    new_position = position + velocity / sampling_rate_hz + 0.5 * (rotated_acc - GRAV_VEC) / sampling_rate_hz ** 2
    new_velocity = velocity + (rotated_acc - GRAV_VEC) / sampling_rate_hz
    return new_position, new_velocity, new_orientation


@njit()
def default_rts_kalman_forward_pass(  # noqa: too-many-statements, too-many-branches
    accel,
    gyro,
    initial_orientation,
    sampling_rate_hz,
    meas_noise,
    process_noise,
    zupts,
    parameters: SimpleZuptParameter,
    dependencies: ForwardPassDependencies,
):
    """Run the forward pass of an RTSKalman filter.

    The forward pass function that is used for the default RTSKalman filter.
    """
    prior_covariances = np.empty((accel.shape[0] + 1, 9, 9))
    posterior_covariances = np.empty((accel.shape[0] + 1, 9, 9))
    prior_error_states = np.empty((accel.shape[0] + 1, 9))
    posterior_error_states = np.empty((accel.shape[0] + 1, 9))
    positions = np.empty((accel.shape[0] + 1, 3))
    velocities = np.empty((accel.shape[0] + 1, 3))
    orientations = np.empty((accel.shape[0] + 1, 4))
    state_transitions = np.empty((accel.shape[0], 9, 9))

    # initialize states
    initial_covariance = np.copy(process_noise)
    positions[0, :] = np.zeros(3)
    velocities[0, :] = np.zeros(3)
    orientations[0, :] = np.copy(initial_orientation)
    prior_error_states[0, :] = np.zeros(9)
    posterior_error_states[0, :] = np.zeros(9)
    prior_covariances[0, :, :] = initial_covariance
    posterior_covariances[0, :, :] = initial_covariance
    transition_matrix = np.eye(9)
    transition_matrix[:3, 3:6] = np.eye(3) / sampling_rate_hz

    # During zupt we measure 4 values:
    # 1-3: v_{x,y,z} = 0
    # 4 : p_z = 0
    #
    # All values are directly part of the stater space.
    # This means we have a trivial measurement function h and jacobian H
    zupt_measurement = np.zeros(4)
    # meas_jacob dh/d(\delta x) maps from measurement space into the error space
    # Because we directly observe the values from our error vector without any conversion, it just consists of 1 in the
    # right places.
    meas_jacob = np.zeros((4, 9))
    # The meas_func is used to mask only the correction values we want to have.
    meas_func = np.zeros(4)
    # Zero Velocity update
    meas_func[0:3] = 1
    meas_jacob[0:3, 3:6] = np.eye(3)
    if parameters.level_walking is True:
        # Zero elevation update
        meas_jacob[3, 2] = 1
        meas_func[3] = 1

    for i, zupt in enumerate(zupts):
        acc = np.ascontiguousarray(accel[i])
        omega = np.ascontiguousarray(gyro[i])

        # calculate the new nominal position, velocity and orientation by integrating the acc and gyro measurement
        position, velocity, orientation = dependencies.motion_update_func(
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
            if parameters.level_walking:
                # z-position error
                zupt_measurement[3] = position[2]

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
            factor = np.eye(initial_covariance.shape[0]) - gain @ meas_jacob
            covariance_adj = factor @ prior_covariances[i + 1] @ factor.T
            posterior_covariances[i + 1, :, :] = covariance_adj + gain @ meas_noise @ gain.T
        else:
            posterior_error_states[i + 1, :] = prior_error_states[i + 1]
            posterior_covariances[i + 1, :, :] = prior_covariances[i + 1]

    return (
        (prior_covariances, posterior_covariances, prior_error_states, posterior_error_states, state_transitions),
        (positions, velocities, orientations),
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
    process_noise,
    zupts,
    parameters: _RtsParameter,
    forward_pass_func: Callable,
    forward_pass_dependencies: ForwardPassDependencies,
):
    forward_eskf_results, forward_nominal_states = forward_pass_func(
        acc,
        gyro,
        initial_orientation,
        sampling_rate_hz,
        meas_noise,
        process_noise,
        zupts,
        parameters=parameters,
        dependencies=forward_pass_dependencies,
    )
    corrected_error_states, corrected_covariances = _rts_kalman_backward_pass(*forward_eskf_results)

    corrected_states = _rts_kalman_correction_pass(*forward_nominal_states, corrected_error_states)
    return corrected_states, corrected_covariances
