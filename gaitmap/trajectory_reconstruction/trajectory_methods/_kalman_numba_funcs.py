"""Helper functions for the RTS Kalman filter."""

from typing import Any, Callable, NamedTuple, Tuple

import numpy as np
from numba import njit

from gaitmap.utils.consts import GRAV_VEC
from gaitmap.utils.fast_quaternion_math import multiply, quat_from_rotvec, rotate_vector


class SimpleZuptParameter(NamedTuple):
    """Parameter for a simple Zupt aided RtsKalman filter."""

    level_walking: bool


class ForwardPassDependencies(NamedTuple):
    """Required dependency functions for the forward pass."""

    motion_update_func: Callable
    # This needs to be a tuple and not a dict, as numba can not process dicts
    motion_update_func_parameters: Tuple


def rts_kalman_update_series(
    acc,
    gyro,
    initial_orientation,
    sampling_rate_hz,
    meas_noise,
    process_noise,
    zupts,
    parameters: Tuple[Any, ...],
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
def simple_navigation_equations(acc, gyro, orientation, position, velocity, sampling_rate_hz, _):
    """Calculate the next state using simple navigation equations."""
    sigma = gyro / sampling_rate_hz
    r = quat_from_rotvec(sigma)
    new_orientation = multiply(orientation, r)

    rotated_acc = rotate_vector(new_orientation, acc)
    new_position = position + velocity / sampling_rate_hz + 0.5 * (rotated_acc - GRAV_VEC) / sampling_rate_hz**2
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

    Notes
    -----
    These equations are based on the following papers:
    _[1] describes the overall structure of the open loop filter.
    However, they do not consider the level walking case.
    We expand the proposed algorithm (Algorithm 1 Pseudo Code) by using also a corrective measurement of the position
    p and not just the velocity v.
    Further, we use the "fully" open loop variant of the algorithm that is explained in the text.
    Hence, we first calculate the full forward pass (until the end of the recording) and then apply smoothing,
    instead of breaking the loop at every ZUPT.
    The main reason for that is just simplicity of implementation as we do not have any live requirement.
    _[2] gives more details about the individual equations (in the closed looped form).
    From _[2] we implement the state equations in the global coordinate system (chapter 7).

    .. [1] D. Simón Colomar, J. Nilsson and P. Händel, "Smoothing for ZUPT-aided INSs,"
    .. [2] Solà, Joan. (2015). Quaternion kinematics for the error-state KF


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
            acc,
            omega,
            orientations[i],
            positions[i],
            velocities[i],
            sampling_rate_hz,
            dependencies.motion_update_func_parameters,
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
        # We try to annotate each equation with the respective symbols Algorithm 1 from paper _[1].
        if zupt:
            meas_space_error = meas_jacob @ prior_error_states[i + 1]

            # Our zupt_measurement is directly the error and not really the measurement.
            # We assume that velocity an z-position (in level walking case) are all 0.
            # This means every value that is still measured/estimated must be an error.
            #
            # velocity error
            zupt_measurement[0:3] = velocity  # \hat{v}_n
            if parameters.level_walking:
                # z-position error
                zupt_measurement[3] = position[2]  # not in paper but would be \hat{p_z}_n

            # Instead of using the calculated error, we calculate how much the error has increased since the last time
            # step.
            # This value is than used to calculate the change in covariance.
            # This is needed, because we do not apply the innovation to the actual state as we would do in a regular
            # kalman filter.
            # The error is just summed up and only applied to the state in the final smoothing pass.
            #
            # We use "naive" measurement function here as we directly observe the error.
            # But this would be equivalent h(x_t) from eq. 271 in paper _[1].
            innovation = meas_func.copy()
            # innovation = \delta\hat{x}_{n|n-1} - h(\hat{x}_n)
            # (_[1] only uses \hat{v}_n and we use the meas. func h from _[2])
            innovation *= zupt_measurement - meas_space_error
            # innovation_cov = HP_{n|n-1}H^T + R
            innovation_cov = meas_jacob @ prior_covariances[i + 1] @ meas_jacob.T + meas_noise
            # K_n = P_{n|n-1}H^T(HP_{n|n-1}H^T + R)^{-1} = P_{n|n-1}H^T(innovation_cov)^{-1}
            # Note that we use the Moore-Penrose Pseudo inverse instead of the "real" inverse to avoid issues with
            # rounding errors that lead to matrizes that are not well defined.
            # The pseudo inverse might still have issues, as it might happen that the internal SVD will not converge.
            # So far we couldn't see any obvious errors on the data we tested on.
            gain = prior_covariances[i + 1] @ meas_jacob.T @ np.linalg.pinv(innovation_cov)
            # posterior_error_state = \delta\hat{x}_{n|n}
            #                       = \delta\hat{x}_{n|n-1} - K_n(\delta\hat{x}_{n|n-1} - h(\hat{x}_n)
            #                       = \delta\hat{x}_{n|n-1} - K_n(innovation)
            posterior_error_states[i + 1, :] = prior_error_states[i + 1] + gain @ innovation
            # To calculate the posterior_covariance P_{n|n-1} we do not use the equation from _[1], but use the
            # positive and symmetric Joseph from to avoid numerical instability (see footnote 26 in _[2] page 61).
            #
            # factor = I - K_nH
            factor = np.eye(initial_covariance.shape[0]) - gain @ meas_jacob
            # covariance_adj = (I - K_nH)P_{n|n-1}(I - K_nH)^T = (factor)P_{n|n-1}(factor)^T
            covariance_adj = factor @ prior_covariances[i + 1] @ factor.T
            # posterior_covariance = P_{n|n} = (I - K_nH)P_{n|n-1}(I - K_nH)^T + K_nRK_n^T
            # (Note that we use R for the measurement noise as in _[1] instead of V as in _[2])
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


@njit()
def _rts_kalman_update_series(
    acc,
    gyro,
    initial_orientation,
    sampling_rate_hz,
    meas_noise,
    process_noise,
    zupts,
    parameters: Tuple[Any, ...],
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
