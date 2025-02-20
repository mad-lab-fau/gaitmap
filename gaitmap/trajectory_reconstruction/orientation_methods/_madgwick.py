"""Implementation of the MadgwickAHRS."""

from typing import Optional, Union

import numpy as np
from joblib import Memory
from numba import njit
from scipy.spatial.transform import Rotation
from tpcp import cf
from typing_extensions import Self

from gaitmap.base import BaseOrientationMethod
from gaitmap.utils.consts import SF_ACC, SF_GYR, SF_MAG
from gaitmap.utils.datatype_helper import SingleSensorData, is_single_sensor_data
from gaitmap.utils.fast_quaternion_math import rate_of_change_from_gyro


class MadgwickAHRS(BaseOrientationMethod):
    """The MadwickAHRS algorithm to estimate the orientation of an IMU.

    This method applies a simple gyro integration with an additional correction step that tries to align the estimated
    orientation of the z-axis with gravity direction estimated from the acceleration data.
    This implementation is based on the paper [1]_.
    An open source C-implementation of the algorithm can be found at [2]_.
    The original code is published under GNU-GPL.

    Parameters
    ----------
    beta : flat between 0 and 1
        This parameter controls how harsh the acceleration based correction is.
        A high value performs large corrections and a small value small and gradual correction.
        A high value should only be used if the sensor is moved slowly.
        A value of 0 is identical to just the Gyro Integration (see also
        :class:`gaitmap.trajectory_reconstruction.SimpleGyroIntegration` for a separate implementation).
    initial_orientation
        The initial orientation of the sensor that is assumed.
        It is critical that this value is close to the actual orientation.
        Otherwise, the estimated orientation will drift until the real orientation is found.
        In some cases, the algorithm will not be able to converge if the initial orientation is too far off and the
        orientation will slowly oscillate.
        If you pass a array, remember that the order of elements must be x, y, z, w.
    use_magnetometer
        If the magnetometer data should be used to correct the orientation.
        Obvisouly, this requires a magnetometer to be present in the data.
        If True, the version of the Madgwick algorithm that uses the magnetometer data will be used.
    memory
        An optional `joblib.Memory` object that can be provided to cache the calls to madgwick series.

    Attributes
    ----------
    orientation_
        The rotations as a *SingleSensorOrientationList*, including the initial orientation.
        This means the there are len(data) + 1 orientations.
    orientation_object_
        The orientations as a single scipy Rotation object
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
    This class uses *Numba* as a just-in-time-compiler to achieve fast run times.
    In result, the first execution of the algorithm will take longer as the methods need to be compiled first.

    .. [1] Madgwick, S. O. H., Harrison, A. J. L., & Vaidyanathan, R. (2011).
           Estimation of IMU and MARG orientation using a gradient descent algorithm. IEEE International Conference on
           Rehabilitation Robotics, 1-7. https://doi.org/10.1109/ICORR.2011.5975346
    .. [2] http://x-io.co.uk/open-source-imu-and-ahrs-algorithms/

    Examples
    --------
    Your data must be a pd.DataFrame with columns defined by :obj:`~gaitmap.utils.consts.SF_COLS`.

    >>> import pandas as pd
    >>> from gaitmap.utils.consts import SF_COLS
    >>> data = pd.DataFrame(..., columns=SF_COLS)
    >>> sampling_rate_hz = 100
    >>> # Create an algorithm instance
    >>> mad = MadgwickAHRS(beta=0.2, initial_orientation=np.array([0, 0, 0, 1.0]))
    >>> # Apply the algorithm
    >>> mad = mad.estimate(data, sampling_rate_hz=sampling_rate_hz)
    >>> # Inspect the results
    >>> mad.orientation_
    <pd.Dataframe with resulting quaternions>
    >>> mad.orientation_object_
    <scipy.Rotation object>

    See Also
    --------
    gaitmap.trajectory_reconstruction: Other implemented algorithms for orientation and position estimation
    gaitmap.trajectory_reconstruction.StrideLevelTrajectory: Apply the method for each stride of a stride list.

    """

    initial_orientation: Union[np.ndarray, Rotation]
    beta: float
    use_magnetometer: bool
    memory: Optional[Memory]

    def __init__(
        self,
        beta: float = 0.2,
        initial_orientation: Union[np.ndarray, Rotation] = cf(np.array([0, 0, 0, 1.0])),
        use_magnetometer: bool = False,
        memory: Optional[Memory] = None,
    ) -> None:
        self.initial_orientation = initial_orientation
        self.beta = beta
        self.use_magnetometer = use_magnetometer
        self.memory = memory

    def estimate(self, data: SingleSensorData, *, sampling_rate_hz: float, **_) -> Self:
        """Estimate the orientation of the sensor.

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

        memory = self.memory
        if memory is None:
            memory = Memory(None)

        is_single_sensor_data(self.data, frame="sensor", raise_exception=True, check_mag=self.use_magnetometer)
        if isinstance(initial_orientation, Rotation):
            initial_orientation = Rotation.as_quat(initial_orientation)
        initial_orientation = initial_orientation.copy()
        gyro_data = np.deg2rad(data[SF_GYR].to_numpy())
        acc_data = data[SF_ACC].to_numpy()
        if self.use_magnetometer:
            mag_data = data[SF_MAG].to_numpy()
            madgwick_update_series = memory.cache(_madgwick_update_series_mag)
            rots = madgwick_update_series(
                gyro=gyro_data,
                acc=acc_data,
                mag=mag_data,
                initial_orientation=initial_orientation,
                sampling_rate_hz=sampling_rate_hz,
                beta=self.beta,
            )
        else:
            madgwick_update_series = memory.cache(_madgwick_update_series)
            rots = madgwick_update_series(
                gyro=gyro_data,
                acc=acc_data,
                initial_orientation=initial_orientation,
                sampling_rate_hz=sampling_rate_hz,
                beta=self.beta,
            )
        self.orientation_object_ = Rotation.from_quat(rots)
        return self


@njit()
def _madgwick_update(gyro, acc, initial_orientation, sampling_rate_hz, beta):
    q = np.copy(initial_orientation)
    qdot = rate_of_change_from_gyro(gyro, q)

    # Note that we change the order of q components here as we use a different quaternion definition.
    q1, q2, q3, q0 = q

    if beta > 0.0 and not np.all(acc == 0.0):
        acc = acc / np.sqrt(np.sum(acc**2))
        ax, ay, az = acc

        # Auxiliary variables to avoid repeated arithmetic
        _2q0 = 2.0 * q0
        _2q1 = 2.0 * q1
        _2q2 = 2.0 * q2
        _2q3 = 2.0 * q3
        _4q0 = 4.0 * q0
        _4q1 = 4.0 * q1
        _4q2 = 4.0 * q2
        _8q1 = 8.0 * q1
        _8q2 = 8.0 * q2
        q0q0 = q0 * q0
        q1q1 = q1 * q1
        q2q2 = q2 * q2
        q3q3 = q3 * q3

        # Gradient decent algorithm corrective step
        s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay
        s1 = _4q1 * q3q3 - _2q3 * ax + 4.0 * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az
        s2 = 4.0 * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az
        s3 = 4.0 * q1q1 * q3 - _2q1 * ax + 4.0 * q2q2 * q3 - _2q2 * ay

        # Switch the component order back
        s = np.array([s1, s2, s3, s0])
        norm_s = np.sqrt(np.sum(s**2))
        if norm_s != 0.0:
            s /= norm_s

        # Apply feedback step
        qdot -= beta * s

    # Integrate rate of change of quaternion to yield quaternion
    q = q + qdot / sampling_rate_hz
    q /= np.sqrt(np.sum(q**2))

    return q


@njit()
def _madgwick_update_mag(gyro, acc, mag, initial_orientation, sampling_rate_hz, beta):
    q = np.copy(initial_orientation)
    qdot = rate_of_change_from_gyro(gyro, q)

    # Note that we change the order of q components here as we use a different quaternion definition.
    q1, q2, q3, q0 = q

    if beta > 0.0 and not np.all(acc == 0.0):
        acc = acc / np.sqrt(np.sum(acc**2))
        ax, ay, az = acc

        mag = mag / np.sqrt(np.sum(mag**2))
        mx, my, mz = mag

        # Auxiliary variables to avoid repeated arithmetic
        _2q0mx = 2.0 * q0 * mx
        _2q0my = 2.0 * q0 * my
        _2q0mz = 2.0 * q0 * mz
        _2q1mx = 2.0 * q1 * mx
        _2q0 = 2.0 * q0
        _2q1 = 2.0 * q1
        _2q2 = 2.0 * q2
        _2q3 = 2.0 * q3
        _2q0q2 = 2.0 * q0 * q2
        _2q2q3 = 2.0 * q2 * q3
        q0q0 = q0 * q0
        q0q1 = q0 * q1
        q0q2 = q0 * q2
        q0q3 = q0 * q3
        q1q1 = q1 * q1
        q1q2 = q1 * q2
        q1q3 = q1 * q3
        q2q2 = q2 * q2
        q2q3 = q2 * q3
        q3q3 = q3 * q3

        # Gradient decent algorithm corrective step
        hx = mx * q0q0 - _2q0my * q3 + _2q0mz * q2 + mx * q1q1 + _2q1 * my * q2 + _2q1 * mz * q3 - mx * q2q2 - mx * q3q3
        hy = _2q0mx * q3 + my * q0q0 - _2q0mz * q1 + _2q1mx * q2 - my * q1q1 + my * q2q2 + _2q2 * mz * q3 - my * q3q3
        _2bx = np.sqrt(hx * hx + hy * hy)
        _2bz = -_2q0mx * q2 + _2q0my * q1 + mz * q0q0 + _2q1mx * q3 - mz * q1q1 + _2q2 * my * q3 - mz * q2q2 + mz * q3q3
        _4bx = 2.0 * _2bx
        _4bz = 2.0 * _2bz

        s0 = (
            -_2q2 * (2.0 * q1q3 - _2q0q2 - ax)
            + _2q1 * (2.0 * q0q1 + _2q2q3 - ay)
            - _2bz * q2 * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
            + (-_2bx * q3 + _2bz * q1) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
            + _2bx * q2 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz)
        )
        s1 = (
            _2q3 * (2.0 * q1q3 - _2q0q2 - ax)
            + _2q0 * (2.0 * q0q1 + _2q2q3 - ay)
            - 4.0 * q1 * (1 - 2.0 * q1q1 - 2.0 * q2q2 - az)
            + _2bz * q3 * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
            + (_2bx * q2 + _2bz * q0) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
            + (_2bx * q3 - _4bz * q1) * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz)
        )
        s2 = (
            -_2q0 * (2.0 * q1q3 - _2q0q2 - ax)
            + _2q3 * (2.0 * q0q1 + _2q2q3 - ay)
            - 4.0 * q2 * (1 - 2.0 * q1q1 - 2.0 * q2q2 - az)
            + (-_4bx * q2 - _2bz * q0) * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
            + (_2bx * q1 + _2bz * q3) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
            + (_2bx * q0 - _4bz * q2) * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz)
        )
        s3 = (
            _2q1 * (2.0 * q1q3 - _2q0q2 - ax)
            + _2q2 * (2.0 * q0q1 + _2q2q3 - ay)
            + (-_4bx * q3 + _2bz * q1) * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
            + (-_2bx * q0 + _2bz * q2) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
            + _2bx * q1 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz)
        )

        # Switch the component order back
        s = np.array([s1, s2, s3, s0])
        norm_s = np.sqrt(np.sum(s**2))
        if norm_s != 0.0:
            s /= norm_s

        # Apply feedback step
        qdot -= beta * s

    # Integrate rate of change of quaternion to yield quaternion
    q = q + qdot / sampling_rate_hz
    q /= np.sqrt(np.sum(q**2))

    return q


@njit(cache=True)
def _madgwick_update_series(gyro, acc, initial_orientation, sampling_rate_hz, beta):
    out = np.empty((len(gyro) + 1, 4))
    out[0] = initial_orientation
    for i, (gyro_val, acc_val) in enumerate(zip(gyro, acc)):
        initial_orientation = _madgwick_update(
            gyro=gyro_val,
            acc=acc_val,
            initial_orientation=initial_orientation,
            sampling_rate_hz=sampling_rate_hz,
            beta=beta,
        )
        out[i + 1] = initial_orientation

    return out


@njit(cache=True)
def _madgwick_update_series_mag(gyro, acc, mag, initial_orientation, sampling_rate_hz, beta):
    out = np.empty((len(gyro) + 1, 4))
    out[0] = initial_orientation
    for i, (gyro_val, acc_val, mag_val) in enumerate(zip(gyro, acc, mag)):
        initial_orientation = _madgwick_update_mag(
            gyro=gyro_val,
            acc=acc_val,
            mag=mag_val,
            initial_orientation=initial_orientation,
            sampling_rate_hz=sampling_rate_hz,
            beta=beta,
        )
        out[i + 1] = initial_orientation

    return out
