"""Implementation of the MahonyAHRS."""

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


class MahonyAHRS(BaseOrientationMethod):
    """The MahonyAHRS algorithm to estimate the orientation of an IMU.

    This method applies gyroscope integration with proportional-integral feedback based on the estimated gravity
    direction from the accelerometer data.
    If requested, the magnetometer can be used as an additional heading reference.
    This implementation is based on the paper [1]_ and the open source x-io reference implementation [2]_.

    Parameters
    ----------
    kp
        Proportional gain of the feedback term.
        Higher values make the algorithm react faster to accelerometer or magnetometer based corrections.
        A value of 0 disables the proportional correction.
    ki
        Integral gain of the feedback term.
        This term can compensate constant gyro bias over time, but can also introduce drift or overshoot if chosen too
        large.
        A value of 0 disables the integral correction.
    initial_orientation
        The initial orientation of the sensor that is assumed.
        It is critical that this value is close to the actual orientation.
        Otherwise, the estimated orientation will drift until the real orientation is found.
        If you pass an array, the element order must be x, y, z, w.
    use_magnetometer
        If the magnetometer data should be used to correct the orientation.
        This requires magnetometer columns to be present in the input data.
        If True, the Mahony AHRS variant with magnetic heading correction will be used.
    memory
        An optional `joblib.Memory` object that can be provided to cache the calls to the Mahony update series.

    Attributes
    ----------
    orientation_
        The rotations as a *SingleSensorOrientationList*, including the initial orientation.
        This means that there are ``len(data) + 1`` orientations.
    orientation_object_
        The orientations as a single scipy Rotation object.
    rotated_data_
        The rotated data after applying the estimated orientation to the data.
        The first sample of the data remains unrotated and only reflects the initial orientation.

    Other Parameters
    ----------------
    data
        The data passed to :meth:`estimate`.
        It must be a single-sensor dataframe with accelerometer and gyroscope columns in the sensor frame.
        If ``use_magnetometer=True``, magnetometer columns are required as well.
    sampling_rate_hz
        The sampling rate of ``data`` in Hz.

    Notes
    -----
    This class uses *Numba* as a just-in-time compiler to achieve fast run times.
    As a result, the first execution of the algorithm will take longer because the methods need to be compiled first.

    .. [1] Mahony, R., Hamel, T., & Pflimlin, J.-M. (2008).
           Nonlinear complementary filters on the special orthogonal group.
           IEEE Transactions on Automatic Control, 53(5), 1203-1218.
           https://doi.org/10.1109/TAC.2008.923738
    .. [2] https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/

    Examples
    --------
    Your data must be a pd.DataFrame with columns defined by :obj:`~gaitmap.utils.consts.SF_COLS` or
    :obj:`~gaitmap.utils.consts.SF_COLS_WITH_MAG`.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from gaitmap.utils.consts import SF_COLS
    >>> data = pd.DataFrame(..., columns=SF_COLS)
    >>> sampling_rate_hz = 100
    >>> mahony = MahonyAHRS(kp=0.8, ki=0.05, initial_orientation=np.array([0, 0, 0, 1.0]))
    >>> mahony = mahony.estimate(data, sampling_rate_hz=sampling_rate_hz)
    >>> mahony.orientation_
    <pd.DataFrame with resulting quaternions>
    >>> mahony.orientation_object_
    <scipy.Rotation object>

    See Also
    --------
    gaitmap.trajectory_reconstruction: Other implemented algorithms for orientation and position estimation
    gaitmap.trajectory_reconstruction.StrideLevelTrajectory: Apply the method for each stride of a stride list.

    """

    initial_orientation: Union[np.ndarray, Rotation]
    kp: float
    ki: float
    use_magnetometer: bool
    memory: Optional[Memory]

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        initial_orientation: Union[np.ndarray, Rotation] = cf(np.array([0, 0, 0, 1.0])),
        use_magnetometer: bool = False,
        memory: Optional[Memory] = None,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.initial_orientation = initial_orientation
        self.use_magnetometer = use_magnetometer
        self.memory = memory

    def estimate(self, data: SingleSensorData, *, sampling_rate_hz: float, **_) -> Self:
        """Estimate the orientation of the sensor.

        Parameters
        ----------
        data
            Continuous sensor data including gyro and acc values.
            If ``use_magnetometer=True``, magnetometer values must be present as well.
            The gyroscope data is expected to be in deg/s.
        sampling_rate_hz
            The sampling rate of the data in Hz.

        Returns
        -------
        self
            The class instance with all result attributes populated.

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
            mahony_update_series = memory.cache(_mahony_update_series_mag)
            rots = mahony_update_series(
                gyro=gyro_data,
                acc=acc_data,
                mag=data[SF_MAG].to_numpy(),
                initial_orientation=initial_orientation,
                sampling_rate_hz=sampling_rate_hz,
                kp=self.kp,
                ki=self.ki,
            )
        else:
            mahony_update_series = memory.cache(_mahony_update_series)
            rots = mahony_update_series(
                gyro=gyro_data,
                acc=acc_data,
                initial_orientation=initial_orientation,
                sampling_rate_hz=sampling_rate_hz,
                kp=self.kp,
                ki=self.ki,
            )
        self.orientation_object_ = Rotation.from_quat(rots)
        return self


@njit()
def _mahony_update(gyro, acc, initial_orientation, sampling_rate_hz, kp, ki, integral_error):
    q = np.copy(initial_orientation)
    integral_error = np.copy(integral_error)
    corrected_gyro = np.copy(gyro)

    if not np.all(acc == 0.0):
        acc = acc / np.sqrt(np.sum(acc**2))
        ax, ay, az = acc

        qx, qy, qz, qw = q

        # Estimated direction of gravity from the current orientation.
        vx = 2.0 * (qx * qz - qw * qy)
        vy = 2.0 * (qw * qx + qy * qz)
        vz = qw * qw - qx * qx - qy * qy + qz * qz

        error = np.array([ay * vz - az * vy, az * vx - ax * vz, ax * vy - ay * vx])

        if ki > 0.0:
            integral_error += error
        else:
            integral_error[:] = 0.0

        corrected_gyro += kp * error

    elif ki <= 0.0:
        integral_error[:] = 0.0

    if ki > 0.0:
        corrected_gyro += ki * integral_error

    qdot = rate_of_change_from_gyro(corrected_gyro, q)

    q = q + qdot / sampling_rate_hz
    q /= np.sqrt(np.sum(q**2))

    return q, integral_error


@njit()
def _mahony_update_mag(gyro, acc, mag, initial_orientation, sampling_rate_hz, kp, ki, integral_error):
    if np.all(mag == 0.0):
        return _mahony_update(gyro, acc, initial_orientation, sampling_rate_hz, kp, ki, integral_error)

    q = np.copy(initial_orientation)
    integral_error = np.copy(integral_error)
    corrected_gyro = np.copy(gyro)

    if not np.all(acc == 0.0):
        acc = acc / np.sqrt(np.sum(acc**2))
        ax, ay, az = acc

        mag = mag / np.sqrt(np.sum(mag**2))
        mx, my, mz = mag

        qx, qy, qz, qw = q
        qxqx = qx * qx
        qxqy = qx * qy
        qxqz = qx * qz
        qxqw = qx * qw
        qyqy = qy * qy
        qyqz = qy * qz
        qyqw = qy * qw
        qzqz = qz * qz
        qzqw = qz * qw

        hx = 2.0 * mx * (0.5 - qyqy - qzqz) + 2.0 * my * (qxqy - qzqw) + 2.0 * mz * (qxqz + qyqw)
        hy = 2.0 * mx * (qxqy + qzqw) + 2.0 * my * (0.5 - qxqx - qzqz) + 2.0 * mz * (qyqz - qxqw)
        bx = np.sqrt(hx * hx + hy * hy)
        bz = 2.0 * mx * (qxqz - qyqw) + 2.0 * my * (qyqz + qxqw) + 2.0 * mz * (0.5 - qxqx - qyqy)

        # Estimated directions of gravity and the magnetic field from the current orientation.
        vx = 2.0 * (qx * qz - qw * qy)
        vy = 2.0 * (qw * qx + qy * qz)
        vz = qw * qw - qx * qx - qy * qy + qz * qz
        wx = 2.0 * bx * (0.5 - qyqy - qzqz) + 2.0 * bz * (qxqz - qyqw)
        wy = 2.0 * bx * (qxqy - qzqw) + 2.0 * bz * (qxqw + qyqz)
        wz = 2.0 * bx * (qyqw + qxqz) + 2.0 * bz * (0.5 - qxqx - qyqy)

        error = np.array(
            [
                (ay * vz - az * vy) + (my * wz - mz * wy),
                (az * vx - ax * vz) + (mz * wx - mx * wz),
                (ax * vy - ay * vx) + (mx * wy - my * wx),
            ]
        )

        if ki > 0.0:
            integral_error += error
        else:
            integral_error[:] = 0.0

        corrected_gyro += kp * error

    elif ki <= 0.0:
        integral_error[:] = 0.0

    if ki > 0.0:
        corrected_gyro += ki * integral_error

    qdot = rate_of_change_from_gyro(corrected_gyro, q)

    q = q + qdot / sampling_rate_hz
    q /= np.sqrt(np.sum(q**2))

    return q, integral_error


@njit(cache=True)
def _mahony_update_series(gyro, acc, initial_orientation, sampling_rate_hz, kp, ki):
    out = np.empty((len(gyro) + 1, 4))
    out[0] = initial_orientation
    integral_error = np.zeros(3)
    for i, (gyro_val, acc_val) in enumerate(zip(gyro, acc)):
        initial_orientation, integral_error = _mahony_update(
            gyro=gyro_val,
            acc=acc_val,
            initial_orientation=initial_orientation,
            sampling_rate_hz=sampling_rate_hz,
            kp=kp,
            ki=ki,
            integral_error=integral_error,
        )
        out[i + 1] = initial_orientation

    return out


@njit(cache=True)
def _mahony_update_series_mag(gyro, acc, mag, initial_orientation, sampling_rate_hz, kp, ki):
    out = np.empty((len(gyro) + 1, 4))
    out[0] = initial_orientation
    integral_error = np.zeros(3)
    for i, (gyro_val, acc_val, mag_val) in enumerate(zip(gyro, acc, mag)):
        initial_orientation, integral_error = _mahony_update_mag(
            gyro=gyro_val,
            acc=acc_val,
            mag=mag_val,
            initial_orientation=initial_orientation,
            sampling_rate_hz=sampling_rate_hz,
            kp=kp,
            ki=ki,
            integral_error=integral_error,
        )
        out[i + 1] = initial_orientation

    return out
