"""Naive Integration of Gyroscope to estimate the orientation."""
from typing import Optional, Union

import numpy as np
from joblib import Memory
from numba import njit
from scipy.spatial.transform import Rotation
from tpcp import cf
from typing_extensions import Self

from gaitmap.base import BaseOrientationMethod
from gaitmap.utils.consts import SF_GYR
from gaitmap.utils.datatype_helper import SingleSensorData, is_single_sensor_data
from gaitmap.utils.fast_quaternion_math import rate_of_change_from_gyro


class SimpleGyroIntegration(BaseOrientationMethod):
    """Integrate Gyro values without any drift correction.

    Parameters
    ----------
    initial_orientation : Rotation or (1 x 4) quaternion array
        The initial orientation used.
        If you pass a array, remember that the order of elements must be x, y, z, w.
    memory
        An optional `joblib.Memory` object that can be provided to cache the results of the integration.

    Attributes
    ----------
    orientation_
        The rotations as a *SingleSensorOrientationList*, including the initial orientation.
        This means the there are len(data) + 1 orientations.
    orientation_object_
        The orientations as a single scipy Rotation object

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

    Examples
    --------
    Your data must be a pd.DataFrame with at least columns defined by :obj:`~gaitmap.utils.consts.SF_GYR`.

    >>> import pandas as pd
    >>> from gaitmap.utils.consts import SF_GYR
    >>> data = pd.DataFrame(..., columns=SF_GYR)
    >>> sampling_rate_hz = 100
    >>> # Create an algorithm instance
    >>> sgi = SimpleGyroIntegration(initial_orientation=np.array([0, 0, 0, 1.0]))
    >>> # Apply the algorithm
    >>> sgi = sgi.estimate(data, sampling_rate_hz=sampling_rate_hz)
    >>> # Inspect the results
    >>> sgi.orientation_
    <pd.Dataframe with resulting quaternions>
    >>> sgi.orientation_object_
    <scipy.Rotation object>

    See Also
    --------
    gaitmap.trajectory_reconstruction: Other implemented algorithms for orientation and position estimation
    gaitmap.trajectory_reconstruction.StrideLevelTrajectory: Apply the method for each stride of a stride list.

    """

    initial_orientation: Union[np.ndarray, Rotation]
    memory: Optional[Memory]

    orientation_: Rotation

    data: SingleSensorData
    sampling_rate_hz: float

    def __init__(
        self,
        initial_orientation: Union[np.ndarray, Rotation] = cf(np.array([0, 0, 0, 1.0])),
        memory: Optional[Memory] = None,
    ):
        self.initial_orientation = initial_orientation
        self.memory = memory

    def estimate(self, data: SingleSensorData, sampling_rate_hz: float) -> Self:
        """Estimate the orientation of the sensor by simple integration of the Gyro data.

        Parameters
        ----------
        data
            Continuous sensor data that includes at least a Gyro.
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

        is_single_sensor_data(self.data, check_acc=False, frame="sensor", raise_exception=True)
        if isinstance(initial_orientation, Rotation):
            initial_orientation = Rotation.as_quat(initial_orientation)
        initial_orientation = initial_orientation.copy()
        gyro_data = np.deg2rad(data[SF_GYR].to_numpy())
        simple_gyro_integration_series = memory.cache(_simple_gyro_integration_series)

        rots = simple_gyro_integration_series(
            gyro=gyro_data, initial_orientation=initial_orientation, sampling_rate_hz=sampling_rate_hz
        )
        self.orientation_object_ = Rotation.from_quat(rots)

        return self


@njit(cache=True)
def _simple_gyro_integration_series(gyro, initial_orientation, sampling_rate_hz):
    out = np.empty((len(gyro) + 1, 4))
    q = initial_orientation
    out[0] = q
    for i in range(len(gyro)):  # noqa: consider-using-enumerate
        qdot = rate_of_change_from_gyro(gyro[i], q)
        # Integrate rate of change of quaternion to yield quaternion
        q += qdot / sampling_rate_hz
        q /= np.sqrt(np.sum(q**2))
        out[i + 1] = q

    return out
