import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from gaitmap.utils.consts import SF_GYR, SF_ACC
from gaitmap.trajectory_reconstruction.orientation_estimation import GyroIntegration


class TestGyroIntegration:
    def test_180_x(self):
        sensor_data = pd.DataFrame(columns=SF_GYR)
        fs = 100

        # 180 degree rotation around y-axis
        sensor_data[SF_GYR[0]] = [0] * fs
        sensor_data[SF_GYR[1]] = [np.pi] * fs
        sensor_data[SF_GYR[2]] = [0] * fs
        sensor_data[SF_ACC[0]] = [0] * fs
        sensor_data[SF_ACC[1]] = [1] * fs
        sensor_data[SF_ACC[2]] = [0] * fs

        # create initial "spatial quaternion" that describes a vector along z-axis
        gyr_integrator = GyroIntegration(Rotation([0, 0, 1, 0]))

        gyr_integrator.estimate_orientation_sequence(sensor_data, fs)
        orientations = gyr_integrator.estimated_orientations_
        # for i_sample in orientations:
        # print(i_sample.as_quat())

        # rotating around x-axis by 180 degree should result in a vector along negative z
        np.testing.assert_array_almost_equal(gyr_integrator.estimated_orientations_[-1].as_quat(), [0, 0, -1, 0])
