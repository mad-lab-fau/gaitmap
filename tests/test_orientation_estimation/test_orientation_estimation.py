import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from gaitmap.utils.consts import SF_GYR, SF_ACC
from gaitmap.trajectory_reconstruction.orientation_estimation import GyroIntegration


class TestGyroIntegration:

    def test_180_x(self):
        sensor_data = pd.DataFrame(columns=SF_GYR)
        fs = 100
        # 180 degree rotation around first axis
        sensor_data[SF_GYR[0]] = [0] * fs
        sensor_data[SF_GYR[1]] = [np.pi] * fs
        sensor_data[SF_GYR[2]] = [0] * fs
        sensor_data[SF_ACC[0]] = [0] * fs
        sensor_data[SF_ACC[1]] = [1] * fs
        sensor_data[SF_ACC[2]] = [0] * fs

        gyr_integrator = GyroIntegration(Rotation([0, 0, 1, 0]))
        gyr_integrator.estimate_orientation_sequence(sensor_data, fs)
        print('\n')
        print('Result:')
        orientations = gyr_integrator.estimated_orientations_
        for i_sample in orientations:
            print(i_sample.as_quat())

        #np.testing.assert_array_almost_equal(gyr_integrator.estimated_orientations_[-1], )