import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from gaitmap.utils.consts import SF_GYR, SF_ACC
from gaitmap.trajectory_reconstruction.orientation_estimation import GyroIntegration

# TODO: @to08kece @Arne, add metatest once DTW is merged


class TestGyroIntegration:
    @pytest.mark.parametrize(
        "axis_to_rotate, vector_to_rotate, expected_result",
        ((0, [0, 0, 1], [0, 0, -1]), (1, [0, 0, 1], [0, 0, -1]), (2, [1, 0, 0], [-1, 0, 0])),
    )
    def test_180_x(self, axis_to_rotate: int, vector_to_rotate: list, expected_result: list):
        """Rotate by 180 degree around one axis and check resulting rotation by transforming a 3D vector with start
        and final rotation.

        Parameters
        ----------
        axis_to_rotate
            the axis around which should be rotated

        vector_to_rotate
            test vector that will be transformed using the initial orientation and the updated/final orientation

        expected_result
            the result that is to be expected

        """
        sensor_data = pd.DataFrame(columns=SF_GYR)
        fs = 100

        # 180 degree rotation around i_axis
        for i_axis in SF_GYR:
            if i_axis == SF_GYR[axis_to_rotate]:
                sensor_data[i_axis] = [np.pi] * fs
            else:
                sensor_data[i_axis] = [0] * fs

        # create initial "spatial quaternion" that describes a vector along z-axis
        if axis_to_rotate == 0 or axis_to_rotate == 1:
            rot_init = Rotation([0, 0, 1, 0])
        else:
            rot_init = Rotation([1, 0, 0, 0])

        gyr_integrator = GyroIntegration(rot_init)
        gyr_integrator.estimate(sensor_data, fs)
        rot_final = gyr_integrator.estimated_orientations_[-1]
        np.testing.assert_array_almost_equal(rot_final.apply(vector_to_rotate), expected_result)
