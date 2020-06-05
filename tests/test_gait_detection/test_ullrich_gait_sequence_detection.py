from gaitmap.gait_detection import UllrichGaitSequenceDetection
from gaitmap.utils import coordinate_conversion
import pandas as pd
import numpy as np


class TestUllrichGaitSequenceDetection:
    """Test the gait sequence detection by Ullrich."""

    def test_multi_sensor_input(self, healthy_example_imu_data, snapshot):
        """Dummy test to see if the algorithm is generally working on the example data"""
        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )

        gsd = UllrichGaitSequenceDetection()
        gsd = gsd.detect(data, 204.8)

        return None

    def test_different_activities(self, healthy_example_imu_data, snapshot):
        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )

        # induce rest
        rest_df = pd.DataFrame([[0] * data.shape[1]], columns=data.columns)
        rest_df = pd.concat([rest_df] * 2048)

        # induce non-gait cyclic activity
        # create a sine signal to mimic non-gait
        sampling_rate = 204.8
        samples = 2048
        t = np.arange(samples) / sampling_rate
        freq = 1
        test_signal = np.sin(2 * np.pi * freq * t) * 200

        test_signal_reshaped = np.tile(test_signal, (data.shape[1], 1)).T
        non_gait_df = pd.DataFrame(test_signal_reshaped, columns=data.columns)

        test_data_df = pd.concat([rest_df, data, non_gait_df, data, rest_df], ignore_index=True)

        gsd = UllrichGaitSequenceDetection()
        gsd = gsd.detect(test_data_df, 204.8)

        return None
