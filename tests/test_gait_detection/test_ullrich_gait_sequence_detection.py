from gaitmap.gait_detection import UllrichGaitSequenceDetection
from gaitmap.utils import coordinate_conversion


class TestUllrichGaitSequenceDetection:
    """Test the gait sequence detection by Ullrich."""

    def test_multi_sensor_input(self, healthy_example_imu_data, snapshot):
        """Dummy test to see if the algorithm is generally working on the example data"""
        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )

        data.iloc[:2048] = 0
        data.iloc[-3072:] = 0

        gsd = UllrichGaitSequenceDetection(peak_prominence=17)
        gsd.detect(data, 204.8)

        return None
