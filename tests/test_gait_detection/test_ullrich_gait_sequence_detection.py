from gaitmap.gait_detection import UllrichGaitSequenceDetection
from gaitmap.utils import coordinate_conversion


class TestUllrichGaitSequenceDetection:
    """Test the gait sequence detection by Ullrich."""

    def test_multi_sensor_input(self, healthy_example_imu_data, snapshot):
        """Dummy test to see if the algorithm is generally working on the example data"""
        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )

        gsd = UllrichGaitSequenceDetection()
        gsd.detect(data, 204.8)

        # snapshot.assert_match(ed.stride_events_["left_sensor"], "left", check_dtype=False)
        # snapshot.assert_match(ed.stride_events_["right_sensor"], "right", check_dtype=False)
