import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gaitmap.utils.consts import SF_COLS, BF_COLS
from gaitmap.utils.coordinate_conversion import convert_left_foot_to_fbf, convert_right_foot_to_fbf, convert_to_fbf


class TestConvertAxes:
    """Test the functions for converting either left or right foot"""

    @pytest.fixture(autouse=True)
    def _sample_sensor_data(self):
        """Create some sample dummy data frames"""
        self.data_left = pd.DataFrame([[1, 2, 3, 4, 5, 6]], columns=SF_COLS)
        self.data_right = pd.DataFrame([[7, 8, 9, 10, 11, 12]], columns=SF_COLS)
        self.data_dict = {"left_sensor": self.data_left, "right_sensor": self.data_right}
        self.data_df = pd.concat(self.data_dict, axis=1)

        # Create expected output
        self.data_left_expected = pd.DataFrame([[1, 2, -3, -4, -5, -6]], columns=BF_COLS)
        self.data_right_expected = pd.DataFrame([[7, -8, -9, 10, -11, 12]], columns=BF_COLS)

        self.data_dict_expected = {"left_sensor": self.data_left_expected, "right_sensor": self.data_right_expected}
        self.data_df_expected = pd.concat(self.data_dict_expected, axis=1)

    def test_convert_left_foot(self):
        data_converted = convert_left_foot_to_fbf(self.data_left)
        assert_frame_equal(data_converted, self.data_left_expected)

    def test_convert_right_foot(self):
        data_converted = convert_right_foot_to_fbf(self.data_right)
        assert_frame_equal(data_converted, self.data_right_expected)

    def test_no_position_arguments(self):
        with pytest.raises(ValueError):
            convert_to_fbf(self.data_df)

    def test_wrong_key_arguments(self):
        with pytest.raises(KeyError):
            convert_to_fbf(self.data_df, left=["abc"])
        with pytest.raises(KeyError):
            convert_to_fbf(self.data_df, right=["abc"])

    def test_rotate_multisensor(self):
        data_converted = convert_to_fbf(self.data_df, left=["left_sensor"], right=["right_sensor"])
        assert_frame_equal(data_converted, self.data_df_expected)

    def test_rotate_multisensor_left(self):
        data_converted = convert_to_fbf(self.data_df, left=["left_sensor"])
        assert_frame_equal(data_converted["left_sensor"], self.data_left_expected)
        assert_frame_equal(data_converted["right_sensor"], self.data_df["right_sensor"])

    def test_rotate_multisensor_right(self):
        data_converted = convert_to_fbf(self.data_df, right=["right_sensor"])
        assert_frame_equal(data_converted["right_sensor"], self.data_right_expected)
        assert_frame_equal(data_converted["left_sensor"], self.data_df["left_sensor"])

    def test_rotate_multisensor_dict(self):
        data_converted = convert_to_fbf(self.data_dict, left=["left_sensor"], right=["right_sensor"])

        for sensor in self.data_dict_expected:
            assert_frame_equal(data_converted[sensor], self.data_dict_expected[sensor])

    def test_like_argument(self):
        data_converted = convert_to_fbf(self.data_df, right_like="right_", left_like="left_")
        assert_frame_equal(data_converted["right_sensor"], self.data_right_expected)

    def test_like_argument_error(self):
        with pytest.raises(ValueError):
            convert_to_fbf(self.data_df, right=["right_sensor"], right_like="right_")

    def test_does_not_match_warning(self):
        with pytest.warns(UserWarning) as w:
            convert_to_fbf(self.data_df, right_like="not_in_any_name_")

        assert "not_in_any_name_" in w[0].message.args[0]

    def test_only_multisensor_dataset_supported(self):
        with pytest.raises(ValueError) as e:
            convert_to_fbf(self.data_df["left_sensor"])

        assert "MultiSensorDataset" in str(e)
