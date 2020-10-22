import numpy as np
import pandas as pd
import pytest

from gaitmap.gait_detection import UllrichGaitSequenceDetection
from gaitmap.utils import coordinate_conversion
from gaitmap.utils.consts import BF_COLS


class TestUllrichGaitSequenceDetection:
    """Test the gait sequence detection by Ullrich."""

    def test_single_sensor_input(self, healthy_example_imu_data, snapshot):
        """Dummy test to see if the algorithm is generally working on the example data"""
        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )

        data = data["left_sensor"]

        gsd = UllrichGaitSequenceDetection()
        gsd = gsd.detect(data, 204.8)

        assert len(gsd.gait_sequences_) == 1

    def test_multi_sensor_input(self, healthy_example_imu_data, snapshot):
        """Dummy test to see if the algorithm is generally working on the example data"""
        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )

        gsd = UllrichGaitSequenceDetection()
        gsd = gsd.detect(data, 204.8)

        assert len(gsd.gait_sequences_["left_sensor"]) == 1
        assert len(gsd.gait_sequences_["right_sensor"]) == 1

    @pytest.mark.parametrize(
        "sensor_channel_config,peak_prominence", (("gyr_ml", 17), ("acc_si", 8), ("acc", 13), ("gyr", 11))
    )
    def test_different_activities_different_configs(
        self, healthy_example_imu_data, sensor_channel_config, peak_prominence, snapshot
    ):
        """Test if the algorithm is generally working with different sensor channel configs and their respective
                optimal peak prominence thresholds."""
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

        gsd = UllrichGaitSequenceDetection(sensor_channel_config=sensor_channel_config, peak_prominence=peak_prominence)
        gsd = gsd.detect(test_data_df, 204.8)

        filename = sensor_channel_config
        if isinstance(filename, (tuple, list)):
            filename = "_".join(filename)
        filename = filename + "_" + str(peak_prominence)
        snapshot.assert_match(gsd.gait_sequences_["left_sensor"], filename, check_dtype=False)

    def test_signal_length_one_window_size(self, healthy_example_imu_data, snapshot):
        """Test to see if the algorithm is working if the signal length equals to one window size"""
        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )

        data = data["left_sensor"]

        sampling_rate_hz = 204.8
        window_size_s = 10
        data = data.iloc[2048 : 2048 + int(window_size_s * sampling_rate_hz)]

        gsd = UllrichGaitSequenceDetection(window_size_s=window_size_s)
        gsd = gsd.detect(data, 204.8)

        assert len(gsd.gait_sequences_) == 1

    def test_on_signal_without_activity(self, snapshot):
        """Test to see if the algorithm is working if the signal contains no activity at all"""

        data_columns = BF_COLS

        # induce rest
        rest_df = pd.DataFrame([[0] * 6], columns=data_columns)
        rest_df = pd.concat([rest_df] * 2048)

        gsd = UllrichGaitSequenceDetection()
        gsd = gsd.detect(rest_df, 204.8)

        assert len(gsd.gait_sequences_) == 0

    def test_on_signal_with_only_nongait(self, snapshot):
        """Test to see if the algorithm is working if the signal contains only non-gait activity"""

        data_columns = BF_COLS

        # induce non-gait cyclic activity
        # create a sine signal to mimic non-gait
        sampling_rate = 204.8
        samples = 2048
        t = np.arange(samples) / sampling_rate
        freq = 1
        test_signal = np.sin(2 * np.pi * freq * t) * 200

        test_signal_reshaped = np.tile(test_signal, (6, 1)).T
        non_gait_df = pd.DataFrame(test_signal_reshaped, columns=data_columns)

        gsd = UllrichGaitSequenceDetection()
        gsd = gsd.detect(non_gait_df, 204.8)

        assert len(gsd.gait_sequences_) == 0

    def test_invalid_sensor_channel_config_type(self, healthy_example_imu_data):
        """Check if ValueError is raised for wrong sensor_channel_config data type."""
        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )
        # use an int instead of str or list
        sensor_channel_config = 1
        with pytest.raises(ValueError, match=r".* must be a str."):
            gsd = UllrichGaitSequenceDetection(sensor_channel_config=sensor_channel_config)
            gsd.detect(data, 204.8)

    @pytest.mark.parametrize("sensor_channel_config", "dummy")
    def test_invalid_sensor_channel_config_value(self, healthy_example_imu_data, sensor_channel_config):
        """Check if ValueError is raised for wrong sensor_channel_config data type."""
        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )

        with pytest.raises(ValueError, match=r".* you have passed is invalid. .*"):
            gsd = UllrichGaitSequenceDetection(sensor_channel_config=sensor_channel_config)
            gsd.detect(data, 204.8)

    def test_invalid_window_size(self, healthy_example_imu_data):
        """Check if ValueError is raised for window size higher than len of signal."""
        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )

        # cut the data to 500 samples
        data = data.iloc[0:500]
        window_size_s = 10
        with pytest.raises(ValueError, match=r".* window size .*"):
            gsd = UllrichGaitSequenceDetection(window_size_s=window_size_s)
            gsd.detect(data, 204.8)

    @pytest.mark.parametrize("locomotion_band", ([1], (0, 1, 2)))
    def test_invalid_locomotion_band_size(self, healthy_example_imu_data, locomotion_band):
        """Check if ValueError is raised for locomotion band with other than two values."""
        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )

        with pytest.raises(ValueError, match=r".* exactly two values."):
            gsd1 = UllrichGaitSequenceDetection(locomotion_band=locomotion_band)
            gsd1.detect(data, 204.8)

    @pytest.mark.parametrize("locomotion_band", ((3, 0.5), (0.5, 0.5)))
    def test_invalid_locomotion_value_order(self, healthy_example_imu_data, locomotion_band):
        """Check if ValueError is raised for locomotion band where second value is smaller or equal than first."""
        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )

        with pytest.raises(ValueError, match=r".* smaller than the second value."):
            gsd = UllrichGaitSequenceDetection(locomotion_band=locomotion_band)
            gsd.detect(data, 204.8)

    def test_invalid_locomotion_upper_value(self, healthy_example_imu_data):
        """Check if ValueError is raised for locomotion band where the upper limit is too close to Nyquist freq."""
        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )

        locomotion_band = (3, 100)
        with pytest.raises(ValueError, match=r".* Nyquist frequency .*"):
            gsd = UllrichGaitSequenceDetection(locomotion_band=locomotion_band)
            gsd.detect(data, 204.8)

    @pytest.mark.parametrize("harmonic_tolerance_hz", (-3, 0))
    def test_invalid_harmonic_tolerance(self, healthy_example_imu_data, harmonic_tolerance_hz):
        """Check if ValueError is raised for harmonic tolerance of being too small Hz."""
        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )

        with pytest.raises(ValueError, match=r"Value for harmonic_tolerance_hz too small. .*"):
            gsd_1 = UllrichGaitSequenceDetection(harmonic_tolerance_hz=harmonic_tolerance_hz)
            gsd_1.detect(data, 204.8)

    def test_invalid_merging_gait_sequences(self, healthy_example_imu_data):
        """Check if data and value for merge_gait_sequences_from_sensors fit to each other. Only gait sequences detected
        from synced data can be merge."""
        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )

        # create dict of dfs to mock up non-synced data
        data_dict = {"left": data["left_sensor"], "right": data["right_sensor"]}
        merge_gait_sequences_from_sensors = True
        with pytest.raises(ValueError, match=r".* synchronized data sets."):
            gsd = UllrichGaitSequenceDetection(merge_gait_sequences_from_sensors=merge_gait_sequences_from_sensors)
            gsd.detect(data_dict, 204.8)

    def test_merging_gait_sequences(self, healthy_example_imu_data):
        """Check if merging of gait sequences works for synchronized data."""

        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )

        # if merging is turned off, result will be a dict
        merge_gait_sequences_from_sensors = False
        gsd = UllrichGaitSequenceDetection(merge_gait_sequences_from_sensors=merge_gait_sequences_from_sensors)
        gsd.detect(data, 204.8)
        assert type(gsd.gait_sequences_) == dict

        # if merging is turned on, result will be a pd.DataFrame
        merge_gait_sequences_from_sensors = True
        gsd = UllrichGaitSequenceDetection(merge_gait_sequences_from_sensors=merge_gait_sequences_from_sensors)
        gsd.detect(data, 204.8)
        assert type(gsd.gait_sequences_) == pd.DataFrame

    def test_merging_for_no_activity(self):
        """Test to see if the merging is working if the signal contains no activity at all"""

        data_columns = BF_COLS

        # induce rest
        rest_df = pd.DataFrame([[0] * 6], columns=data_columns)
        rest_df = pd.concat([rest_df] * 2048)

        dict_of_df = {"left_sensor": rest_df, "right_sensor": rest_df}
        synced_rest_df = pd.concat(dict_of_df, axis=1)

        merge_gait_sequences_from_sensors = True
        gsd = UllrichGaitSequenceDetection(merge_gait_sequences_from_sensors=merge_gait_sequences_from_sensors)
        gsd = gsd.detect(synced_rest_df, 204.8)

        assert type(gsd.gait_sequences_) == pd.DataFrame

    def test_merging_on_signal_with_only_nongait(self):
        """Test to see if the merging is working if the signal contains only non-gait activity"""

        data_columns = BF_COLS

        # induce non-gait cyclic activity
        # create a sine signal to mimic non-gait
        sampling_rate = 204.8
        samples = 2048
        t = np.arange(samples) / sampling_rate
        freq = 1
        test_signal = np.sin(2 * np.pi * freq * t) * 200

        test_signal_reshaped = np.tile(test_signal, (6, 1)).T
        non_gait_df = pd.DataFrame(test_signal_reshaped, columns=data_columns)

        dict_of_df = {"left_sensor": non_gait_df, "right_sensor": non_gait_df}
        synced_non_gait_df = pd.concat(dict_of_df, axis=1)

        merge_gait_sequences_from_sensors = True
        gsd = UllrichGaitSequenceDetection(merge_gait_sequences_from_sensors=merge_gait_sequences_from_sensors)
        gsd = gsd.detect(synced_non_gait_df, 204.8)

        assert type(gsd.gait_sequences_) == pd.DataFrame
        assert gsd.gait_sequences_.empty
