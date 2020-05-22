import numpy as np
import pandas as pd
import pytest
from numpy.linalg import norm
from numpy.testing import assert_almost_equal

from gaitmap.preprocessing.sensor_alignment import align_dataset_to_gravity, align_heading_of_sensors
from gaitmap.utils.consts import SF_ACC, SF_COLS
from gaitmap.utils.dataset_helper import MultiSensorDataset
from gaitmap.utils.rotations import rotation_from_angle, rotate_dataset


class TestAlignToGravity:
    """Test the function `align_to_gravity`."""

    sample_sensor_data: pd.DataFrame
    sample_sensor_dataset: MultiSensorDataset

    @pytest.fixture(autouse=True, params=("dict", "frame"))
    def _sample_sensor_data(self, request):
        """Create some sample data.

        This data is recreated before each test (using pytest.fixture).
        """
        acc = [0.0, 0.0, 1.0]
        gyr = [0.11, 0.12, 0.13]
        all_data = np.repeat(np.array([*acc, *gyr])[None, :], 5, axis=0)
        self.sample_sensor_data = pd.DataFrame(all_data, columns=SF_COLS)
        dataset = {"s1": self.sample_sensor_data, "s2": self.sample_sensor_data}
        if request.param == "dict":
            self.sample_sensor_dataset = dataset
        elif request.param == "frame":
            self.sample_sensor_dataset = pd.concat(dataset, axis=1)

    def test_no_static_moments_in_dataset(self):
        """Test if value error is raised correctly if no static window can be found on dataset with given user
        settings."""
        with pytest.raises(ValueError, match=r".*No static windows .*"):
            align_dataset_to_gravity(
                self.sample_sensor_dataset,
                sampling_rate_hz=1,
                window_length_s=3,
                static_signal_th=0.0,
                metric="maximum",
            )

    def test_mulit_sensor_dataset_misaligned(self):
        """Test basic alignment using different 180 deg rotations on each dataset."""
        gravity = np.array([0.0, 0.0, 1.0])

        rot = {
            "s1": rotation_from_angle(np.array([0, 1, 0]), np.deg2rad(180)),
            "s2": rotation_from_angle(np.array([0, 0, 1]), np.deg2rad(180)),
        }
        miss_aligned_dataset = rotate_dataset(self.sample_sensor_dataset, rot)

        aligned_dataset = align_dataset_to_gravity(
            miss_aligned_dataset, sampling_rate_hz=1, window_length_s=3, static_signal_th=1.0, gravity=gravity
        )

        assert_almost_equal(aligned_dataset["s1"][SF_ACC].to_numpy(), np.repeat(gravity[None, :], 5, axis=0))
        assert_almost_equal(aligned_dataset["s2"][SF_ACC].to_numpy(), np.repeat(gravity[None, :], 5, axis=0))

    def test_single_sensor_dataset_misaligned(self):
        """Test basic alignment using different 180 deg rotations on single sensor."""
        gravity = np.array([0.0, 0.0, 1.0])

        miss_aligned_data = rotate_dataset(
            self.sample_sensor_data, rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(180))
        )

        aligned_data = align_dataset_to_gravity(
            miss_aligned_data, sampling_rate_hz=1, window_length_s=3, static_signal_th=1.0, gravity=gravity
        )

        assert_almost_equal(aligned_data[SF_ACC].to_numpy(), np.repeat(gravity[None, :], 5, axis=0))


class TestXYAlignment:
    @pytest.mark.parametrize("angle", [90, 180.0, 22.0, 45.0, -90, -45])
    def test_xy_alignment_simple(self, angle):
        signal = np.random.normal(scale=1000, size=(500, 3))
        rot_signal = rotation_from_angle(np.array([0, 0, 1]), np.deg2rad(angle)).apply(signal)
        rot = align_heading_of_sensors(signal, rot_signal)
        rotvec = rot.as_rotvec()

        assert_almost_equal(np.rad2deg(norm(rotvec)) * np.sign(rotvec[-1]), -angle)
        assert_almost_equal(np.abs(rotvec / norm(rotvec) @ [0, 0, 1]), 1)

        assert_almost_equal(rot.apply(rot_signal), signal)

    def test_xy_alignment_dummy(self,):
        signal = np.random.normal(scale=1000, size=(500, 3))
        rot_signal = rotation_from_angle(np.array([0, 0, 1]), 0).apply(signal)
        rot = align_heading_of_sensors(signal, rot_signal)
        rotvec = rot.as_rotvec()

        assert_almost_equal(rotvec, [0, 0, 0])

    @pytest.mark.parametrize("angle", [90, 180.0, 22.0, 45.0, -90, -45])
    def test_xy_alignment_with_noise(self, angle):
        signal = np.random.normal(scale=1000, size=(500, 3))
        rot_signal = rotation_from_angle(np.array([0, 0, 1]), np.deg2rad(angle)).apply(signal)

        noise = np.random.normal(scale=5, size=(500, 3))

        rot = align_heading_of_sensors(signal, rot_signal + noise)
        rotvec = rot.as_rotvec()

        angle_error = ((np.rad2deg(norm(rotvec)) * np.sign(rotvec[-1]) - -angle) + 180) % 360 - 180
        # Allow error of 2 deg
        assert np.abs(angle_error) < 2.0
        assert_almost_equal(np.abs(rotvec / norm(rotvec) @ [0, 0, 1]), 1, 3)
