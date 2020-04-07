import numpy as np
import pandas as pd
import pytest
from gaitmap.utils.consts import SF_ACC, SF_COLS
from numpy.testing import assert_almost_equal

import gaitmap.utils.rotations as rotations
from gaitmap.preprocessing.align_to_gravity import align_dataset

from gaitmap.utils.dataset_helper import MultiSensorDataset


class TestAlignToGravity:
    """Test the function `align_to_gravity`."""

    # TODO: add regression test on real dataset

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
            align_dataset(self.sample_sensor_dataset, window_length=3, static_signal_th=0.0, metric="maximum")

    def test_mulit_sensor_dataset_misaligned(self):
        """Test basic alignment using different 180 deg rotations on each dataset."""
        gravity = np.array([0.0, 0.0, 1.0])

        rot = {
            "s1": rotations.rotation_from_angle(np.array([0, 1, 0]), np.deg2rad(180)),
            "s2": rotations.rotation_from_angle(np.array([0, 0, 1]), np.deg2rad(180)),
        }
        miss_aligned_dataset = rotations.rotate_dataset(self.sample_sensor_dataset, rot)

        aligned_dataset = align_dataset(miss_aligned_dataset, window_length=3, static_signal_th=1.0, gravity=gravity)

        assert_almost_equal(aligned_dataset["s1"][SF_ACC].to_numpy(), np.repeat(gravity[None, :], 5, axis=0))
        assert_almost_equal(aligned_dataset["s2"][SF_ACC].to_numpy(), np.repeat(gravity[None, :], 5, axis=0))

    def test_single_sensor_dataset_misaligned(self):
        """Test basic alignment using different 180 deg rotations on single sensor."""
        gravity = np.array([0.0, 0.0, 1.0])

        miss_aligned_data = rotations.rotate_dataset(
            self.sample_sensor_data, rotations.rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(180))
        )

        aligned_data = align_dataset(miss_aligned_data, window_length=3, static_signal_th=1.0, gravity=gravity)

        assert_almost_equal(aligned_data[SF_ACC].to_numpy(), np.repeat(gravity[None, :], 5, axis=0))
