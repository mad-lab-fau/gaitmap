import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.spatial.transform import Rotation
from gaitmap.preprocessing.align_to_gravity import align_dataset


class TestAlignToGravity:
    """Test the function `align_to_gravity`."""

    @pytest.fixture(autouse=True)
    def _sample_sensor_data(self):
        """Create some sample data.

        This data is recreated before each test (using pytest.fixture).
        """
        acc = [1.0, 2.0, 3.0]
        gyr = [4.0, 5.0, 6.0]
        all_data = np.repeat(np.array([*acc, *gyr])[None, :], 3, axis=0)
        self.sample_sensor_data = pd.DataFrame(all_data, columns=SF_COLS)
        self.sample_sensor_dataset = pd.concat(
            [self.sample_sensor_data, self.sample_sensor_data + 0.5], keys=["s1", "s2"], axis=1
        )

    def test_align_to_gravity(self):
        """Test some basic functionality."""
        test_data = np.array([1, 2, 3])
        assert True
