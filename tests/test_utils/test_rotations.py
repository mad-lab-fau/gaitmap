import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from pandas._testing import assert_frame_equal

from gaitmap.utils.rotations import rotation_from_angle, _rotate_sensor, rotate_dataset
from gaitmap.utils.consts import SF_COLS, SF_ACC, SF_GYR


@pytest.fixture()
def cyclic_rotation():
    """Rotation that turns x to y, y to z, and z to x."""
    return rotation_from_angle(np.array([0, 0, 1.0]), np.pi / 2) * rotation_from_angle(np.array([1, 0, 0.0]), np.pi / 2)


class TestRotationFromAngle:
    def test_single_angle(self):
        """Test single axis, single angle."""
        assert_almost_equal(rotation_from_angle(np.array([1, 0, 0]), np.pi).as_quat(), [1.0, 0, 0, 0])

    def test_multiple_axis_and_angles(self):
        """Test multiple axis, multiple angles."""
        start = np.repeat(np.array([1.0, 0, 0])[None, :], 5, axis=0)
        goal = np.repeat(np.array([1.0, 0, 0, 0])[None, :], 5, axis=0)
        angle = np.array([np.pi] * 5)
        assert_almost_equal(rotation_from_angle(start, angle).as_quat(), goal)

    def test_multiple_axis_single_angle(self):
        """Test multiple axis, single angles."""
        start = np.repeat(np.array([1.0, 0, 0])[None, :], 5, axis=0)
        goal = np.repeat(np.array([1.0, 0, 0, 0])[None, :], 5, axis=0)
        angle = np.array(np.pi)
        assert_almost_equal(rotation_from_angle(start, angle).as_quat(), goal)

    def test_single_axis_multiple_angle(self):
        """Test single axis, multiple angles."""
        start = np.array([1.0, 0, 0])[None, :]
        goal = np.repeat(np.array([1.0, 0, 0, 0])[None, :], 5, axis=0)
        angle = np.array([np.pi] * 5)
        assert_almost_equal(rotation_from_angle(start, angle).as_quat(), goal)


class TestRotateDataset:
    sample_sensor_data: pd.DataFrame
    sample_sensor_dataset: pd.DataFrame

    def _compare_cyclic(self, data, rotated_data, cycles=1):
        # The cyclic rotation should be equivalent to just shifting columns
        shifted_acc_cols = SF_ACC[-cycles:] + SF_ACC[:-cycles]
        shifted_gyr_cols = SF_GYR[-cycles:] + SF_GYR[:-cycles]

        assert_almost_equal(rotated_data[SF_ACC].to_numpy(), data[shifted_acc_cols].to_numpy())
        assert_almost_equal(rotated_data[SF_GYR].to_numpy(), data[shifted_gyr_cols].to_numpy())

    @pytest.fixture(autouse=True)
    def _sample_sensor_data(self):
        acc = [1.0, 2.0, 3.0]
        gyr = [4.0, 5.0, 6.0]
        all_data = np.repeat(np.array([*acc, *gyr])[None, :], 3, axis=0)
        self.sample_sensor_data = pd.DataFrame(all_data, columns=SF_COLS)
        self.sample_sensor_dataset = pd.concat(
            [self.sample_sensor_data, self.sample_sensor_data + 0.5], keys=["s1", "s2"], axis=1
        )

    def test_rotate_sensor(self, cyclic_rotation):
        """Test if rotation is correctly applied to gyr and acc."""
        rotated_data = _rotate_sensor(self.sample_sensor_data, cyclic_rotation)

        self._compare_cyclic(self.sample_sensor_data, rotated_data)

    @pytest.mark.parametrize("inplace,equal", ((None, False), (False, False), (True, True)))
    def test_rotate_sensor_inplace(self, inplace, equal, cyclic_rotation):
        if inplace is None:
            # Test default
            rotated_data = _rotate_sensor(self.sample_sensor_data, cyclic_rotation)
        else:
            rotated_data = _rotate_sensor(self.sample_sensor_data, cyclic_rotation, inplace=inplace)

        if equal is True:
            assert rotated_data is self.sample_sensor_data
        else:
            assert rotated_data is not self.sample_sensor_data

    def test_rotate_dataset_single(self, cyclic_rotation):
        rotated_data = rotate_dataset(self.sample_sensor_data, cyclic_rotation)

        self._compare_cyclic(self.sample_sensor_data, rotated_data)

    def test_rotate_single_named_dataset(self, cyclic_rotation):
        test_data = self.sample_sensor_dataset[["s1"]]
        rotated_data = rotate_dataset(test_data, cyclic_rotation)

        self._compare_cyclic(test_data["s1"], rotated_data["s1"])

    def test_rotate_multiple_named_dataset(self, cyclic_rotation):
        test_data = self.sample_sensor_dataset
        rotated_data = rotate_dataset(test_data, cyclic_rotation)

        self._compare_cyclic(test_data["s1"], rotated_data["s1"])
        self._compare_cyclic(test_data["s2"], rotated_data["s2"])

    def test_rotate_multiple_named_dataset_with_multiple_rotations(self, cyclic_rotation):
        test_data = self.sample_sensor_dataset
        rotated_data = rotate_dataset(test_data, {"s1": cyclic_rotation, "s2": cyclic_rotation * cyclic_rotation})

        self._compare_cyclic(test_data["s1"], rotated_data["s1"])
        self._compare_cyclic(test_data["s2"], rotated_data["s2"], cycles=2)

    def test_only_rotate_some_sensors(self, cyclic_rotation):
        test_data = self.sample_sensor_dataset
        rotated_data = rotate_dataset(test_data, {"s1": cyclic_rotation})

        self._compare_cyclic(test_data["s1"], rotated_data["s1"])
        assert_frame_equal(test_data["s2"], rotated_data["s2"])

    def test_rotate_dataset_is_copy(self, cyclic_rotation):
        rotated_data = rotate_dataset(self.sample_sensor_dataset, cyclic_rotation)

        assert rotated_data is not self.sample_sensor_dataset

    @pytest.mark.parametrize("ascending", (True, False))
    def test_order_is_preserved(self, cyclic_rotation, ascending):
        changed_sorted_data = self.sample_sensor_dataset.sort_index(ascending=ascending, axis=1)
        rotated_data = rotate_dataset(changed_sorted_data, cyclic_rotation)

        assert_frame_equal(changed_sorted_data.columns.to_frame(), rotated_data.columns.to_frame())

    @pytest.mark.parametrize("inputs", ({"dataset": "single", "rotation": {}},))
    def test_invalid_inputs(self, inputs):
        if inputs["dataset"] == "single":
            inputs["dataset"] = self.sample_sensor_data

        with pytest.raises(ValueError):
            rotate_dataset(**inputs)
