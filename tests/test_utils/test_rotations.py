import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pandas._testing import assert_frame_equal

from gaitmap.utils.dataset_helper import Dataset, MultiSensorDataset, get_multi_sensor_dataset_names
from gaitmap.utils.rotations import (
    rotation_from_angle,
    _rotate_sensor,
    rotate_dataset,
    find_shortest_rotation,
    get_gravity_rotation,
)
from gaitmap.utils.consts import SF_COLS, SF_ACC, SF_GYR


@pytest.fixture()
def cyclic_rotation():
    """Rotation that turns x to y, y to z, and z to x."""
    return rotation_from_angle(np.array([0, 0, 1.0]), np.pi / 2) * rotation_from_angle(np.array([1, 0, 0.0]), np.pi / 2)


class TestRotationFromAngle:
    """Test the function `rotation_from_angle`."""

    def test_single_angle(self):
        """Test single axis, single angle."""
        assert_almost_equal(rotation_from_angle(np.array([1, 0, 0]), np.pi).as_quat(), [1.0, 0, 0, 0])

    def test_multiple_axis_and_angles(self):
        """Test multiple axes, multiple angles."""
        start = np.repeat(np.array([1.0, 0, 0])[None, :], 5, axis=0)
        goal = np.repeat(np.array([1.0, 0, 0, 0])[None, :], 5, axis=0)
        angle = np.array([np.pi] * 5)
        assert_almost_equal(rotation_from_angle(start, angle).as_quat(), goal)

    def test_multiple_axis_single_angle(self):
        """Test multiple axes, single angles."""
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


def _compare_cyclic(data, rotated_data, cycles=1):
    """Quickly check if rotated data was rotated by a cyclic axis rotation.

    This can be used in combination with :func:`cyclic_rotation fixture`, to test if this rotation was correctly
    applied to the data of a sensor.
    See tests below for examples.
    """
    # The cyclic rotation should be equivalent to just shifting columns
    shifted_acc_cols = SF_ACC[-cycles:] + SF_ACC[:-cycles]
    shifted_gyr_cols = SF_GYR[-cycles:] + SF_GYR[:-cycles]

    assert_almost_equal(rotated_data[SF_ACC].to_numpy(), data[shifted_acc_cols].to_numpy())
    assert_almost_equal(rotated_data[SF_GYR].to_numpy(), data[shifted_gyr_cols].to_numpy())


class TestRotateDfDataset:
    sample_sensor_data: pd.DataFrame
    sample_sensor_dataset: MultiSensorDataset

    @pytest.fixture(autouse=True)
    def _sample_sensor_data(self):
        """Create some sample data.

        This data is recreated before each test (using pytest.fixture).
        """
        acc = [1.0, 2.0, 3.0]
        gyr = [4.0, 5.0, 6.0]
        all_data = np.repeat(np.array([*acc, *gyr])[None, :], 3, axis=0)
        self.sample_sensor_data = pd.DataFrame(all_data, columns=SF_COLS)
        dataset = {"s1": self.sample_sensor_data, "s2": self.sample_sensor_data + 0.5}
        self.sample_sensor_dataset = pd.concat(dataset, axis=1)

    @pytest.mark.parametrize("inplace,equal", ((None, False), (False, False), (True, True)))
    def test_rotate_sensor_inplace(self, inplace, equal, cyclic_rotation):
        """Test if `rotate_sensor` correctly copies the data, if indicated by its arguments."""
        if inplace is None:
            # Test default
            rotated_data = _rotate_sensor(self.sample_sensor_data, cyclic_rotation)
        else:
            rotated_data = _rotate_sensor(self.sample_sensor_data, cyclic_rotation, inplace=inplace)

        if equal is True:
            assert rotated_data is self.sample_sensor_data
        else:
            assert rotated_data is not self.sample_sensor_data

    @pytest.mark.parametrize("inputs", ({"dataset": "single", "rotation": {}},))
    def test_invalid_inputs(self, inputs):
        """Test input combinations that should lead to ValueErrors."""
        # Select the dataset for test using strings, as you can not use self-parameters in the decorator.
        if inputs["dataset"] == "single":
            inputs["dataset"] = self.sample_sensor_data

        with pytest.raises(ValueError):
            rotate_dataset(**inputs)

    @pytest.mark.parametrize("ascending", (True, False))
    def test_order_is_preserved_multiple_datasets(self, cyclic_rotation, ascending):
        """Test if the function preserves the order of columns, if they are not sorted in the beginning.

        Different orders are simulated by sorting the columns once in ascending and once in descending order.
        After the rotations, the columns should still be in the same order, but the rotation is applied.

        This tests the MultiIndex input.
        """
        changed_sorted_data = self.sample_sensor_dataset.sort_index(ascending=ascending, axis=1)
        rotated_data = rotate_dataset(changed_sorted_data, cyclic_rotation)

        assert_frame_equal(changed_sorted_data.columns.to_frame(), rotated_data.columns.to_frame())

        # Test rotation worked
        _compare_cyclic(changed_sorted_data["s1"], rotated_data["s1"])
        _compare_cyclic(changed_sorted_data["s2"], rotated_data["s2"])


class TestRotateDataset:
    """Test the functions `rotate_dataset` and `_rotate_sensor`."""

    sample_sensor_data: pd.DataFrame
    sample_sensor_dataset: MultiSensorDataset

    @pytest.fixture(autouse=True, params=("dict", "frame"))
    def _sample_sensor_data(self, request):
        """Create some sample data.

        This data is recreated before each test (using pytest.fixture).
        """
        acc = [1.0, 2.0, 3.0]
        gyr = [4.0, 5.0, 6.0]
        all_data = np.repeat(np.array([*acc, *gyr])[None, :], 3, axis=0)
        self.sample_sensor_data = pd.DataFrame(all_data, columns=SF_COLS)
        dataset = {"s1": self.sample_sensor_data, "s2": self.sample_sensor_data + 0.5}
        if request.param == "dict":
            self.sample_sensor_dataset = dataset
        elif request.param == "frame":
            self.sample_sensor_dataset = pd.concat(dataset, axis=1)

    def test_rotate_sensor(self, cyclic_rotation):
        """Test if rotation is correctly applied to gyr and acc of single sensor data."""
        rotated_data = _rotate_sensor(self.sample_sensor_data, cyclic_rotation)

        _compare_cyclic(self.sample_sensor_data, rotated_data)

    def test_rotate_dataset_single(self, cyclic_rotation):
        """Rotate a single dataset with `rotate_dataset`.

        This tests the input  option where no MultiIndex df is used.
        """
        rotated_data = rotate_dataset(self.sample_sensor_data, cyclic_rotation)

        _compare_cyclic(self.sample_sensor_data, rotated_data)

    def test_rotate_single_named_dataset(self, cyclic_rotation):
        """Rotate a single dataset with a named sensor.

        This tests MultiIndex input with a single sensor.
        """
        if isinstance(self.sample_sensor_dataset, dict):
            test_data = {"s1": self.sample_sensor_dataset["s1"]}
        else:
            test_data = self.sample_sensor_dataset[["s1"]]
        rotated_data = rotate_dataset(test_data, cyclic_rotation)

        _compare_cyclic(test_data["s1"], rotated_data["s1"])

    def test_rotate_multiple_named_dataset(self, cyclic_rotation):
        """Rotate multiple dataset with a named sensors.

        This tests MultiIndex input with multiple sensors.
        """
        test_data = self.sample_sensor_dataset
        rotated_data = rotate_dataset(test_data, cyclic_rotation)

        _compare_cyclic(test_data["s1"], rotated_data["s1"])
        _compare_cyclic(test_data["s2"], rotated_data["s2"])

    def test_rotate_multiple_named_dataset_with_multiple_rotations(self, cyclic_rotation):
        """Apply different rotations to each dataset."""
        test_data = self.sample_sensor_dataset
        # Apply single cycle to "s1" and cycle twice to "s2"
        rotated_data = rotate_dataset(test_data, {"s1": cyclic_rotation, "s2": cyclic_rotation * cyclic_rotation})

        _compare_cyclic(test_data["s1"], rotated_data["s1"])
        _compare_cyclic(test_data["s2"], rotated_data["s2"], cycles=2)

    def test_only_rotate_some_sensors(self, cyclic_rotation):
        """Only apply rotation to some sensors and not all.

        This uses the dict input to only provide a rotation for s1 and not s2.
        """
        test_data = self.sample_sensor_dataset
        rotated_data = rotate_dataset(test_data, {"s1": cyclic_rotation})

        _compare_cyclic(test_data["s1"], rotated_data["s1"])
        assert_frame_equal(test_data["s2"], rotated_data["s2"])

    def test_rotate_dataset_is_copy(self, cyclic_rotation):
        """Test if the output is indeed a copy and the original dataset was not modified."""
        org_data = self.sample_sensor_dataset.copy()
        rotated_data = rotate_dataset(self.sample_sensor_dataset, cyclic_rotation)

        assert rotated_data is not self.sample_sensor_dataset
        # sample_sensor_dataset is unchanged
        for k in get_multi_sensor_dataset_names(org_data):
            assert_frame_equal(org_data[k], self.sample_sensor_dataset[k])

    @pytest.mark.parametrize("ascending", (True, False))
    def test_order_is_preserved_single_sensor(self, cyclic_rotation, ascending):
        """Test if the function preserves the order of columns, if they are not sorted in the beginning.

        Different orders are simulated by sorting the columns once in ascending and once in descending order.
        After the rotations, the columns should still be in the same order, but the rotation is applied.

        This version tests the non-MultiIndex input.
        """
        changed_sorted_data = self.sample_sensor_data.sort_index(ascending=ascending, axis=1)
        rotated_data = rotate_dataset(changed_sorted_data, cyclic_rotation)

        assert_frame_equal(changed_sorted_data.columns.to_frame(), rotated_data.columns.to_frame())

        # Test rotation worked
        _compare_cyclic(changed_sorted_data, rotated_data)


class TestFindShortestRotation:
    """Test the function `find_shortest_rotation`."""

    def test_find_shortest_rotation(self):
        """Test shortest rotation between two vectors."""
        goal = np.array([0, 0, 1])
        start = np.array([1, 0, 0])
        rot = find_shortest_rotation(start, goal)
        rotated = rot.apply(start)
        assert_almost_equal(rotated, goal)

    def test_find_shortest_rotation_unnormalized_vector(self):
        """Test shortest rotation for invalid input (one of the vectors is not normalized)."""
        with pytest.raises(ValueError):
            find_shortest_rotation([2, 0, 0], [0, 1, 0])


class TestGetGravityRotation:
    """Test the function `get_gravity_rotation`."""

    def test_gravity_rotation_simple(self):
        """Test simple gravity rotation."""
        rotation_quad = get_gravity_rotation(np.array([1, 0, 0]))
        rotated_vector = rotation_quad.apply(np.array([1, 0, 0]))
        assert_almost_equal(rotated_vector, np.array([0, 0, 1]))
