import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gaitmap.utils.consts import SL_EVENT_ORDER
from gaitmap.utils.exceptions import ValidationError
from gaitmap.utils.stride_list_conversion import (
    _stride_list_to_min_vel_single_sensor,
    convert_stride_list,
    enforce_stride_list_consistency,
    intersect_stride_list,
)


class TestEnforceStrideListConsistency:
    def _create_example_stride_list(self, stride_type: str):
        events = SL_EVENT_ORDER[stride_type]
        event_dict = {}
        event_dict["start"] = np.arange(20) * 4
        event_dict["s_id"] = np.arange(len(event_dict["start"]))
        event_dict["end"] = np.arange(1, 21) * 4
        for i, e in enumerate(events):
            event_dict[e] = event_dict["start"] + i
        if stride_type != "segmented":
            event_dict["start"] = event_dict[stride_type]
        return pd.DataFrame(event_dict)

    @pytest.mark.parametrize(
        "stride_type",
        ["segmented", "min_vel", "ic"],
    )
    def test_all_good(self, stride_type) -> None:
        event_list = self._create_example_stride_list(stride_type)
        filtered_event_list, removed_strides = enforce_stride_list_consistency(event_list, stride_type)
        assert_frame_equal(event_list, filtered_event_list)
        assert len(removed_strides) == 0

    @pytest.mark.parametrize(
        "stride_type",
        ["segmented", "min_vel", "ic"],
    )
    def test_simple_error(self, stride_type) -> None:
        event_list = self._create_example_stride_list(stride_type)
        wrong_s_ids = [0, 3, 5, 19]
        modified = SL_EVENT_ORDER[stride_type][-1]
        wrong = event_list[modified].astype("Int64")
        wrong[event_list["s_id"].isin(wrong_s_ids)] = -1
        event_list[modified] = wrong

        filtered_event_list, removed_strides = enforce_stride_list_consistency(event_list, stride_type)

        assert_frame_equal(event_list[~event_list["s_id"].isin(wrong_s_ids)], filtered_event_list)
        assert_frame_equal(event_list[event_list["s_id"].isin(wrong_s_ids)], removed_strides)
        assert len(removed_strides) == len(wrong_s_ids)

    @pytest.mark.parametrize(
        "stride_type",
        ["segmented", "min_vel", "ic"],
    )
    def test_nan_removal(self, stride_type) -> None:
        """Test that strides that contain NaN in any column are removed."""
        event_list = self._create_example_stride_list(stride_type)
        nan_s_ids = [0, 3, 5, 19]
        modified = SL_EVENT_ORDER[stride_type][-1]
        wrong = event_list[modified].astype("Int64")
        wrong[event_list["s_id"].isin(nan_s_ids)] = np.nan
        event_list[modified] = wrong

        filtered_event_list, removed_strides = enforce_stride_list_consistency(event_list, stride_type)

        assert_frame_equal(event_list[~event_list["s_id"].isin(nan_s_ids)], filtered_event_list)
        assert_frame_equal(event_list[event_list["s_id"].isin(nan_s_ids)], removed_strides)
        assert len(removed_strides) == len(nan_s_ids)

    @pytest.mark.parametrize(
        "stride_type",
        ("segmented", "ic"),
    )
    def test_check_stride_list(self, stride_type):
        # First use a stride list that works
        event_list = self._create_example_stride_list(stride_type)
        try:
            enforce_stride_list_consistency(event_list, stride_type)
        except ValidationError as e:
            pytest.fail(str(e))

        # Remove the start column to fail the check
        failing_event_list = event_list.drop("start", axis=1)
        with pytest.raises(ValidationError):
            enforce_stride_list_consistency(failing_event_list, stride_type)

        # Disable the check and see that it works again
        try:
            enforce_stride_list_consistency(failing_event_list, stride_type, check_stride_list=False)
        except ValidationError as e:
            pytest.fail(str(e))

    def test_input_stride_type_has_no_default(self) -> None:
        event_list = self._create_example_stride_list("segmented")

        with pytest.raises(TypeError):
            enforce_stride_list_consistency(event_list)


class TestConvertSegmentedStrideList:
    def _create_example_stride_list_with_pause(self):
        events = SL_EVENT_ORDER["segmented"]
        event_dict = {}
        event_dict["start"] = np.arange(10) * 4
        event_dict["s_id"] = np.arange(len(event_dict["start"]))
        event_dict["end"] = np.arange(1, 11) * 4
        for i, e in enumerate(events):
            event_dict[e] = event_dict["start"] + i
        stride_list = pd.DataFrame(event_dict)
        stride_list = stride_list.drop(5)
        return stride_list

    @pytest.mark.parametrize("target", ["ic", "min_vel"])
    def test_simple_conversion(self, target) -> None:
        stride_list = self._create_example_stride_list_with_pause()

        converted, dropped = _stride_list_to_min_vel_single_sensor(
            stride_list, source_stride_type="segmented", target_stride_type=target
        )

        # We do not test everything here, but just see if it passes the basic checks.
        assert np.all(converted["start"] == converted[target])
        # We drop two strides, one at the end of both sequences
        assert len(dropped) == 2
        assert list(dropped.index) == [4, 9]
        # check consistency
        _, tmp = enforce_stride_list_consistency(converted, input_stride_type=target)
        assert len(tmp) == 0
        # Check that the length of all strides is still 4
        assert np.all((converted["end"] - converted["start"]) == 4)

    def test_second_to_last_stride_is_break(self) -> None:
        """Test an edge case where there is a break right before the last stride."""
        stride_list = self._create_example_stride_list_with_pause()
        # Drop the second to last stride to create a pause
        stride_list = stride_list.drop(8)
        converted, dropped = _stride_list_to_min_vel_single_sensor(
            stride_list, source_stride_type="segmented", target_stride_type="min_vel"
        )

        # Check that the length of all strides is still 4
        assert np.all((converted["end"] - converted["start"]) == 4)
        # We should have dropped 3 strides
        assert len(dropped) == 3
        assert list(dropped.index) == [4, 7, 9]

    @pytest.mark.parametrize("target", ["ic", "min_vel"])
    def test_simple_conversion_multiple(self, target) -> None:
        stride_list = self._create_example_stride_list_with_pause()
        converted = convert_stride_list(stride_list, target_stride_type=target)

        stride_list = {"s1": stride_list}
        converted_multiple = convert_stride_list(stride_list, target_stride_type=target)
        converted_multiple = converted_multiple["s1"]

        # We do not test everything here, but just see if it passes the basic checks.
        assert np.all(converted["start"] == converted[target])
        _, tmp = enforce_stride_list_consistency(converted, input_stride_type=target)
        assert len(tmp) == 0

        assert_frame_equal(converted, converted_multiple)


class TestConvertIcStrideList:
    def _create_example_stride_list_with_pause(self):
        events = SL_EVENT_ORDER["ic"]
        event_dict = {}
        event_dict["start"] = np.arange(10) * 4
        event_dict["s_id"] = np.arange(len(event_dict["start"]))
        event_dict["end"] = np.arange(1, 11) * 4
        for i, e in enumerate(events):
            event_dict[e] = event_dict["start"] + i
        stride_list = pd.DataFrame(event_dict)
        stride_list = stride_list.drop(5)
        return stride_list

    def test_simple_conversion(self):
        stride_list = self._create_example_stride_list_with_pause()

        converted, dropped = _stride_list_to_min_vel_single_sensor(
            stride_list, source_stride_type="ic", target_stride_type="min_vel"
        )

        # We do not test everything here, but just see if it passes the basic checks.
        assert np.all(converted["start"] == converted["min_vel"])
        # We drop two strides, one at the end of both sequences
        assert len(dropped) == 2
        assert list(dropped.index) == [4, 9]
        # check consistency
        _, tmp = enforce_stride_list_consistency(converted, input_stride_type="min_vel")
        assert len(tmp) == 0
        # Check that the length of all strides is still 4
        assert np.all((converted["end"] - converted["start"]) == 4)

    def test_second_to_last_stride_is_break(self):
        """Test an edge case where there is a break right before the last stride."""
        stride_list = self._create_example_stride_list_with_pause()
        # Drop the second to last stride to create a pause
        stride_list = stride_list.drop(8)
        converted, dropped = _stride_list_to_min_vel_single_sensor(
            stride_list, source_stride_type="ic", target_stride_type="min_vel"
        )

        # Check that the length of all strides is still 4
        assert np.all((converted["end"] - converted["start"]) == 4)
        # We should have dropped 3 strides
        assert len(dropped) == 3
        assert list(dropped.index) == [4, 7, 9]

    def test_simple_conversion_multiple(self):
        stride_list = self._create_example_stride_list_with_pause()
        converted = convert_stride_list(stride_list, source_stride_type="ic", target_stride_type="min_vel")

        stride_list = {"s1": stride_list}
        converted_multiple = convert_stride_list(stride_list, source_stride_type="ic", target_stride_type="min_vel")
        converted_multiple = converted_multiple["s1"]

        # We do not test everything here, but just see if it passes the basic checks.
        assert np.all(converted["start"] == converted["min_vel"])
        _, tmp = enforce_stride_list_consistency(converted, input_stride_type="min_vel")
        assert len(tmp) == 0

        assert_frame_equal(converted, converted_multiple)


class TestIntersectStrideList:
    def test_simple_with_overlap(self) -> None:
        stride_list = pd.DataFrame(
            {"start": [10, 25, 30, 50], "end": [20, 30, 40, 55]}, index=pd.Series([0, 1, 2, 3], name="s_id")
        )
        roi = pd.DataFrame({"start": [5, 28], "end": [32, 45]}, index=pd.Series([0, 1], name="roi_id"))

        expected = [
            pd.DataFrame({"start": [10, 25], "end": [20, 30]}, index=pd.Series([0, 1], name="s_id")) - 5,
            pd.DataFrame({"start": [30], "end": [40]}, index=pd.Series([2], name="s_id")) - 28,
        ]

        out = intersect_stride_list(stride_list, roi)
        assert len(out) == 2

        for i, o in enumerate(out):
            assert_frame_equal(o, expected[i])


@pytest.mark.parametrize("as_index", [False, True])
def test_intersection_preserves_integer_samples_with_float_roi_metadata(as_index):
    """ROI IDs and confidence values cannot alter sample coordinates or their representation."""
    strides = pd.DataFrame({"start": [10], "end": [20], "ic": [13]}, index=pd.Index(["step"], name="s_id"))
    regions = pd.DataFrame({"roi_id": [0.5], "start": [5], "end": [25], "confidence": [0.75]})
    if as_index:
        regions = regions.set_index("roi_id")
    original_strides, original_regions = strides.copy(deep=True), regions.copy(deep=True)

    (actual,) = intersect_stride_list(strides, regions)

    expected = pd.DataFrame({"start": [5], "end": [15], "ic": [8]}, index=strides.index)
    assert_frame_equal(actual, expected)
    assert_frame_equal(strides, original_strides)
    assert_frame_equal(regions, original_regions)


@pytest.mark.parametrize("missing_boundary", ["start", "end"])
def test_intersection_keeps_complete_regions_when_another_boundary_is_missing(missing_boundary):
    """An incomplete annotation contributes no strides without discarding other regions."""
    strides = pd.DataFrame({"start": [10], "end": [20]}, index=pd.Index(["step"], name="s_id"))
    regions = pd.DataFrame({"start": [5, 5], "end": [25, 25]}, dtype="Int64").rename_axis("roi_id")
    regions.loc[0, missing_boundary] = pd.NA

    empty, complete = intersect_stride_list(strides, regions)

    assert_frame_equal(empty, strides.iloc[:0])
    assert_frame_equal(complete, pd.DataFrame({"start": [5], "end": [15]}, index=strides.index))


@pytest.mark.parametrize("dtype", ["uint32", "uint64", "UInt32", "UInt64"])
def test_intersection_preserves_events_before_roi_start(dtype):
    """A previous contact remains negative in ROI coordinates, rather than wrapping."""
    strides = pd.DataFrame(
        {"start": [10], "end": [20], "pre_ic": [8], "min_vel": [10], "tc": [12], "ic": [15]},
        dtype=dtype,
    ).rename_axis("s_id")
    regions = pd.DataFrame({"start": [9], "end": [25]}, index=pd.Index([0.5], name="roi_id"))
    original = strides.copy(deep=True)

    (actual,) = intersect_stride_list(strides, regions)

    np.testing.assert_array_equal(actual.iloc[0], [1, 11, -1, 1, 3, 6])
    valid, invalid = enforce_stride_list_consistency(actual, input_stride_type="min_vel")
    assert len(valid) == 1
    assert invalid.empty
    assert_frame_equal(strides, original)


@pytest.mark.parametrize("dtype", ["uint64", "UInt64"])
def test_intersection_shifts_large_absolute_positions_before_signed_conversion(dtype):
    """Only ROI-relative coordinates need to fit signed storage."""
    origin = 2**63
    strides = pd.DataFrame({"start": [origin], "end": [origin + 10], "pre_ic": [origin - 2]}, dtype=dtype).rename_axis(
        "s_id"
    )
    regions = pd.DataFrame({"start": [origin - 1], "end": [origin + 20]}, dtype=dtype).rename_axis("roi_id")

    (actual,) = intersect_stride_list(strides, regions)

    np.testing.assert_array_equal(actual.iloc[0], [1, 11, -1])


@pytest.mark.parametrize("dtype", ["uint64", "UInt64"])
@pytest.mark.parametrize("target", ["min_vel", "ic"])
def test_conversion_preserves_unsigned_absolute_event_positions(dtype, target):
    """Changing stride alignment must preserve events across the full unsigned range."""
    origin = 2**63
    strides = (
        pd.DataFrame(
            {
                "start": [origin, origin + 20],
                "end": [origin + 20, origin + 40],
                "ic": [origin + 10, origin + 30],
                "tc": [origin + 5, origin + 25],
                "min_vel": [origin + 15, origin + 35],
            },
            dtype=dtype,
        )
        .rename_axis("s_id")
        .assign(confidence=[0.5, 0.75])
    )
    original = strides.copy(deep=True)

    actual = convert_stride_list(strides, target_stride_type=target)

    expected = {
        "start": [origin + (15 if target == "min_vel" else 10)],
        "end": [origin + (35 if target == "min_vel" else 30)],
        "ic": [origin + (30 if target == "min_vel" else 10)],
        "tc": [origin + 25],
        "min_vel": [origin + 15],
        "confidence": [0.5],
    }
    if target == "min_vel":
        expected["pre_ic"] = [origin + 10]
    assert actual.to_dict("list") == expected
    assert_frame_equal(strides, original)
