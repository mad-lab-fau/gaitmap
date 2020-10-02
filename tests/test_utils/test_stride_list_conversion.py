import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from gaitmap.utils.consts import SL_EVENT_ORDER
from gaitmap.utils.exceptions import ValidationError
from gaitmap.utils.stride_list_conversion import (
    enforce_stride_list_consistency,
    _segmented_stride_list_to_min_vel_single_sensor,
    convert_segmented_stride_list,
)


class TestEnforceStrideListConsistency:
    def _create_example_stride_list(self, stride_type: str):
        events = SL_EVENT_ORDER[stride_type]
        event_dict = dict()
        event_dict["start"] = np.array(list(range(20)))
        event_dict["s_id"] = event_dict["start"]
        event_dict["end"] = np.array(list(range(1, 21)))
        for i, e in enumerate(events):
            event_dict[e] = event_dict["start"] + i * 1 / len(events)
        if stride_type != "segmented":
            event_dict["start"] = event_dict[stride_type]
        return pd.DataFrame(event_dict)

    @pytest.mark.parametrize(
        "stride_type", ("segmented", "min_vel", "ic"),
    )
    def test_all_good(self, stride_type):
        event_list = self._create_example_stride_list(stride_type)
        filtered_event_list, removed_strides = enforce_stride_list_consistency(event_list, stride_type)
        assert_frame_equal(event_list, filtered_event_list)
        assert len(removed_strides) == 0

    @pytest.mark.parametrize(
        "stride_type", ("segmented", "min_vel", "ic"),
    )
    def test_simple_error(self, stride_type):
        event_list = self._create_example_stride_list(stride_type)
        wrong_s_ids = [0, 3, 5, 19]
        modified = SL_EVENT_ORDER[stride_type][-1]
        wrong = event_list[modified].copy()
        wrong[event_list["s_id"].isin(wrong_s_ids)] = -1
        event_list[modified] = wrong

        filtered_event_list, removed_strides = enforce_stride_list_consistency(event_list, stride_type)

        assert_frame_equal(event_list[~event_list["s_id"].isin(wrong_s_ids)], filtered_event_list)
        assert_frame_equal(event_list[event_list["s_id"].isin(wrong_s_ids)], removed_strides)
        assert len(removed_strides) == len(wrong_s_ids)

    def test_check_stride_list(self):
        stride_type = "segmented"
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


class TestConvertSegmentedStrideList:
    def _create_example_stride_list_with_pause(self):
        events = SL_EVENT_ORDER["segmented"]
        event_dict = dict()
        event_dict["start"] = np.array(list(range(10)))
        event_dict["s_id"] = event_dict["start"]
        event_dict["end"] = np.array(list(range(1, 11)))
        for i, e in enumerate(events):
            event_dict[e] = event_dict["start"] + i * 1 / len(events)
        stride_list = pd.DataFrame(event_dict)
        stride_list = stride_list.drop(5)
        return stride_list

    @pytest.mark.parametrize("target", ("ic", "min_vel"))
    def test_simple_conversion(self, target):
        stride_list = self._create_example_stride_list_with_pause()

        converted, dropped = _segmented_stride_list_to_min_vel_single_sensor(stride_list, target_stride_type=target)

        # We do not test everything here, but just see if it passes the basic checks.
        assert np.all(converted["start"] == converted[target])
        # We drop two strides, one at the end of both sequences
        assert len(dropped) == 2
        assert list(dropped.index) == [4, 9]
        # check consistency
        _, tmp = enforce_stride_list_consistency(converted, stride_type=target)
        assert len(tmp) == 0
        # Check that the length of all strides is still 1
        assert np.all((converted["end"] - converted["start"]).round(2) == 1.0)

    def test_second_to_last_stride_is_break(self):
        """Test an edge case where there is a break right before the last stride."""
        stride_list = self._create_example_stride_list_with_pause()
        # Drop the second to last stride to create a pause
        stride_list = stride_list.drop(8)
        converted, dropped = _segmented_stride_list_to_min_vel_single_sensor(stride_list, "min_vel")

        # Check that the length of all strides is still 1
        assert np.all((converted["end"] - converted["start"]).round(2) == 1.0)
        # We should have dropped 3 strides
        assert len(dropped) == 3
        assert list(dropped.index) == [4, 7, 9]

    @pytest.mark.parametrize("target", ("ic", "min_vel"))
    def test_simple_conversion_multiple(self, target):
        stride_list = self._create_example_stride_list_with_pause()
        converted = convert_segmented_stride_list(stride_list, target_stride_type=target)

        stride_list = {"s1": stride_list}
        converted_multiple = convert_segmented_stride_list(stride_list, target_stride_type=target)
        converted_multiple = converted_multiple["s1"]

        # We do not test everything here, but just see if it passes the basic checks.
        assert np.all(converted["start"] == converted[target])
        _, tmp = enforce_stride_list_consistency(converted, stride_type=target)
        assert len(tmp) == 0

        assert_frame_equal(converted, converted_multiple)
