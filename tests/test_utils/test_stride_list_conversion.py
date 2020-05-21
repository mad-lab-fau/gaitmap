from typing import List

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from gaitmap.utils.consts import SL_EVENT_ORDER
from gaitmap.utils.stride_list_conversion import enforce_stride_list_consistency


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
        except ValueError as e:
            pytest.fail(str(e))

        # Remove the start column to fail the check
        failing_event_list = event_list.drop("start", axis=1)
        with pytest.raises(ValueError):
            enforce_stride_list_consistency(failing_event_list, stride_type)

        # Disable the check and see that it works again
        try:
            enforce_stride_list_consistency(failing_event_list, stride_type, check_stride_list=False)
        except ValueError as e:
            pytest.fail(str(e))
