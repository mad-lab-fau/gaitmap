from typing import Optional, Union, Dict

from typing_extensions import Literal

from gaitmap.stride_segmentation.barth_dtw import BarthDtw
from gaitmap.stride_segmentation.dtw_templates import DtwTemplate, BarthOriginalTemplate


class ConstrainedBarthDtw(BarthDtw):
    def __init__(
        self,
        template: Optional[Union[DtwTemplate, Dict[str, DtwTemplate]]] = BarthOriginalTemplate(),
        resample_template: bool = True,
        find_matches_method: Literal["min_under_thres", "find_peaks"] = "find_peaks",
        max_cost: Optional[float] = 4.0,
        min_match_length_s: Optional[float] = 0.6,
        max_match_length_s: Optional[float] = 3.0,
        max_template_stretch_ms: Optional[float] = 20,
        max_signal_stretch_ms: Optional[float] = 30,
        snap_to_min_win_ms: Optional[float] = 300,
        snap_to_min_axis: Optional[str] = "gyr_ml",
        conflict_resolution: bool = True,
    ):
        super().__init__(
            template=template,
            max_cost=max_cost,
            min_match_length_s=min_match_length_s,
            max_match_length_s=max_match_length_s,
            max_template_stretch_ms=max_template_stretch_ms,
            max_signal_stretch_ms=max_signal_stretch_ms,
            resample_template=resample_template,
            find_matches_method=find_matches_method,
            snap_to_min_win_ms=snap_to_min_win_ms,
            snap_to_min_axis=snap_to_min_axis,
            conflict_resolution=conflict_resolution,
        )
