"""Historical constrained-DTW behavior through public segmentation."""

import numpy as np
import pytest
from joblib import Memory

from gaitmap.stride_segmentation import (
    BarthDtw,
    ConstrainedBarthDtw,
    DtwTemplate,
    IncorrectButBetterConstrainedBarthDtw,
)
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.mixins.test_caching_mixin import TestCachingMixin
from tests.test_stride_segmentation.test_constrained_barth_dtw import MetaTestConfig


@pytest.mark.parametrize("use_cache", [False, True])
def test_preserves_historical_costs_and_inconsistent_traceback(use_cache, tmp_path):
    """Reproduce the pre-fix issue-93 result, including its known defects."""
    template = np.array([3.0, 3.0, 3.0, 3.0, 0.0, 3.0])
    signal = np.array([3.0, 1.0, 0.0, 0.0, 3.0, 3.0, 2.0, 2.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0, 3.0])
    dtw = IncorrectButBetterConstrainedBarthDtw(
        template=DtwTemplate(data=template, sampling_rate_hz=1),
        resample_template=False,
        min_match_length_s=None,
        max_match_length_s=None,
        max_cost=100,
        find_matches_method="min_under_thres",
        max_template_stretch_ms=1000,
        max_signal_stretch_ms=1000,
        snap_to_min_win_ms=None,
        memory=Memory(tmp_path if use_cache else None, verbose=0),
    ).segment(signal, sampling_rate_hz=1)
    if use_cache:
        dtw = dtw.clone().segment(signal, sampling_rate_hz=1)

    # Captured from the implementation preceding the issue-93 correction.
    np.testing.assert_array_equal(
        dtw.acc_cost_mat_[-1], [np.inf, np.inf, np.inf, 26, 17, 17, 14, 14, 6, 11, 2, 2, 18, 9, 9]
    )
    np.testing.assert_array_equal(dtw.paths_, [[[0, 8], [1, 8], [2, 8], [3, 8], [4, 9], [5, 10]]])
    np.testing.assert_array_equal(dtw.matches_start_end_, [[8, 11]])
    np.testing.assert_allclose(dtw.costs_, [np.sqrt(2)])


def test_preserves_historical_exclusion_of_recording_boundary_minima():
    """The compatibility class must not add the corrected boundary candidates."""
    kwargs = {
        "template": DtwTemplate(data=np.array([0.0, 1.0, 0.0]), sampling_rate_hz=1),
        "resample_template": False,
        "min_match_length_s": None,
        "max_match_length_s": None,
        "snap_to_min_win_ms": None,
        "max_cost": 0.1,
        "max_template_stretch_ms": 1000,
        "max_signal_stretch_ms": 1000,
    }
    signal = np.array([9.0, 0.0, 1.0, 0.0])
    historical = IncorrectButBetterConstrainedBarthDtw(**kwargs).segment(signal, sampling_rate_hz=1)
    corrected = ConstrainedBarthDtw(**kwargs).segment(signal, sampling_rate_hz=1)
    assert historical.paths_ == []
    np.testing.assert_array_equal(corrected.matches_start_end_, [[1, 4]])


@pytest.mark.parametrize("method", ["find_peaks", "min_under_thres"])
def test_unbounded_limits_preserve_ordinary_dtw(method):
    """The historical unconstrained branch still uses ordinary DTW."""
    kwargs = {
        "template": DtwTemplate(data=np.array([0.0, 1.0, 0.0]), sampling_rate_hz=1),
        "resample_template": False,
        "min_match_length_s": None,
        "max_match_length_s": None,
        "snap_to_min_win_ms": None,
        "max_template_stretch_ms": None,
        "max_signal_stretch_ms": None,
        "find_matches_method": method,
    }
    signal = np.array([9.0, 0.0, 1.0, 0.0, 9.0])
    historical = IncorrectButBetterConstrainedBarthDtw(**kwargs).segment(signal, sampling_rate_hz=1)
    ordinary = BarthDtw(**kwargs).segment(signal, sampling_rate_hz=1)
    np.testing.assert_array_equal(historical.acc_cost_mat_, ordinary.acc_cost_mat_)
    np.testing.assert_array_equal(historical.matches_start_end_, ordinary.matches_start_end_)
    for actual, expected in zip(historical.paths_, ordinary.paths_, strict=True):
        np.testing.assert_array_equal(actual, expected)


class TestMetaFunctionality(MetaTestConfig, TestAlgorithmMixin):
    """The compatibility class supports the standard algorithm interface."""

    __test__ = True
    algorithm_class = IncorrectButBetterConstrainedBarthDtw


class TestCachingFunctionality(MetaTestConfig, TestCachingMixin):
    """Cached public calls retain historical results."""

    __test__ = True
    algorithm_class = IncorrectButBetterConstrainedBarthDtw
