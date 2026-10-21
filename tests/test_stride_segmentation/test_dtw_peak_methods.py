"""Select historical and boundary-complete peak handling through segment()."""

import joblib
import numpy as np
import pytest

from gaitmap.stride_segmentation import (
    BarthDtw,
    BaseDtw,
    ConstrainedBarthDtw,
    DtwTemplate,
    IncorrectButBetterConstrainedBarthDtw,
)


def _segment(algorithm, method=None, signal=None, **kwargs):
    parameters = {
        "template": DtwTemplate(data=np.array([0.0, 1.0, 0.0]), sampling_rate_hz=1),
        "resample_template": False,
        "min_match_length_s": None,
        "max_match_length_s": None,
        "max_cost": 0.1,
        **({"snap_to_min_win_ms": None} if algorithm is not BaseDtw else {}),
        **kwargs,
    }
    if method is not None:
        parameters["find_matches_method"] = method
    if signal is None:
        signal = np.array([9.0, 0.0, 1.0, 0.0])
    return algorithm(**parameters).segment(signal, sampling_rate_hz=1)


@pytest.mark.parametrize("algorithm", [BaseDtw, BarthDtw, ConstrainedBarthDtw, IncorrectButBetterConstrainedBarthDtw])
def test_boundary_peak_option_detects_a_match_at_the_recording_end(algorithm):
    """An explicitly selected peak method can detect a valid last-sample minimum."""
    dtw = _segment(algorithm, "find_peaks_with_boundaries")
    np.testing.assert_array_equal(dtw.matches_start_end_, [[1, 4]])


@pytest.mark.parametrize("algorithm", [BaseDtw, BarthDtw, ConstrainedBarthDtw])
def test_main_algorithms_detect_boundary_matches_by_default(algorithm):
    """The shared default includes a valid final-sample match."""
    np.testing.assert_array_equal(_segment(algorithm).matches_start_end_, [[1, 4]])


@pytest.mark.parametrize("algorithm", [BaseDtw, BarthDtw, ConstrainedBarthDtw, IncorrectButBetterConstrainedBarthDtw])
def test_explicit_historical_peak_option_excludes_boundary_matches(algorithm):
    """Selecting the original method preserves interior-only peak handling."""
    assert _segment(algorithm, "find_peaks").paths_ == []


def test_compatibility_class_keeps_its_historical_default():
    """The compatibility class must not silently inherit the new peak default."""
    assert _segment(IncorrectButBetterConstrainedBarthDtw).paths_ == []


@pytest.mark.parametrize("algorithm", [BaseDtw, BarthDtw, ConstrainedBarthDtw])
def test_boundary_method_detects_a_flat_minimum_touching_both_edges(algorithm):
    """A boundary plateau still represents a possible complete template match."""
    dtw = _segment(
        algorithm,
        "find_peaks_with_boundaries",
        signal=np.zeros(5),
        template=DtwTemplate(data=np.zeros(3), sampling_rate_hz=1),
        max_template_stretch_ms=None,
        max_signal_stretch_ms=None,
    )
    np.testing.assert_array_equal(dtw.matches_start_end_, [[0, 3]])


@pytest.mark.parametrize("method", ["find_peaks", "find_peaks_with_boundaries", "min_under_thres"])
def test_selected_matching_method_survives_cloning_and_persistence(method, tmp_path):
    """Persisted explicit choices continue to control boundary detection."""
    original = _segment(ConstrainedBarthDtw, method)
    path = tmp_path / "algorithm.joblib"
    joblib.dump(original.clone(), path)
    for restored in (original.clone(), joblib.load(path)):
        restored.segment(original.data, sampling_rate_hz=1)
        np.testing.assert_array_equal(restored.matches_start_end_, original.matches_start_end_)
