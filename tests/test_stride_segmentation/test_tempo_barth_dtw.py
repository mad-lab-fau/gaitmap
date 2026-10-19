"""Public segment() behavior of the experimental tempo segmentation sibling."""

import numpy as np
import pandas as pd
import pytest
from joblib import Memory
from tpcp import make_action_safe

from gaitmap.stride_segmentation import DtwTemplate, ExperimentalTempoBarthDtw


def test_segment_reports_weighted_objective_and_exclusive_intervals():
    """The public wrapper exposes the same weighted alignment as its solver."""
    algorithm = ExperimentalTempoBarthDtw(
        template=DtwTemplate(data=np.array([0.0, 2.0, 0.0]), sampling_rate_hz=1),
        resample_template=False,
        n_segments=3,
        tempo_bounds=(1, 1),
        max_cost=0.1,
        find_matches_method="min_under_thres",
    ).segment(np.array([9.0, 0.0, 2.0, 0.0]), sampling_rate_hz=1)
    np.testing.assert_array_equal(algorithm.matches_start_end_, [[1, 4]])
    np.testing.assert_array_equal(algorithm.stride_list_[["start", "end"]], [[1, 4]])
    np.testing.assert_allclose(algorithm.path_weights_[0], [1 / 3] * 3)
    np.testing.assert_array_equal(algorithm.block_edges_[0], [1, 2, 3, 4])
    assert algorithm.costs_[0] == algorithm.alignments_[0].cost == 0


def test_nonzero_regularized_cost_and_duration_optimum_agree_with_metadata():
    """Duration-valid alternate starts are optimized before endpoint detection."""
    algorithm = ExperimentalTempoBarthDtw(
        template=DtwTemplate(data=np.array([0.0, 1.0, 1.0, 0.0]), sampling_rate_hz=4),
        resample_template=False,
        n_segments=4,
        tempo_bounds=(1, 2),
        max_cost=10,
        tempo_change_penalty=2,
        min_match_length_s=5.5 / 4,
        max_match_length_s=6.5 / 4,
        find_matches_method="min_under_thres",
    ).segment(np.array([1.0, 3.0, 3.0, 0.0, 0.0, 0.0]), sampling_rate_hz=4)
    np.testing.assert_array_equal(algorithm.matches_start_end_, [[0, 6]])
    expected = 1.875 + 2 * np.log(2) ** 2
    np.testing.assert_allclose(algorithm.costs_, [np.sqrt(expected)])
    np.testing.assert_allclose(algorithm.alignments_[0].cost, expected)
    assert algorithm.duration_fallback_count_ > 0


def test_multisensor_cache_clone_and_json_roundtrip_preserve_selected_paths(tmp_path):
    """Real caching and tpcp cloning preserve numerical decisions without mocks."""
    template = DtwTemplate(data=pd.DataFrame({"gyr_ml": [0.0, 2.0, 0.0]}), sampling_rate_hz=1)
    algorithm = ExperimentalTempoBarthDtw(
        template=template,
        resample_template=False,
        n_segments=3,
        tempo_bounds=(1, 1),
        max_cost=0.1,
        memory=Memory(tmp_path, mmap_mode="r", verbose=0),
    )
    data = {
        "left": pd.DataFrame({"gyr_ml": [9.0, 0.0, 2.0, 0.0]}),
        "right": pd.DataFrame({"gyr_ml": [0.0, 2.0, 0.0, 9.0]}),
    }
    first = make_action_safe(type(algorithm).segment)(algorithm, data, sampling_rate_hz=1)
    cached = algorithm.clone().segment(data, sampling_rate_hz=1)
    for sensor, expected in [("left", [[1, 4]]), ("right", [[0, 3]])]:
        np.testing.assert_array_equal(first.stride_list_[sensor][["start", "end"]], expected)
        np.testing.assert_array_equal(first.paths_[sensor][0], cached.paths_[sensor][0])
        np.testing.assert_allclose(first.path_weights_[sensor][0], cached.path_weights_[sensor][0])
    restored = ExperimentalTempoBarthDtw.from_json(algorithm.clone().set_params(memory=None).to_json())
    result = restored.segment(data, sampling_rate_hz=1)
    np.testing.assert_array_equal(result.paths_["left"][0], first.paths_["left"][0])


def test_no_valid_match_has_empty_outputs_and_infinite_curve():
    """Threshold-free peak finding cannot fabricate paths at infinite costs."""
    algorithm = ExperimentalTempoBarthDtw(
        template=DtwTemplate(data=np.array([0.0, 1.0, 0.0]), sampling_rate_hz=1),
        resample_template=False,
        n_segments=3,
        tempo_bounds=(1, 1),
        max_cost=None,
    ).segment(np.array([0.0, 1.0]), sampling_rate_hz=1)
    assert algorithm.paths_ == algorithm.alignments_ == []
    assert algorithm.stride_list_.empty
    assert np.all(np.isinf(algorithm.cost_function_))


def test_resampling_and_native_unequal_rates_both_use_physical_tempo():
    """Resampled and native-rate templates expose their actual tempo lattice."""
    data = pd.DataFrame({"gyr_ml": np.ones(8)})
    template = DtwTemplate(data=pd.DataFrame({"gyr_ml": np.ones(4)}), sampling_rate_hz=4)
    for resample in (True, False):
        algorithm = ExperimentalTempoBarthDtw(
            template=template,
            resample_template=resample,
            n_segments=2,
            tempo_bounds=(1, 1),
            max_cost=0.1,
            find_matches_method="min_under_thres",
        ).segment(data, sampling_rate_hz=8)
        np.testing.assert_array_equal(algorithm.matches_start_end_, [[0, 8]])
        np.testing.assert_array_equal(algorithm.tempo_profiles_[0], [1, 1])
        np.testing.assert_allclose(algorithm.path_weights_[0].sum(), 1)


def test_snapping_preserves_original_alignment_metadata():
    """Postprocessing may move borders while costs retain their original path."""
    template = DtwTemplate(data=pd.DataFrame({"gyr_ml": [0.0, 2.0, 0.0]}), sampling_rate_hz=1)
    data = pd.DataFrame({"gyr_ml": [9.0, 0.0, 2.0, 0.0, 9.0]})
    algorithm = ExperimentalTempoBarthDtw(
        template=template,
        resample_template=False,
        n_segments=3,
        tempo_bounds=(1, 1),
        max_cost=0.1,
        find_matches_method="min_under_thres",
        snap_to_min_win_ms=2000,
    ).segment(data, sampling_rate_hz=1)
    np.testing.assert_array_equal(algorithm.matches_start_end_original_, [[1, 4]])
    np.testing.assert_array_equal(algorithm.matches_start_end_, [[1, 3]])
    np.testing.assert_array_equal(algorithm.block_edges_[0], [1, 2, 3, 4])
    assert algorithm.costs_[0] == 0


@pytest.mark.parametrize("method", ["min_under_thres", "find_peaks"])
def test_empty_candidates_with_snapping_return_empty_results(method):
    """Neither a float empty candidate array nor snapping may invent a match."""
    template = DtwTemplate(data=pd.DataFrame({"gyr_ml": [0.0, 2.0, 0.0]}), sampling_rate_hz=1)
    algorithm = ExperimentalTempoBarthDtw(
        template=template,
        resample_template=False,
        n_segments=3,
        tempo_bounds=(1, 1),
        max_cost=0.1,
        find_matches_method=method,
        snap_to_min_win_ms=2000,
    ).segment(pd.DataFrame({"gyr_ml": [9.0, 9.0, 9.0, 9.0]}), sampling_rate_hz=1)
    assert algorithm.paths_ == algorithm.alignments_ == []
    assert algorithm.stride_list_.empty
    assert algorithm.matches_start_end_.shape == (0, 2)
    assert algorithm.costs_.size == 0
