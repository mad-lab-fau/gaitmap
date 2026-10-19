"""Constrained DTW correctness through the public segmentation interface."""

import numpy as np
import pandas as pd
import pytest
from joblib import Memory

from gaitmap.stride_segmentation import BaseDtw, DtwTemplate


def _segment(template, signal, vertical, horizontal, **kwargs):
    if template.ndim == 2:
        template, signal = pd.DataFrame(template), pd.DataFrame(signal)
    kwargs = {"max_cost": 100, "find_matches_method": "min_under_thres", "min_match_length_s": None, **kwargs}
    return BaseDtw(
        template=DtwTemplate(data=template, sampling_rate_hz=1),
        resample_template=False,
        max_template_stretch_ms=vertical * 1000 if vertical else 0.1 if vertical == 0 else None,
        max_signal_stretch_ms=horizontal * 1000 if horizontal else 0.1 if horizontal == 0 else None,
        **kwargs,
    ).segment(signal, sampling_rate_hz=1)


def _enumerated_costs(template, signal, vertical, horizontal):
    """Enumerate whole valid paths; deliberately do not merge intermediate states."""
    costs = np.full((len(template), len(signal)), np.inf)
    vertical = np.inf if vertical is None else vertical
    horizontal = np.inf if horizontal is None else horizontal
    pending = [(0, start, (0, 0), 0, 0.0) for start in range(len(signal))]
    while pending:
        i, j, previous_move, run, prefix_cost = pending.pop()
        cost = prefix_cost + np.sum((template[i] - signal[j]) ** 2)
        costs[i, j] = min(costs[i, j], cost)
        for move in ((1, 1), (1, 0), (0, 1)):
            next_i, next_j = i + move[0], j + move[1]
            if next_i >= len(template) or next_j >= len(signal):
                continue
            next_run = run + 1 if move == previous_move else 1
            if move == (1, 0) and next_run > vertical:
                continue
            if move == (0, 1) and next_run > horizontal:
                continue
            pending.append((next_i, next_j, move, next_run, cost))
    return costs


def _assert_paths(dtw, template, signal, vertical, horizontal):
    for path, cost, interval in zip(dtw.paths_, dtw.costs_, dtw.matches_start_end_, strict=True):
        assert path[0, 0] == 0
        assert path[-1, 0] == len(template) - 1
        np.testing.assert_array_equal(interval, [path[0, 1], path[-1, 1] + 1])
        moves = np.diff(path, axis=0)
        run, previous = 0, None
        for move in map(tuple, moves):
            assert move in ((1, 1), (1, 0), (0, 1))
            run = run + 1 if move == previous else 1
            if move == (1, 0) and vertical is not None:
                assert run <= vertical
            if move == (0, 1) and horizontal is not None:
                assert run <= horizontal
            previous = move
        actual = np.sqrt(np.sum((template[path[:, 0]] - signal[path[:, 1]]) ** 2))
        np.testing.assert_allclose(cost, actual, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(cost**2, dtw.acc_cost_mat_[-1, path[-1, 1]], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("threshold", [0.1, 100])
@pytest.mark.parametrize("method", ["find_peaks", "min_under_thres"])
def test_issue_93_retains_the_legal_zero_cost_match(threshold, method):
    """A legal zero-cost path must survive forward calculation and thresholding."""
    template = np.array([3.0, 3.0, 3.0, 3.0, 0.0, 3.0])
    signal = np.array([3.0, 1.0, 0.0, 0.0, 3.0, 3.0, 2.0, 2.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0, 3.0])
    dtw = BaseDtw(
        template=DtwTemplate(data=template, sampling_rate_hz=1),
        resample_template=False,
        min_match_length_s=None,
        max_cost=threshold,
        find_matches_method=method,
        max_template_stretch_ms=1000,
        max_signal_stretch_ms=1000,
    ).segment(signal, sampling_rate_hz=1)

    assert len(dtw.paths_) > 0
    assert np.any(dtw.costs_ == 0)
    assert dtw.acc_cost_mat_[-1, 13] == 0
    _assert_paths(dtw, template, signal, 1, 1)


@pytest.mark.parametrize(
    ("vertical", "horizontal"),
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (1, 2), (None, 1), (1, None), (None, None), (10, 10)],
)
@pytest.mark.parametrize("shape", [(1, 4), (4, 1), (3, 5), (4, 4)])
def test_costs_agree_with_exhaustive_valid_path_enumeration(vertical, horizontal, shape):
    """The public accumulated matrix must contain constrained optima at every cell."""
    rng = np.random.default_rng(93)
    for _ in range(4):
        template = rng.integers(0, 4, size=(shape[0], 2)).astype(float)
        signal = rng.integers(0, 4, size=(shape[1], 2)).astype(float)
        dtw = _segment(template, signal, vertical, horizontal)
        np.testing.assert_array_equal(dtw.acc_cost_mat_, _enumerated_costs(template, signal, vertical, horizontal))
        _assert_paths(dtw, template, signal, vertical, horizontal)


def test_ties_are_deterministic_and_cached_paths_remain_consistent(tmp_path):
    """Caching must retain the selected state decisions, including zero-cost ties."""
    template = np.zeros(6)
    signal = np.zeros(9)
    first = _segment(template, signal, 1, 1, memory=Memory(tmp_path, mmap_mode="r", verbose=0))
    second = _segment(template, signal, 1, 1, memory=Memory(tmp_path, mmap_mode="r", verbose=0))
    assert len(first.paths_) > 0
    np.testing.assert_array_equal(first.acc_cost_mat_, second.acc_cost_mat_)
    for a, b in zip(first.paths_, second.paths_, strict=True):
        np.testing.assert_array_equal(a, b)
    _assert_paths(second, template, signal, 1, 1)


def test_nonbinding_limits_agree_with_unconstrained_paths():
    """Limits that cannot bind preserve ordinary DTW costs and selected paths."""
    template = np.array([0.0, 1.0, 1.0, 0.0])
    signal = np.array([2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0])
    constrained = _segment(template, signal, 20, 20)
    unconstrained = _segment(template, signal, None, None)
    np.testing.assert_array_equal(constrained.acc_cost_mat_, unconstrained.acc_cost_mat_)
    for a, b in zip(constrained.paths_, unconstrained.paths_, strict=True):
        np.testing.assert_array_equal(a, b)


def test_large_stretch_limit_keeps_traceback_state_ids_intact():
    """More than 255 states must not truncate stored predecessor IDs."""
    template = np.zeros(300)
    signal = np.zeros(3)
    dtw = _segment(template, signal, 257, 0)
    assert len(dtw.paths_) > 0
    _assert_paths(dtw, template, signal, 257, 0)


def test_infinite_threshold_does_not_create_unreachable_matches():
    """A template longer than the signal cannot match when all stretching is forbidden."""
    dtw = _segment(np.zeros(6), np.zeros(2), 0, 0, max_cost=np.inf)
    assert np.all(np.isinf(dtw.acc_cost_mat_[-1]))
    assert dtw.paths_ == []


@pytest.mark.parametrize("max_cost", [1, None])
def test_short_boundary_candidate_cannot_suppress_a_valid_nearby_match(max_cost):
    """Reject short candidates before enforcing the distance between valid peaks."""
    dtw = _segment(
        np.array([0.0, 1.0, 0.0]),
        np.array([1 / 3, 1.8, 0.0, 10.0]),
        2,
        0,
        max_cost=max_cost,
        min_match_length_s=2.5,
        find_matches_method="find_peaks",
    )
    np.testing.assert_array_equal(dtw.matches_start_end_, [[0, 3]])
