"""Public numerical alignment contract for the experimental block-tempo model."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from gaitmap_mad.stride_segmentation.dtw import tempo_dtw_paths, tempo_regularized_subsequence_dtw


def test_identity_match_has_reproducible_path_cost():
    """Free starts and exclusive edges must describe the same zero-cost match."""
    template = np.array([0.0, 2.0, 0.0])
    signal = np.array([9.0, 0.0, 2.0, 0.0])
    result = tempo_regularized_subsequence_dtw(
        template,
        signal,
        template_sampling_rate_hz=1,
        signal_sampling_rate_hz=1,
        n_segments=3,
        tempo_bounds=(1, 1),
        tempo_change_penalty=1,
    )
    alignment = tempo_dtw_paths(template, signal, result, [3])[0]
    np.testing.assert_array_equal(alignment.path, [[0, 1], [1, 2], [2, 3]])
    np.testing.assert_allclose(alignment.weights, np.full(3, 1 / 3))
    np.testing.assert_array_equal(alignment.block_edges, [1, 2, 3, 4])
    assert alignment.data_cost == alignment.tempo_cost == alignment.cost == 0
    assert result.costs[-1, 3] == alignment.cost


def _whole_warp_cost(template, signal, edges, penalty):
    """Integrate a complete warp via rational global breakpoints, without DP."""
    m, k_count = len(template), len(edges) - 1
    knots = [Fraction(k * m, k_count) for k in range(k_count + 1)]
    breakpoints = set(map(Fraction, range(m + 1)))
    for k, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        for j in range(left, right + 1):
            breakpoints.add(knots[k] + (knots[k + 1] - knots[k]) * Fraction(int(j - left), int(right - left)))
    points = sorted(breakpoints)
    cost = 0.0
    for left, right in zip(points[:-1], points[1:]):
        mid = (left + right) / 2
        k = int(mid * k_count / m)
        position = edges[k] + (mid - knots[k]) / (knots[k + 1] - knots[k]) * (edges[k + 1] - edges[k])
        cost += float((right - left) / m) * np.sum((template[int(mid)] - signal[int(position)]) ** 2)
    return cost + penalty * np.sum(np.diff(np.log(np.diff(edges))) ** 2)


def _enumerate(template, signal, k_count, advances, penalty, minimum=0, maximum=None):
    """Enumerate complete start/advance sequences; never merge arrivals."""
    costs = np.full(len(signal), np.inf)
    maximum = len(signal) if maximum is None else maximum
    for steps in product(advances, repeat=k_count):
        duration = sum(steps)
        if not minimum <= duration <= maximum:
            continue
        for start in range(len(signal) - duration + 1):
            edges = np.r_[start, start + np.cumsum(steps)]
            cost = _whole_warp_cost(template, signal, edges, penalty)
            end = edges[-1] - 1
            costs[end] = min(costs[end], cost)
    return costs


def test_duration_constraint_retains_alternative_starts_against_exhaustive_oracle():
    """A feasible alternative start must survive an invalid unrestricted winner."""
    template = np.array([0.0, 1.0, 2.0, 0.0, 1.0])
    signal = np.array([0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0])
    result = tempo_regularized_subsequence_dtw(
        template,
        signal,
        template_sampling_rate_hz=5,
        signal_sampling_rate_hz=3,
        n_segments=3,
        tempo_bounds=(1, 2),
        tempo_change_penalty=0.3,
        min_duration_samples=5,
        max_duration_samples=5,
    )
    expected = _enumerate(template, signal, 3, (1, 2), 0.3, 5, 5)
    np.testing.assert_allclose(result.costs[-1], expected, atol=1e-13)
    assert result.fallback_count > 0
    for end in np.flatnonzero(np.isfinite(expected)):
        alignment = tempo_dtw_paths(template, signal, result, [end])[0]
        assert alignment.block_edges[-1] - alignment.block_edges[0] == 5
        np.testing.assert_allclose(alignment.cost, expected[end], atol=1e-13)


@pytest.mark.parametrize("penalty", [0, 0.3, 5])
@pytest.mark.parametrize(("m", "n", "blocks"), [(1, 4, 1), (5, 8, 3), (4, 7, 2), (3, 2, 3)])
@pytest.mark.parametrize(("minimum", "maximum"), [(None, None), (4, 5), (6, 6), (9, 4)])
def test_tiny_endpoint_optima_and_paths_agree_with_whole_warp_enumeration(penalty, m, n, blocks, minimum, maximum):
    """Exhaust complete warps including split template bins and multiple features."""
    rng = np.random.default_rng(193)
    template = rng.integers(0, 4, (m, 2)).astype(float)
    signal = rng.integers(0, 4, (n, 2)).astype(float)
    result = tempo_regularized_subsequence_dtw(
        template,
        signal,
        template_sampling_rate_hz=m,
        signal_sampling_rate_hz=blocks,
        n_segments=blocks,
        tempo_bounds=(1, 3),
        tempo_change_penalty=penalty,
        min_duration_samples=minimum,
        max_duration_samples=maximum,
    )
    expected = _enumerate(template, signal, blocks, (1, 2, 3), penalty, minimum or 0, maximum)
    np.testing.assert_allclose(result.costs[-1], expected, atol=2e-14)
    for end in np.flatnonzero(np.isfinite(expected)):
        alignment = tempo_dtw_paths(template, signal, result, [end])[0]
        np.testing.assert_allclose(alignment.cost, expected[end], atol=2e-14)
        np.testing.assert_allclose(_whole_warp_cost(template, signal, alignment.block_edges, penalty), alignment.cost)
        np.testing.assert_allclose(alignment.weights.sum(), 1)
        assert np.all(alignment.weights > 0)
        np.testing.assert_array_equal(np.unique(alignment.path[:, 0]), np.arange(m))
        np.testing.assert_array_equal(np.unique(alignment.path[:, 1]), np.arange(alignment.block_edges[0], end + 1))
        np.testing.assert_array_equal(alignment.path[[0, -1]], [[0, alignment.block_edges[0]], [m - 1, end]])


def test_more_expensive_tempo_arrival_wins_after_a_shared_continuation():
    """Discarding a competing tempo would change the complete optimum."""
    template = np.array([2.0, 1.0, 0.0])
    signal = np.array([3.0, 2.0, 2.0, 1.0, 2.0, 2.0])
    # Both prefixes arrive at template edge 2 and signal edge 3. The [2, 1]
    # arrival fits better, but [1, 2] avoids a change for the final advance 2.
    assert _whole_warp_cost(template[:2], signal, [0, 2, 3], 0.5) < _whole_warp_cost(
        template[:2], signal, [0, 1, 3], 0.5
    )
    assert _whole_warp_cost(template, signal, [0, 1, 3, 5], 0.5) < _whole_warp_cost(template, signal, [0, 2, 3, 5], 0.5)
    result = tempo_regularized_subsequence_dtw(
        template,
        signal,
        template_sampling_rate_hz=3,
        signal_sampling_rate_hz=3,
        n_segments=3,
        tempo_bounds=(1, 2),
        tempo_change_penalty=0.5,
        min_duration_samples=5,
        max_duration_samples=5,
    )
    np.testing.assert_array_equal(tempo_dtw_paths(template, signal, result, [4])[0].block_edges, [0, 1, 3, 5])


@pytest.mark.parametrize(("signal", "tempo"), [([0.0, 1.0], 0.5), ([0.0] * 4 + [1.0] * 4, 2.0)])
def test_uniform_fast_and_slow_tempos_have_no_change_penalty(signal, tempo):
    """Constant speeds have no regularization, including far from unit speed."""
    template = np.array([0.0, 0.0, 1.0, 1.0])
    result = tempo_regularized_subsequence_dtw(
        template,
        np.array(signal),
        template_sampling_rate_hz=4,
        signal_sampling_rate_hz=4,
        n_segments=2,
        tempo_bounds=(0.5, 2),
        tempo_change_penalty=10,
    )
    alignment = tempo_dtw_paths(template, signal, result, [len(signal) - 1])[0]
    np.testing.assert_array_equal(alignment.tempos, [tempo, tempo])
    assert alignment.cost == alignment.tempo_cost == 0


def test_penalty_discourages_sudden_compression_and_expansion():
    """Regularization trades a closer signal fit for fewer tempo changes."""
    template = np.array([0.0, 1.0, 1.0, 0.0])
    signal = np.array([1.0, 3.0, 3.0, 0.0, 0.0, 0.0])
    alignments = []
    for penalty in (0, 2):
        result = tempo_regularized_subsequence_dtw(
            template,
            signal,
            template_sampling_rate_hz=4,
            signal_sampling_rate_hz=4,
            n_segments=4,
            tempo_bounds=(1, 2),
            tempo_change_penalty=penalty,
            min_duration_samples=6,
            max_duration_samples=6,
        )
        alignments.append(tempo_dtw_paths(template, signal, result, [5])[0])
    unregularized, regularized = alignments
    np.testing.assert_array_equal(unregularized.block_edges, [0, 1, 3, 5, 6])
    np.testing.assert_array_equal(regularized.block_edges, [0, 1, 2, 4, 6])
    assert unregularized.data_cost == 1.5
    assert regularized.data_cost == 1.875
    assert np.sum(np.diff(np.log(regularized.tempos)) ** 2) < np.sum(np.diff(np.log(unregularized.tempos)) ** 2)


def test_large_advance_accounts_for_hidden_mismatching_sample():
    """Endpoint-only scoring would incorrectly make this block cost zero."""
    result = tempo_regularized_subsequence_dtw(
        [0.0],
        [0.0, 3.0, 0.0],
        template_sampling_rate_hz=1,
        signal_sampling_rate_hz=1,
        n_segments=1,
        tempo_bounds=(3, 3),
    )
    alignment = tempo_dtw_paths([0.0], [0.0, 3.0, 0.0], result, [2])[0]
    assert alignment.cost == 3
    np.testing.assert_allclose(alignment.weights, [1 / 3] * 3)


def test_equal_costs_prefer_earliest_start_and_paths_are_repeatable():
    """Tied start and tempo decisions must be stable during recomputation."""
    template, signal = np.zeros(3), np.zeros(6)
    result = tempo_regularized_subsequence_dtw(
        template,
        signal,
        template_sampling_rate_hz=3,
        signal_sampling_rate_hz=3,
        n_segments=3,
        tempo_bounds=(1, 2),
    )
    first = tempo_dtw_paths(template, signal, result, [5])[0]
    second = tempo_dtw_paths(template, signal, result, [5])[0]
    np.testing.assert_array_equal(first.block_edges, [0, 2, 4, 6])
    np.testing.assert_array_equal(first.path, second.path)


def test_unreachable_endpoints_have_no_path():
    """Insufficient signal bins cannot produce fabricated paths."""
    result = tempo_regularized_subsequence_dtw(
        [0.0, 1.0, 0.0],
        [0.0, 1.0],
        template_sampling_rate_hz=3,
        signal_sampling_rate_hz=3,
        n_segments=3,
        tempo_bounds=(1, 1),
    )
    assert np.all(np.isinf(result.costs[-1]))
    with pytest.raises(ValueError, match="unreachable"):
        tempo_dtw_paths([0.0, 1.0, 0.0], [0.0, 1.0], result, [1])


def test_tempo_uses_physical_sampling_rates_not_raw_sample_counts():
    """Equal physical tempo at different rates consumes different sample counts."""
    template, signal = [0.0, 0.0, 1.0, 1.0], [0.0] * 4 + [1.0] * 4
    result = tempo_regularized_subsequence_dtw(
        template,
        signal,
        template_sampling_rate_hz=4,
        signal_sampling_rate_hz=8,
        n_segments=2,
        tempo_bounds=(1, 1),
    )
    alignment = tempo_dtw_paths(template, signal, result, [7])[0]
    np.testing.assert_array_equal(alignment.block_edges, [0, 4, 8])
    np.testing.assert_array_equal(alignment.tempos, [1, 1])
    assert alignment.cost == 0


@pytest.mark.parametrize("advance", [1, 3])
def test_data_loss_normalization_is_independent_of_uniform_tempo(advance):
    """Identical mismatch over the whole template has a tempo-independent cost."""
    result = tempo_regularized_subsequence_dtw(
        [0.0],
        [2.0] * advance,
        template_sampling_rate_hz=1,
        signal_sampling_rate_hz=1,
        n_segments=1,
        tempo_bounds=(advance, advance),
        tempo_change_penalty=0,
    )
    assert result.costs[-1, -1] == 4


@pytest.mark.parametrize("bounds", [(0, 1), (1, np.inf), (1, 0.5), (0.1, 0.2), (1, 65)])
def test_undefined_or_unbounded_tempo_alphabets_are_rejected(bounds):
    """Zero slopes, infinite slopes, empty and oversized lattices are undefined."""
    with pytest.raises(ValueError, match="tempo"):
        tempo_regularized_subsequence_dtw(
            [0.0],
            [0.0],
            template_sampling_rate_hz=1,
            signal_sampling_rate_hz=1,
            n_segments=1,
            tempo_bounds=bounds,
        )


def test_unit_tempo_boundary_survives_time_conversion_roundoff():
    """Seven bins at 100 Hz must retain the exact unit-tempo advance seven."""
    samples = np.arange(7, dtype=float)
    result = tempo_regularized_subsequence_dtw(
        samples,
        samples,
        template_sampling_rate_hz=100,
        signal_sampling_rate_hz=100,
        n_segments=1,
        tempo_bounds=(1, 1),
    )
    np.testing.assert_array_equal(result.advances, [7])
    assert result.costs[-1, -1] == 0
