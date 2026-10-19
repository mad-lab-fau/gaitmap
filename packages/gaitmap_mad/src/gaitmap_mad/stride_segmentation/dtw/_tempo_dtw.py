"""Exact alignment on an experimental positive block-tempo lattice.

See ``docs/source/development/tempo_regularized_dtw.rst`` for the model.
This module deliberately does not change the existing DTW solvers.
"""

from typing import NamedTuple

import numpy as np
from numba import njit


class TempoDtwResult(NamedTuple):
    """Projected block costs and endpoint decisions, without full state history.

    Columns use inclusive sample indices; internal positions are exclusive edges.
    Interior rows are unrestricted partial optima. Duration limits, if supplied,
    apply only to the complete-template final row. ``fallback_count`` counts
    duration-invalid winners recomputed with endpoint-specific start constraints.
    """

    costs: np.ndarray
    endpoint_starts: np.ndarray
    endpoint_states: np.ndarray
    advances: np.ndarray
    tempos: np.ndarray
    tempo_change_penalty: float
    fallback_count: int


class TempoDtwAlignment(NamedTuple):
    """A complete weighted alignment; all costs here are squared objectives.

    ``path`` contains sample-index pairs, ``weights`` their fractions of template
    time, and ``block_edges`` exclusive signal edges. The weights sum to one.
    ``cost = data_cost + tempo_cost``; segmentation reports its square root.
    """

    path: np.ndarray
    weights: np.ndarray
    block_edges: np.ndarray
    tempos: np.ndarray
    data_cost: float
    tempo_cost: float
    cost: float


def _arrays(template, signal):
    x, y = np.asarray(template, dtype=float), np.asarray(signal, dtype=float)
    x, y = x.reshape(len(x), -1), y.reshape(len(y), -1)
    if not len(x) or not len(y) or x.shape[1] != y.shape[1] or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Template and signal must be nonempty finite arrays with equal feature counts.")
    return x, y


def _overlaps(m, k_count, advances, n):
    """Precompute exact bin intersections using integer boundary coordinates."""
    indices, offsets, weights = [], [], []
    pointers = np.empty((k_count, len(advances) + 1), dtype=np.int64)
    for k in range(k_count):
        for q, advance in enumerate(advances):
            pointers[k, q] = len(weights)
            r = int(advance)
            if r > n:
                continue
            position, stop = 0, m * r
            while position < stop:
                i = (k * m * r + position) // (k_count * r)
                j = position // m
                next_position = min(((i + 1) * k_count - k * m) * r, (j + 1) * m, stop)
                indices.append(i)
                offsets.append(j)
                weights.append((next_position - position) / (stop * k_count))
                position = next_position
        pointers[k, -1] = len(weights)
    return pointers, np.asarray(indices, dtype=np.int64), np.asarray(offsets, dtype=np.int64), np.asarray(weights)


@njit(cache=True)
def _block_loss(x, y, start, first, last, indices, offsets, weights):
    loss = 0.0
    for a in range(first, last):
        squared = 0.0
        for feature in range(x.shape[1]):
            squared += (x[indices[a], feature] - y[start + offsets[a], feature]) ** 2
        loss += weights[a] * squared
    return loss


@njit(cache=True)
def _run(x, y, advances, penalties, overlaps, start_low, start_high, trace):  # noqa: C901, PLR0912
    pointers, indices, offsets, weights = overlaps
    k_count, q_count = pointers.shape[0], len(advances)
    n = len(y)
    lookback = min(int(advances[-1]), n) + 1
    ring = np.full((lookback, k_count, q_count), np.inf)
    starts = np.full((lookback, k_count, q_count), n + 1, dtype=np.int64)
    costs = np.full((k_count, n), np.inf)
    endpoint_starts = np.full(n, -1, dtype=np.int64)
    endpoint_states = np.zeros(n, dtype=np.uint8)
    parents = np.zeros((k_count, n if trace else 0, q_count), dtype=np.uint8)
    for j in range(1, n + 1):
        column = j % lookback
        ring[column, :, :] = np.inf
        starts[column, :, :] = n + 1
        for k in range(k_count):
            best, best_start, best_state = np.inf, n + 1, 0
            for q in range(q_count):
                b = j - advances[q]
                if b < 0:
                    continue
                value, initial, parent = np.inf, n + 1, 0
                if k == 0:
                    if start_low <= b <= start_high:
                        value, initial = 0.0, b
                else:
                    previous = b % lookback
                    for p in range(q_count):
                        candidate = ring[previous, k - 1, p] + penalties[p, q]
                        candidate_start = starts[previous, k - 1, p]
                        if candidate < value or (candidate == value and candidate_start < initial):
                            value, initial, parent = candidate, candidate_start, p
                if not np.isfinite(value):
                    continue
                value += _block_loss(x, y, b, pointers[k, q], pointers[k, q + 1], indices, offsets, weights)
                ring[column, k, q], starts[column, k, q] = value, initial
                if trace:
                    parents[k, j - 1, q] = parent
                if value < best or (value == best and initial < best_start):
                    best, best_start, best_state = value, initial, q
            costs[k, j - 1] = best
            if k == k_count - 1 and np.isfinite(best):
                endpoint_starts[j - 1], endpoint_states[j - 1] = best_start, best_state
    return costs, endpoint_starts, endpoint_states, parents


def tempo_regularized_subsequence_dtw(
    template,
    signal,
    *,
    template_sampling_rate_hz,
    signal_sampling_rate_hz,
    n_segments=20,
    tempo_bounds=(0.5, 2.0),
    tempo_change_penalty=0.1,
    min_duration_samples=None,
    max_duration_samples=None,
) -> TempoDtwResult:
    """Align a whole template to free signal starts on a finite tempo lattice.

    Samples represent constant time bins. K equal template-time blocks consume
    positive integer signal advances. Actual tempos are signal seconds per
    template second, within ``tempo_bounds``. At most 64 tempo states are allowed;
    choose more segments to reduce the alphabet rather than silently prune it.

    The objective is mean squared feature mismatch over template time plus
    ``tempo_change_penalty * sum(diff(log(tempos)) ** 2)``. The first tempo is
    unpenalized. The coefficient has squared feature units and no implicit grid
    scaling. Zero penalty recovers this block lattice, not ordinary DTW.

    Optional duration limits are inclusive positive integer signal-sample counts.
    A duration-invalid unrestricted winner triggers exact recomputation with all
    feasible starts for its fixed endpoint. Worst-case time grows by a factor up
    to K * max(advances), while state memory remains bounded. ``fallback_count``
    exposes how often this happened.

    Floating state costs use bounded lookback; traceback deliberately recomputes
    requested paths with :func:`tempo_dtw_paths`. No full float state tensor is
    retained. Ties prefer earliest starts then ascending tempo indices.
    """
    x, y = _arrays(template, signal)
    if not isinstance(n_segments, (int, np.integer)) or n_segments < 1:
        raise ValueError("n_segments must be a positive integer.")
    rates = np.asarray([template_sampling_rate_hz, signal_sampling_rate_hz], dtype=float)
    bounds = np.asarray(tempo_bounds, dtype=float)
    if np.any(~np.isfinite(rates)) or np.any(rates <= 0):
        raise ValueError("Sampling rates must be finite and positive.")
    if bounds.shape != (2,) or np.any(~np.isfinite(bounds)) or not 0 < bounds[0] <= bounds[1]:
        raise ValueError("tempo_bounds must be ordered finite positive values.")
    if not np.isfinite(tempo_change_penalty) or tempo_change_penalty < 0:
        raise ValueError("tempo_change_penalty must be finite and nonnegative.")
    signal_per_block = (rates[1] / rates[0]) * (len(x) / n_segments)
    scaled = bounds * signal_per_block
    # Only arithmetic roundoff at integer boundaries is tolerated, not a grid
    # approximation or pruning. Eight ulps cover the rate/product operations.
    tolerance = 8 * np.spacing(scaled)
    low = max(1, int(np.ceil(scaled[0] - tolerance[0])))
    high = int(np.floor(scaled[1] + tolerance[1]))
    if high < low or high - low + 1 > 64:
        raise ValueError("The tempo alphabet must contain 1 to 64 states; adjust n_segments or bounds.")
    advances = np.arange(low, high + 1, dtype=np.int64)
    tempos = advances / signal_per_block
    penalties = tempo_change_penalty * (np.log(tempos[:, None]) - np.log(tempos[None, :])) ** 2
    overlaps = _overlaps(len(x), n_segments, advances, len(y))
    costs, starts, states, _ = _run(x, y, advances, penalties, overlaps, 0, len(y), False)
    fallback_count = _apply_duration_limits(
        x, y, advances, penalties, overlaps, costs, starts, states, min_duration_samples, max_duration_samples
    )
    return TempoDtwResult(costs, starts, states, advances, tempos, tempo_change_penalty, fallback_count)


def _apply_duration_limits(x, y, advances, penalties, overlaps, costs, starts, states, minimum, maximum):
    min_duration_samples, max_duration_samples = minimum, maximum
    minimum = (
        len(costs) * advances[0]
        if min_duration_samples is None
        else max(len(costs) * advances[0], min_duration_samples)
    )
    maximum = (
        len(costs) * advances[-1]
        if max_duration_samples is None
        else min(len(costs) * advances[-1], max_duration_samples)
    )
    fallback_count = 0
    for end in range(len(y)):
        if not np.isfinite(costs[-1, end]):
            continue
        duration = end + 1 - starts[end]
        if minimum <= duration <= maximum:
            continue
        left, last_start = max(0, end + 1 - maximum), end + 1 - minimum
        costs[-1, end], starts[end] = np.inf, -1
        if last_start < left:
            continue
        # Endpoint-specific allowed starts all have the same future validity.
        # Merging their equal-tempo arrivals is now safe, unlike duration filtering
        # after the unrestricted traceback. No float history is retained.
        local, local_starts, local_states, _ = _run(
            x,
            y[left : end + 1],
            advances,
            penalties,
            overlaps,
            0,
            last_start - left,
            False,
        )
        fallback_count += 1
        costs[-1, end] = local[-1, -1]
        if np.isfinite(costs[-1, end]):
            starts[end], states[end] = left + local_starts[-1], local_states[-1]
    return fallback_count


def tempo_dtw_paths(template, signal, result: TempoDtwResult, endpoints) -> list[TempoDtwAlignment]:
    """Recompute selected paths; endpoints are inclusive signal sample indices.

    Inputs must be the same samples used to calculate ``result``. Unreachable
    endpoints raise ValueError instead of producing a fabricated path.
    """
    x, y = _arrays(template, signal)
    k_count = len(result.costs)
    overlaps = _overlaps(len(x), k_count, result.advances, len(y))
    pointers, indices, offsets, weights = overlaps
    penalties = result.tempo_change_penalty * (np.log(result.tempos[:, None]) - np.log(result.tempos[None, :])) ** 2
    alignments = []
    for endpoint in endpoints:
        end = int(endpoint)
        if not 0 <= end < len(y) or not np.isfinite(result.costs[-1, end]):
            raise ValueError("Cannot trace an unreachable endpoint.")
        start = int(result.endpoint_starts[end])
        _, _, _, parents = _run(x, y[start : end + 1], result.advances, penalties, overlaps, 0, 0, True)
        edges = np.empty(k_count + 1, dtype=np.int64)
        states = np.empty(k_count, dtype=np.int64)
        j, q = end - start + 1, int(result.endpoint_states[end])
        edges[-1] = end + 1
        for k in range(k_count - 1, -1, -1):
            states[k] = q
            parent = int(parents[k, j - 1, q])
            j -= result.advances[q]
            edges[k] = j + start
            q = parent
        pairs, path_weights = [], []
        for k, q in enumerate(states):
            for a in range(pointers[k, q], pointers[k, q + 1]):
                pairs.append((indices[a], edges[k] + offsets[a]))
                path_weights.append(weights[a])
        path = np.asarray(pairs, dtype=np.int64)
        path_weights = np.asarray(path_weights)
        data_cost = float(np.sum(path_weights * np.sum((x[path[:, 0]] - y[path[:, 1]]) ** 2, axis=1)))
        tempos = result.tempos[states]
        tempo_cost = float(result.tempo_change_penalty * np.sum(np.diff(np.log(tempos)) ** 2))
        alignments.append(
            TempoDtwAlignment(path, path_weights, edges, tempos, data_cost, tempo_cost, data_cost + tempo_cost)
        )
    return alignments
