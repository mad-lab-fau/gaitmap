"""Exact bounded-run subsequence DTW with rolling costs and compact traceback."""

from typing import NamedTuple

import numpy as np
from numba import njit

from gaitmap_mad.stride_segmentation.dtw._vendored_tslearn import _local_squared_dist


class ConstrainedDtwResult(NamedTuple):
    """Cost matrix and the information needed to reconstruct its selected paths."""

    costs: np.ndarray
    predecessors: np.ndarray
    endpoint_states: np.ndarray
    n_vertical: int
    vertical_unbounded: bool
    horizontal_unbounded: bool


def constrained_subsequence_dtw(subseq, longseq, max_vertical, max_horizontal):
    """Keep the cheapest arrival for every direction/run-length state.

    States are ordered diagonal/start, vertical runs, horizontal runs. Strict
    comparisons retain that order on ties, with shorter runs first. A direction
    whose limit cannot bind needs only one state, including self-transitions.

    Only two columns of state costs are retained. Traceback stores predecessor
    state IDs for diagonal, vertical-run-1 and horizontal-run-1 arrivals. Longer
    runs have an implicit predecessor in the same direction with length r - 1.
    """
    m, n = len(subseq), len(longseq)
    vertical_unbounded = max_vertical >= m - 1
    horizontal_unbounded = max_horizontal >= n - 1
    n_vertical = min(1, max(m - 1, 0)) if vertical_unbounded else int(max_vertical)
    n_horizontal = min(1, max(n - 1, 0)) if horizontal_unbounded else int(max_horizontal)
    n_states = 1 + n_vertical + n_horizontal
    state_dtype = np.min_scalar_type(n_states - 1)
    predecessors = np.zeros((m, n, 3), dtype=state_dtype)
    endpoint_states = np.empty(n, dtype=state_dtype)
    costs = _accumulate(
        subseq,
        longseq,
        n_vertical,
        n_horizontal,
        vertical_unbounded,
        horizontal_unbounded,
        predecessors,
        endpoint_states,
    )
    return ConstrainedDtwResult(
        costs, predecessors, endpoint_states, n_vertical, vertical_unbounded, horizontal_unbounded
    )


@njit(cache=True)
def _accumulate(  # noqa: C901, PLR0912, PLR0915
    subseq, longseq, n_vertical, n_horizontal, vertical_unbounded, horizontal_unbounded, predecessors, endpoint_states
):
    m, n = len(subseq), len(longseq)
    n_states = 1 + n_vertical + n_horizontal
    horizontal_start = 1 + n_vertical
    previous = np.full((m, n_states), np.inf)
    current = np.full((m, n_states), np.inf)
    # Three summaries per cell: best overall, best excluding vertical arrivals,
    # and best excluding horizontal arrivals. They avoid rescanning predecessors
    # for each new state and retain the same deterministic tie order.
    previous_min = np.full((m, 3), np.inf)
    current_min = np.full((m, 3), np.inf)
    previous_best = np.zeros((m, 3), dtype=np.int64)
    current_best = np.zeros((m, 3), dtype=np.int64)
    costs = np.empty((m, n))

    for j in range(n):
        for i in range(m):
            local = _local_squared_dist(subseq[i], longseq[j])
            # A match may start at any signal sample without consuming a step.
            diagonal_cost = np.inf
            diagonal_parent = 0
            if i == 0:
                diagonal_cost = 0.0
            elif j > 0:
                diagonal_cost = previous_min[i - 1, 0]
                diagonal_parent = previous_best[i - 1, 0]
            current[i, 0] = local + diagonal_cost
            predecessors[i, j, 0] = diagonal_parent

            if n_vertical:
                summary = 0 if vertical_unbounded else 1
                parent = 0
                value = np.inf
                if i > 0:
                    value = current_min[i - 1, summary]
                    parent = current_best[i - 1, summary]
                current[i, 1] = local + value
                predecessors[i, j, 1] = parent
                for run in range(2, n_vertical + 1):
                    current[i, run] = local + current[i - 1, run - 1] if i > 0 else np.inf

            if n_horizontal:
                summary = 0 if horizontal_unbounded else 2
                parent = 0
                value = np.inf
                if j > 0:
                    value = previous_min[i, summary]
                    parent = previous_best[i, summary]
                current[i, horizontal_start] = local + value
                predecessors[i, j, 2] = parent
                for run in range(2, n_horizontal + 1):
                    state = horizontal_start + run - 1
                    current[i, state] = local + previous[i, state - 1] if j > 0 else np.inf

            for summary in range(3):
                current_min[i, summary] = current[i, 0]
                current_best[i, summary] = 0
            for state in range(1, n_states):
                value = current[i, state]
                if value < current_min[i, 0]:
                    current_min[i, 0] = value
                    current_best[i, 0] = state
                summary = 2 if state <= n_vertical else 1
                if value < current_min[i, summary]:
                    current_min[i, summary] = value
                    current_best[i, summary] = state
            costs[i, j] = current_min[i, 0]
        endpoint_states[j] = current_best[m - 1, 0]
        previous, current = current, previous
        previous_min, current_min = current_min, previous_min
        previous_best, current_best = current_best, previous_best
    return costs


def constrained_paths(result: ConstrainedDtwResult, endpoints: np.ndarray) -> list[np.ndarray]:
    """Follow the forward decisions, including implicit longer-run predecessors."""
    return [
        np.asarray(
            _traceback(
                result.predecessors,
                int(end),
                int(result.endpoint_states[end]),
                result.n_vertical,
                result.vertical_unbounded,
                result.horizontal_unbounded,
            )
        )
        for end in endpoints
    ]


@njit(cache=True)
def _traceback(predecessors, end, state, n_vertical, vertical_unbounded, horizontal_unbounded):
    i, j = predecessors.shape[0] - 1, end
    path = [(i, j)]
    while not (i == 0 and state == 0):
        if state == 0:
            state = int(predecessors[i, j, 0])
            i -= 1
            j -= 1
        elif state <= n_vertical:
            state = int(predecessors[i, j, 1]) if state == 1 or vertical_unbounded else state - 1
            i -= 1
        else:
            first_horizontal = n_vertical + 1
            state = int(predecessors[i, j, 2]) if state == first_horizontal or horizontal_unbounded else state - 1
            j -= 1
        path.append((i, j))
    return path[::-1]
