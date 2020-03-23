import numpy as np
import pytest

from gaitmap.stride_segmentation.BarthDtw import BarthDtw


@pytest.fixture(params=list(BarthDtw._allowed_methods_map.keys()))
def method(request):
    return request.param


def test_sdtw_simple_multi_match(method):
    template = np.array([0, 1.0, 0])
    sequence = [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]

    dtw = BarthDtw(
        template=template,
        template_sampling_rate_hz=100.0,
        max_cost=0.5,
        min_stride_time_s=None,
        find_matches_method=method,
    )
    dtw = dtw.segment(np.array(sequence), sampling_rate=100.0,)

    np.testing.assert_array_equal(dtw.paths_, [[(0, 5), (1, 6), (2, 7)]])
    assert dtw.costs_ == [0.0]
    np.testing.assert_array_equal(dtw.paths_start_end_, [[5, 7]])
    np.testing.assert_array_equal(
        dtw.acc_cost_mat_,
        [
            [4.0, 4.0, 4.0, 4.0, 4.0, 0.0, 1.0, 0.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [9.0, 9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ],
    )

    np.testing.assert_array_equal(dtw.data, sequence)
    np.testing.assert_array_equal(dtw.template, template)


def test_sdtw_multi_match(method):
    template = np.array([0, 1.0, 0])
    sequence = 2 * [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]

    dtw = BarthDtw(
        template=template,
        template_sampling_rate_hz=100.0,
        max_cost=0.5,
        min_stride_time_s=None,
        find_matches_method=method,
    )
    dtw = dtw.segment(np.array(sequence), sampling_rate=100.0,)

    np.testing.assert_array_equal(dtw.paths_, [[(0, 5), (1, 6), (2, 7)], [(0, 18), (1, 19), (2, 20)]])
    np.testing.assert_array_equal(dtw.paths_start_end_, [[5, 7], [18, 20]])
    np.testing.assert_array_equal(dtw.costs_, [0.0, 0.0])

    np.testing.assert_array_equal(dtw.data, sequence)
    np.testing.assert_array_equal(dtw.template, template)


# TODO; Test template interpolate
# TODO; TEST min distance
# TODO: Test max_cost
