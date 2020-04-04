import numpy as np
import pytest

from gaitmap.base import BaseType
from gaitmap.stride_segmentation.base_dtw import BaseDtw
from gaitmap.stride_segmentation.dtw_templates import create_dtw_template
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin


class TestMetaFunctionality(TestAlgorithmMixin):
    algorithm_class = BaseDtw
    __test__ = True

    @pytest.fixture()
    def after_action_instance(self) -> BaseType:
        template = create_dtw_template(np.array([0, 1.0, 0]), sampling_rate_hz=100.0)
        dtw = BaseDtw(template=template, max_cost=0.5, min_stride_time_s=None,)
        data = np.array([0, 1.0, 0])
        dtw.segment(data, sampling_rate_hz=100)
        return dtw


class TestSimpleSegment:
    template = create_dtw_template(np.array([0, 1.0, 0]), sampling_rate_hz=100.0)

    @pytest.fixture(params=list(BaseDtw._allowed_methods_map.keys()), autouse=True)
    def _create_instance(self, request):
        dtw = BaseDtw(template=self.template, max_cost=0.5, min_stride_time_s=None, find_matches_method=request.param,)
        self.dtw = dtw

    def test_sdtw_simple_multi_match(self):
        sequence = [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]

        dtw = self.dtw.segment(np.array(sequence), sampling_rate_hz=100.0,)

        np.testing.assert_array_equal(dtw.paths_, [[(0, 5), (1, 6), (2, 7)]])
        assert dtw.costs_ == [0.0]
        np.testing.assert_array_equal(dtw.matches_start_end_, [[5, 7]])
        np.testing.assert_array_equal(
            dtw.acc_cost_mat_,
            [
                [4.0, 4.0, 4.0, 4.0, 4.0, 0.0, 1.0, 0.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [9.0, 9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ],
        )

        np.testing.assert_array_equal(dtw.data, sequence)

    def test_sdtw_multi_match(self):
        sequence = 2 * [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]

        dtw = self.dtw.segment(np.array(sequence), sampling_rate_hz=100.0,)

        np.testing.assert_array_equal(dtw.paths_, [[(0, 5), (1, 6), (2, 7)], [(0, 18), (1, 19), (2, 20)]])
        np.testing.assert_array_equal(dtw.matches_start_end_, [[5, 7], [18, 20]])
        np.testing.assert_array_equal(dtw.costs_, [0.0, 0.0])

        np.testing.assert_array_equal(dtw.data, sequence)


# TODO; Test template interpolate
# TODO; TEST min distance
# TODO: Test max_cost
