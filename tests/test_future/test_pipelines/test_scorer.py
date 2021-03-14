from unittest.mock import Mock

import numpy as np
import pytest

from gaitmap.future.pipelines import GaitmapScorer
from gaitmap.future.pipelines._scorer import _passthrough_scoring, _validate_scorer
from tests.test_future.test_pipelines.conftest import (
    DummyPipeline,
    DummyDataset,
    dummy_single_score_func,
    dummy_multi_score_func,
    dummy_error_score_func,
    dummy_error_score_func_multi,
)


class TestGaitmapScorerCalls:
    scorer: GaitmapScorer

    @pytest.fixture(autouse=True)
    def create_scorer(self):
        self.scorer = GaitmapScorer(lambda x: x)

    def test_score_func_called(self):
        """Test that the score func is called once per dataset"""
        mock_score_func = Mock(return_value=1)
        scorer = GaitmapScorer(mock_score_func)
        pipe = DummyPipeline()
        scorer(pipeline=pipe, data=DummyDataset(), error_score=np.nan)

        assert mock_score_func.call_count == len(DummyDataset())
        for call, d in zip(mock_score_func.call_args_list, DummyDataset()):
            assert call[0][1].groups == d.groups
            assert isinstance(call[0][0], DummyPipeline)
            # Check that the pipeline was cloned before calling
            assert id(pipe) != id(call[0][0])


class TestGaitmapScorer:
    def test_score_return_val_single_score(self):
        scorer = GaitmapScorer(dummy_single_score_func)
        pipe = DummyPipeline()
        data = DummyDataset()
        agg, single = scorer(pipe, data, np.nan)
        assert len(single) == len(data)
        # Our Dummy scorer, returns the groupname of the dataset
        assert all(single == data.groups)
        assert agg == np.mean(data.groups)

    def test_score_return_val_multi_score(self):
        scorer = GaitmapScorer(dummy_multi_score_func)
        pipe = DummyPipeline()
        data = DummyDataset()
        agg, single = scorer(pipe, data, np.nan)
        assert isinstance(single, dict)
        for k, v in single.items():
            assert len(v) == len(data)
            # Our Dummy scorer, returns the groupname of the dataset
            if k == "score_2":
                assert all(v == np.array(data.groups) + 1)
            else:
                assert all(v == data.groups)
        assert isinstance(agg, dict)
        for k, v in agg.items():
            if k == "score_2":
                assert v == np.mean(data.groups) + 1
            else:
                assert v == np.mean(data.groups)

    @pytest.mark.parametrize("err_val", (np.nan, 1))
    def test_scoring_return_err_val(self, err_val):
        scorer = GaitmapScorer(dummy_error_score_func)
        pipe = DummyPipeline()
        data = DummyDataset()
        with pytest.warns(UserWarning) as ws:
            agg, single = scorer(pipe, data, err_val)

        assert len(ws) == 3
        for w, n in zip(ws, [0, 2, 4]):
            assert str(n) in str(w)
            assert str(err_val) in str(w)

        expected = np.array([err_val, 1, err_val, 3, err_val])

        assert len(single) == len(data)
        nan_vals = np.isnan(single)
        assert all(nan_vals == np.isnan(expected))
        assert all(single[~nan_vals] == expected[~nan_vals])

        # agg should become nan if a single value is nan
        if sum(nan_vals) > 0:
            assert np.isnan(agg)
        else:
            assert agg == np.mean(expected)

    @pytest.mark.parametrize("err_val", (np.nan, 1))
    def test_scoring_return_err_val_multi(self, err_val):
        scorer = GaitmapScorer(dummy_error_score_func_multi)
        pipe = DummyPipeline()
        data = DummyDataset()
        agg, single = scorer(pipe, data, err_val)

        expected = np.array([err_val, 1, err_val, 3, err_val])

        for v in single.values():
            assert len(v) == len(data)
            nan_vals = np.isnan(v)
            assert all(nan_vals == np.isnan(expected))
            assert all(v[~nan_vals] == expected[~nan_vals])

        for v in agg.values():
            # agg should become nan if a single value is nan
            if sum(nan_vals) > 0:
                assert np.isnan(v)
            else:
                assert v == np.mean(expected)

    def test_err_val_raises(self):
        scorer = GaitmapScorer(dummy_error_score_func)
        pipe = DummyPipeline()
        data = DummyDataset()
        with pytest.raises(ValueError) as e:
            scorer(pipe, data, "raise")

        assert str(e.value) == "Dummy Error for 0"


def _dummy_func(x):
    return x


def _dummy_func_2(x):
    return x


class TestScorerUtils:
    @pytest.mark.parametrize(
        "scoring, expected",
        (
            (None, GaitmapScorer(_passthrough_scoring)),
            (_dummy_func, GaitmapScorer(_dummy_func)),
            (GaitmapScorer(_dummy_func_2), GaitmapScorer(_dummy_func_2)),
        ),
    )
    def test_validate_scorer(self, scoring, expected):
        out = _validate_scorer(scoring)
        assert isinstance(out, type(expected))
        assert out._score_func == expected._score_func
