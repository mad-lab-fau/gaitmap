from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from gaitmap.future.dataset import Dataset
from gaitmap.future.pipelines import GaitmapScorer, SimplePipeline


class DummyPipeline(SimplePipeline):
    pass


class DummyDataset(Dataset):
    def create_index(self) -> pd.DataFrame:
        return pd.DataFrame({"value": list(range(5))})


def dummy_single_score_func(pipeline, data_point):
    return data_point.groups[0]


def dummy_multi_score_func(pipeline, data_point):
    return {"score_1": data_point.groups[0], "score_2": data_point.groups[0]}


def dummy_error_score_func(pipeline, data_point):
    if data_point.groups[0] in [0, 2, 4]:
        raise ValueError("Dummy Error for {}".format(data_point.groups[0]))
    return data_point.groups[0]


def dummy_error_score_func_multi(pipeline, data_point):
    tmp = dummy_error_score_func(pipeline, data_point)
    return {"score_1": tmp, "score_2": tmp}


@pytest.fixture(params=(dummy_single_score_func, dummy_multi_score_func))
def scorer(request):
    return request.param


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
    # scorer: GaitmapScorer
    #
    # @pytest.fixture(autouse=True)
    # def create_scorer(self, scorer):
    #     self.scorer = GaitmapScorer(scorer)

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
        for v in single.values():
            assert len(v) == len(data)
            # Our Dummy scorer, returns the groupname of the dataset
            assert all(v == data.groups)
        assert isinstance(agg, dict)
        for v in agg.values():
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
