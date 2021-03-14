from unittest.mock import patch

import pytest
import pandas as pd
from sklearn.model_selection import ParameterGrid

from gaitmap.future.pipelines import GridSearch, Optimize
from gaitmap.future.pipelines._optimize import _BaseOptimize
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.test_future.test_pipelines.conftest import (
    DummyPipeline,
    dummy_single_score_func,
    DummyDataset,
    dummy_multi_score_func,
)


class TestMetaFunctionalityGridSearch(TestAlgorithmMixin):
    __test__ = True
    algorithm_class = GridSearch

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data) -> GridSearch:
        gs = GridSearch(DummyPipeline(), ParameterGrid({"para_1": [1]}), scoring=dummy_single_score_func)
        gs.optimize(DummyDataset())
        return gs

    def test_empty_init(self):
        pytest.skip()

    def test_json_roundtrip(self, after_action_instance):
        # TODO: Implement json serialazable for sklearn objects
        pytest.skip("TODO: This needs to be fixed!")


class TestGridSearch:
    def test_single_score(self):
        gs = GridSearch(DummyPipeline(), ParameterGrid({"para_1": [1, 2]}), scoring=dummy_single_score_func)
        gs.optimize(DummyDataset())
        results = gs.gs_results_
        results_df = pd.DataFrame(results)

        assert len(results_df) == 2  # Parameters
        assert all(
            s in results for s in ["data_labels", "score", "rank_score", "single_score", "params", "param_para_1"]
        )
        assert all(len(v) == 5 for v in results_df["single_score"])  # 5 data points
        assert all(len(v) == 5 for v in results_df["data_labels"])  # 5 data points
        assert list(results["param_para_1"]) == [1, 2]
        assert list(results["params"]) == [{"para_1": 1}, {"para_1": 2}]
        # In this case the dummy scorer returns the same mean value (2) for each para.
        # Therefore, the ranking should be the same.
        assert list(results["rank_score"]) == [1, 1]
        assert list(results["score"]) == [2, 2]

        assert gs.multi_metric_ is False

    def test_multi_score(self):
        gs = GridSearch(
            DummyPipeline(), ParameterGrid({"para_1": [1, 2]}), scoring=dummy_multi_score_func, return_optimized=False
        )
        gs.optimize(DummyDataset())
        results = gs.gs_results_
        results_df = pd.DataFrame(results)

        assert len(results_df) == 2  # Parameters
        assert all(
            s in results
            for s in [
                "data_labels",
                "score_1",
                "rank_score_1",
                "single_score_1",
                "score_2",
                "rank_score_2",
                "single_score_2",
                "params",
                "param_para_1",
            ]
        )
        assert all(len(v) == 5 for c in ["single_score_1", "single_score_2"] for v in results_df[c])  # 5 data points
        assert all(len(v) == 5 for v in results_df["data_labels"])  # 5 data points
        assert list(results["param_para_1"]) == [1, 2]
        assert list(results["params"]) == [{"para_1": 1}, {"para_1": 2}]
        # In this case the dummy scorer returns the same mean value (2) for each para.
        # Therefore, the ranking should be the same.
        assert list(results["rank_score_1"]) == [1, 1]
        assert list(results["rank_score_2"]) == [1, 1]
        assert list(results["score_1"]) == [2, 2]
        assert list(results["score_2"]) == [3, 3]

        assert gs.multi_metric_ is True

    @pytest.mark.parametrize("return_optimized", (True, False, "some_str"))
    def test_return_optimized_single(self, return_optimized):
        gs = GridSearch(
            DummyPipeline(),
            ParameterGrid({"para_1": [1, 2]}),
            scoring=dummy_single_score_func,
            return_optimized=return_optimized,
        )
        warning = None
        if isinstance(return_optimized, str):
            warning = UserWarning

        with pytest.warns(warning) as w:
            gs.optimize(DummyDataset())

        if isinstance(return_optimized, str):
            assert "return_optimize" in str(w[0])
        else:
            assert len(w) == 0

        if return_optimized:  # True or str
            assert gs.best_params_ == {"para_1": 1}
            assert gs.best_index_ == 0
            assert gs.best_score_ == 2
            assert isinstance(gs.optimized_pipeline_, DummyPipeline)
            assert gs.optimized_pipeline_.para_1 == gs.best_params_["para_1"]
        else:
            assert not hasattr(gs, "best_params_")
            assert not hasattr(gs, "best_index_")
            assert not hasattr(gs, "best_score_")
            assert not hasattr(gs, "optimized_pipeline_")

    @pytest.mark.parametrize("return_optimized", (False, "score_1", "score_2"))
    def test_return_optimized_multi(self, return_optimized):
        gs = GridSearch(
            DummyPipeline(),
            ParameterGrid({"para_1": [1, 2]}),
            scoring=dummy_multi_score_func,
            return_optimized=return_optimized,
        )
        gs.optimize(DummyDataset())

        if return_optimized in ("score_1", "score_2"):
            assert gs.best_params_ == {"para_1": 1}
            assert gs.best_index_ == 0
            assert gs.best_score_ == {"score_1": 2, "score_2": 3}[return_optimized]
            assert isinstance(gs.optimized_pipeline_, DummyPipeline)
            assert gs.optimized_pipeline_.para_1 == gs.best_params_["para_1"]
        else:
            assert not hasattr(gs, "best_params_")
            assert not hasattr(gs, "best_index_")
            assert not hasattr(gs, "best_score_")
            assert not hasattr(gs, "optimized_pipeline_")

    @pytest.mark.parametrize("return_optimized", (True, "some_str"))
    def test_return_optimized_multi_exception(self, return_optimized):
        gs = GridSearch(
            DummyPipeline(),
            ParameterGrid({"para_1": [1, 2]}),
            scoring=dummy_multi_score_func,
            return_optimized=return_optimized,
        )

        with pytest.raises(ValueError):
            gs.optimize(DummyDataset())

    @pytest.mark.parametrize("best_value", (1, 2))
    def test_rank(self, best_value):
        def dummy_best_scorer(best):
            def scoring(pipe, ds):
                if pipe.para_1 == best:
                    return 1
                return 0

            return scoring

        paras = [1, 2]
        gs = GridSearch(
            DummyPipeline(),
            ParameterGrid({"para_1": paras}),
            scoring=dummy_best_scorer(best_value),
            return_optimized=True,
        )
        gs.optimize(DummyDataset())

        assert gs.best_score_ == 1
        assert gs.best_index_ == paras.index(best_value)
        assert gs.best_params_ == {"para_1": best_value}
        expected_ranking = [2, 2]
        expected_ranking[paras.index(best_value)] = 1
        assert list(gs.gs_results_["rank_score"]) == expected_ranking


class TestOptimizeBase:
    optimzer: _BaseOptimize

    @pytest.fixture(
        autouse=True, params=(Optimize(DummyPipeline()), GridSearch(DummyPipeline(), ParameterGrid({"para_1": [1]})))
    )
    def optimzer_instance(self, request):
        self.optimizer = request.param

    @pytest.mark.parametrize("method", ("run", "score"))
    def test_run_and_score(self, method):
        ds = DummyDataset()[0]
        return_val = "return_val"
        self.optimizer.optimized_pipeline_ = DummyPipeline()
        with patch.object(DummyPipeline, method, return_value=return_val) as mock_method:
            out = getattr(self.optimizer, method)(ds)

        assert mock_method.called_with(ds)
        assert return_val == out
