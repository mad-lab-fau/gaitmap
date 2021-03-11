import pytest
from sklearn.model_selection import ParameterGrid

from gaitmap.future.pipelines import GridSearch
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.test_future.test_pipelines.conftest import DummyPipeline, dummy_single_score_func, DummyDataset


class TestMetaFunctionality(TestAlgorithmMixin):
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
    pass
