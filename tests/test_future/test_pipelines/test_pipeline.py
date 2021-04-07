"""Test basic pipeline functionality."""
from unittest.mock import patch

import pytest

from gaitmap.future.dataset import Dataset
from gaitmap.future.pipelines import SimplePipeline
from tests.test_future.test_pipelines.conftest import DummyDataset, DummyPipeline


class PipelineInputModify(SimplePipeline):
    def __init__(self, test="a value"):
        self.test = test

    def run(self, datapoint: Dataset):
        self.test = "another value"
        return self


class PipelineInputModifyNested(SimplePipeline):
    def __init__(self, pipe=PipelineInputModify()):
        self.pipe = pipe

    def run(self, datapoint: Dataset):
        self.pipe.run(datapoint)
        return self


class PipelineNoOutput(SimplePipeline):
    def __init__(self, test="a value"):
        self.test = test

    def run(self, datapoint: Dataset):
        self.not_a_output_paras = "something"
        return self


class TestSafeRun:
    @pytest.mark.parametrize("pipe", (PipelineInputModify, PipelineInputModifyNested))
    def test_modify_input_paras_simple(self, pipe):
        with pytest.raises(ValueError) as e:
            pipe().safe_run(DummyDataset()[0])

        assert "Running the pipeline did modify the parameters of the pipeline." in str(e)

    def test_no_self_return(self):
        pipe = DummyPipeline()
        pipe.run = lambda d: "some Value"
        with pytest.raises(ValueError) as e:
            pipe.safe_run(DummyDataset()[0])

        assert "The `run` method of the pipeline must return `self`" in str(e)

    def test_no_output(self):
        with pytest.raises(ValueError) as e:
            PipelineNoOutput().safe_run(DummyDataset()[0])

        assert "Running the pipeline did not set any results on the output." in str(e)

    def test_output(self):
        pipe = DummyPipeline()
        pipe.result_ = "some result"
        ds = DummyDataset()[0]
        with patch.object(DummyPipeline, "run", return_value=pipe) as mock:
            result = DummyPipeline().safe_run(ds)

        mock.assert_called_with(ds)
        assert id(result) == id(pipe)
