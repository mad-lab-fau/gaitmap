"""Check pooled challenge metrics through tpcp's scoring interface."""

import pandas as pd
import pytest
from tpcp import Dataset, Pipeline
from tpcp.validate import Scorer

from gaitmap.parameters.evaluation import SingleValueErrors
from gaitmap.pipeline.evaluation import ParameterErrors
from gaitmap.stride_segmentation.evaluation import SingleValuePrecisionRecallF1


class Recordings(Dataset):
    def create_index(self):
        return pd.DataFrame({"recording": ["a", "b"]})


def test_precision_recall_pool_matches_instead_of_recording_scores():
    def score(_pipeline, datapoint):
        matches = ["tp", "tp", "fp"] if datapoint.group_label.recording == "a" else ["fn"]
        return {"pooled": SingleValuePrecisionRecallF1(return_raw_scores=False)(pd.DataFrame({"match_type": matches}))}

    aggregate, raw = Scorer(score, progress_bar=False)(Pipeline(), Recordings())
    assert aggregate == pytest.approx({"pooled__precision": 2 / 3, "pooled__recall": 2 / 3, "pooled__f1_score": 2 / 3})
    assert raw == {}


def test_parameter_errors_weight_each_stride_equally():
    def score(_pipeline, datapoint):
        reference = [1.0, 1.0] if datapoint.group_label.recording == "a" else [1.0]
        predicted = [2.0, 2.0] if datapoint.group_label.recording == "a" else [5.0]
        frames = {
            name: {"left_sensor": pd.DataFrame({"stride_length": values}).rename_axis("s_id")}
            for name, values in [("reference", reference), ("predicted", predicted)]
        }
        return {"pooled": SingleValueErrors(return_raw_scores=False)(("stride_length", frames))}

    aggregate, _ = Scorer(score, progress_bar=False)(Pipeline(), Recordings())
    assert aggregate["pooled__error_mean"] == pytest.approx(2.0)
    assert aggregate["pooled__abs_error_mean"] == pytest.approx(2.0)


def test_pipeline_parameter_errors_compare_recordings_with_distinct_labels():
    def score(_pipeline, datapoint):
        predicted = 1.0 if datapoint.group_label.recording == "a" else 4.0
        return {
            "pooled": ParameterErrors(return_raw_scores=False)(
                {"reference": pd.Series({"stride_length": 2.0}), "predicted": pd.Series({"stride_length": predicted})}
            )
        }

    aggregate, _ = Scorer(score, progress_bar=False)(Pipeline(), Recordings())
    assert aggregate["pooled__stride_length__error_mean"] == pytest.approx(0.5)
    assert aggregate["pooled__stride_length__abs_error_mean"] == pytest.approx(1.5)
