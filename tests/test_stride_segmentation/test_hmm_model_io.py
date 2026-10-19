"""Saved-model compatibility is tested at the public loading interface."""

import json
from importlib.resources import files
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from gaitmap.stride_segmentation.hmm import GaussianMixture, HiddenMarkovModel, load_legacy_hmm_json

SAVED_MODEL = Path(__file__).parent / "fixtures" / "hmm_finite_saved_model.json"


@pytest.mark.parametrize("algorithm", ["viterbi", "map"])
def test_saved_finite_simple_hmm_keeps_all_emitting_samples(algorithm):
    """Keep the last emitting sample when the model includes explicit termination."""
    serialized = SAVED_MODEL.read_text()
    model = load_legacy_hmm_json(serialized)
    data = pd.DataFrame({"feature": [-2.0, -1.0, 2.0]})

    result = model.predict_hidden_state_sequence(data, algorithm=algorithm)

    assert_array_equal(result, [0, 0, 1])


def test_saved_stride_segmentation_preserves_model_and_postprocessing(tmp_path):
    """Migrate a complete segmentation configuration, including nondefault settings."""
    saved_model = json.loads(
        files("gaitmap_mad.stride_segmentation.hmm._pre_trained_models")
        .joinpath("fallriskpd_at_lab_model.json")
        .read_text()
    )
    saved_model["params"]["name"] = "archived_model"
    serialized = json.dumps(
        {
            "_gaitmap_obj": "HmmStrideSegmentation",
            "params": {"model": saved_model, "snap_to_min_win_ms": 35, "snap_to_min_axis": "gyr_pa"},
        }
    )

    migrated = load_legacy_hmm_json(serialized)
    joblib.dump(migrated.clone(), tmp_path / "model.joblib")
    restored = joblib.load(tmp_path / "model.joblib")

    assert restored.snap_to_min_win_ms == 35
    assert restored.snap_to_min_axis == "gyr_pa"
    assert restored.model.name == "archived_model"
    assert restored.model.model.n_states == 25
    assert restored.model.data_columns == ["raw__gyr_ml", "gradient__gradient__gyr_ml"]


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan])
def test_native_model_loader_rejects_invalid_persisted_probabilities(value):
    """Malformed saved probabilities fail at loading rather than corrupting inference."""
    emission = GaussianMixture([[0.0]], [[[1.0]]], [1.0])
    data = HiddenMarkovModel([emission], [[1.0]], [1.0]).to_dict()
    data["transitions"][0][0] = value

    with pytest.raises(ValueError, match="probabilities"):
        HiddenMarkovModel.from_dict(data)


@pytest.mark.parametrize("unsupported", ["distribution", "silent state", "distribution ties", "transition ties"])
def test_unsupported_persisted_model_features_fail_clearly(unsupported):
    """Unsupported saved model types do not silently change inference semantics."""
    serialized = json.loads(SAVED_MODEL.read_text())["params"]["model"]["hmm"]
    if unsupported == "distribution":
        serialized["states"][0]["distribution"]["name"] = "NormalDistribution"
    elif unsupported == "silent state":
        serialized["states"][0]["distribution"] = None
    elif unsupported == "distribution ties":
        serialized["distribution ties"] = [[0, 0]]
    else:
        serialized["edges"][0][4] = "group"
    with pytest.raises(ValueError, match="supported"):
        HiddenMarkovModel.from_dict(serialized)
