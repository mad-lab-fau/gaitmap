"""Saved-model compatibility is tested at the public loading interface."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from gaitmap.stride_segmentation.hmm import GaussianMixture, HiddenMarkovModel, SimpleHmm

SAVED_MODEL = Path(__file__).parent / "fixtures" / "hmm_finite_saved_model.json"


@pytest.mark.parametrize("algorithm", ["viterbi", "map"])
def test_saved_finite_simple_hmm_keeps_all_emitting_samples(algorithm):
    """Keep the last emitting sample when the model includes explicit termination."""
    serialized = SAVED_MODEL.read_text()
    model = SimpleHmm.from_json(serialized)
    data = pd.DataFrame({"feature": [-2.0, -1.0, 2.0]})

    result = model.predict_hidden_state_sequence(data, algorithm=algorithm)

    assert_array_equal(result, [0, 0, 1])


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
