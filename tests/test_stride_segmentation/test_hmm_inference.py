"""Public HMM decoding and saved-model compatibility regressions."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from gaitmap.stride_segmentation.hmm import (
    GaussianMixture,
    HiddenMarkovModel,
    PreTrainedRothSegmentationModel,
    RothHmmFeatureTransformer,
    SimpleHmm,
)
from gaitmap.utils.coordinate_conversion import convert_to_fbf


@pytest.mark.parametrize("n_states", [1, 2])
@pytest.mark.parametrize("algorithm", ["viterbi", "map"])
def test_public_prediction_keeps_every_observation_in_a_short_valid_path(n_states, algorithm):
    """A valid one-sample path survives wrappers even with more states than observations."""
    emissions = [GaussianMixture([[float(state)]], [[[1.0]]], [1.0]) for state in range(n_states)]
    model = SimpleHmm(
        n_states,
        1,
        model=HiddenMarkovModel(emissions, np.eye(n_states), np.eye(n_states)[0]),
        data_columns=("x",),
    )

    assert_array_equal(model.predict_hidden_state_sequence(pd.DataFrame({"x": [0.0]}), algorithm=algorithm), [0])


@pytest.mark.parametrize("states", [[0], [0, 1, 2], [1, 1, 0, 2, 3]])
def test_inverse_feature_transform_preserves_full_length_identity(states):
    """Decoding without downsampling needs one interpolation coordinate per label."""
    actual = RothHmmFeatureTransformer().inverse_transform_state_sequence(
        np.asarray(states), data=np.zeros(len(states))
    )

    assert_array_equal(actual, states)


@pytest.mark.parametrize("restore", ["clone", "json"])
def test_saved_and_cloned_pretrained_models_preserve_predictions(healthy_example_imu_data, restore):
    """Saving or cloning a trained model preserves the decoded gait sequence."""
    original = PreTrainedRothSegmentationModel()
    restored = original.clone() if restore == "clone" else type(original).from_json(original.to_json())
    data = convert_to_fbf(healthy_example_imu_data, left_like="left_", right_like="right_")["left_sensor"]

    expected = original.predict(data, sampling_rate_hz=204.8).hidden_state_sequence_
    actual = restored.predict(data, sampling_rate_hz=204.8).hidden_state_sequence_

    assert_array_equal(actual, expected)


def test_public_hmm_training_prediction_and_serialization_do_not_require_pomegranate():
    """Train, load and save HMMs when the old dependency is unavailable."""
    code = """
import importlib.abc
import sys

class RejectPomegranate(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "pomegranate":
            raise AssertionError("Inference attempted to import pomegranate")

sys.meta_path.insert(0, RejectPomegranate())
import numpy as np
import pandas as pd
from gaitmap.stride_segmentation.hmm import PreTrainedRothSegmentationModel, SimpleHmm
model = PreTrainedRothSegmentationModel()
model.clone()
type(model).from_json(model.to_json())
trained = SimpleHmm(2, 1, algo_train="labeled", max_iterations=1, verbose=False)
trained.self_optimize([pd.DataFrame({"feature": [-3., -1., 1., 3.]})], [np.array([0, 0, 1, 1])])
restored = SimpleHmm.from_json(trained.to_json()).clone()
np.testing.assert_array_equal(restored.predict_hidden_state_sequence(pd.DataFrame({"feature": [-3., -1., 1., 3.]})), [0, 0, 1, 1])
"""

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
