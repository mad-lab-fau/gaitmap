"""Regenerate the reference with the documented legacy revision and Python environment."""

# ruff: noqa: INP001

from pathlib import Path

import numpy as np
import pandas as pd
import pomegranate

from gaitmap.example_data import get_healthy_example_imu_data
from gaitmap.stride_segmentation.hmm import HmmStrideSegmentation, PreTrainedRothSegmentationModel, SimpleHmm
from gaitmap.utils.coordinate_conversion import convert_to_fbf


def main():
    """Capture legacy results for both feet and decoding algorithms."""
    if pomegranate.__version__ != "0.14.6":
        raise RuntimeError(
            "Generate this reference with pomegranate 0.14.6 and the documented legacy gaitmap revision."
        )
    data = convert_to_fbf(get_healthy_example_imu_data(), left_like="left_", right_like="right_")
    values = {}
    for algorithm in ["viterbi", "map"]:
        model = PreTrainedRothSegmentationModel().set_params(algo_predict=algorithm)
        if not isinstance(model.model, pomegranate.HiddenMarkovModel):
            raise TypeError("The documented legacy gaitmap revision must be installed to regenerate this reference.")
        result = HmmStrideSegmentation(model, snap_to_min_win_ms=300).segment(data, sampling_rate_hz=204.8)
        for sensor in ["left_sensor", "right_sensor"]:
            prediction = result.result_model_[sensor]
            prefix = f"{sensor}_{algorithm}"
            values[prefix + "_feature_states"] = prediction.hidden_state_sequence_feature_space_.astype(np.int8)
            values[prefix + "_states"] = prediction.hidden_state_sequence_.astype(np.int8)
            values[prefix + "_strides"] = result.stride_list_[sensor].to_numpy()
            values[prefix + "_original_strides"] = result.matches_start_end_original_[sensor]
            values[prefix + "_likelihood"] = prediction.model.log_probability(
                np.ascontiguousarray(prediction.feature_space_data_.to_numpy())
            )
            values[sensor + "_features"] = prediction.feature_space_data_.to_numpy()
    np.savez_compressed(Path(__file__).with_name("hmm_legacy_inference.npz"), **values)
    finite_hmm = pomegranate.HiddenMarkovModel.from_matrix(
        [[0.6, 0.4], [0.0, 0.5]],
        [
            pomegranate.MultivariateGaussianDistribution([-2.0], [[1.0]]),
            pomegranate.MultivariateGaussianDistribution([2.0], [[1.0]]),
        ],
        [1.0, 0.0],
        [0.0, 0.5],
        name="finite_reference",
    )
    finite_model = SimpleHmm(n_states=2, n_gmm_components=1, model=finite_hmm, data_columns=("feature",), verbose=False)
    for algorithm in ["viterbi", "map"]:
        np.testing.assert_array_equal(
            finite_model.predict_hidden_state_sequence(
                pd.DataFrame({"feature": [-2.0, -1.0, 2.0]}), algorithm=algorithm
            ),
            [0, 0, 1],
        )
    Path(__file__).with_name("hmm_legacy_finite.json").write_text(finite_model.to_json())


if __name__ == "__main__":
    main()
