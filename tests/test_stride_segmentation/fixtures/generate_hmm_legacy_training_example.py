"""Capture the training example with the documented legacy revision and environment."""

# ruff: noqa: INP001

from pathlib import Path

import matplotlib
import numpy as np
import pomegranate


def main():
    """Store legacy state labels and stride boundaries for the complete example."""
    matplotlib.use("Agg")
    from examples.stride_segmentation.segmentation_hmm_training import hmm, segmentation_model

    if pomegranate.__version__ != "0.14.6" or not isinstance(segmentation_model.model, pomegranate.HiddenMarkovModel):
        raise RuntimeError(
            "Generate this reference with pomegranate 0.14.6 and the documented legacy gaitmap revision."
        )
    values = {}
    for sensor in ["left_sensor", "right_sensor"]:
        values[sensor + "_states"] = hmm.hidden_state_sequence_[sensor].astype(np.int8)
        values[sensor + "_feature_states"] = hmm.result_model_[sensor].hidden_state_sequence_feature_space_.astype(
            np.int8
        )
        values[sensor + "_strides"] = hmm.stride_list_[sensor].to_numpy()
        values[sensor + "_original_strides"] = hmm.matches_start_end_original_[sensor]
    np.savez_compressed(Path(__file__).with_name("hmm_legacy_training_example.npz"), **values)


if __name__ == "__main__":
    main()
