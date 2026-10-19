r"""
.. _example_custom_hierarchical_hmms:

Custom hierarchical HMMs
=======================

This example demonstrates how to train and combine three HMM submodels, each
with multiple internal states, into one segmentation model.

.. warning::

   We do not recommend using this HMM to detect stance/swing transitions without
   further validation.

With suitable annotations and features, you can also train models for movement
classes such as walking, stair walking, and rest, or walking, running, and rest.
For published multiclass approaches, Roth et al. [1]_ combine stride models for
level walking, stair ascent, and stair descent with a transition model, while
Martindale et al. [2]_ model walking, running, and rest. Those approaches use
multiple internal states per movement class. This tutorial follows that
composition pattern using three phase submodels and the available walking data.

Train stance, swing, and transition/null HMMs on both feet of the healthy walking
recording, converted to the same foot body frame.
The stance and swing submodels each have five internal states; the transition/null
submodel has three. The combined model therefore has 13 hidden states.
As in the Roth setup, everything outside the annotated strides belongs to the
transition/null class.
Within each stride, initial contact separates swing from stance. These labels
are derived from the supplied annotations; no extra rest annotations are needed.

.. [1] Roth, N., Küderle, A., Prossel, D., Gassner, H., Eskofier, B. M., & Kluge, F. (2021).
   An Inertial Sensor-Based Gait Analysis Pipeline for the Assessment of Real-World
   Stair Ambulation Parameters. Sensors, 21(19), 6559. https://doi.org/10.3390/s21196559

.. [2] Martindale, C. F., Hoenig, F., Strohrmann, C., & Eskofier, B. M. (2017).
   Smart Annotation of Cyclic Data Using Hierarchical Hidden Markov Models.
   Sensors, 17(10), 2328. https://doi.org/10.3390/s17102328
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from gaitmap.example_data import (
    get_healthy_example_imu_data,
    get_healthy_example_stride_borders,
    get_healthy_example_stride_events,
)
from gaitmap.stride_segmentation.hmm import CompositeHmm, SimpleHmm
from gaitmap.utils.coordinate_conversion import convert_to_fbf

# %%
# Derive the three reference classes
# ------------------------------------
#
# Stride borders are aligned near toe-off. Label stride start to initial contact
# as swing, and initial contact to stride end as stance. The motion-capture event
# lists use different stride boundaries, so match contacts by sample position
# rather than joining the two lists by stride ID. Convert both feet to the foot
# body frame so the same feature columns describe the same motion directions.

sampling_rate_hz = 204.8
stride_lists = get_healthy_example_stride_borders()
recording = get_healthy_example_imu_data()
# The final right-foot stride has no motion-capture contact annotation. End the
# example before that stride starts; the left-foot reference may end mid-stride.
end_sample = int(stride_lists["right_sensor"].iloc[-1].start)
data = convert_to_fbf(
    {sensor: recording[sensor].iloc[:end_sample] for sensor in stride_lists}, left_like="left_", right_like="right_"
)
event_lists = get_healthy_example_stride_events()
phase_names = np.array(["stance", "swing", "transition/null"])

reference_labels = {}
for sensor, frame in data.items():
    labels = np.full(len(frame), 2, dtype=np.int64)
    contacts = np.unique(event_lists[sensor][["pre_ic", "ic"]].to_numpy()).astype(int)
    for start, end in stride_lists[sensor][["start", "end"]].to_numpy():
        if start >= len(frame):
            break
        contact = contacts[(contacts >= start) & (contacts < end)].item()
        labels[start:contact] = 1
        labels[contact:end] = 0
    reference_labels[sensor] = labels

# %%
# Hold out the second walking bout on both feet
# ---------------------------------------------
#
# Split in the middle of the left foot's largest gap between strides. The right
# foot has a stride annotation crossing that split. Exclude it from both parts,
# so every training stride ends before the test interval on either foot starts.
# Extract features separately on each partition to keep context windows apart.

left_strides = stride_lists["left_sensor"]
gap = np.argmax(left_strides.start.to_numpy()[1:] - left_strides.end.to_numpy()[:-1])
split_sample = int((left_strides.end.iloc[gap] + left_strides.start.iloc[gap + 1]) // 2)
train_data, test_data, test_starts = {}, {}, {}
for sensor, frame in data.items():
    strides = stride_lists[sensor]
    crossing = strides.loc[(strides.start < split_sample) & (strides.end > split_sample)]
    train_end = split_sample if crossing.empty else int(crossing.start.iloc[0])
    test_start = split_sample if crossing.empty else int(crossing.end.iloc[0])
    train_data[sensor], test_data[sensor] = frame.iloc[:train_end], frame.iloc[test_start:]
    test_starts[sensor] = test_start

# %%
# Add movement context
# --------------------
# A stationary foot can mean either stance during walking or a transition/null
# period. One second of surrounding gyroscope energy helps distinguish them.
# Centered windows make this an offline example. Compute features separately on
# each partition, and fit scaling on training data only.


def extract_features(frame):
    """Combine short-term foot motion with surrounding movement intensity."""
    smooth = frame.gyr_ml.rolling(9, center=True, min_periods=1).mean()
    energy = np.sqrt(
        frame[["gyr_pa", "gyr_ml", "gyr_si"]]
        .pow(2)
        .sum(axis=1)
        .rolling(round(sampling_rate_hz), center=True, min_periods=1)
        .mean()
    )
    return pd.DataFrame(
        {"gyr_ml": smooth, "gradient": np.gradient(smooth), "log_energy": np.log1p(energy)}, index=frame.index
    )


train_features = {sensor: extract_features(frame) for sensor, frame in train_data.items()}
test_features = {sensor: extract_features(frame) for sensor, frame in test_data.items()}
pooled_training_features = pd.concat(train_features.values(), ignore_index=True)
feature_mean, feature_scale = pooled_training_features.mean(), pooled_training_features.std()
train_features = {sensor: (frame - feature_mean) / feature_scale for sensor, frame in train_features.items()}
test_features = {sensor: (frame - feature_mean) / feature_scale for sensor, frame in test_features.items()}

# %%
# Train the three submodels separately
# ------------------------------------
#
# Split the training recording into contiguous regions of each reference class.
# Initialize internal states by dividing each region into equal-length sections;
# Baum-Welch then learns their alignment. No internal-state annotations are needed.
# Stance and swing use strict left-to-right chains. Transition/null is fully
# connected because movement outside strides need not follow a fixed order.
# Each internal state uses three Gaussian mixture components.

training_regions = {}
for sensor, labels in reference_labels.items():
    train_labels = labels[: len(train_data[sensor])]
    boundaries = np.r_[0, np.flatnonzero(np.diff(train_labels)) + 1, len(train_labels)]
    training_regions[sensor] = pd.DataFrame(
        {"start": boundaries[:-1], "end": boundaries[1:], "label": phase_names[train_labels[boundaries[:-1]]]}
    )
phase_models = {}
for name, n_states in zip(phase_names, [5, 5, 3], strict=True):
    sequences = []
    for sensor, regions in training_regions.items():
        phase_regions = regions.loc[regions.label == name, ["start", "end"]]
        sequences.extend(
            train_features[sensor].iloc[start:end] for start, end in phase_regions.itertuples(index=False, name=None)
        )
    initial_labels = [np.arange(len(sequence)) * n_states // len(sequence) for sequence in sequences]
    phase_models[name] = SimpleHmm(
        n_states=n_states,
        n_gmm_components=3,
        architecture="fully-connected" if name == "transition/null" else "left-right-strict",
        algo_train="baum-welch",
        max_iterations=10,
        min_covar=1e-4,
        verbose=False,
    ).self_optimize(sequences, initial_labels)

# %%
# Compose the submodels
# ------------------------------------
#
# Pass the trained submodels and labeled intervals to CompositeHmm. Each row is
# one instance of a submodel; separate rows can also describe repeated instances
# of the same movement, such as consecutive strides. Both connections and their
# probabilities are derived from the training data, with emissions held fixed.
# The held-out labels are used only for evaluation.

combined, history = CompositeHmm({name: submodel.model for name, submodel in phase_models.items()}).fit_transitions(
    sequences=list(train_features.values()), labels=list(training_regions.values())
)

predicted_states = {sensor: combined.predict_hidden_state_sequence(frame) for sensor, frame in test_features.items()}
predictions = {
    sensor: pd.Categorical(combined.predict_labels(frame), categories=phase_names).codes
    for sensor, frame in test_features.items()
}
held_out_labels = {sensor: labels[test_starts[sensor] :] for sensor, labels in reference_labels.items()}

# %%
# Evaluate each class
# -------------------
# Report all three classes so a good overall score cannot hide a missing phase.
# This is a held-out portion of one participant's recording, not an evaluation
# on new participants. "Transition/null" means outside the supplied strides.

counts = np.zeros((3, 3), dtype=int)
for sensor, labels in held_out_labels.items():
    np.add.at(counts, (labels, predictions[sensor]), 1)
confusion_matrix = pd.DataFrame(counts, index=phase_names, columns=phase_names)
precision = counts.diagonal() / counts.sum(axis=0)
recall = counts.diagonal() / counts.sum(axis=1)
scores = pd.DataFrame(
    {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall)},
    index=phase_names,
)
print("Confusion matrix: rows are reference labels, columns are predictions.")
print(confusion_matrix)
print(scores.round(3))

# %%
# Compare the phase sequences
# ---------------------------

colors = ["#377eb8", "#ff7f00", "#4daf4a"]
fig, axes = plt.subplots(4, 2, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 2]})
for column, (sensor, frame) in enumerate(test_data.items()):
    frame.gyr_ml.plot(ax=axes[0, column], color="black", linewidth=0.8)
    axes[0, column].set(ylabel="gyr_ml [deg/s]", title=f"Held-out second walking bout: {sensor}")
    extent = (frame.index[0], frame.index[-1] + 1 / sampling_rate_hz, 0, 1)
    for axis, labels, title in zip(
        axes[1:3, column], [held_out_labels[sensor], predictions[sensor]], ["Reference", "Prediction"], strict=True
    ):
        axis.imshow(labels[None, :], aspect="auto", extent=extent, cmap=ListedColormap(colors), vmin=0, vmax=2)
        axis.set(ylabel=title, yticks=[])
    axes[-1, column].step(frame.index, predicted_states[sensor], where="post", color="black", linewidth=0.8)
    axes[-1, column].set(
        ylabel="Internal state",
        yticks=[states[0] for states in combined.state_groups.values()],
        ylim=(-0.5, combined.model.n_states - 0.5),
        xlabel="Time [s]",
    )
fig.legend(
    handles=[Patch(color=color, label=name) for color, name in zip(colors, phase_names, strict=True)],
    loc="upper center",
    ncol=3,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()
