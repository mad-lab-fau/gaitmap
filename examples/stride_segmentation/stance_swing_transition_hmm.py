r"""
.. _example_stance_swing_transition_hmm:

Composing stance, swing, and transition HMMs
============================================

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

Train stance, swing, and transition/null HMMs on the healthy walking recording.
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
from gaitmap.utils.coordinate_conversion import convert_left_foot_to_fbf

# %%
# Derive the three reference classes
# ------------------------------------
#
# The left-foot stride borders are aligned near toe-off. Label stride start to
# initial contact as swing, and initial contact to stride end as stance. The
# motion-capture event list uses different stride boundaries, so match contacts
# by their sample positions rather than joining the two lists by stride ID.

sampling_rate_hz = 204.8
data = convert_left_foot_to_fbf(get_healthy_example_imu_data()["left_sensor"])
stride_list = get_healthy_example_stride_borders()["left_sensor"]
event_list = get_healthy_example_stride_events()["left_sensor"]
phase_names = np.array(["stance", "swing", "transition/null"])

reference_labels = np.full(len(data), 2, dtype=np.int64)
contacts = np.unique(event_list[["pre_ic", "ic"]].to_numpy()).astype(int)
for start, end in stride_list[["start", "end"]].to_numpy():
    contact = contacts[(contacts >= start) & (contacts < end)].item()
    reference_labels[start:contact] = 1
    reference_labels[contact:end] = 0

# %%
# Hold out the second walking bout
# ------------------------------------
#
# Split in the middle of the largest gap between annotated strides. This keeps
# complete strides in either training or testing. Use only the left foot, so
# synchronized observations from the other foot cannot enter the training set.

gap = np.argmax(stride_list.start.to_numpy()[1:] - stride_list.end.to_numpy()[:-1])
split_sample = int((stride_list.end.iloc[gap] + stride_list.start.iloc[gap + 1]) // 2)
train_data, test_data = data.iloc[:split_sample], data.iloc[split_sample:]

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


train_features, test_features = extract_features(train_data), extract_features(test_data)
feature_mean, feature_scale = train_features.mean(), train_features.std()
train_features = (train_features - feature_mean) / feature_scale
test_features = (test_features - feature_mean) / feature_scale

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

train_labels = reference_labels[:split_sample]
boundaries = np.r_[0, np.flatnonzero(np.diff(train_labels)) + 1, len(train_labels)]
training_regions = pd.DataFrame(
    {"start": boundaries[:-1], "end": boundaries[1:], "label": phase_names[train_labels[boundaries[:-1]]]}
)
phase_models = {}
for name, n_states in zip(phase_names, [5, 5, 3]):
    regions = training_regions.loc[training_regions.label == name, ["start", "end"]]
    sequences = [train_features.iloc[start:end] for start, end in regions.itertuples(index=False, name=None)]
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
    sequences=[train_features], labels=[training_regions]
)

predicted_states = combined.predict_hidden_state_sequence(test_features)
predicted_labels = combined.predict_labels(test_features)
predictions = pd.Categorical(predicted_labels, categories=phase_names).codes
held_out_labels = reference_labels[split_sample:]

# %%
# Evaluate each class
# -------------------
# Report all three classes so a good overall score cannot hide a missing phase.
# This is a held-out portion of one participant's recording, not an evaluation
# on new participants. "Transition/null" means outside the supplied strides.

counts = np.zeros((3, 3), dtype=int)
np.add.at(counts, (held_out_labels, predictions), 1)
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
fig, axes = plt.subplots(4, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 2]})
test_data.gyr_ml.plot(ax=axes[0], color="black", linewidth=0.8)
axes[0].set(ylabel="gyr_ml [deg/s]", title="Held-out second walking bout")
extent = (test_data.index[0], test_data.index[-1] + 1 / sampling_rate_hz, 0, 1)
for axis, labels, title in zip(axes[1:3], [held_out_labels, predictions], ["Reference", "Prediction"]):
    axis.imshow(labels[None, :], aspect="auto", extent=extent, cmap=ListedColormap(colors), vmin=0, vmax=2)
    axis.set(ylabel=title, yticks=[])
axes[-1].step(test_data.index, predicted_states, where="post", color="black", linewidth=0.8)
axes[-1].set(
    ylabel="Internal state",
    yticks=[states[0] for states in combined.state_groups.values()],
    ylim=(-0.5, combined.model.n_states - 0.5),
)
axes[-1].set_xlabel("Time [s]")
axes[0].legend(handles=[Patch(color=color, label=name) for color, name in zip(colors, phase_names)], loc="upper right")
fig.tight_layout()
plt.show()
