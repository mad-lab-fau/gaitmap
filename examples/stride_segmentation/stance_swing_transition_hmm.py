r"""
.. _example_stance_swing_transition_hmm:

Stance, swing, and transition states
====================================

This example demonstrates how to train HMMs with more than two states.

.. warning::

   We do not recommend using this HMM to detect stance/swing transitions without
   further validation.

With suitable annotations and features, you can also train models for movement
classes such as walking, stair walking, and rest, or walking, running, and rest.
For a published example of HMMs for walking, running, and rest, see Martindale
et al. [1]_. That work uses a hierarchical model; this tutorial trains a simpler
three-state model to demonstrate the training workflow with the available data.

Train a three-state HMM on the healthy walking recording. As in the Roth setup,
everything outside the annotated strides belongs to the transition/null class.
Within each stride, initial contact separates swing from stance. These labels
are derived from the supplied annotations; no extra rest annotations are needed.

.. [1] Martindale, C. F., Hoenig, F., Strohrmann, C., & Eskofier, B. M. (2017).
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
from gaitmap.stride_segmentation.hmm import SimpleHmm
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
state_names = np.array(["stance", "swing", "transition/null"])

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
# Train and predict
# -----------------
# Each of the three hidden states has three Gaussian mixture components. Known
# phase labels identify the states; training estimates their emission parameters
# and transition probabilities. The held-out labels are used only for evaluation.

model, history = SimpleHmm(
    n_states=3,
    n_gmm_components=3,
    architecture="fully-connected",
    algo_train="labeled",
    max_iterations=2,
    min_covar=1e-4,
    verbose=False,
).self_optimize_with_info([train_features], [reference_labels[:split_sample]])

predictions = model.predict_hidden_state_sequence(test_features, algorithm="viterbi")
held_out_labels = reference_labels[split_sample:]

# %%
# Evaluate each class
# -------------------
# Report all three classes so a good overall score cannot hide a missing phase.
# This is a held-out portion of one participant's recording, not an evaluation
# on new participants. "Transition/null" means outside the supplied strides.

counts = np.zeros((3, 3), dtype=int)
np.add.at(counts, (held_out_labels, predictions), 1)
confusion_matrix = pd.DataFrame(counts, index=state_names, columns=state_names)
precision = counts.diagonal() / counts.sum(axis=0)
recall = counts.diagonal() / counts.sum(axis=1)
scores = pd.DataFrame(
    {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall)},
    index=state_names,
)
print("Confusion matrix: rows are reference labels, columns are predictions.")
print(confusion_matrix)
print(scores.round(3))

# %%
# Compare the phase sequences
# ---------------------------

colors = ["#377eb8", "#ff7f00", "#4daf4a"]
fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
test_data.gyr_ml.plot(ax=axes[0], color="black", linewidth=0.8)
axes[0].set(ylabel="gyr_ml [deg/s]", title="Held-out second walking bout")
extent = (test_data.index[0], test_data.index[-1] + 1 / sampling_rate_hz, 0, 1)
for axis, labels, title in zip(axes[1:], [held_out_labels, predictions], ["Reference", "Prediction"]):
    axis.imshow(labels[None, :], aspect="auto", extent=extent, cmap=ListedColormap(colors), vmin=0, vmax=2)
    axis.set(ylabel=title, yticks=[])
axes[-1].set_xlabel("Time [s]")
axes[0].legend(handles=[Patch(color=color, label=name) for color, name in zip(colors, state_names)], loc="upper right")
fig.tight_layout()
plt.show()
