r"""
.. _example_segmentation_hmm_training:

Training the Roth HMM
=====================

This example trains a backend-neutral Roth segmentation model with pomegranate
1.x. Pomegranate and torch are only involved while fitting; the resulting
:class:`~gaitmap.stride_segmentation.hmm.HmmModel` contains plain NumPy
parameters and can be serialized or decoded with the SciPy backend.

The modern training backend requires Python 3.10 or newer.
"""

import matplotlib.pyplot as plt
import numpy as np

# %%
# Prepare labeled training data
# -----------------------------
# The Roth model expects body-frame sensor data and matching stride borders.
# Each sensor recording is one training sequence.
from gaitmap.example_data import get_healthy_example_imu_data, get_healthy_example_stride_borders
from gaitmap.utils.coordinate_conversion import convert_to_fbf

sampling_rate_hz = 204.8
bf_data = convert_to_fbf(get_healthy_example_imu_data(), left_like="left_", right_like="right_")
stride_lists = get_healthy_example_stride_borders()

data_train_sequence = [bf_data["left_sensor"], bf_data["right_sensor"]]
stride_list_sequence = [stride_lists["left_sensor"], stride_lists["right_sensor"]]

# %%
# Configure and train
# -------------------
# Feature extraction, training, and inference are independent parameters. The
# defaults reproduce the Roth feature space, use pomegranate 1.x for supervised
# fitting, and use SciPy for portable inference.
from gaitmap.stride_segmentation.hmm import PomegranateHmmTrainer, RothSegmentationHmm

segmentation_model = RothSegmentationHmm(
    training_backend=PomegranateHmmTrainer(
        n_gmm_components=3,
        max_iterations=10,
        covariance_regularization=1e-6,
        random_state=0,
    )
).self_optimize(data_train_sequence, stride_list_sequence, sampling_rate_hz=sampling_rate_hz)

print(f"Compiled states: {segmentation_model.model.n_states}")
np.set_printoptions(precision=3, linewidth=180, suppress=True)
print(segmentation_model.model.transition_probabilities)

# The fitted model is ordinary gaitmap/tpcp state and can be persisted without
# pomegranate-specific serialization.
model_json = segmentation_model.to_json()
restored_model = RothSegmentationHmm.from_json(model_json)

# %%
# Apply the trained model
# -----------------------
from gaitmap.stride_segmentation.hmm import HmmStrideSegmentation

hmm = HmmStrideSegmentation(restored_model).segment(bf_data, sampling_rate_hz=sampling_rate_hz)
print(hmm.stride_list_["left_sensor"].head())

# %%
# Inspect one result
# ------------------
sensor = "left_sensor"
fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(10, 5))
axs[0].plot(bf_data.reset_index(drop=True)[sensor]["gyr_ml"])
for start, end in hmm.stride_list_[sensor].to_numpy():
    axs[0].axvspan(start, end, alpha=0.2)
axs[0].set_ylabel("gyr-ml [deg/s]")

axs[1].plot(hmm.hidden_state_sequence_[sensor])
axs[1].set_ylabel("Hidden state")
axs[1].set_xlabel(f"Samples @ {sampling_rate_hz:g} Hz")
plt.xlim([6000, 7200])
fig.tight_layout()
plt.show()
