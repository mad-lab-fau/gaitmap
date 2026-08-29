r"""
.. _example_composite_hmm_training:

Training a multi-part HMM
=========================

This example trains transition, walking, stair, and running HMMs without a
domain-specific orchestration class. Each annotation table covers the complete
recording and names the model responsible for every region.
"""

import numpy as np
import pandas as pd

from gaitmap.stride_segmentation.hmm import CompositeHmm, HmmModel, RothHmmFeatureTransformer

# %%
# Define the topology
# -------------------
part_names = ("transition", "walking", "stairs", "running")
topology = HmmModel.compose(
    parts={name: HmmModel.left_right(1, n_gmm_components=1) for name in part_names},
    routes={
        "transition": ("walking", "stairs", "running"),
        "walking": ("transition",),
        "stairs": ("transition",),
        "running": ("transition",),
    },
    starts=("transition",),
    ends=("transition",),
)

# %%
# Create complete annotations
# ---------------------------
region_names = ["transition", "walking", "transition", "stairs", "transition", "running", "transition"]
regions = pd.DataFrame(
    {
        "start": np.arange(0, 140, 20),
        "end": np.arange(20, 160, 20),
        "model": region_names,
    }
)
means = {"transition": 0.0, "walking": 3.0, "stairs": 6.0, "running": 9.0}
rng = np.random.default_rng(4)
data = pd.DataFrame({"gyr_ml": np.concatenate([rng.normal(means[name], 0.1, 20) for name in region_names])})

# %%
# Train all parts and the combined routing matrix
# ------------------------------------------------
feature_transform = RothHmmFeatureTransformer(
    sampling_rate_feature_space_hz=20,
    low_pass_filter=None,
    axes=["gyr_ml"],
    features=["raw"],
    standardization=False,
)
trained = CompositeHmm(model=topology, feature_transform=feature_transform).self_optimize(
    [data], [regions], sampling_rate_hz=20
)

print(trained.model.state_groups)
print(trained.model.transition_probabilities)
