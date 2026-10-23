"""
A reusable pipeline with interchangeable algorithms
===================================================

``GenericGaitPipeline`` combines stride segmentation, event detection, trajectory
reconstruction, and temporal and spatial parameter calculation. Each algorithm
can be supplied independently and configured through tpcp's nested parameters.
"""

from dataclasses import dataclass

from gaitmap.event_detection import HerzerEventDetection
from gaitmap.example_data import get_healthy_example_imu_data
from gaitmap.pipeline import BaseGaitDataset, GenericGaitPipeline
from gaitmap.stride_segmentation import BarthDtw
from gaitmap.utils.datatype_helper import SensorData

# %%
# A recording only needs the two properties in the BaseGaitDataset protocol.
# It does not need to inherit from gaitmap or tpcp. Existing gaitmap-datasets
# loaders also expose these properties on single-recording dataset subsets.


@dataclass
class Recording:
    """One recording with data already aligned to the gaitmap foot sensor frame."""

    data: SensorData
    sampling_rate_hz: float


recording: BaseGaitDataset = Recording(get_healthy_example_imu_data(), 204.8)

# %%
# Stride segmentation and event detection are required. The other stages default
# to StrideLevelTrajectory, TemporalParameterCalculation, and
# SpatialParameterCalculation. Pass different algorithm instances to replace
# them. All stages execute on clones, leaving the supplied templates unchanged.
pipeline = GenericGaitPipeline(
    stride_segmentation=BarthDtw(),
    event_detection=HerzerEventDetection(),
)
pipeline.safe_run(recording)

print(pipeline.parameters_["left_sensor"].head())
print(pipeline.aggregated_gait_parameters_)

# %%
# Nested parameters remain accessible to tpcp parameter searches.
pipeline.set_params(stride_segmentation__max_cost=3.0)
