"""A set of functions that help to align the sensor orientation and prepare the dataset for the use with gaitmap."""
from gaitmap.preprocessing.sensor_alignment import align_dataset_to_gravity, get_xy_alignment_from_gyro

__all__ = ["align_dataset_to_gravity", "get_xy_alignment_from_gyro"]
