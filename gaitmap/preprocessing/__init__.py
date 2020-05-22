"""A set of functions that help to align the sensor orientation and prepare the dataset for the use with gaitmap."""
from gaitmap.preprocessing.sensor_alignment import align_dataset_to_gravity, align_heading_of_sensors

__all__ = ["align_dataset_to_gravity", "align_heading_of_sensors"]
