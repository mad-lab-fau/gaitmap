"""Methods to perform coordinate system transformation and alignments."""

from gaitmap.preprocessing.sensor_alignment._gravity_alignment import align_dataset_to_gravity
from gaitmap.preprocessing.sensor_alignment._mulisensor_alignment import align_heading_of_sensors
from gaitmap.preprocessing.sensor_alignment._pca_alignment import PcaAlignment
