"""Methods to perform coordinate system transformation and alignments."""

from gaitmap.preprocessing.sensor_alignment._gravity_alignment import align_dataset_to_gravity
from gaitmap.preprocessing.sensor_alignment._mulisensor_alignment import align_heading_of_sensors
from gaitmap.preprocessing.sensor_alignment._pca_alignment import PcaAlignment
from gaitmap.utils._gaitmap_mad import patch_gaitmap_mad_import

_gaitmap_mad_modules = {
    "ForwardDirectionSignAlignment",
}

if not (__getattr__ := patch_gaitmap_mad_import(_gaitmap_mad_modules, __name__)):
    from gaitmap_mad.preprocessing.sensor_alignment import ForwardDirectionSignAlignment

__all__ = ["align_dataset_to_gravity", "align_heading_of_sensors", "PcaAlignment", "ForwardDirectionSignAlignment"]
