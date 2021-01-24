r"""
.. _algo_serialize:

Export Algorithms to JSON
=========================

This example demonstrates how to save algorithms including their configuration to json and load them again to ensure
reproducibility.

The example will use :class:`~gaitmap.trajectory_reconstruction.StrideLevelParameter` as example, but the same will
work with all other algorithms (and Pipelines).

.. warning:: This json export only stores the Parameters of an algorithm and **not** any potential results!

.. warning:: While calling an Algorithms with the same parameters should produce the same results, you also need to
            ensure that the same version of gaitmap (and all other dependencies) is used to ensure full
            reproducibility.
            This means you should save the exact library version together with the json version of the used algorithms.
"""
from pprint import pprint

# %%
# To JSON
# -------
# In the following we will create an algorithm instance that itself has other nested algorithm objects.
# In the from the json output
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseAlgorithm
from gaitmap.trajectory_reconstruction import StrideLevelTrajectory, MadgwickAHRS, ForwardBackwardIntegration

# Setting an initial orientation here, is pointless as it will be overwritten by StrideLevelTrajectory
# It is used here to demonstrate the ability to serialize Rotation objects.
custom_ori_method = MadgwickAHRS(beta=0.5, initial_orientation=Rotation.from_quat([0, 3, 3, 0]))
custom_pos_method = ForwardBackwardIntegration(turning_point=0.8)

slt = StrideLevelTrajectory(ori_method=custom_ori_method, pos_method=custom_pos_method, align_window_width=10)
pprint(slt.get_params())

# %%
json_str = slt.to_json()

print(json_str)

# %%
# This output could now be stored in a file to document the exact parameters that were used for a specific analysis.
#
#
# From JSON
# ---------
# All algorithms can be loaded from json as well using the `from_json` method.
# This method can be called on any algorithm class and the Algorithm class specified in the json object is returned.
# To avoid confusion it is advisable to use either the exact algorithm class that was stored or `BaseAlgorithm`.
loaded_slt = BaseAlgorithm.from_json(json_str)
pprint(loaded_slt.get_params())

# %%
# To show that you can call `from_json` from any Algorithm class we will perform the same import using the
# `StrideLevelTrajectory`.
loaded_slt = StrideLevelTrajectory.from_json(json_str)
pprint(loaded_slt.get_params())

# %%
# Caching Support
# ---------------
# Note that the json export does not cover `joblib.memory` objects that some algorithms use to cache results.
# When you attempt to do this, you will get a warning with further information.
from gaitmap.stride_segmentation import BarthDtw
from joblib import Memory

# Create a memory object. Usually you would pass the path to a cache dir.
mem = Memory()

instance = BarthDtw(memory=Memory())
pprint(instance.get_params())

# %%
json_str_cached = instance.to_json()
print(json_str_cached)

# %%
# If we now load the object again, the memory argument is empty.
# We need to set it again.
# If we use the same memory object or recreate a new memory object that points to the same cache-location, the same
# cache will be used as before.
loaded_instance = BarthDtw.from_json(json_str_cached)
print(loaded_instance.memory)

# %%
# We can simply reactivate the memory.
loaded_instance = loaded_instance.set_params(memory=mem)
pprint(loaded_instance.get_params())
