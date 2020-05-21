r"""
.. _mulit_process:

Running multiple pipelines in parallel
=======================================

When working on large datasets, it can dramatically speed up calculations if multiple processing cores on a modern CPU
are used in parallel.
Python is usually extremely bad at this, as the Global Interpreter Lock (GIL) limits every Python process to a single
Core.
While gaitmap tries to make use of lower level C-implementations (that release the GIL) for many of the heavy lifting
tasks, this does not result in the expected performance increase one might expect from going from a 2-core to a 4-core
processor.
To make proper use of multiple cores, Python - and in turn gaitmap - need to run multiple separte processes, each bound
to a different core.

The following example shows how a gaitmap Algorithms (or pipeline of algorithms) can be run in parallel with different
parameter combinations.
A similar multiprocessing approach could be used to compute multiple subjects or recordings in parallel.

.. note:: To get the best performance you need to select a number of parallel processes that make sense for your CPU.
          While, it might be tempting to set this number to the number of available processing thread, this might not
          always yield the best results.
          Modern CPU have adaptive clock speeds and hitting the processor with a all core load usually results in a
          reduction of per core performance.
          Hence, more parallel processes might not always result in the best overall performance.

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

# First, let's load some data and set up the Stride detection algorithm we want to run

from sklearn
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
