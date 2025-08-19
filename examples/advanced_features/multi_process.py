r"""
.. _example_mulit_process:

Running multiple pipelines in parallel
=======================================

When working on large datasets, it can dramatically speed up calculations if multiple processing cores on a modern CPU
are used in parallel.
Python is usually extremely bad at this, as the Global Interpreter Lock (GIL) limits every Python process to a single
Core.
While gaitmap tries to make use of lower level C-implementations (that release the GIL) for many of the heavy lifting
tasks, this does not result in the expected performance increase one might expect from going from a 2-core to a 4-core
processor.
To make proper use of multiple cores, Python - and in turn gaitmap - need to run multiple separate processes, each bound
to a different core.

The following example shows how a gaitmap algorithm (or pipeline of algorithms) can be run in parallel with different
parameter combinations.
A similar multiprocessing approach could be used to compute multiple subjects or recordings in parallel.

.. note:: To get the best performance you need to select a number of parallel processes that make sense for your CPU.
          While it might be tempting to set this number to the number of available processing threads, this might not
          always yield the best results.
          Modern CPU have adaptive clock speeds and hitting the processor with an all core load usually results in a
          reduction of per core performance.
          Hence, more parallel processes might not always result in the best overall performance.

The following example shows how you can make a parameter sweep for the Stride Segmentation Algorithm using `joblib`
as a helper module.
Other Python helpers to spawn multiple processes will of course work as well.

"""

from pprint import pprint
from typing import Any

# %%
# Load some example data
# ----------------------
from gaitmap.example_data import get_healthy_example_imu_data
from gaitmap.utils.coordinate_conversion import convert_to_fbf

data = get_healthy_example_imu_data()
bf_data = convert_to_fbf(data, left_like="left_", right_like="right_")
sampling_rate_hz = 204.8

from sklearn.model_selection import ParameterGrid

# %%
# Preparing the stride segmentation
# ---------------------------------
# In this example we simulate a Gridsearch.
# To make this example as real-life as possible, we use the sklearn `ParameterGrid` to set up our parameter sweep.
# Note, that we make use of gaitmaps `set_params` methods later.
# Hence, we can specify parameters for the nested template object in the parameter grid using the double "_" notation.
from gaitmap.stride_segmentation import BarthDtw

dtw = BarthDtw()
parameter_grid = ParameterGrid({"max_cost": [1800, 2200], "template__use_cols": [("gyr_ml",), ("gyr_ml", "gyr_si")]})

pprint(list(parameter_grid))

# %%
# Creating a function for the Multi-processing
# --------------------------------------------
# To run something in parallel, we need to write a function that will be executed in every process.
# The function below returns the entire dtw object, which might not be desired, as each object contains a copy of the
# entire data.
# Further, the current concept will copy the entire data to each process.
# This could be further optimized by using a read-only shared memory object for the data.


def run(dtw: BarthDtw, parameter: dict[str, Any]) -> BarthDtw:
    # For this run, change the parameters on the dtw object
    dtw = dtw.set_params(**parameter)
    dtw = dtw.segment(data=bf_data, sampling_rate_hz=sampling_rate_hz)
    return dtw


# %%
# Do the Parallel Run
# -------------------
# Finally we use joblib to run the code in parallel using 2 workers.
from joblib import Parallel, delayed

results = Parallel(n_jobs=2)(delayed(run)(dtw, para) for para in parameter_grid)

# %%
# We will not inspect the results here, but we can see that each dtw object has different parameters set.
for r in results:
    pprint(r.get_params())
