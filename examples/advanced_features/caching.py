r"""

.. _caching:

Caching algorithm outputs
=========================

Many algorithms implemented in gaitmap have a runtime of multiple seconds on larger datasets.
In the context of algorithm evaluation, for example when performing a cross-validation, algorithms are sometimes
repeatedly called on the same data and even with the same parameters.
In these cases, it can be helpful to cache results to ensure that you do not need to recalculate values.

The `joblib` Python package makes cashing extremely easy and you should read their
`guide <https://joblib.readthedocs.io/en/latest/memory.html>`__ first, before continuing with this example.
However, one of the caveats with joblib caching is that it only works on pure functions without side effects and should
not be used with methods.

Unfortunately, gaitmap is mostly object oriented and all the computational expensive things you might want to do are
hidden behind a method call.

Therefore, many gaitmap algorithms have caching built-in.
These algorithms support an additional keyword argument called `memory` in their init-function.
If you pass a `joblib.Memory` object to these, it will be used to cache the most time consuming function calls.
Note, that this will usually not cache all the calculations in a method, but only the ones that are considered worth
caching by the algorithm developer.

If you really want to cache the full method calls (on your own risk), see the last section of this example.
"""

# %%
# Example Pipeline
# ----------------
# We will simply copy the stride segmentation example to have some data to work with.
from gaitmap.example_data import get_healthy_example_imu_data
from gaitmap.stride_segmentation import BarthOriginalTemplate
from gaitmap.utils.coordinate_conversion import convert_to_fbf

data = get_healthy_example_imu_data()
sampling_rate_hz = 204.8

data = convert_to_fbf(data, left_like="left_", right_like="right_")

# %%
# Creating the cash
# -----------------
# First we will create a memory instance for our cash.
# We can use the same cash to cash the output of multiple algorithms.
# The cash stays valid even after you restart Python, if you didn't delete the folder.
#
# However, in this example, we will use a temp-directory that will be deleted at the end of the example.
from tempfile import TemporaryDirectory

from joblib import Memory

tmp_dir = TemporaryDirectory()
# We will activate some more debug output for this example
mem = Memory(tmp_dir.name, verbose=2)

# %%
# Initialize algorithm
# --------------------
# We initialize our algorithm as normal, but pass the memory instance as an additional parameter.
from gaitmap.stride_segmentation import BarthDtw

dtw = BarthDtw(memory=mem)

# %%
# Calling cached methods
# ----------------------
# The first time we call `segment` now, all calculation will run as normal, but the output of certain calculations will
# be cached.
# They are then reused when we call `segment` again with the same data and configuration.
#
# Observe the print output to see what happens.
first_call_results = dtw.segment(data=data, sampling_rate_hz=204.8)
first_call_stride_list = first_call_results.stride_list_.copy()

# %%
# According to the debug output, two internal functions of `BarthDtw` are cached.
# Each twice with different value inputs, because our data had two sensors.
# It depends on the actual algorithm, which and how internal components are cached.
#
# Independent of that if we call the method again, we can see in the debug output that the results of these methods
# are now loaded from disk.
# If we would use a larger dataset, we would see dramatic speed improvements.
second_call_results = dtw.segment(data=data, sampling_rate_hz=204.8)
second_call_stride_list = second_call_results.stride_list_.copy()

# %%
# We can verify that the results are actually identical
first_call_stride_list["left_sensor"].equals(second_call_stride_list["left_sensor"])

# %%
# Partially cached calls
# ----------------------
# As you have seen before, `BarthDtw` caches multiple steps individually.
# This ensures that we can change some parameters while still making use of the some cached results.
# For `BarthDtw` we cache the creation of the cost matrix and the identification of strides within the cost matrix
# separatly.
# As the cost-matrix only depends on the template and the constrains, we can reuse the cash, if we change any other
# parameter.
#
# If we change the `max_cost` for example, only the stride detection part needs to be recalculated.
new_instance = BarthDtw(max_cost=5.0, memory=mem)
new_instance.segment(data=data, sampling_rate_hz=204.8)

# %%
# As you can see in the debug output, we loaded the results of `subsequence_cost_matrix`, but recalculated the second
# step.

# %%
# Some Note
# ---------
# - Caching support will vary from algorithm to algorithm
# - Caching supports multi-processing
# - Do **not** use you cache as permanent storage of results. It is way too easy to delete it.
# - If you try a lot of things with a lot of data, your cache can become really large.
# - Clear your cache, before you do your final calculations for a publication!
# - Make sure you add you cache dir to your ".gitignore" file.


# %%
# Caching Full method calls
# -------------------------
# In some cases it might still be desirable to cache the entire output of an algorithm.
# To do this safely you need to be aware of how cashing works under the hood.
# The `Memory` class calculates a hash of all inputs to a function and stores a pickeled version of the results together
# with this input-hast.
# If the function is called again, the hash of the input is compared with hashes stored on the disk.
# Depending on this, a cached result can be selected.
import joblib

# %%
# We can calculate the hash of our algorithm.
joblib.hash(dtw)

# %%
# If we recreate the object with the same parameters, the hash is identical.
joblib.hash(BarthDtw())

# %%
# The same is true for cloning
joblib.hash(dtw.clone())

# %%
# However, if we change any parameters the hash of the object changes.
joblib.hash(BarthDtw(max_cost=100))

# %%
# It is important to note that the hash always changes, if **any** of the attributes are modified, not just the ones
# accessible through the init.
# This means, if e.g. after you call `segment` and the algorithm object will have all results stored, the hash will
# change.
# The same will happen, if you add custom attributes to the instance.
# The hash will change and the cache would be invalidated.
test_dtw = BarthDtw()
test_dtw.custom_value = 4
joblib.hash(test_dtw)

# %%
# This observation becomes an issue when caching class methods.
# As python passes the class instance itself as the first argument to this method.
# This means the input-hash used for caching will change whenever anything on the class instance changes, even if the
# change might not affect the actual output of the method.
# In many cases this is less of an issue with gaitmap, as we can reasonably assume that the main action method should
# only depend on the params of an algorithm (`self.get_params()`) and the actual action method.
#
# Therefore, we can cache action methods reliably when cloning the algorithm before hand and using a wrapper method.
# Cloning the algorithm instance ensures that all instance data, except the params are reset.


def call_segment(algo, data, sampling_rate_hz):
    return algo.segment(data=data, sampling_rate_hz=sampling_rate_hz)


# Cache the wrapper:
cached_call_segment = Memory(tmp_dir.name, verbose=2).cache(call_segment)

# Then we need to clone the algorithm every time we call the cached wrapper, to reset the params:
reset_dtw = dtw.clone()
results = cached_call_segment(reset_dtw, data, sampling_rate_hz)

# %%
# On this first call, we can see that the cached call actually modified the `reset_dtw` object in place.
id(reset_dtw) == id(results)

# %%
# However, on the second call, it will return a copy (loaded from the cache)
reset_dtw = dtw.clone()
results = cached_call_segment(reset_dtw, data, sampling_rate_hz)
id(reset_dtw) == id(results)

# %%
# While it is possible to cache methods this way, this might be error prone.
# The safest option (and remember, we are already in the unsafe territory), is to use a nested wrapper resolve potential
# user errors.
#
# In the general case you can use the recipe below.
# It will always ensure that the algo object is cloned and will return a copy of the algorithm in any case.
#
# .. warning::
#    While this expected to work, cashing an entire algorithm object as return value can take **a lot** of storage space
#    as it usually stores a copy of the input data.
#    Whenever possible you should only return the parts of the result you are really interested inside the cached
#    function.


def cached_call_method(_algo, _method_name: str, _memory: Memory, *args, **kwargs):
    """Call a method on the algo object and cache the output.

    Repeated calls to this function with the same algorithm and the same args, and kwargs, will return cached results
    saved on disk.

    .. warning ::
        This method will clone the algorithm object before calling the method.
        This ensures that the cache is not invalidated because of results stored on the object.

    Parameters
    ----------
    _algo
        The algorithm instance to use
    _method_name
        The name of the method to call
    _memory
        A instance of `joblib.memory` used for caching
    args
        Positional arguments passed to the called method
    kwargs
        Keyword arguments passed to the called method.

    Returns
    -------
    method_return
        The return value of the called methods either calculated or cached.

    See Also
    --------
    gaitmap.utils.caching.cached_call_action

    """

    def _call_method(_algo, _method_name, *args, **kwargs):
        return getattr(_algo, _method_name)(*args, **kwargs)

    _algo = _algo.clone()
    return _memory.cache(_call_method)(_algo, _method_name, *args, **kwargs)


mem = Memory(tmp_dir.name, verbose=2)

cached_result = cached_call_method(
    BarthDtw(), _method_name="segment", _memory=mem, data=data, sampling_rate_hz=sampling_rate_hz
)

# %%
# And the second call will load the results.
cached_result = cached_call_method(
    BarthDtw(), _method_name="segment", _memory=mem, data=data, sampling_rate_hz=sampling_rate_hz
)

# %%
# Finally remove the tempdir
tmp_dir.cleanup()
