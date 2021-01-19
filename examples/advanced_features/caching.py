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
This example shows, how you can still use caching when keeping a couple of things in mind.
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
# Here we initialize our algorithm.
from gaitmap.stride_segmentation import BarthDtw

dtw = BarthDtw()

# %%
# Caching the method call
# -----------------------
# Normally we would now call `dtw.segment(...)` with the data we have loaded.
# But, in this example we want to cache the result of this method call.
# This means, we need to wrap the method in a `joblib.Memory` object.
#
# However, as mentioned above, we want to avoid caching methods, but rather cache functions.
# Therefore, we will create a function that performs the method call and has all "dependencies" as parameters.


def call_segment(algo, data, sampling_rate_hz):
    return algo.segment(data=data, sampling_rate_hz=sampling_rate_hz)


# %%
# This function can now be cached and joblib will store a hash of all the input parameters.
# If the hashes match in a future call, the result will simply be loaded from disk.
# If the hashes don't match, the function will actually be called.
#
# Before we set up the caching, we should better understand the concept of hashing.
import joblib

# %%
# We can calculate the hash of our algorithm
joblib.hash(dtw)

# %%
# If we recreate the object with the same parameters, the hash is identical
joblib.hash(BarthDtw())

# %%
# The same is true for cloning
joblib.hash(dtw.clone())

# %%
# However, if we change any parameters the hash of the object changes
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
# We will keep that in mind for the next step
#
# Creating a Disk Cache
# ---------------------
# We can create a cache in any directory on our hard-drive.
# It will stay there until we delete it or it becomes invalid because we updated our code.
# Note, that this directory can grow quite large and you should be mindful of what you cache and clean the cache
# directory.
# Further, it is a good idea to clean the cache when you made changes to your code that might affect the algorithm you
# are using.
# There is no way for joblib to check if you changed the implementation, if you changed how the algorithm works in the
# background, which would require a recalculation of all results.
#
# In this example we will store the cache in a tempfile to make sure we don't leave any traces.
# In you code, you will just pick a folder that you wouldn't accidentally change or delete (e.g. your-project/.cache).
import tempfile

cache_dir = tempfile.TemporaryDirectory()
mem = joblib.Memory(cache_dir.name)

# %%
# Now we can use `mem` as a decorator to wrap our function

cached_call_segment = mem.cache(call_segment)

# %%
# Using the cache
# ---------------
# The cashed function can now be used like the normal function before.
# However, repeated calls with the same argument, should be much faster.
#
# Note, by default joblib prints some information whenever, the function is actually called and not loaded from cache.
# We can use that as indicator to see if caching works.

result_dtw = cached_call_segment(dtw, data, sampling_rate_hz)

# %%
# The second call
# ^^^^^^^^^^^^^^^
# We would assume that repeating the above line would load results from disk.
# However, in this case it will **not**.
# Calling `segment` on the `dtw` function has stored the results as attributes and hence, changed the cache.
# To ensure that we actually get a cached call, we will create a clean copy of the dtw object without any results first.
cloned_dtw = dtw.clone()
result_dtw_2 = cached_call_segment(cloned_dtw, data, sampling_rate_hz)

# %%
# As you can see, the second call did not produce any debug output, indicating that the cashed result was used.
#
# Simplifying things
# ------------------
# To ensure that you don't accidentally call the function without cloning the algorithm before, gaitmap has some helper
# functions.
# Both versions below are identical to what we did before.
from gaitmap.utils.caching import cached_call_method, cached_call_action

result_dtw_1 = cached_call_method(dtw, "segment", mem, data=data, sampling_rate_hz=sampling_rate_hz)
result_dtw_2 = cached_call_method(dtw, "segment", mem, data=data, sampling_rate_hz=sampling_rate_hz)

# %%
# Or if you only want to call the primary action method ("segment" in case of dtw), you can use the `cached_call_action`
# function without specifying a method name.
#
# Note, that without clearing the cache before calling these functions, they would actually use the cached results from
# `cached_call_method`, as `cached_call_action` is just a wrapper around it.
mem.clear()

result_dtw_1 = cached_call_action(dtw, mem, data=data, sampling_rate_hz=sampling_rate_hz)
result_dtw_2 = cached_call_action(dtw, mem, data=data, sampling_rate_hz=sampling_rate_hz)

# %%
# Some Note
# ---------
# - If you want to get specific values from the cache, just call the cached function with the same arguments as before
# - Do **not** use you cache as permanent storage of results. It is way to easy to delete it.
# - Clear your cache, before you do your final calculations for a publication!
# - Make sure you add you cache dir to your ".gitignore" file.

# %%
# Finally remove the tempdir
cache_dir.cleanup()
