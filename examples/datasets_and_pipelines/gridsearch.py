r"""
.. _grid_search:

Grid Search optimal Algorithm Parameter
=======================================

.. warning:: GridSearch and Pipelines are still an experimental feature and the API might change at any time.


"""
import pandas as pd

# %%
# To perform a GridSearch (or any other form of parameter optimization in Gaitmap), we first need to have a
# **Dataset**, a **Pipeline** and a **score** function.
#
# 1. The Dataset
# --------------
# Datsets wrap multiple gait recordings into an easy to use interface that can be passed around between the higher
# level gaitmap functions.
# Learn more about this :ref:`here <custom_dataset>`.
# If you are lucky, you do not need to create the dataset on your own, but someone has already created a gaitmap dataset
# for the data you want to use.
#
# Here we are going to create a simple "dummy" dataset, that just uses the left and the right-foot recording of the
# example data to simulate a dataset of one participant with two recordings.
# In addition to the raw IMU data, the dataset also exposes reference stride borders that we will use to judge the
# performance of our algorithm.
# (Note, usually, you wouldn't split the left and the right foot into separate recordings, as gaitmap can handle
# multiple sensor recordings at once).
#
# For our GridSearch, we need an instance of this dataset.
from gaitmap.example_data import get_healthy_example_imu_data, get_healthy_example_stride_borders
from gaitmap.future.dataset import Dataset
from gaitmap.utils.datatype_helper import SingleSensorStrideList


class MyDataset(Dataset):
    @property
    def sampling_rate_hz(self) -> float:
        return 204.8

    @property
    def data(self):
        self.assert_is_single(None, "data")
        return get_healthy_example_imu_data()[self.index.iloc[0]["foot"] + "_sensor"]

    @property
    def segmented_stride_list_(self):
        self.assert_is_single(None, "data")
        return get_healthy_example_stride_borders()[self.index.iloc[0]["foot"] + "_sensor"].set_index("s_id")

    def create_index(self) -> pd.DataFrame:
        return pd.DataFrame({"participant": ["test", "test"], "foot": ["left", "right"]})


dataset = MyDataset()
dataset

# %%
# 2. The Pipeline
# ---------------
# The pipeline simply defines what algorithms we want to run on our data and defines, which parameters of the pipeline
# you still want to be able to modify (e.g. to optimize in the GridSearch).
#
# The pipeline usually needs 3 things:
#
# 1. It needs to be subclass of `SimplePipeline`.
# 2. It needs to have a `run` method that runs all the algorithmic steps and stores the results as class attributes.
#    The `run` method should expect only a single data point (in our case a single recording of one sensor) as input.
# 3. A `init` that defines all parameters that should be adjustable. Note, that the names in the function signature of
#    the `init` method, **must** match the corresponding attribute names (e.g. `max_cost` -> `self.max_cost`).
#
# Here we simply transform the data into the correct coordinate system depending on the foot and apply `BarthDtw` to
# identify the start and the end of all strides in the recording.
# The parameter `max_cost` of this algorithm is exposed and will be optimized as part of the GridSearch.
#
# For the final GridSearch, we need an instance of the pipeline object.
from gaitmap.future.pipelines import SimplePipeline
from gaitmap.stride_segmentation import BarthDtw
from gaitmap.utils.coordinate_conversion import convert_left_foot_to_fbf, convert_right_foot_to_fbf


class MyPipeline(SimplePipeline):
    max_cost: float

    segmented_stride_list_: SingleSensorStrideList

    def __init__(self, max_cost: float = 3):
        self.max_cost = max_cost

    def run(self, datapoint: MyDataset):
        converter = {"left": convert_left_foot_to_fbf, "right": convert_right_foot_to_fbf}
        # `datapoint.groups[0]` gives us the identifier of the datapoint (e.g. `("test", "left")`).
        # And ``datapoint.groups[0][1]` is the foot.
        data = converter[datapoint.groups[0][1]](datapoint.data)

        dtw = BarthDtw(max_cost=self.max_cost)
        dtw.segment(data, datapoint.sampling_rate_hz)

        self.segmented_stride_list_ = dtw.stride_list_
        return self


pipe = MyPipeline()


# %%
# 3. The scorer
# -------------
# In the context of a gridsearch, we want to calculate the performance of our algorithm and rank the different
# parameter candidates accordingly.
# This is what our score function is for.
# It gets a pipeline object (**without** results!) and a data point (i.e. a single recording) as input and should
# return a some sort of performance metric.
# A higher value is always considered better.
# If you want to calculate multiple performance measures, you can also return a dictionary of such values.
# In any case, the performance for a specific parameter combination in the GridSearch will be calculated as the mean
# over all datapoints.
# (Note, if you want to change this, you can create custom subclasses of `GaitmapScorer`).
#
# A typical score function will first call `safe_run` (which calls `run` internally) on the pipeline and then
# compare the output with some reference.
# This reference should be supplied as part of the dataset.
#
# Instead of using a function as scorer (shown here), you can also implement a method called `score` on your pipeline.
# Then just pass `None` (which is the default) for the `scoring` parameter in the GridSearch (and other optimizers).
# However, a function is usually more flexible.
#
# In this case we compare the calculated stride lists with the reference and identify which strides were correctly
# found.
# Based on these matches, we calculate the precision, the recall, and the f1-score using some helper functions.
from gaitmap.evaluation_utils import evaluate_segmented_stride_list, precision_recall_f1_score


def score(pipeline: MyPipeline, datapoint: MyDataset):
    # We use the `safe_run` wrapper instead of just run. This is always a good idea.
    pipeline.safe_run(datapoint)
    matches_df = evaluate_segmented_stride_list(
        ground_truth=datapoint.segmented_stride_list_, segmented_stride_list=pipeline.segmented_stride_list_
    )
    return precision_recall_f1_score(matches_df)


# %%
# The Parameters
# --------------
# The last step before running the GridSearch, is to select the parameters we want to test for each dataset.
# For this, we can directly use sklearn's `ParameterGrid`.
#
# In this example, we will just test two values for the `max_cost` threshold.
from sklearn.model_selection import ParameterGrid

parameters = ParameterGrid({"max_cost": [3, 5]})

# %%
# Running the GridSearch
# ----------------------
# Now we have all the pieces to run the GridSearch.
# After initializing, we can use `optimize` to run the GridSearch.
#
# .. note:: If the score function returns a dictionary of scores, `rank_scorer` must be set to the name of the score,
#           that should be used to decide on the best parameter set.
from gaitmap.future.pipelines import GridSearch

gs = GridSearch(pipe, parameters, scoring=score, return_optimized="f1_score")
gs = gs.optimize(MyDataset())

# %%
# The main results are stored in `gs_results_`.
# It shows the mean performance per parameter combination, the rank for each parameter combination and the
# performance for each individual data point (in our case a single recording of one sensor).
results = gs.gs_results_
pd.DataFrame(results)

# %%
# Further, the `optimized_pipeline_` parameter holds an instance of the pipeline initialized with the best parameter
# combination.
print("Best Para Combi:", gs.best_params_)
print("Paras of optimized Pipeline:", gs.optimized_pipeline_.get_params())

# %%
# To run the optmized pipeline, we can directly use the `run`/`safe_run` method on the GridSearch object.
# This makes it possible to use the `GridSearch` as a replacement for your pipeline object with minimal code changes.
#
# If you would try to call `run`/`safe_run` (or `score` for that matter), before the optimization, an error is raised.
segmented_stride_list = gs.safe_run(dataset[0]).segmented_stride_list_
segmented_stride_list
