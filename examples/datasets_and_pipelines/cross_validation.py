r"""
.. _cross_validation:

Cross Validation
================

.. warning:: GridSearch and Pipelines in gaitmap are deprecated! Use tpcp instead!

Whenever using some sort of trainable algorithm it is important to clearly separate the training and the testing data to
get an unbiased result.
Usually this is achieved by a train-test split.
However, if you don't have that much data, there is always a risk that one random train-test split, will provide
better (or worse) results than another.
In this cases it is a good idea to use cross-validation.
In this procedure, you perform multiple train-test splits and average the results over all "folds".
For more information see our :ref:`evaluation guide <algorithm_evaluation>` and the `sklearn guide on cross
validation <https://scikit-learn.org/stable/modules/cross_validation.html>`_.

In this example, we will learn how to use the :func:`~tpcp.optimize.cross_validate` function implemented in
gaitmap.
For this, we will redo the example on :ref:`optimizable pipelines <optimize_pipelines>` but we will perform the final
evaluation via cross-validation.
If you want to have more information on how the dataset and pipeline is built, head over to this example.
Here we will just copy the code over.
"""
import numpy as np
import pandas as pd
from tpcp import CloneFactory, Dataset, OptimizableAlgorithm, OptimizableParameter, OptimizablePipeline, Parameter

from gaitmap.data_transform import TrainableAbsMaxScaler
from gaitmap.example_data import get_healthy_example_imu_data, get_healthy_example_stride_borders
from gaitmap.stride_segmentation import BarthDtw, BarthOriginalTemplate, DtwTemplate, InterpolatedDtwTemplate
from gaitmap.utils.array_handling import iterate_region_data
from gaitmap.utils.coordinate_conversion import convert_left_foot_to_fbf, convert_right_foot_to_fbf
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


class MyPipeline(OptimizablePipeline):
    max_cost: Parameter[float]
    template: OptimizableParameter[DtwTemplate]

    segmented_stride_list_: SingleSensorStrideList
    cost_func_: np.ndarray

    # We need to wrap the template in a `CloneFactory` call here to prevent issues with mutable defaults!
    def __init__(self, max_cost: float = 3, template: DtwTemplate = CloneFactory(BarthOriginalTemplate())):
        self.max_cost = max_cost
        self.template = template

    def self_optimize(self, dataset: MyDataset, **kwargs):
        if not isinstance(self.template, OptimizableAlgorithm):
            raise ValueError(
                "The template must be optimizable! If you are using a fixed template (e.g. "
                "BarthOriginalTemplate), switch to an optimizable base classe."
            )
        # Our training consists of cutting all strides from the dataset and then creating a new template from all
        # strides in the dataset

        # We expect multiple datapoints in the dataset
        sampling_rate = dataset[0].sampling_rate_hz
        # We create a generator for the data and the stride labels
        data_sequences = (
            self._convert_cord_system(datapoint.data, datapoint.groups[0][1]).filter(like="gyr")
            for datapoint in dataset
        )
        stride_labels = (datapoint.segmented_stride_list_ for datapoint in dataset)

        stride_generator = iterate_region_data(data_sequences, stride_labels)

        self.template.self_optimize(
            stride_generator, columns=["gyr_pa", "gyr_ml", "gyr_si"], sampling_rate_hz=sampling_rate
        )
        return self

    def _convert_cord_system(self, data, foot):
        converter = {"left": convert_left_foot_to_fbf, "right": convert_right_foot_to_fbf}
        return converter[foot](data)

    def run(self, datapoint: MyDataset):
        # `datapoint.groups[0]` gives us the identifier of the datapoint (e.g. `("test", "left")`).
        # And `datapoint.groups[0][1]` is the foot.
        data = self._convert_cord_system(datapoint.data, datapoint.groups[0][1])

        dtw = BarthDtw(max_cost=self.max_cost, template=self.template)
        dtw.segment(data, datapoint.sampling_rate_hz)

        self.segmented_stride_list_ = dtw.stride_list_
        self.cost_func_ = dtw.cost_function_
        return self


# %%
# The Scorer
# ----------
# When using cross validation, we usually want to calculate performance parameters for each fold, so that we can
# calculate the average performance as our expected "generalization" error.
# For this example, we will use the "precision", the "recall" and the "f1_score" to score the stride detection
# performance.
from gaitmap.evaluation_utils import evaluate_segmented_stride_list, precision_recall_f1_score


def score(pipeline: MyPipeline, datapoint: MyDataset):
    pipeline.safe_run(datapoint)
    matches_df = evaluate_segmented_stride_list(
        ground_truth=datapoint.segmented_stride_list_, segmented_stride_list=pipeline.segmented_stride_list_
    )
    return precision_recall_f1_score(matches_df)


# %%
# Data Splitting
# --------------
# Before performing a cross validation, we need to decide on the number of folds and type of splits.
# In gaitmap we support all cross validation iterators provided in :ref:`sklearn
# <https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators>`.
#
# In this example we only have two datapoints.
# This means, we can only use a 2-fold cross-validation:
from sklearn.model_selection import KFold

cv = KFold(n_splits=2)

# %%
# Cross Validation
# ----------------
# Now we have all the pieces for the final cross validation.
# First we need to create instances of our data and pipeline.
# Then we need to wrap our pipeline instance into an :class:`~tpcp.Optimize` wrapper.
# Finally we can call `cross_validate`.
from tpcp.optimize import Optimize
from tpcp.validate import cross_validate

ds = MyDataset()
pipe = MyPipeline(template=InterpolatedDtwTemplate(scaling=TrainableAbsMaxScaler()))
optimizable_pipe = Optimize(pipe)

results = cross_validate(optimizable_pipe, ds, scoring=score, cv=cv, return_optimizer=True, return_train_score=True)
result_df = pd.DataFrame(results)
result_df

# %%
# Understanding the Results
# -------------------------
# The cross validation provides a lot of outputs (some of them can be disabled using the function parameters).
# To simplify things a little, we will split the output into four parts:
#
# The main output are the test set performance values.
# Each row corresponds to performance in respective fold.
performance = result_df[["test_precision", "test_recall", "test_f1_score"]]
performance

# %%
# The final generalization performance you would report is usually the average over all folds.
# The STD can also be interesting, as it tells you how stable your optimization is and if your splits provide
# comparable data distributions.
generalization_performance = performance.agg(["mean", "std"])
generalization_performance

# %%
# If you need more insight into the results (e.g. when the std of your results is high), you can inspect the
# individual score for each data point.
# In this example this is only a list with a single element per score, as we only had a single datapoint per fold.
# In a real scenario, this will be a list of all datapoints.
# Inspecting this list can help to identify potential issues with certain parts of your dataset.
# To link the performance values to a specific datapoint, you can look at the `test_data_labels` field.
single_performance = result_df[
    ["test_single_precision", "test_single_recall", "test_single_f1_score", "test_data_labels"]
]
single_performance

# %%
# Even further insight is provided by the train results (if activated in parameters).
# These are the performance results on the train set and can indicate if the training provided meaningful results and
# can also indicate over-fitting, if the performance of the test set is much worse than the performance on the train
# set.
train_performance = result_df[
    [
        "train_precision",
        "train_recall",
        "train_f1_score",
        "train_single_precision",
        "train_single_recall",
        "train_single_f1_score",
        "train_data_labels",
    ]
]
train_performance

# %%
# The final level of debug information is provided via the timings (note the long runtime in fold 0 can be explained
# by the jit-compiler used in `BarthDtw`) ...
timings = result_df[["score_time", "optimize_time"]]
timings

# %%
# ... and the optimized pipeline object.
# This is the actual trained object generated in this fold.
# You can apply it to other data for testing or inspect the actual object for further debug information that might be
# stored on it.
optimized_pipeline = result_df["optimizer"][0]
optimized_pipeline

#%%
optimized_pipeline.optimized_pipeline_.get_params()

# %%
# Further Notes
# -------------
# We also support grouped cross validation.
# Check the :ref:`dataset guide <custom_dataset>` on how you can group the data before cross-validaiton or generate
# data labels to be used with `GroupedKFold`.
#
# `Optimize` is just an example of an optimizer that can be passed to cross validation.
# You can pass any gaitmap optimizer like `GridSearch` or `GridSearchCV`.
