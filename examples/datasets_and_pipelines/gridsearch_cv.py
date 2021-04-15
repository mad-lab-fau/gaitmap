r"""
.. _gridsearch_cv:

GridSearchCV
============

.. warning:: GridSearch and Pipelines are still an experimental feature and the API might change at any time.

When trying to optimize parameters for algorithms that have trainable components, it is required to perform
the parameter search on a validation set (that is separate from the test set used for the final validation).
Even better, is to use a cross validation for this step.
In gaitmap this can be done by using :class:`~gaitmap.future.pipelines.GridSearchCv`.

This example explains how to use this method.
To learn more about the concept, review the :ref:`evaluation guide <algorithm_evaluation>` and the `sklearn guide on
tuning hyperparameters <https://scikit-learn.org/stable/modules/grid_search.html#grid-search>`_.

"""
import random
from typing import Optional

import numpy as np
import pandas as pd


random.seed(1)  # We set the random seed for repeatable results

# %%
# Dataset
# -------
# As always, we need a dataset, a pipeline, and a scoring method for a parameter search.
# We ruse the dataset used in other pipeline examples.
from gaitmap.future.dataset import Dataset
from gaitmap.example_data import get_healthy_example_imu_data, get_healthy_example_stride_borders


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


# %%
# The Pipeline
# ------------
# We use a gait segmentation pipeline, that is explained in more detail in the :ref:`optimize_pipelines` example.
# However, we modify this pipeline in one key way:
# We add an additional parameter `n_train_strides` that controls how many randomly selected strides should be used
# during training.
# Modifying this parameter, will change the result of the `self_optimize` step.
from gaitmap.future.pipelines import OptimizablePipeline
from gaitmap.stride_segmentation import DtwTemplate, BarthOriginalTemplate, create_interpolated_dtw_template, BarthDtw
from gaitmap.utils.datatype_helper import SingleSensorStrideList
from gaitmap.utils.coordinate_conversion import convert_left_foot_to_fbf, convert_right_foot_to_fbf


class MyPipeline(OptimizablePipeline):
    max_cost: float
    template: DtwTemplate
    n_train_strides: Optional[int]

    segmented_stride_list_: SingleSensorStrideList
    cost_func_: np.ndarray

    def __init__(
        self,
        max_cost: float = 3,
        template: DtwTemplate = BarthOriginalTemplate(),
        n_train_strides: Optional[int] = None,
    ):
        self.max_cost = max_cost
        self.template = template
        self.n_train_strides = n_train_strides

    def self_optimize(self, dataset: MyDataset, **kwargs):
        # Our training consists of cutting all strides from the dataset and then creating a new template from all
        # strides in the dataset

        # We expect multiple datapoints in the dataset
        all_strides = []
        sampling_rate = dataset[0].sampling_rate_hz
        for dp in dataset:
            data = self._convert_cord_system(dp.data, dp.groups[0][1]).filter(like="gyr")
            # We take the segmented stride list from the reference/ground truth here
            reference_stride_list = dp.segmented_stride_list_
            all_strides.extend([data.iloc[s:e] for _, (s, e) in reference_stride_list[["start", "end"]].iterrows()])
        # This is the new part:
        if self.n_train_strides:
            all_strides = random.sample(all_strides, self.n_train_strides)

        template = create_interpolated_dtw_template(all_strides, sampling_rate_hz=sampling_rate)
        # We normalize the template the same way as `BarthOriginalTemplate` is normalized to make them comparable
        scaling = np.max(np.abs(template.get_data().to_numpy()))
        template.scaling = scaling
        template.data /= scaling

        # We modify the pipeline instance and set the new template
        self.template = template
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
# The scorer is identical to the scoring function used in the other examples.
# The F1-score is still the most important parameter for our comparison.
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
# Like with a normal cross validation, we need to decide on the number of folds and type of splits.
# In gaitmap we support all cross validation iterators provided in :ref:`sklearn
# <https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators>`.
#
# In this example we only have two datapoints.
# This means, we can only use a 2-fold cross-validation:
from sklearn.model_selection import KFold

cv = KFold(n_splits=2)

# %%
# The Parameters
# --------------
# The pipeline above exposes a couple of parameters.
# The `template` will be modified during training.
# The `n_train_strides` controls how many strides are used during training and hence, directly effects the outcome.
# Therefore, we consider it a **Hyper**parameter.
# The `max_cost` parameter is important when matching, but does not used, and hence does not influence the result of
# the training.
# Therefore, we consider it a normal or **pure** parameter.
#
# The Gridsearch will be performed over the `n_train_strides` and the `max_cost` parameter.
# For the `n_train_strides` we test the values `None` (all strides) and 1 (single stride) to make sure that we will
# see a performance difference between the two options.
# Note, that we must not Gridsearch parameters that are modified during training.
from sklearn.model_selection import ParameterGrid


parameters = ParameterGrid({"max_cost": [3, 5], "n_train_strides": [None, 1]})  # None means all strides.

# %%
# GridSearchCV
# ------------
# Setting up the GridSearchCV object is similar to the normal GridSearch, we just need to add the additional `cv`
# parameter.
# Then we can simply run the search using the `optimize` method.
from gaitmap.future.pipelines import GridSearchCV

gs = GridSearchCV(pipeline=MyPipeline(), parameter_grid=parameters, scoring=score, cv=cv, return_optimized="f1_score")
gs = gs.optimize(MyDataset())

# %%
# Results
# -------
# The output is also comparable to the output of the GridSearch.
# The main results are stored in the `cv_results_` parameter.
# But instead of just a single performance value per parameter, we get one value per fold and the mean and std over
# all folds.
results = gs.cv_results_
results_df = pd.DataFrame(results)

results_df

# %%
# The mean score is the primary parameter used to select the best parameter combi (if `return_optimized` is True).
# All other values performance values are just there to provide further inside.

results_df[["mean_test_precision", "mean_test_recall", "mean_test_f1_score"]]

# %%
# For even more insight, you can inspect the scores per datapoint:

results_df.filter(like="test_single")

# %%
# If `return_optimized` was set to True (or the name of a score), a final optimization is performed using the best
# set of parameters and **all** the available data.
# The resulting pipeline will be stored in `optimizable_pipeline_`.
print("Best Para Combi:", gs.best_params_)
print("Paras of optimized Pipeline:", gs.optimized_pipeline_.get_params())

# %%
# To run the optmized pipeline, we can directly use the `run`/`safe_run` method on the GridSearch object.
# This makes it possible to use the `GridSearch` as a replacement for your pipeline object with minimal code changes.
#
# If you would try to call `run`/`safe_run` (or `score` for that matter), before the optimization, an error is raised.
segmented_stride_list = gs.safe_run(MyDataset()[0]).segmented_stride_list_
segmented_stride_list


# %%
# Pure Parameters
# ---------------
# As mentioned above, some parameters in this search do not affect the outcome of the optimization step.
# We call these parameters *pure* parameters.
# Still, `self_optimize` was called once for each parameter combination above.
# In this example this didn't really matter, because the optimization was fast, but in other cases it could be very
# wasteful to rerun the optimization multiple times, even though the outcome would be identical.
#
# A better approach would be to only run the training for all parameter combinations that are actually expected to
# change its output and set the rest of the parameters only during the `run` step.
# To learn more about this approach review the concept of *Group 3 algorithms* in the
# :ref:`evaluation guide <algorithm_evaluation>`.
#
# `GridSearchCV` has the option to make exactly this optimization.
# However, it can not magically know, which parameters should be considered "pure".
# This information needs to be provided manually via the `pure_parameter_names` parameter.
# If provided, the output of the optimization will be cached and reused, if only pure parameters are modified.
#
# .. warning :: Setting the wrong parameters as *pure* can result in hard to debug issues.
#               Make sure you fully understand your pipeline, before using this option and compare the results of you
#               pipeline on a subset of your data with and without the option before using it!
#
# In our case, `max_cost` is a pure parameter.
# We will rerun the pipeline below and mark `max_cost` as a pure parameter explicitly.
# We will also set the verbosity to 2, to see the caching in action.
# Note, that we also need to reset the random seed, otherwise we would get different results than above.
random.seed(1)

gs_cached = GridSearchCV(
    pipeline=MyPipeline(),
    parameter_grid=parameters,
    scoring=score,
    pure_parameter_names=["max_cost"],
    cv=cv,
    return_optimized="f1_score",
    verbose=2,
)
gs_cached = gs_cached.optimize(MyDataset())
cached_results = gs_cached.cv_results_

# %%
# When inspecting the debug output above, we can see that the function `cachable_optimize` (which handles the
# optimization internally) was called 4 times for all combinations of the hyper parameter and data folds.
# Then these cached results were used in 4 further cases, which correspond to the already run combinations,
# but with a different value for `max_cost`.
# This means, we saved 50% of all calls to `self_optimize` of our pipeline.
#
# Just to make sure, you can see that the results below are still identical to our first run.
pd.DataFrame(cached_results).filter(like="mean")
