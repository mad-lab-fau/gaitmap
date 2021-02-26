r"""
.. _custom_dataset:

Custom Dataset
==============

.. warn:: Datasets are still an experimental feature and the API might change at any time.

Gaitmap has the concept of Dataset.
Datasets represent a set of recordings that should all be processed in the same way.
For example the data of multiple participants in a study, multiple days of recording, or multiple gait tests.
The goal of datasets is to provide a consistent interface to access the raw data, meta data and potential reference
information in an object oriented way.
It is up to you to define, what is considered a single "data-point" for your dataset.
Note, that datasets can be arbitrarily nested (e.g. multiple subjects with multiple recordings).

Datasets work best in combination with `Pipelines` and are further compatible with concepts like `GridSearch` and
`cross_validation`.
"""

# %%
# Defining your own dataset
# -------------------------
# Fundamentally you only need to create a subclass of :func:`~gaitmap.future.dataset.Dataset` and define the
# `create_index` method.
# This method should return a dataframe describing all the data-points that should be available in the dataset.
#
# In the following we will create an example dataset, that is independent of the concept of gait analysis,
# but can be used to demonstrate most functionality.
# At the end we will discuss, how gait specific data should be integrated.
#
# We will define an index that contains 5 participants, with 3 recordings each. Recording 3 has 2 trials,
# while the others have only one.
# Note, that we implement this a static index here, but most of the time, you would create the index by e.g. scanning
# and listing the files in your data directory.
# It is important that you don't want to load the entire actual data (e.g. the imu samples) in memory, but just list
# the available data-points in the index.
# Then you can filter the dataset first and load the data once you know which data-points you want to access.
# We will discuss this later in the example.
from itertools import product
from typing import Optional, Union, List

import pandas as pd


trials = list(product(("rec_1", "rec_2", "rec_3"), ("trial_1",)))
trials.append(("rec_3", "trial_2"))
index = [(p, *t) for p, t in product(("p{}".format(i) for i in range(1, 6)), trials)]
index = pd.DataFrame(index, columns=["participant", "recording", "trial"])
index

# %%
# Now we use this index as index for our new dataset.
# To see the dataset in action, we need to create a instance of it.
# It's string representation will show us the most important information.
from gaitmap.future.dataset import Dataset


class CustomDataset(Dataset):
    def create_index(self):
        return index


dataset = CustomDataset()
dataset

# %%
# Subsets
# -------
# When working with a dataset, the first thing is usually to select the data you want to use.
# For this you can primarily use the method `get_subset`.
# Here we want to select only recording 2 and 3 from participant 1 to 4.
# Note that the returned subset is an instance of your dataset class as well.
subset = dataset.get_subset(participant=["p1", "p2", "p3", "p4"], recording=["rec_2", "rec_3"])
subset

# %%
# The subset can then be further filtered.
# For more advanced filter approaches you can also filter the index directly and use a bool-map to index the dataset
example_bool_map = subset.index["participant"].isin(["p1", "p2"])
final_subset = subset.get_subset(bool_map=example_bool_map)
final_subset

# %%
# Iteration and Groups
# --------------------------------
# After selecting the part of the data you want to use, you usually want/need to iterate over the data to apply your
# processing steps.
#
# By default, you can simply iterate over all rows.
# Note, that each row itself is a dataset again, but just with a single entry.
# This can actually be checked using the `is_single` method.
for row in final_subset:
    print(row)
print("This row contains {} data-point".format(len(row)))
# %%
# However, in many cases, we don't want to iterate over all rows, but rather iterate over groups of the datasets (
# e.g. all participants or all tests) individually.
# We can do that in 2 ways (depending on what is needed).
# For example, if we want to iterate over all recordings, we can do this:
for trial in final_subset.iter_level("recording"):
    print(trial)

# %%
# You can see that we get two subsets, one for each recording label.
# But what, if we want to iterate over the participants and the recordings together?
# In these cases, we need to group our dataset first.
# Note that the grouped_subset shows the new groupby columns as index in the representation and the length of the
# dataset is reported to be the number of groups.
grouped_subset = final_subset.groupby(["participant", "recording"])
print("The dataset contains {} groups.".format(len(grouped_subset)))
grouped_subset

# %%
# If we now iterate the dataset, it will iterate over the unique groups.
#
# Grouping also changes the meaning of a "single datapoint".
# Each group reports a shape of `(1,)` independent of the number of rows in each group.
for group in grouped_subset:
    print("This group has the shape {}".format(group.shape))
    print(group)

# %%
# At any point, you can view all unique groups/rows in the dataset using the `groups` attribute.
# The order shown here, is the same order used when iterating the dataset.
# When creating a new subset, the order might change!
grouped_subset.groups

# %%
# Note that for an "un-grouped" dataset, this corresponds to all rows.
final_subset.groups

# %%
# If you want you can also ungroup a dataset again.
# This can be usefull for nested iteration:
for outer, group in enumerate(grouped_subset):
    ungrouped = group.groupby(None)
    for inner, subgroup in enumerate(ungrouped):
        print(outer, inner)
        print(subgroup)

# %%
# Splitting
# ---------
# If you are evaluating algorithms, it is often important to split your data into a train and a test set, or multiple
# distinct sets for a cross-validation.
#
# The `Dataset` objects directly support the `sklearn` helper functions for this.
# For example to split our subset into training and testing we can do the following:
from sklearn.model_selection import train_test_split

train, test = train_test_split(final_subset, train_size=0.5)
print("Train:\n", train)
print("Test:\n", test)

# %%
# Such splitting always occures on data-point level and can therefore be influenced by grouping.
# If we want to split our datasets into training and testing, but only based on the participants, we can do this:
train, test = train_test_split(final_subset.groupby("participant"), train_size=0.5)
print("Train:\n", train)
print("Test:\n", test)

# %%
# In the same way you can use the dataset (grouped or not) with the cross-validation helper functions:
from sklearn.model_selection import KFold

cv = KFold(n_splits=2)
grouped_subset = final_subset.groupby("participant")
for train, test in cv.split(grouped_subset):
    # We only print the train set here
    print(grouped_subset[train])


# %%
# While this works well, it is not always what we want.
# Sometimes, we still want to consider each row a single datapoint, but want to prevent that data of e.g. a single
# participant is partially put into train- and partially into the test-split.
# For this, we can use `GroupKFold` in combination with `dataset.create_group_labels`.
#
# `create_group_labels` generates a unique identifier for each row/group:
group_labels = final_subset.create_group_labels("participant")
group_labels

# %%
# They can then be used as the `group` parameter in `GroupKFold`.
# Now the data of the two participants is never split between train and test set.
from sklearn.model_selection import GroupKFold

cv = GroupKFold(n_splits=2)
for train, test in cv.split(final_subset, groups=group_labels):
    # We only print the train set here
    print(final_subset[train])

# %%
# Adding Data
# -----------
# So far we only operated on the index of the dataset.
# But if we want to run algorithms, we need the actual data (i.e. IMU samples, clinical data, ...).
#
# Because the data and the structure of the data can vary widely from dataset to dataset, it is up to you to implement
# data access.
# It comes down to documentation to ensure that users access the data in the correct way.
#
# However, if you want to make sure your dataset "feels" like part of gaitmap, you should follow these recommendations:
#
# - Data access should be provided via `@properties` on the dataset objects, loading the data on demand.
# - The names of these properties should follow naming used in gaitmap (e.g. `data` for IMU-data) and should return
#   values using the established gaitmap datatypes.
# - The names of values that represents gold standard information (i.e. values you would only have in an evaluation
#   dataset), should have a trailing `_`, which marks them as result in the context of gaitmap.
#
# This should look something like this:
from gaitmap.utils.datatype_helper import MultiSensorData, MultiSensorStrideList


class CustomDataset(Dataset):
    @property
    def data(self) -> MultiSensorData:
        # Some logic to load data from disc
        raise NotImplementedError()

    @property
    def sampling_rate_hz(self) -> float:
        return 204.8

    @property
    def segmented_stride_list_(self) -> MultiSensorStrideList:
        # Some custom logic to load the gold-standard stride list of this validation dataset.
        # Note the trailing `_` in the name.
        raise NotImplementedError()

    def create_index(self):
        return index


# %%
# For each of the data-values you need to decide, on which "level" you provide data access.
# Meaning, do you want/can return data, when there are still multiple participants/recordings in the dataset, or can you
# only return the data, when there is only a single trail of a single participant left.
#
# Usually, we recommend to always return the data on the lowest logical level (e.g. if you recorded separate IMU per
# trail, you should provide access only, if there is just a single trail by a single participant left in the dataset).
# Otherwise, you should throw an error.
# This pattern can be simplified using the `is_single` or `assert_is_single` helper method.
# These helper check based on the provided `groupby_cols` if there is really just a single group/row left with the
# given groupby settings.
#
# Let's say `data` can be accessed on either a `recording` or a `trail` level, and `segmented_stride_list` can only
# be accessed on a `trail` level.
# Than we could do something like this:


class CustomDataset(Dataset):
    @property
    def data(self) -> MultiSensorData:
        # Note that we need to make our checks from the least restrictive to the most restrictive (if there is only a
        # single trail, there is only just a single recording).
        if self.is_single(["participant", "recording"]):
            return "This is the data for participant {} and rec {}".format(*self.groups[0])
        # None -> single row
        if self.is_single(None):
            return "This is the data for participant {}, rec {} and trial {}".format(*self.groups[0])
        raise ValueError(
            "Data can only be accessed when their is only a single recording of a single participant in " "the subset"
        )

    @property
    def sampling_rate_hz(self) -> float:
        return 204.8

    @property
    def segmented_stride_list_(self) -> MultiSensorStrideList:
        # We use assert here, as we don't have multiple options.
        # (We could also used `None` for the `groupby_cols` here)
        self.assert_is_single(["participant", "recording", "trial"], "segmented_stride_list_")
        return "This is the segmented stride list for participant {}, rec {} and trial {}".format(*self.groups[0])

    def create_index(self):
        return index


# %%
# If we select a single trial (row), we can get data and the stride list:
test_dataset = CustomDataset()
single_trial = test_dataset[0]
print(single_trial.data)
print(single_trial.segmented_stride_list_)

# %%
# If we only select a recording, we get an error for the stride list:

# We select only recording 3 here, as it has 2 trials.
single_recording = test_dataset.get_subset(recording="rec_3").groupby(["participant", "recording"])[0]
print(single_recording.data)
try:
    print(single_recording.segmented_stride_list_)
except Exception as e:
    print("ValueError: ", e)

# %%
# Custom parameter
# ----------------
# Often it is required to pass some parameters/configuration to the dataset.
# This could be for example the place where the data is stored or if a specific part of the dataset should be included,
# if some preprocessing should be applied to the data, ... .
#
# Such additional configuration can be provided via a custom `__init__` and is then available for all methods to be
# used.
# Note that you **must** assign the configuration values to attributes with the same name and **must not** forget to
# call `super().__init__`


class CustomDatasetWithConfig(Dataset):
    data_folder: str
    custom_config_para: bool

    def __init__(
        self,
        data_folder: str,
        custom_config_para: bool = False,
        *,
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        self.data_folder = data_folder
        self.custom_config_para = custom_config_para
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        # Use e.g. `self.data_folder` to load the data.
        return index
