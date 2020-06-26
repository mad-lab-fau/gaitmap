===========================
Common Datatypes in Gaitmap
===========================

Gaitmap tries to stick to common data-containers - namely `np.arrays`, `pd.DataFrames`, `dict` - to store all in- and
outputs of the used algorithm.
However, based on the above mentioned containers, a set of certain data-types are defined and used throughout the
library.
This makes it easy for users to handle complex problems (e.g. the analysis of multiple sensors at the same time) and
makes it possible to perform sanity checks that prevent common issues.
The following explains these data-structures in details to ease to process of preparing your data for the use of gaitmap
and help to understand the outputs.

Units
=====

Before talking about data-types the physical units for all values stored in these data-types should be clear.
The following table provides an overview over the commonly used values types and there units.

.. table:: Common Units

   =============  ======================
   Value          Unit
   =============  ======================
   Acceleration   m/s^2
   Rotation Rate  deg/s
   Velocity       m/s
   Distance       m
   Time           s or # (see "Further Rules" below)
   Sampling Rate  Hz
   =============  ======================

Further rules:

- If a method requires a parameter in a given unit, append it with a common short-hand name in this unit (e.g.
  `windowsize_ms` would expect a value in milliseconds).
- Time is either specified in seconds (s) for user facing durations (e.g. stride time), but time points in intermediate
  results (e.g. biomechanical events) are typically specified in samples since the start of the measurement (#).

Datasets
========

The term Dataset is used to describe the data-container that holds the raw IMU data.
Six different versions of this container exist aimed at combination of different use cases.

Single-Sensor Datasets
----------------------

The base container structure is a `pd.DataFrame` with a preset name of columns (:obj:`~gaitmap.utils.consts.SF_COLS`),
which are defined in the `consts` module.
The names (shown below) should be self explanatory.

>>> from gaitmap.utils.consts import SF_COLS
>>> SF_COLS
['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']

Every pandas dataframe that has at least these columns is considered a *SingleSensorDataset* in gaitmap.
To check the compliance with this rule :func:`~gaitmap.utils.dataset_helper.is_single_sensor_dataset` can be used.

>>> from gaitmap.utils.dataset_helper import is_single_sensor_dataset
>>> dataset = ...
>>> is_single_sensor_dataset(dataset, frame="sensor")
True

The above set of columns describes a dataset in the Sensor Frame.
An additional version of the *SingleSensorDataset* exists in the Body Frame.
Its definition is identical to dataset in the sensor frame, except different column names
(:obj:`~gaitmap.utils.consts.BF_COLS`) are expected.
For the concept of Sensor and Body Frame and how to convert between these frames, refer to
:ref:`Coordinate System Guide <coordinate_systems>`.

>>> from gaitmap.utils.consts import BF_COLS
>>> BF_COLS
['acc_pa', 'acc_ml', 'acc_si', 'gyr_pa', 'gyr_ml', 'gyr_si']

Algorithms that require data to be in the BF will use the following check to ensure that correct data is passed.

>>> from gaitmap.utils.dataset_helper import is_single_sensor_dataset
>>> bf_dataset = ...
>>> is_single_sensor_dataset(bf_dataset, frame="body")
True

Multi-Sensor Datasets
---------------------

*MultiSensorDatasets* are combinations of multiple *SingleSensorDatasets*.
Hence, they need to carry the data of each sensor and a unique sensor name to address the data of each sensor.
Gaitmap supports two types of data-containers for this use-case:

First, for sensor data that is fully synchronised (i.e. the data of all sensors have the same index and the same number
of samples), gaitmap uses a `pd.DataFrame` with a :class:`~pandas.MultiIndex` as columns.
The first level (`level=0`) provides the sensor name and the second level the typical columns for the sensor data.

>>> from gaitmap.example_data import get_healthy_example_imu_data
>>> multi_dataset = get_healthy_example_imu_data()
>>> multi_dataset.head(1).sort_index(axis=1)
sensor left_sensor                         right_sensor
axis         acc_x     acc_y    ...        acc_x    acc_y     ...
0.0       0.880811  2.762208    ...        0.311553 -2.398646 ...

Second, for sensor data that is not synchronised gaitmap also supports a dictionary based *MultiSensorDatasets*.
Instead of a single dataframe with `MultiIndex` it consists of a dictionary with the sensor names as keys and valid
*SingleSensorDatasets* as values.

For both types simply indexing with the sensor name should return a valid *SingleSensorDatasets*.

>>> is_single_sensor_dataset(multi_dataset["left_sensor"])
True

To allow for consistent iteration over all sensors the following function can be used to obtain the sensor names
independent of the format of the dataset:

>>> from gaitmap.utils.dataset_helper import get_multi_sensor_dataset_names
>>> get_multi_sensor_dataset_names(multi_dataset)
["left_sensor", "right_sensor"]

Like *SingleSensorDatasets*, *MultiSensorDatasets* can exist in the Body or the Sensor Frame.
However, all single-sensor datasets in a *MultiSensorDataset* must be in the same frame.
This can be checked using :func:`~gaitmap.utils.dataset_helper.is_multi_sensor_dataset`.

>>> from gaitmap.utils.dataset_helper import is_multi_sensor_dataset
>>> is_multi_sensor_dataset(multi_dataset, frame="sensor")
True
>>> is_multi_sensor_dataset(multi_dataset, frame="body")
False

All core methods support a *MultiSensorDataset* as input.
This usually means that the method simply iterates over all sensors and provides a separate output for each sensor.
The sensor names can be chosen arbitrarily.
For the future, methods are planned that make active use of multiple sensors at the same time.
These might handle multi-sensor input differently.

.. _stride_list_guide:

Stride Lists
============

At some point during most gait analysis pipelines it is important to extract the start and end of each stride as well as
relevant events within these strides.
Such information is stored in a *StrideList*.

A *SingleSensorStrideList* is just a `pd.DataFrame` that should at least have the columns defined by
:obj:`~gaitmap.utils.consts.SL_COLS`.
Instead of beeing a column, `s_id` can also be the name of the index.
This is the default in all cases were stride lists are generated during the pipeline.
However, all aglorithms that take a stride list as input support both formats (index or column)
Independent of that, `s_id` index or column should contain a unique identifier for each stride in the stride list.
All other columns should provide values in samples since the start of the recording (not the start of the stride!)

>>> from gaitmap.utils.consts import SL_COLS
>>> SL_COLS
['s_id', 'start', 'end']

Depending of the type of stride list, more columns are expected.
Required additional columns are documented in :obj:`~gaitmap.utils.consts.SL_ADDITIONAL_COLS`.

>>> from gaitmap.utils.consts import SL_ADDITIONAL_COLS
>>> SL_ADDITIONAL_COLS
{
    "min_vel": ["pre_ic", "ic", "min_vel", "tc"],
    "segmented": ["ic", "min_vel", "tc"],
    "ic": ["ic", "min_vel", "tc"],
}

At the moment three types of strides lists are supported besides the basic one.
The `min_vel` and the `ic` describe stride lists in which each stride starts and stops with the respective event.
The `segmented` stride list expects that the start and the end of each stride corresponds to some time point between the
`min_vel` and the `tc`.
For more details on the `min_vel` strides see :class:`~gaitmap.event_detection.RamppEventDetection` and for the
`segmented` strides see :class:`~gaitmap.stride_segmentation.BarthDtw`.

The format of a stride list can be checked using :func:`~gaitmap.utils.dataset_helper.is_single_sensor_stride_list`.

>>> from gaitmap.utils.dataset_helper import is_single_sensor_stride_list
>>> simple_stride_list = ...
>>> is_single_sensor_stride_list(simple_stride_list, stride_type="any")
True

>>> min_vel_stride_list = ...
>>> is_single_sensor_stride_list(simple_stride_list, stride_type="min_vel")
True

As for the dataset types, a multi-sensor of the *StrideList* exists, too.
Because even two synchronised sensors can contain a different amount of strides, only a dictionary based version of the
*MultiSensorStrideList* is supported.
It consists of a dictionary with the sensor names as keys and valid *SingleSensorStrideLists* as values.
Its format can be validated using :func:`~gaitmap.utils.dataset_helper.is_multi_sensor_stride_list`.

>>> from gaitmap.utils.dataset_helper import is_multi_sensor_stride_list
>>> multi_sensor_stride_list = {"sensor1": ..., "sensor2": ...}
>>> is_multi_sensor_stride_list(multi_sensor_stride_list, stride_type="any")
True

Depending on the stride type the expected order of events changes as well.
This order is documented in :obj:`~gaitmap.utils.consts.SL_EVENT_ORDER`.

>>> from gaitmap.utils.consts import SL_EVENT_ORDER
>>> SL_EVENT_ORDER
{
    "segmented": ["tc", "ic", "min_vel"],
    "min_vel": ["pre_ic", "min_vel", "tc", "ic"],
    "ic": ["ic", "min_vel", "tc"],
}

The normal format check shown above does not check if the values in the stride list follow this order.
However, you can use :func:`~gaitmap.utils.stride_list_conversion.enforce_stride_list_consistency` to remove strides
with invalid event order.

Further, it is possible to convert a segmented stride list into an "min_vel" or "ic" stride list using
:func:`~gaitmap.utils.stride_list_conversion.convert_segmented_stride_list`.
Note that conversions between "min_vel" and "ic" is not supported as this would lead to the unneeded removal of strides.

Position and Orientation Lists
==============================

# TODO: Update to reflect proper world frame coordinates. Also change names of columns in the entire package.

To calculate spatial parameters usually the orientation and the position of a sensor need to be estimated first.
This can usually not be done over the entire duration of a recording, as this would result in a large drift error.
Therefore, this estimation is rather just performed for shorter sections such as a single stride or a gait sequence.
The structure of the position and orientation lists reflect these.

Both, the *SingleSensorOrientationList* and the *SingleSensorPositionList* are `pd.DataFrames` with a
:class:`~pandas.MultiIndex` index.
The first level of this double index (`level=0`) is a unique identifier of the stride or gait sequence that is used as
basis of the estimation.
The difference between stride and gaitsequence level estimations is indicated based on the level name of the index,
which is either `s_id` for strides, or `gs_id` for gait sequences.
However, only stride based lists are properly supported at the moment.
Note that the exact definition of a gait sequence depends on the algorithm that detected it.
The second level of the index indicates the sample (starting from 0) within each integration period.

>>> from gaitmap.example_data import get_healthy_example_orientation
>>> get_healthy_example_orientation()['left_sensor']
                   qx        qy        qz        qw
s_id sample
0    0      -0.077640 -0.025560 -0.080004 -0.993438
     1      -0.077347 -0.025167 -0.080207 -0.993454
...               ...       ...       ...       ...
1    0     0.405476  0.132966  0.886753 -0.177700
     1     0.442030  0.126231  0.868311 -0.186309
...               ...       ...       ...       ...

Alternatively to being part of the index, `s_id` and `sample` can also be regular columns.
Methods that take Orientation and Postion lists as inputs can use :func:`~gaitmap.utils.dataset_helper.set_correct_index`
to unify both formats.

>>> from gaitmap.utils.dataset_helper import set_correct_index
>>> orientation_list = ...
>>> unified_format_orientation_list = set_correct_index(orientation_list, ["s_id", "sample"])

Orientation and Position lists only differ based on their expected columns.
Orientation lists are expected to have all columns specified in :obj:`~gaitmap.utils.consts.GF_ORI` and Position lists
all columns specified in :obj:`~gaitmap.utils.consts.GF_POS`.

# TODO: Add for Orientation
>>> from gaitmap.utils.consts import GF_POS
>>> GF_POS
['pos_x', 'pos_y', 'pos_z']

>>> from gaitmap.utils.consts import GF_ORI
>>> GF_ORI
['q_x', 'q_y', 'q_z', 'q_w']

To validate the correctness of these data objectes, :func:`~gaitmap.utils.dataset_helper.is_single_sensor_position_list`
and :func:`~gaitmap.utils.dataset_helper.is_single_sensor_orientation_list` can be used, respectively.
These functions call `:func:`~gaitmap.utils.dataset_helper.set_correct_index` internally and hence, support both
possible dataframe formats that are described above.

>>> from gaitmap.utils.dataset_helper import is_single_sensor_orientation_list
>>> orientation_list = ...
>>> is_single_sensor_orientation_list(orientation_list)
True

Additionally, a multi-sensor version exists for both types of lists.
They follow the dictionary structure introduced for the stride list.
:func:`~gaitmap.utils.dataset_helper.is_multi_sensor_position_list` and
:func:`~gaitmap.utils.dataset_helper.is_multi_sensor_orientation_list` can be used to validate these formats.

>>> from gaitmap.utils.dataset_helper import is_single_sensor_orientation_list
>>> multi_sensor_orientation_list = {"sensor1": ..., "sensor2": ...}
>>> is_single_sensor_orientation_list(multi_sensor_orientation_list, stride_type="any")
True
