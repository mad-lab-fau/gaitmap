===========================
Common Datatypes in Gaitmap
===========================

Gaitmap tries to stick to common data-containers - namely `np.arrays`, `pd.DataFrames`, `dict` - to store all in- and
outputs of the used algorithm.
However, to make it easy for users to handle complex problems (e.g. the analysis of multiple sensors at the same time)
and to make it possible to perform some sanity checks that prevent common issues, a set of certain datatypes - based on
the above mentioned containers - are defined and used throughout the library.
The following explains these data-structures in details to ease to process of preparing your data for the use of gaitmap
and help to understand the outputs.

Datasets
========

Single-Sensor Datasets
----------------------

The term Dataset is used to describe the data-container that holds the raw IMU data.
Six different versions of this container exist aimed at combination of different use cases.

The base container structure is a `pd.DataFrame` with a preset name of columns (:mod:`~gaitmap.utils.consts.SF_COLS`),
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

The above set of columns describe a dataset in the Sensor Frame.
An additional version of the *SingleSensorDataset* exists in the Body Frame.
Its definition is identical to dataset in the sensor frame, except different column names
(:mod:`~gaitmap.utils.consts.BF_COLS`) are expected.
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

Second, for sensor data that is not synchronised gaitmap also support a dictionary based *MultiSensorDatasets*.
Instead of a single dataframe with `MultiIndex` it consists of a dictionary with the sensor names as keys and valid
*SingleSensorDatasets* as values.

For both types simply indexing with the sensor name should returns a valid *SingleSensorDatasets*.

>>> is_single_sensor_dataset(multi_dataset["left_sensor"])
True

To allow for consistent iteration over all sensors the following function can be used to obtain the sensor names
independent of the format of the dataset:

>>> from gaitmap.utils.dataset_helper import get_multi_sensor_dataset_names
>>> get_multi_sensor_dataset_names(multi_dataset)
["left_sensor", "right_sensor"]

Like *SingleSensorDatasets*, *MultiSensorDatasets* can exist in the Body or the Sensor Frame.
However, all single datasets must be in the same frame.
This can be checked using :func:`~gaitmap.utils.dataset_helper.is_multi_sensor_dataset`.

>>> from gaitmap.utils.dataset_helper import  is_multi_sensor_dataset
>>> is_multi_sensor_dataset(multi_dataset, frame="sensor")
True
>>> is_multi_sensor_dataset(multi_dataset, frame="body")
False

All core methods support a *MultiSensorDataset* as input.
This usually means that the method simply iterates over all sensors and provides a separate output for each sensor.
The sensor names can be chosen arbitrarily.
For the future, methods are planned that make active use of multiple sensors at the same time.
These might handle multi-sensor input differently.



