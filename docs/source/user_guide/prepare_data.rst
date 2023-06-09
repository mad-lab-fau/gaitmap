.. _prepare_data:

Prepare your own data
=====================

Gaitmap explicitly does not support any specific sensors of data formats.
Instead, we assume that the user loads their data into `supported data objects <datatypes>`_ and prepare the data to
have the `correct units <units>`_ and `coordinate systems <coordinate_systems>`_.

In the following, we will describe in detail, how to get your data ready to work with gaitmap.

Some testing data
-----------------
In case you don't have data on your own, but you want to play around with the algorithms, we provide some testing data
that can be loaded using `gaitmap.example_data` (see :ref:`here <example_data>`).

Before you can use this data, you need to download the test files from `here <https://github.com/mad-lab-fau/gaitmap/tree/master/example_data>`__.
and store them in a `.gaitmap_data/` in your home directory (you can also just run the loader function and it will tell you where to put the files).
Then you can load the data using the following code:

.. code-block:: python

    from gaitmap import example_data
    imu_data = example_data.get_healthy_example_imu_data()

    stride_borders = example_data.get_healthy_example_stride_borders()
    stride_events = example_data.get_healthy_example_stride_events()

There are more example data sets available, which you can find in the `gaitmap.example_data` module.

All examples in this documentation use the example data.

Existing datasets
-----------------
As part of the `gaitmap-datasets <https://github.com/mad-lab-fau/gaitmap-datasets>`_ python package we provide loader
functions for some publicly available datasets.
Using these loaders will automatically convert the data into the correct format and units.
Note, you still need to download the dataset yourself first!
Then you can check the `gaitmap-datasets example page <https://mad-lab-fau.github.io/gaitmap-datasets/auto_examples/index.html>`__
on the specifics of how to load the dataset you are interested in.

All datsets available there use the :class:`tpcp.Dataset` interface to represent the data.
This means each dataset class contains the data from multiple participants and trials.
The exact structure of the dataset and what data is available will vary between datasets.

To provide data to gaitmap, you always need to select a single trial as gaitmap is designed to work on a single
continuous sensor signal at the time and does not handle any concepts of your overall dataset/experiment structure.

Below an example for the `Kluge2017` dataset:

.. code-block:: python

    from gaitmap.datasets import Kluge2017

    dataset = Kluge2017()
    single_trial = dataset.get_subset(participant="216060", repetition="0", speed="normal")

    # Imu data
    imu_data = single_trial.imu_data

    # Event data from reference system (Note that this assumes a different sampling rate than the imu data!)
    gait_events = single_trial.mocap_events_

Custom data
-----------
Most the time you likely want to use your own data.
In this case, you need to load your data into the correct data objects yourself.
Here are the steps we recommend:

1. Loading an individual sensor file
++++++++++++++++++++++++++++++++++++
Gaitmap is designed to work on a single continuous sensor signal (from one or multiple sensors) at the time.
This means we don't handle multiple recordings (e.g. from different participants or trials) explicitly.

To start preparing your data, we recommend to start with a single recording consisting of one continuous IMU datastream.
As a first step you need to load this data into memory using Python.
How this work will depend on how you stored the sensor data.

If your sensor data is not stored in any common format (e.g. csv, hdf5) that can easily be loaded into Python, you
first need to convert it (e.g. using the manufacturer's software) before loading it into Python.

.. warning:: We only support data with a constant sampling rate without any missing values!

2. Converting into the basic data objects
+++++++++++++++++++++++++++++++++++++++++
Once you have loaded your data into Python, you need to convert it into the basic data objects.
For raw sensor data this means pandas DataFrames/a dict of pandas DataFrames.

First, we need to differentiate between three cases:

1. Your recording only contains the data from a single sensor: In this case you can simply convert your data into a
   single pandas DataFrame.
2. Your recording contains data from multiple sensors, but all sensors are synchronised (i.e. it would make sense for
   them to all share the same time-index): In this case you can convert your data into a single pandas DataFrame with a
   multilevel column index (first level is the sensor name, second level is the sensor channel name).
3. Your recording contains data from multiple sensors, but the sensors are not synchronised: In this case you need to
   convert your data into a dict of pandas DataFrames (one DataFrame per sensor).

In all cases the columns of the dataframes should be the sensor channels named `acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z`.
If you have additional sensor channels, you can leave them in the dataframe, but they will be ignored by gaitmap.
The index of the dataframe is not really standardized, but we recommend to use a index representing the time in seconds
relative to a relevant event in your data (e.g. the start of the recording).

You can learn more about the data objects in the :ref:`data objects section <datatypes>`.

3. Converting into the correct units
++++++++++++++++++++++++++++++++++++
This step should be relatively straight forward.
You need to convert your data into the correct units.
In gaitmap, we assume that acceleration is in m/s^2 and angular velocity is in deg/s.
If your data is in different units, you need to convert it.
For example if your data is in g, you need to multiply it by 9.81 to convert it into m/s^2 and your gyroscope data in
rad/s, you need to multiply it by 180/pi to convert it into deg/s.

.. code-block:: python

    import numpy as np

    data.loc[:, ['acc_x', 'acc_y', 'acc_z']] *= 9.81
    data.loc[:, ['gyr_x', 'gyr_y', 'gyr_z']] *= 180/np.pi

Of course you need to do that for all sensors in your data.

.. code-block:: python

    import pandas as pd
    import numpy as np

    # For multiindex versions
    data.loc[:, pd.IndexSlice[:, ['acc_x', 'acc_y', 'acc_z']]] *= 9.81
    data.loc[:, pd.IndexSlice[:, ['gyr_x', 'gyr_y', 'gyr_z']]] *= 180/np.pi

    # For dict versions
    for sensor_name, sensor_data in data.items():
        sensor_data.loc[['acc_x', 'acc_y', 'acc_z']] *= 9.81
        sensor_data.loc[['gyr_x', 'gyr_y', 'gyr_z']] *= 180/np.pi

4. Converting into the correct coordinate system
+++++++++++++++++++++++++++++++++++++++++++++++++
This is the most complicated step and will depend on your sensor setup.
In gaitmap (so far) we have fixed expected coordinate systems for the feet (see :ref:`here <coordinate_systems>`).
This means, you need to transform your sensor coordinate system to match the expected coordinate system of gaitmap
**during mid-stance**.
It is not relevant, if the coordinate systems match perfectly, but the rough alignment should be correct.

In general there are two approaches you can take here:

1. (recommended) You have some knowledge on the rough mounting orientation of your sensors and you can simply transform
   your data into the correct coordinate system by establishing a fixed rotation matrix (per foot).
2. You don't have any knowledge on the mounting orientation of your sensors and you need to estimate the correct rotation
   matrix from your data.

Known mounting orientation (approach 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you know the rough mounting orientation of you sensors, the process is relatively straight forward.

1. Find the sensor coordinate system of your sensor unit.
   This should be documented in the manufacturer's documentation.
   If you can not find any documentation, you can also take the sensor unit and create a test recording where you
   place it in a known orientation (e.g. flat on the table) and then rotate it around all axes to retrieve the sensor
   coordinate system.

   .. warning:: The sensor coordinate system might change, when you calibrated the sensor (e.g. using a Ferraris
                calibration).
                In this case you need to use the calibrated sensor coordinate system!

2. Double check, that the coordinate system of your gyroscope and accelerometer are identical.
   Older sensors (e.g. Shimmer 2R) often have different coordinate systems for the gyroscope and accelerometer.
   If this is the case, fix this misalignment first before continuing.
   You can use a similar approach to find out the required rotations as described below.

   Also check, if your sensor coordinate system is right-handed.
   If (in a rare case, the coordinate system is left-handed, you need to invert one axis of the coordinate system)
   Again, do this before continuing.
3. With the knowledge of your sensor coordinate system, draw a rough sketch of the coordinate systems of your sensor
   relative to the feet.
   Then next to it draw the expected coordinate system of gaitmap.
   For example like shown in the gaitmap-datasets examples (see `here <https://mad-lab-fau.github.io/gaitmap-datasets/auto_examples/egait_adidas_2014.html>`__).
   Then you can simply read off the required rotation matrix from your sketch as follows:

   Write down the required transformations for each axis as a rotation matrix, by writing the new coordinate axis in
   terms of the old coordinate axis.
   For example for the Shimmer2R example linked above, this would look like this:

   .. code-block::

        Left foot:                       Right foot:
        New -> Old                       New -> Old
          x -> -y                          x -> y
          y ->  z                          y -> -z
          z -> -x                          z -> -x

   Based on this we can construct the rotation matrices by simply writing down the vectors in the "old" column as
   rows of the matrix.
   For example, the first row for the left foot would be `[0, -1, 0]` and the second column
   would be `[0, 0, 1]`. The resulting rotation matrix would be:

   .. code-block::

      Left foot:                       Right foot:
      [[ 0, -1,  0],                   [[ 0,  1,  0],
       [ 0,  0,  1],                    [ 0,  0, -1],
       [-1,  0,  0]]                    [-1,  0,  0]]

   .. note:: In case your sensor mounting requires rotations other than "simply" flipping sensor axis, we recommend to
             split the rotation into multiple steps.
             First, rotate the coordinate system into an orientation where you only need to flip the sensor axis.
             Then use the process described above to find the rotation matrix for the flipped sensor axis.
             Finally, multiply the two rotation matrices to get the final rotation matrix.

   With that knowledge, we can not apply the rotation matrix to our data.
   We can do that as follows:

   .. code-block:: python

      # For multiple sensors, we write down the rotation matrices for each sensor into a dict
      rotation_matrices = {
            "left_sensor_name": np.array([[ 0, -1,  0], [ 0,  0,  1], [-1,  0,  0]]),
            "right_sensor_name": np.array([[ 0,  1,  0], [ 0,  0, -1], [-1,  0,  0]])
      }
      from gaitmap.utils.rotations import rotate_dataset

      # We assume `data` has two sensors with the same names as in the dict above
      data = rotate_dataset(data, rotation_matrices)

   In case all of our rotations are just "flipping" sensor axis, we can also use the `flip_dataset` function, which
   can be much faster for long datasets:

   .. code-block:: python

      from gaitmap.utils.rotations import flip_dataset

      # We assume `data` has two sensors with the same names as in the dict above
      data = flip_dataset(data, rotation_matrices)

4. Finally, double check that the final data looks as expected by comparing the plotted data from a couple of strides
   to the example data shown in the `coordinate systems <coordinate_systems>`_ guide.

Unknown mounting orientation (approach 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In case you have no idea about the mounting orientation of your sensors, you can try to estimate the correct rotation.
We have an entire example on that topic `here <example_automatic_sensor_alignment_detailed>`_.

Note, that this approach is not guaranteed to work and you might need to try different approaches to find the correct
rotation.
In any case, you should carefully double check the final data to make sure, that the rotation is correct.

5. Start using your data
++++++++++++++++++++++++
After going all of these steps, your raw imu data should be ready to be used with gaitmap.
For some operations it might be useful, to perform an additional gravity alignment, but other than that, you should be
good to go.

