.. _coordinate_systems:

===================
Coordinate  Systems
===================

Coordinate systems (or frames) are an important topic when working with IMUs and are getting even more complicated
when these IMUs are attached to a human body.
This library makes a couple of deliberate choices when it comes to the definition of coordinate systems and related
naming conventions.
Therefore, please read this document carefully before using the pipelines available in *gaitlab*.

TL;DR(A)
========

This TL;DR(Again) is intended as quick reference, **after** you already read the full documentation below.

* Accelerometer measures a positive value, if movement occurs in the **positive (+)** direction of the axis
* Gyroscope measures a positive value, if rotation occurs in the direction of the arrow in the :ref:`image below <ff>`
* Gravity results in a measure of **+1g**, if an axis is pointing **upwards**, and **-1g**, if the axis is pointing
  **downwards**

Foot Sensor Frame (FSF)
    * forms a right-handed coordinate system with axes called **X, Y, Z**
    * uses right-hand-rule around each axis for definition of positive direction of the Gyroscope 
    * defines axes' directions as up (Z), to the tip of the shoe (X), and
      to the **left** (Y)

Foot Body Frame (FBF)
    * consists of the 3 axes *ML* (medial to lateral), *PA* (posterior to anterior), and *SI* (superior to inferior)
    * is **not** right handed and should not be used for any physical calculations
    * produces the same sensor signal independent of the foot (right/left) for the same anatomical movement (e.g.
      lateral acceleration = positive acceleration)
    * follows convention of directions from [1]_

World/Global Frame (GF)
    * uses the same axes definitions as the FSF during the stance phase, but
    * is stationary, i.e. in contrast to FSF and FBF not moving
    * is used to describe all positions/velocity and orientations after the *Trajectory Reconstruction*
    * is fixed during an observation period
    * The observation period depends on the algorithm used in the *Trajectory Reconstruction* and is usually either
      a stride or a gait sequence level

.. _ff:

Foot Frame Overview
-------------------

.. figure:: /images/gaitmap_foot_frame.svg
    :alt: gaitmap foot frame
    :figclass: align-center

    The positive directions for the accelerometer (straight arrows) and the gyroscope (circular arrows) for the foot
    sensor frame (left) and foot body frame (right).
    `Click here for fullscreen version of image <../_images/gaitmap_foot_frame.svg>`_

.. table:: Table showing the expected signal (positive or negative and in which axis) when a certain movement
           (displacement or rotation) of a foot happens for the sensor (FSF) and the body frame (FBF).

  +-------------------+------------------------+------------------------+
  |                   |          FSF           |          FBF           |
  +-------------------+-----------+------------+-----------+------------+
  |                   | Left Foot | Right Foot | Left Foot | Right Foot |
  +===================+===========+============+===========+============+
  |                              **Displacements**                      |
  +-------------------+-----------+------------+-----------+------------+
  | anterior          | +acc_x    | +acc_x     | +acc_pa   | +acc_pa    |
  +-------------------+-----------+------------+-----------+------------+
  | posterior         | -acc_x    | -acc_x     | -acc_pa   | -acc_pa    |
  +-------------------+-----------+------------+-----------+------------+
  | lateral           | +acc_y    | -acc_y     | +acc_ml   | +acc_ml    |
  +-------------------+-----------+------------+-----------+------------+
  | medial            | -acc_y    | +acc_y     | -acc_ml   | -acc_ml    |
  +-------------------+-----------+------------+-----------+------------+
  | inferior          | -acc_z    | -acc_z     | +acc_si   | +acc_si    |
  +-------------------+-----------+------------+-----------+------------+
  | superior          | +acc_z    | +acc_z     | -acc_si   | -acc_si    |
  +-------------------+-----------+------------+-----------+------------+
  |                                **Rotations**                        |
  +-------------------+-----------+------------+-----------+------------+
  | eversion          | +gyr_x    | -gyr_x     | -gyr_pa   | -gyr_pa    |
  +-------------------+-----------+------------+-----------+------------+
  | inversion         | -gyr_x    | +gyr_x     | +gyr_pa   | +gyr_pa    |
  +-------------------+-----------+------------+-----------+------------+
  | dorsifelxion      | -gyr_y    | -gyr_y     | +gyr_ml   | +gyr_ml    |
  +-------------------+-----------+------------+-----------+------------+
  | plantarflexion    | +gyr_y    | +gyr_y     | -gyr_ml   | -gyr_ml    |
  +-------------------+-----------+------------+-----------+------------+
  | external rotation | +gyr_z    | -gyr_z     | -gyr_si   | -gyr_si    |
  +-------------------+-----------+------------+-----------+------------+
  | internal rotation | -gyr_z    | +gyr_z     | +gyr_si   | +gyr_si    |
  +-------------------+-----------+------------+-----------+------------+

.. figure:: /images/signal_templates_coordinate_system_definitions.svg
    :alt: gaitmap foot frame signal templates
    :figclass: align-center

    Example signals of 3D-accelerometer and 3D-gyroscope sensor signals for sensor-frame and body-frame coordinate
    system definitions.
    The signal section corresponds to one left-/right stride, corresponding to the DTW template definition.
    Click for `here <../_images/signal_templates_coordinate_system_definitions.svg>`_ for the fullscreen version of
    image.

Sensor vs Body Frame
====================

When working with IMUs you always need to differentiate between the local sensor coordinate system and the global world
coordinate system.
The former describes everything from the view of the sensor (i.e. when the sensor is rotated or moved the coordinate
system is adjusted as well).
The latter describes the sensors orientation and movement relative to a global reference frame set by global objects
relevant for the performed measurement.
If measuring human movement we usually use two separate global frames: (1) A global frame that is relative to the space
the human is moving in.
This frame is important to describe global displacements and rotations of the human during movement.
This "real global frame" is explained in more detail below.
(2) A frame that uses the human body as the reference.
This frame makes it easy to map the motion to anatomical movement directions.
For example, the forward direction should point towards the anterior site of our human and upwards should be defined the
superior direction.

Body Frame
----------

If our sensor is rigidly attached to the trunk (or head) of a human a transformation from the sensor frame (SF) to the
body frame (BF) can be achieved by a single rotation matrix describing the relative orientation from sensor to body.

However, if we attach sensors to two opposite-sided extremities of a human, this is not as easy anymore.
As the movements of our extremities are described as mirror images of each other, it makes sense to choose a BF that
results in the same sensor signal for the same movement, independent of which extremity we look at.
For example, a lateral raise of the leg should produce a positive acceleration signal in both sensors.
This requires that the BF of one sensor is the mirror-image (mirrored at the sagittal plane) of the other's.
As mirroring is a non-cartesian transformation, it is not possible to find rotation matrices that can transform the
local SFs into their respective BFs.
Hence, performing the transformation will require the mirroring of individual axis, which breaks the right-handedness of
the coordinate system.
If we further want to match the positive directions of the Gyroscope with same anatomical rotational directions on both
sides, we also need to break the right-hand-rule for the movement direction around an axis.

.. warning:: Therefore, we must not use such a BF to perform any sort of physical calculations that involve the signal
             for more than one axis!
             In particular rotations cannot be performed in this coordinate system.

This inability to perform certain mathematical options potentially requires us to switch from the SF to the BF multiple
times during an analysis pipeline.
For example, initial sensor alignment, filtering, and calibration needs to be performed in the SF.
The identification of certain movements and events is usually done best in the BF, so that the same algorithm can be
used for each foot.
Orientation and position estimations of the sensors need to be performed in the SF again and the results need to be
converted back to the BF to get final biomechanical measures and joint angles.

Foot Coordinate System
----------------------

Depending on where the sensors are attached at the extremities it might make sense to choose different BFs that align
better with the directions of movement the specific body segment can perform.
The following section describes the chosen SF and BFs for sensors attached anywhere at the feet.
These frames are called *foot sensor frame* (FSF) and *(left/right) foot body frame* ((L/R)FBF).

To make the transformation from the FSF to the FBFs easy, this library uses an FSF that already aligns with all mature
axes of movements (see :ref:`ff`).
Specifically, sensor axes **independent** of the foot are expected to point up (Z), to the tip of the shoe (X), and
to the **left** (Y).
The forward direction for rotations are determined by the right-hand-rule around these axes.

.. note:: This means to use the provided functions for coordinate conversion in this library, you are expected to rotate
          your IMU data to fit this coordinate system.
          This cannot always be done precisely.
          The required precision of alignment will depend on the exact algorithms used and the final biomechanical
          parameters of interest.
          This is discussed further in the section about :ref:`alignment-algorithms`.

To transform FSF into FBFs, only renaming and axis flips are required (see table :ref:`foot-transform`).
FBFs' axes FBFs are denoted by *ML* (medial to lateral), *PA* (posterior to anterior), and *SI* (superior to
inferior).
The order of naming directly indicates the positive direction of the respective axis.
All rotations are named by the axes they occur around.
Note, that positive direction of rotation is not determined by the right-hand-rule.
Rather, forward directions for axes and directions of rotation are directly taken from the recommendations given
in [1]_ (see :ref:`ff`).

.. _foot-transform:

.. table:: Required transformation for accelerometer and gyroscope from FSF to FBF for both feet

   +-----------------+-----------------+
   | Left Foot       | Right Foot      |
   +--------+--------+--------+--------+
   |  LFBF  |  FSF   |  RFBF  |  FSF   |
   +========+========+========+========+
   | acc_pa | acc_x  | acc_pa | acc_x  |
   +--------+--------+--------+--------+
   | acc_ml | acc_y  | acc_ml | -acc_y |
   +--------+--------+--------+--------+
   | acc_si | -acc_z | acc_si | -acc_z |
   +--------+--------+--------+--------+
   | gyr_pa | -gyr_x | gyr_pa | gyr_x  |
   +--------+--------+--------+--------+
   | gyr_ml | -gyr_y | gyr_ml | -gyr_y |
   +--------+--------+--------+--------+
   | gyr_si | -gyr_z | gyr_si | gyr_z  |
   +--------+--------+--------+--------+


World/Global Frame
------------------

At this point, it is important to emphasize again that the SF/BF moves with the sensor/subject. Therefore, the alignment
information for the FSF/FBF provided above, of course only hold true while the test subject is in its default pose (e.g.
flat foot phase for foot mounted IMUs).
During movement the orientation of the sensor can not be easily described by terms like "forward" or "upwards".
Hence, the SF is not suitable to calculate and express measures that require a fixed global frame.
Additionally, the BF is unsuitable to talk about movements of the entire human, as this frame moves with the subject.
For example, to estimate the stride length, we need a clear definition of a forward direction over the entire duration
of a stride and a reference point that does not move with the subject.
Therefore, we need a "global" fixed reference frame (GF) for the duration of our observation.
The GF does not move or rotate if the sensor or the subject move.
For an example, motion capture systems (e.g. camera based systems) always report their information in such a global
reference frame, because the measurement devices themself are fixed in a global frame.

For inside-out systems, like body-attached IMUs, it is difficult to establish such a fixed coordinate system.
Rather complicated algorithms must be used to project the local measurements of an IMU into a fixed global frame.
During the gait pipeline, this step is performed by the *Trajectory Reconstruction*.
The orientation, velocity, and position provided by these algorithm is always expressed in the GF.
However, for the sake of simplicity, no additional prefix is used to indicate that in the column names.
One just has to remember that all positions and orientations provided after the *Trajectory Reconstruction* are
expressed in the global frame.

Note that the orientation of this fixed global frame is usually determined by the orientation of the FSF/FBF at the
beginning of the observation period.
Many algorithms consider a single stride as an observation period.
This means that when comparing two strides the GFs do not necessarily line-up, as the GF for each stride is defined
based on the orientation of the sensor at the beginning of the stride.

Algorithmic Implementation
==========================

.. _alignment-algorithms:

Alignment with the Foot Sensor Frame
------------------------------------

TODO: Add info about transforming the raw sensor frame into the FSF

Transformation into the Foot Body Frame
---------------------------------------

Once the data is properly aligned to the gaitmap-FSF, it is very easy to transform it into the respective BF.
For this you can use the function :func:`~gaitmap.utils.coordinate_conversion.convert_to_fbf` if you have a multi-sensor
dataset or :func:`~gaitmap.utils.coordinate_conversion.convert_left_foot_to_fbf` and
:func:`~gaitmap.utils.coordinate_conversion.convert_right_foot_to_fbf` if you have a single-sensor dataset.
Note that in both cases you need to specify which foot each sensor was attached to, as the transform from the FSF to FBF
is different for each foot (see table :ref:`foot-transform`). For examples please refer directly to the documentation of
these functions.

.. note:: All coordinate transform functions create a full copy of your dataset.

.. [1] Wu, G., Siegler, S., Allard, P., Kirtley, C., Leardini, A., Rosenbaum, D., … Stokes, I. (2002). ISB
       recommendation on definitions of joint coordinate system of various joints for the reporting of human joint
       motion - Part I: Ankle, hip, and spine. Journal of Biomechanics. https://doi.org/10.1016/S0021-9290(01)00222-6
