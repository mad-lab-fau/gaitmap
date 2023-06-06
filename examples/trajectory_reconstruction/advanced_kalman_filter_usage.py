r"""
.. _advanced_kalman:

Advanced Kalman Filter Usage
============================

The RTS Kalman Filter is a powerful tool to estimate the trajectory of a sensor.
However, it has many parameters and options that need to be tuned correctly to get good results.
This example shows how to use the Kalman Filter in a more advanced way.
"""

# %%
# Setup
# -----
# First we get some example data from a participant walking up and down a line.
# We already have the start and end of each stride in the stride list.
# We will use that later.

from gaitmap.example_data import get_healthy_example_imu_data, get_healthy_example_stride_borders

stride_list = get_healthy_example_stride_borders()["left_sensor"]
stride_list = stride_list[stride_list["end"] < 3900]
imu_data = get_healthy_example_imu_data()["left_sensor"].iloc[:3900]
sampling_frequency_hz = 204.8

# %%
# Noise sensitivity
# -----------------
# One of the core parameters of a Kalman filter are the noise parameters.
# They describe the uncertainty associated with each measurement.
# For the RTS Kalman Filter, their are two types of noise parameters:
#
# * `veloctiy_error_variance` and `orientation_error_variance` describe of the actual sensor measurements and should
#   roughly be based on the standard deviation of the actual sensor noise of the accelerometer and gyroscope,
#   respectively.
# * `zupt_variance` and `level_walking_variance` describe the uncertainty of the pseudo measurements used to describe
#   the zero-velocity and level-walking constraints.
#   Basically, how certain are you that the sensor is actually at rest/had no change in elevation during the parts of
#   the signal that are detected to be zero-velocity.
#
# Tuning these noise values is curtial to get good results.
# However, it is difficult to select appropriate values.
# Some literature suggests that the sensor associated noise can be directly derived from the IMU parameters (e.g.
# derived from spec sheet).
# However, this only works in very isolated cases, where the actual electrical noise of the sensor is the primary
# source of error.
# In complex movement cases (like gait), this is not really the case.
# The type and speed of the movement might change the optimal noise values similar to how a different sensor might.
# Hence, tuning parameters using some form of parameter search against some ground truth is the best way to go.
#
# To simplify this approach, it is important to keep in mind, that the noise parameters are not independent.
# More than their absolute values, the relative values between the noise parameters are important.
# Simply speaking, the ratio between two noise parameters describes how much more we trust one measurement (and its
# derived values) over the other.
# So if the error associated with the IMU measurement is 10 times larger than the error associated with the
# zero-velocity, the estimated velocity will very quickly converge to 0 in a zero-velocity region.
# However, if we trust the actual measurements more, it might take some time to converge to 0 or the zero-velocity
# might be effectively ignored.
#
# As an example, below we run the Kalman Filter with different noise parameters.
#
# We start with the default parameters:
from gaitmap.trajectory_reconstruction import RtsKalman

default_rts = RtsKalman().estimate(data=imu_data, sampling_rate_hz=sampling_frequency_hz)

# %%
# Now we decrease the noise associated with the velocity by a factor of 1000, changing the ratio between the velocity
# and orientation noise.
imu_trust_rts = RtsKalman(velocity_error_variance=10e2).estimate(data=imu_data, sampling_rate_hz=sampling_frequency_hz)

# %%
# If we plot the results, we can see the final trajectory starts to differ quite a bit between the two settings and it
# is hard to judge without a reference, which one is better.
# This indicates that tuning with a ground truth is necessary to get good results.
import matplotlib.pyplot as plt

default_rts.position_["pos_z"].plot(label="default")
imu_trust_rts.position_["pos_z"].plot(label="imu trust")
plt.title("z-position (i.e. foot lift)")
plt.xlabel("sample")
plt.ylabel("orientation [a.u.]")
plt.xlim(0, 1000)
plt.legend()
plt.show()

# %%
# Zupt Dependency
# ---------------
# (To learn more about tuning ZUPT-thresholds, also see :ref:`zupt_dependency`)
#
# The Kalmann Filter uses zero-velocity updates (ZUPT) to correct for the drift in velocity and position.
# Without them, no corrections can be performed and the estimated trajectory will be similar to just performing gyro
# and acc integrations without any corrections.
# Hence, it is important to make sure proper Zupts are detected.
#
# This is controlled by the `zupt_detector` parameter.
# This can be set to any `ZuptDetector` class.
# By default, we use the :class:`~gaitmap.zupt_detection.NormZuptDetector`, which basically looks at the norm of the
# gyro signal to determine periods of no movement.
#
# To demonstrate the impact, we will set the threshold to a very low value, so that no ZUPTs are detected and
# once to a very high value, so that basically the whole signal is detected as ZUPT.
from gaitmap.zupt_detection import NormZuptDetector

no_zupt_rts = RtsKalman(zupt_detector=NormZuptDetector(inactive_signal_threshold=0.001)).estimate(
    data=imu_data, sampling_rate_hz=sampling_frequency_hz
)
all_zupt_rts = RtsKalman(zupt_detector=NormZuptDetector(inactive_signal_threshold=100000)).estimate(
    data=imu_data, sampling_rate_hz=sampling_frequency_hz
)

# %%
# We can see that the trajectory calculated without Zupt completely drifts of within a couple of samples and the
# trajectory calculated with all Zupts is basically a straight line.

default_rts.position_["pos_z"].plot(label="default")
no_zupt_rts.position_["pos_z"].plot(label="no zupt")
all_zupt_rts.position_["pos_z"].plot(label="all zupt")
plt.title("z-position (i.e. foot lift)")
plt.xlabel("sample")
plt.ylabel("orientation [a.u.]")
plt.xlim(0, 1000)
plt.ylim(-1, 0.2)
plt.legend()
plt.show()

# %%
# So selecting a proper ZUPT detector is important.
# In many cases, this just means tuning the threshold of one of the existing detectors.
# However, the optimal threshold will always depend on the speed of gait.
# From experience, one approach to deal with that is to force one ZUPT per stride independent of any threshold.
# This can be done using the `min_vel` event per stride that is derived from the event detection (e.g.
# :class:`~gaitmap.event_detection.RamppEventDetection`).
# These events are usually already calculated as part of your pipeline, but in this example, we will just calculate
# here.
from gaitmap.event_detection import RamppEventDetection
from gaitmap.utils.coordinate_conversion import convert_left_foot_to_fbf

min_vel_events = (
    RamppEventDetection(detect_only=("min_vel",))
    .detect(data=convert_left_foot_to_fbf(imu_data), stride_list=stride_list, sampling_rate_hz=sampling_frequency_hz)
    .min_vel_event_list_
)

min_vel_events.head()

# %%
# This can now be used by the :class:`~gaitmap.zupt_detection.StrideEventZuptDetector` to force one ZUPT per stride.
# Note, that we need to provide the `min_vel_events` to the estimate method of the Kalman Filter now.
# This is forwarded to ZUPT detector.

from gaitmap.zupt_detection import StrideEventZuptDetector

per_stride_zupt = RtsKalman(zupt_detector=StrideEventZuptDetector(half_region_size_s=0.05)).estimate(
    data=imu_data, stride_event_list=min_vel_events, sampling_rate_hz=sampling_frequency_hz
)

# %%
# Now we can see that the trajectory is now stable again during walking.
# However, at the end where there are no more strides, and hence, no more ZUPTS with this detector, the trajectory
# drifts again.
# We also visualize the ZUPTs that were detected to visualize the effect of the ZUPT detector.
#
# From experience, this approach also works well to reduce the drift, even if the person did not really had a
# foot-flat phase during the stride.
# However, increasing the ZUPT noise and disabling level-walking in these cases might be beneficial to not provide the
# Kalman Filter with too much "wrong" information.
default_rts.position_["pos_z"].plot(label="default")
per_stride_zupt.position_["pos_z"].plot(label="per stride zupt")

for _, zupt in per_stride_zupt.zupts_.iterrows():
    plt.axvspan(xmin=zupt["start"], xmax=zupt["end"], color="blue", alpha=0.2)

plt.title("z-position (i.e. foot lift)")
plt.xlabel("sample")
plt.ylabel("orientation [a.u.]")
plt.legend()
plt.show()

# %%
# In case you have more advanced requirements, you can also combine multiple ZUPT detectors using
# :class:`~gaitmap.zupt_detection.ComboZuptDetector`.
# For example, in the case above, it might make sense to combine the per-stride Zupt with a threshold-based ZUPT
# detector to also detect the regions before and after the walking periods.
# However, this time we can reduce the threshold, as we only need this additional detector to find real regions of
# full rest.
#
# As we can see below, this now fixes the drift at the end of the walking period.
# Note, this entire approach doesn't really make sense in this example, as the resting periods between the strides are
# quite stable.
# This means, the per-stride ZUPTs don't add anything over a properly tuned threshold-based ZUPT detector.
# However, for faster walking or for patients that don't have a proper foot-flat phase, they might be beneficial.
from gaitmap.zupt_detection import ComboZuptDetector

zupt_detector = ComboZuptDetector(
    detectors=[
        ("stride", StrideEventZuptDetector(half_region_size_s=0.05)),
        ("norm", NormZuptDetector(inactive_signal_threshold=10)),
    ]
)


combo_zupt = RtsKalman(zupt_detector=zupt_detector).estimate(
    data=imu_data, stride_event_list=min_vel_events, sampling_rate_hz=sampling_frequency_hz
)

default_rts.position_["pos_z"].plot(label="default")
combo_zupt.position_["pos_z"].plot(label="combo zupt")

for _, zupt in combo_zupt.zupts_.iterrows():
    plt.axvspan(xmin=zupt["start"], xmax=zupt["end"], color="blue", alpha=0.2)

plt.title("z-position (i.e. foot lift)")
plt.xlabel("sample")
plt.ylabel("orientation [a.u.]")
plt.legend()
plt.show()

# %%
# Madgwick RTS Kalman
# -------------------
# Tuning the ZUPT detector to really detect all ZUPTs that exist in the signal is an important step.
# However, sometimes, there are simply long periods of time when the person is not moving without proper foot-flat
# phases or resting.
# In these cases it would be great to be able to limit drift without relying on ZUPTs.
#
# Outside the Kalman filter concept, complimentary filter like the :class:`~gaitmap.trajectory_reconstruction.Madgwick`
# are often used to provide some form of correction for the orientation drift.
# The :class:`~gaitmap.trajectory_reconstruction.MadgwickRtsKalman` is a variant of the RTS Kalman Filter that uses the
# Madgwick internally to estimate the orientation.
#
# This should result in more stable orientation estimates, even if there are long periods of time without ZUPTs.
# Note, that this version of the Kalman Filter is technically not mathematically correct, as we are not conidering
# the change equation in the error propagation.
# The filter still assumes for the error propagation that the acc only has influence on the velocity and position and
# not the orientation directly.
# In practice, this doesn't seem to be a problem, but noise values might be tuned differently to account for the
# reduced orientation drift.
#
# Below we demonstrate the effectiveness of the Madgwick filter version by comparing the results with and without
# Madgwick for the case without any ZUPTs.
from gaitmap.trajectory_reconstruction import MadgwickRtsKalman

madgwick_rts_no_zupt = MadgwickRtsKalman(zupt_detector=NormZuptDetector(inactive_signal_threshold=0.001)).estimate(
    data=imu_data, sampling_rate_hz=sampling_frequency_hz
)

# %%
# As we can see below, the drift is significantly reduced without the need ZUPTs compared to the no-Zupt case from
# above.
# Of course, it is clear that the drift is still clearly present and ZUPTs are required if you want any reasonable
# results.
default_rts.position_["pos_z"].plot(label="default")
madgwick_rts_no_zupt.position_["pos_z"].plot(label="madgwick no zupt")
no_zupt_rts.position_["pos_z"].plot(label="normal no zupt")

plt.title("z-position (i.e. foot lift)")
plt.xlabel("sample")
plt.ylabel("orientation [a.u.]")
plt.xlim(0, 1000)
plt.ylim(-1, 0.2)
plt.legend()
plt.show()
