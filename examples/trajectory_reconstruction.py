r"""
Trajectory reconstruction
=========================

This example illustrates how a trajectory can be reconstructed for each stride  by
the :class:`~gaitmap.trajectory_reconstruction.StrideLevelTrajectory`. Note that this example does not take care for
any preprocessing steps, which might be necessary for your data, such as alignment to gravity and axis
transformations. To learn more about such preprocessing steps see :class:`examples.preprocessing_example`.

"""

# %%
# Getting input data
# ------------------
#
# For this we need stride event list obtained from event detection method and the sensor data. The sensor data is
# already in the correct gaitmap coordinate system in this case.
from gaitmap.example_data import get_healthy_example_stride_events, get_healthy_example_imu_data
from gaitmap.trajectory_reconstruction import SimpleGyroIntegration, ForwardBackwardIntegration, StrideLevelTrajectory

stride_list = get_healthy_example_stride_events()
imu_data = get_healthy_example_imu_data()

# %%
# Setting up necessary objects
# ----------------------------
# Here, we use simple gyroscopic integration for orientation estimation and forward-backward integration for position
# estimation. You can replace them by any of the methods in
# :class:`~gaitmap.trajectory_reconstruction.orientation_methods`
# and :class:`~gaitmap.trajectory_reconstruction.position_methods`.

ori_method = SimpleGyroIntegration()
pos_method = ForwardBackwardIntegration()
trajectory = StrideLevelTrajectory(ori_method, pos_method)


# %%
# Calculate and inspect results
# -----------------------------
trajectory.estimate(imu_data, stride_list, 204.8)
trajectory.position_["left_sensor"]
