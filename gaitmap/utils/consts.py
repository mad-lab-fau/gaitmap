"""Common constants used in the library."""

import numpy as np

#: The default names of the Gyroscope columns in the sensor frame
SF_GYR = ["gyr_x", "gyr_y", "gyr_z"]
#: The default names of the Accelerometer columns in the sensor frame
SF_ACC = ["acc_x", "acc_y", "acc_z"]
#: The default names of the Magnetometer columns in the sensor frame
SF_MAG = ["mag_x", "mag_y", "mag_z"]
#: The default names of all columns in the sensor frame
SF_COLS = [*SF_ACC, *SF_GYR, *SF_MAG]

#: The default names of the Gyroscope columns in the body frame
BF_GYR = ["gyr_pa", "gyr_ml", "gyr_si"]
#: The default names of the Accelerometer columns in the body frame
BF_ACC = ["acc_pa", "acc_ml", "acc_si"]
#: The default names of the Magnetometer columns in the body frame
BF_MAG = ["mag_pa", "mag_ml", "mag_si"]
#: The default names of all columns in the body frame
BF_COLS = [*BF_ACC, *BF_GYR, *BF_MAG]

#: The minimal required columns for a stride list
SL_COLS = ["start", "end"]
#: Expected index cols for a stride list
SL_INDEX = ["s_id"]
#: Additional Columns of a stride list depending on its type
SL_ADDITIONAL_COLS = {
    "min_vel": ["pre_ic", "ic", "min_vel", "tc"],
    "segmented": ["ic", "min_vel", "tc"],
    "ic": ["ic", "min_vel", "tc"],
}
SL_MINIMAL_COLS = {
    "min_vel": ["min_vel"],
    "ic": ["ic"],
}
#: Expected Order of events based on the stride type
SL_EVENT_ORDER = {
    "segmented": ["tc", "ic", "min_vel"],
    "min_vel": ["pre_ic", "min_vel", "tc", "ic"],
    "ic": ["ic", "min_vel", "tc"],
}

#: The allowed index columns for a regions-of-interest list
ROI_ID_COLS = {"roi": "roi_id", "gs": "gs_id"}

#: The allowed index columns for vel, ori, and pos lists
TRAJ_TYPE_COLS = {**ROI_ID_COLS, "stride": "s_id"}

#: The default names of the Velocity columns in the global frame
GF_VEL = ["vel_x", "vel_y", "vel_z"]
#: The default names of the Position columns in the global frame
GF_POS = ["pos_x", "pos_y", "pos_z"]
#: The default names of the Orientation columns in the global frame
GF_ORI = ["q_x", "q_y", "q_z", "q_w"]
#: The default index names for all global frame spatial paras
GF_INDEX = ["s_id", "sample"]

#: Gravity in m/s^2
GRAV = 9.81
#: The gravity vector in m/s^2 in the FSF
GRAV_VEC = np.array([0.0, 0.0, GRAV])
GRAV_VEC.flags.writeable = False

#: Sensor to body frame conversion for the left foot
FSF_FBF_CONVERSION_LEFT = {
    "acc_x": (1, "acc_pa"),
    "acc_y": (1, "acc_ml"),
    "acc_z": (-1, "acc_si"),
    "gyr_x": (-1, "gyr_pa"),
    "gyr_y": (-1, "gyr_ml"),
    "gyr_z": (-1, "gyr_si"),
}

#: Sensor to body frame conversion for the right foot
FSF_FBF_CONVERSION_RIGHT = {
    "acc_x": (1, "acc_pa"),
    "acc_y": (-1, "acc_ml"),
    "acc_z": (-1, "acc_si"),
    "gyr_x": (1, "gyr_pa"),
    "gyr_y": (-1, "gyr_ml"),
    "gyr_z": (1, "gyr_si"),
}
