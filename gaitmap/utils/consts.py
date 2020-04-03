"""Common constants used in the library."""

# TODO: Add them to documentation

SF_GYR = ["gyr_x", "gyr_y", "gyr_z"]
SF_ACC = ["acc_x", "acc_y", "acc_z"]
SF_COLS = [*SF_ACC, *SF_GYR]

BF_GYR = ["gyr_pa", "gyr_ml", "gyr_si"]
BF_ACC = ["acc_pa", "acc_ml", "acc_si"]
BF_COLS = [*BF_ACC, *BF_GYR]