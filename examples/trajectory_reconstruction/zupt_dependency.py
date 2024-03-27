r"""
.. _zupt_dependency:

ZUPT Dependency of the Trajectory Estimation
============================================

The estimation of the trajectory using the Kalman filter is heavily dependend on the ZUPT detection [1]_.
For this reason we provide various ZUPT detection methods in gaitmap.
Each of these methods has further parameters, that can (and should) be tuned to the specific data.

In this example we show, how to Gridsearch the Parameter space of the ZUPT detection methods to find the best
combination.
This makes heavy use of the `tpcp` optimization methods and the Gridsearch used here can be substituted by any other
optimize method.
This is also a great example of how to use the `tpcp` optimization methods in combination with `gaitmap`.

.. [1] Wagstaff, Peretroukhin, and Kelly, “Robust Data-Driven Zero-Velocity Detection for Foot-Mounted Inertial
       Navigation.”

"""
import pandas as pd

# %%
# The Data
# --------
# We gonna use the healthy example IMU data to find the best settings.
# To use this data with tpcp, we need to wrap it into a `Dataset` class.
# For more details about this see the :ref:`gridsearch <grid_search>` or the :ref:`custom dataset <custom_dataset>`
# example.
# Here we will just copy and paste the dataset used in the gridsearch example and add an additional `mocap_trajectory_`
# property that allows access to the respective mocap reference data.
from tpcp import Dataset

from gaitmap.example_data import get_healthy_example_imu_data, get_healthy_example_mocap_data

# %%
# The Dataset
# -----------
# To use any `tpcp` features, we need to wrap our data into a dataset object.
# For this we need an index.
# Usually you create this by iterating a directory of datafiles you have and list out all participants/tests.
# To keep things simple we just use data from one participant and one test, but treat the left and right foot
# as two independent datasets.
# This is something you would not normally do.


class HealthyImu(Dataset):
    @property
    def sampling_rate_hz(self) -> float:
        return 204.8

    @property
    def data(self) -> pd.DataFrame:
        self.assert_is_single(None, "data")
        return get_healthy_example_imu_data()[self.index.iloc[0]["foot"] + "_sensor"]

    @property
    def mocap_trajectory_(self) -> pd.DataFrame:
        self.assert_is_single(None, "data")
        df = get_healthy_example_mocap_data().filter(like=self.group_label.foot[0].upper())
        # This strips the L_/R_ prefix
        df.columns = pd.MultiIndex.from_tuples((m[2:], a) for m, a in df.columns)
        return df

    def create_index(self) -> pd.DataFrame:
        return pd.DataFrame({"foot": ["left", "right"]})


# %%
# Setting up a pipeline
# ---------------------
# We start by setting up a pipeline, that we can use to estimate the trajectory.
# It takes the Zupt method as parameter and internally uses `RtsKalman` to estimate the trajectory.
from tpcp import Pipeline, cf
from typing_extensions import Self

from gaitmap.base import BaseZuptDetector
from gaitmap.trajectory_reconstruction import RtsKalman
from gaitmap.zupt_detection import AredZuptDetector, ShoeZuptDetector


class TrajectoryPipeline(Pipeline[HealthyImu]):
    trajectory_: pd.DataFrame

    def __init__(self, zupt_method: BaseZuptDetector = cf(ShoeZuptDetector())):
        self.zupt_method = zupt_method

    def run(self, datapoint: HealthyImu) -> Self:
        rts_kalman = RtsKalman(zupt_detector=self.zupt_method)
        self.trajectory_ = rts_kalman.estimate(datapoint.data, sampling_rate_hz=datapoint.sampling_rate_hz).position_
        return self


# %%
# Scorer
# ------
# To decide what parameters we consider good, we need a score function.
# In this case (to keep things simple), we are going to take the walking distance estimated by the kalman filter.
# In the example data, the participant walked 20 m in one direction, turned around and walked back.
# Hence, we are going to check how close our results are to the 20 m.
# To be more precise, we are going to calculate the actual walking distance from the associated mocap data.
import numpy as np


def score(pipeline: TrajectoryPipeline, datapoint: HealthyImu) -> float:
    pipeline.safe_run(datapoint)
    trajectory = pipeline.trajectory_[["pos_x", "pos_y"]]
    walk_distance = np.max(np.linalg.norm(trajectory, axis=1))
    ground_truth_traj = datapoint.mocap_trajectory_["TOE"][["x", "y"]]
    ground_truth_walk_distance = np.max(np.linalg.norm(ground_truth_traj - ground_truth_traj.iloc[0], axis=1))
    return np.abs(walk_distance - ground_truth_walk_distance / 1000)


from sklearn.model_selection import ParameterGrid

# %%
# Gridsearch
# ----------
# Now we can set up a simple GridSearch.
# We just need to specify the parameters we want to use.
# Note, we use the `__` syntax to set parameters of the nested zupt_method object
from tpcp.optimize import GridSearch

# Para-grid used to optimize the default values of the ShoeZuptDetector
# paras = ParameterGrid(
#     {
#         "zupt_method__inactive_signal_threshold": np.logspace(3, 10, 100),
#         "zupt_method__acc_noise_variance": np.logspace(-10, -3, 10),
#         "zupt_method__gyr_noise_variance": np.logspace(-10, -3, 10),
#     }
# )

# Shorter para grid for the example
paras = ParameterGrid(
    {
        "zupt_method__inactive_signal_threshold": np.logspace(5, 10, 10),
    }
)

# %%
# Then we can simply run the GridSearch.
# Note, that we specify `-score` for `return_optimized`, as we want to select the smallest error as best.
gs = GridSearch(TrajectoryPipeline(), paras, scoring=score, return_optimized="-score")
gs.optimize(HealthyImu())

# %%
# Results
# -------
results = gs.gs_results_
results

# %%
# We can also specifically get the best results
print(pd.DataFrame(results))
print(gs.best_params_)
print(gs.best_score_)

# %%
# And plot them
import matplotlib.pyplot as plt

result_df = pd.DataFrame(results)
plt.plot(result_df["param_zupt_method__inactive_signal_threshold"], result_df["score"].abs())
plt.xlabel("inactive_signal_threshold")
plt.ylabel("abs(score)")
plt.yscale("log")
plt.xscale("log")
plt.show()
