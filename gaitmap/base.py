"""Base class for all algorithms."""

from typing import Self, TypeVar

import numpy as np
import pandas as pd
import tpcp
from scipy.spatial.transform import Rotation

from gaitmap.utils.consts import GF_ORI
from gaitmap.utils.datatype_helper import (
    OrientationList,
    PositionList,
    SensorData,
    SingleSensorData,
    SingleSensorOrientationList,
    SingleSensorStrideList,
    StrideList,
    VelocityList,
)
from gaitmap.utils.rotations import rotate_dataset_series

BaseType = TypeVar("BaseType", bound=tpcp.BaseTpcpObject)  # pylint: disable=invalid-name


class BaseAlgorithm(tpcp.Algorithm):
    """Base class for all algorithms.

    All type-specific algorithm classes should inherit from this class and need to

    1. overwrite `_action_method` with the name of the actual action method of this class type
    2. implement a stub for the action method

    Attributes
    ----------
    _action_method
        The name of the action method used by the Childclass

    """


class BaseSensorAlignment(BaseAlgorithm):
    """Base class for all sensor alignment algorithms."""

    _action_methods = ("align",)

    aligned_data_: SensorData

    def align(self, data: SensorData, **kwargs) -> Self:
        """Align sensor data."""
        raise NotImplementedError("Needs to be implemented by child class.")


class BaseStrideSegmentation(BaseAlgorithm):
    """Base class for all stride segmentation algorithms."""

    _action_methods = ("segment",)

    stride_list_: StrideList

    def segment(self, data: SensorData, sampling_rate_hz: float, **kwargs) -> Self:
        """Find stride candidates in data."""
        raise NotImplementedError("Needs to be implemented by child class.")


class BaseEventDetection(BaseAlgorithm):
    """Base class for all event detection algorithms."""

    _action_methods = ("detect",)

    def detect(
        self, data: SensorData, *, stride_event_list: SingleSensorStrideList | None = None, sampling_rate_hz: float
    ) -> Self:
        """Find gait events in data within strides provided by roi_list."""
        raise NotImplementedError("Needs to be implemented by child class.")


class BaseOrientationMethod(BaseAlgorithm):
    """Base class for the individual Orientation estimation methods that work on pd.DataFrame data."""

    _action_methods = ("estimate",)
    orientation_object_: Rotation

    data: SingleSensorData
    sampling_rate_hz: float

    @property
    def orientation_(self) -> SingleSensorOrientationList:
        """Orientations as pd.DataFrame."""
        df = pd.DataFrame(self.orientation_object_.as_quat(), columns=GF_ORI)
        df.index.name = "sample"
        return df

    @property
    def rotated_data_(self) -> SingleSensorData:
        """Rotated data."""
        return rotate_dataset_series(self.data, self.orientation_object_[:-1])

    def estimate(
        self,
        data: SingleSensorData,
        *,
        stride_event_list: SingleSensorStrideList | None = None,
        sampling_rate_hz: float,
    ) -> Self:
        """Estimate the orientation of the sensor based on the input data."""
        raise NotImplementedError("Needs to be implemented by child class.")


class BasePositionMethod(BaseAlgorithm):
    """Base class for the individual Position estimation methods that work on pd.DataFrame data."""

    _action_methods = ("estimate",)
    velocity_: VelocityList
    position_: PositionList

    def estimate(
        self,
        data: SingleSensorData,
        *,
        stride_event_list: SingleSensorStrideList | None = None,
        sampling_rate_hz: float,
    ) -> Self:
        """Estimate the position of the sensor based on the input data.

        Note that the data is assumed to be in the global-frame (i.e. already rotated)
        """
        raise NotImplementedError("Needs to be implemented by child class.")


class BaseTrajectoryMethod(BasePositionMethod, BaseOrientationMethod):
    """Base class for methods that can compute orientation and position in one pass."""


class BaseTrajectoryReconstructionWrapper(BaseAlgorithm):
    """Base class for method that wrap position and orientation methods to be usable with default datatypes."""

    _action_methods = ("estimate",)

    orientation_: OrientationList
    position_: PositionList
    velocity_: VelocityList


class BaseTemporalParameterCalculation(BaseAlgorithm):
    """Base class for temporal parameters calculation."""

    _action_methods = ("calculate",)

    def calculate(self, stride_event_list: StrideList, sampling_rate_hz: float) -> Self:
        """Find temporal parameters in strides after segmentation and detecting events of each stride."""
        raise NotImplementedError("Needs to be implemented by child class.")


class BaseSpatialParameterCalculation(BaseAlgorithm):
    """Base class for spatial parameter calculation."""

    _action_methods = ("calculate",)

    def calculate(
        self,
        stride_event_list: StrideList,
        positions: PositionList,
        orientations: OrientationList,
        sampling_rate_hz: float,
    ) -> Self:
        """Find spatial parameters in strides after segmentation and detecting events of each stride."""
        raise NotImplementedError("Needs to be implemented by child class.")


class BaseGaitDetection(BaseAlgorithm):
    """Base class for all gait detection algorithms."""

    _action_methods = ("detect",)

    def detect(self, data: SensorData, sampling_rate_hz: float) -> Self:
        """Find gait sequences or other regions of interest in data."""
        raise NotImplementedError("Needs to be implemented by child class.")


class BaseZuptDetector(BaseAlgorithm):
    """Base class for all detection algorithms."""

    _action_methods = ("detect",)

    zupts_: pd.DataFrame
    per_sample_zupts_: np.ndarray

    # These two can be implemented optionally
    min_vel_index_: int | None
    min_vel_value_: float | None

    data: SingleSensorData
    stride_event_list: SingleSensorStrideList | None
    sampling_rate_hz: float

    def detect(
        self,
        data: SingleSensorData,
        *,
        stride_event_list: SingleSensorStrideList | None = None,
        sampling_rate_hz: float,
    ) -> Self:
        """Find ZUPTs in data."""
        raise NotImplementedError("Needs to be implemented by child class.")
