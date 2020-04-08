"""Base class for all algorithms."""

import inspect
import types
from typing import Callable, Dict, TypeVar, Type, Any, List, Union

import numpy as np
import pandas as pd

from gaitmap.utils.dataset_helper import (
    Dataset,
    is_multi_sensor_dataset,
    is_single_sensor_dataset,
    get_multi_sensor_dataset_names,
    StrideList,
    PositionList,
    OrienationList,
)

BaseType = TypeVar("BaseType", bound="BaseAlgorithms")


class BaseAlgorithm:
    """Base class for all algorithms.

    All type-specific algorithm classes should inherit from this class and need to

    1. overwrite `_action_method` with the name of the actual action method of this class type
    2. implement a stub for the action method

    Attributes
    ----------
    _action_method
        The name of the action method used by the Childclass

    """

    _action_method: str

    @property
    def _action_is_applied(self) -> bool:
        """Check if the action method was already called/results were generated."""
        if len(self.get_attributes()) == 0:
            return False
        return True

    def _get_action_method(self) -> Callable:
        """Get the action method as callable.

        This is intended to be used by wrappers, that do not know the Type of an algorithm
        """
        return getattr(self, self._action_method)

    @classmethod
    def _get_param_names(cls) -> List[str]:
        """Get parameter names for the estimator.

        The parameters of an algorithm are defined based on its `__init__` method.
        All parameters of this method are considered parameters of the algorithm.

        Notes
        -----
        Adopted based on `sklearn BaseEstimator._get_param_names`.

        Returns
        -------
        param_names
            List of parameter names of the algorithm

        """
        # fetch the constructor or the original constructor before deprecation wrapping if any
        init = cls.__init__
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values() if p.name != "self" and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "gaitmap-algorithms should always specify their parameters in the signature of their "
                    "__init__ (no varargs). {} with constructor {} doesn't follow this convention.".format(
                        cls, init_signature
                    )
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self) -> Dict[str, Any]:
        """Get parameters for this algorithm.

        Returns
        -------
        params
            Parameter names mapped to their values.

        """
        return {k: getattr(self, k) for k in self._get_param_names()}

    def set_params(self: BaseType, **params: Any) -> Type[BaseType]:
        """Set the parameters of this Algorithm."""
        raise NotImplementedError("This will be implemented in the future")

    def get_other_params(self) -> Dict[str, Any]:
        """Get all "Other Parameters" of the Algorithm.

        "Other Parameters" are all parameters set outside of the `__init__` that are not considered results.
        This usually includes the "data" and all other parameters passed to the action method.

        Returns
        -------
        params
            Parameter names mapped to their values.

        """
        params = self.get_params()
        attrs = {
            v: getattr(self, v) for v in vars(self) if not v.endswith("_") and not v.startswith("_") and v not in params
        }
        return attrs

    def get_attributes(self) -> Dict[str, Any]:
        """Get all Attributes of the Algorithm.

        "Attributes" are all values considered results of the algorithm.
        They are indicated by a trailing "_" in their name.
        The values are only populated after the action method of the algorithm was called.

        Returns
        -------
        params
            Parameter names mapped to their values.

        Raises
        ------
        AttributeError
            If one or more of the attributes are not retrievable from the instance.
            This usually indicates that the action method was not called yet.

        """
        all_attributes = dir(self)
        attrs = {
            v: getattr(self, v)
            for v in all_attributes
            if v.endswith("_") and not v.startswith("__") and not isinstance(getattr(self, v), types.MethodType)
        }
        return attrs


class BaseStrideSegmentation(BaseAlgorithm):
    """Base class for all stride segmentation algorithms."""

    _action_method = "segment"

    def segment(self: BaseType, data: np.ndarray, sampling_rate_hz: float, **kwargs) -> BaseType:
        """Find stride candidates in data."""
        raise NotImplementedError("Needs to be implemented by child class.")


class BaseEventDetection(BaseAlgorithm):
    """Base class for all event detection algorithms."""

    _action_method = "detect"

    def detect(self: BaseType, data: Dataset, sampling_rate_hz: float, segmented_stride_list: StrideList) -> BaseType:
        """Find gait events in data within strides provided by stride_list."""
        raise NotImplementedError("Needs to be implemented by child class.")


class BaseOrientationEstimation(BaseAlgorithm):
    """Base class for all algorithms that estimate an orientation from measured sensor signals.

    Attributes
    ----------
    estimated_orientations_
        Holds one quaternion for the initial orientation and `len(data)` quaternions for the subsequent orientations
        as obtained by calling `.estimate(...)` of the specific class.
    estimated_orientations_without_initial_
        Contains `estimated_orientations_` but for each stride, the INITIAL rotation is REMOVED to make it the same
        length as `len(self.data)`.
    estimated_orientations_without_final_
        Contains `estimated_orientations_` but for each stride, the FINAL rotation is REMOVED to make it the same
        length as `len(self.data)`

    """

    # TODO: when implementing a different algorithm, check if initial orientation can be obtained in the same way as
    #       for GyroIntegration. If so, check if that can become part of this base class.

    _action_method = "estimate"

    estimated_orientations_: Union[pd.DataFrame, Dict[str, pd.DataFrame]]

    def estimate(self: BaseType, data: Dataset, stride_event_list: StrideList, sampling_rate_hz: float) -> BaseType:
        """Estimates orientation of the sensor for all samples in sensor data based on the given initial orientation."""
        raise NotImplementedError()

    # TODO: In case a new algorithm is implemented and it turns out, that these algorithms for some reason do not need
    #       the `without_final` and `without_initial` these two methods should be moved to
    #       `gaitmap.trajectory_reconstruction.orientation_estimation`.

    @property
    def estimated_orientations_without_final_(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Return the estimated orientations without initial orientation.

        The number of rotations is equal to the number samples in passed data.
        """

        def remove_final_for_sensor(orientations):
            ori_without_final_sensor = pd.DataFrame()
            for _, i_oris in orientations.groupby(axis=0, level="s_id"):
                ori_without_final_sensor = ori_without_final_sensor.append(i_oris[:-1])
            return ori_without_final_sensor

        if is_multi_sensor_dataset(self.data):
            ori_without_final = dict()
            for i_sensor, i_orientations in self.estimated_orientations_.items():
                ori_without_final[i_sensor] = remove_final_for_sensor(i_orientations)
            return ori_without_final
        if is_single_sensor_dataset(self.data):
            return remove_final_for_sensor(self.estimated_orientations_)
        raise ValueError("Unsuppported datatype.")

    @property
    def estimated_orientations_without_initial_(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Return the estimated orientations without initial orientation.

        The number of rotations is equal to the number samples in passed data.
        """
        if is_multi_sensor_dataset(self.data):
            ori_without_initial = dict()
            for i_sensor in get_multi_sensor_dataset_names(self.data):
                ori_without_initial[i_sensor] = self.estimated_orientations_[i_sensor].drop(
                    axis=0, level="sample", index=0
                )
            return ori_without_initial
        if is_single_sensor_dataset(self.data):
            return self.estimated_orientations_.drop(axis=0, level="sample", index=0)
        raise ValueError("Unsuppported datatype.")


class BasePositionEstimation(BaseAlgorithm):
    """Base class for all position reconstruction methods."""

    _action_method = "estimate"

    def estimate(self: BaseType, data: Dataset, event_list: StrideList, sampling_rate_hz: float) -> BaseType:
        """Estimate position relative to first sample by using sensor data."""
        raise NotImplementedError("Needs to be implemented by child class.")


class BaseTemporalParameterCalculation(BaseAlgorithm):
    """Base class for temporal parameters calculation."""

    _action_method = "calculate"

    def calculate(self: BaseType, stride_event_list: StrideList, sampling_rate_hz: float) -> BaseType:
        """Find temporal parameters in in strides after segmentation and detecting events of each stride."""
        raise NotImplementedError("Needs to be implemented by child class.")


class BaseSpatialParameterCalculation(BaseAlgorithm):
    """Base class for temporal parameters calculation."""

    _action_method = "calculate"

    def calculate(self: BaseType, stride_event_list: StrideList, positions: PositionList,
                  orientations: OrienationList) -> BaseType:
        """Find temporal parameters in in strides after segmentation and detecting events of each stride."""
        raise NotImplementedError("Needs to be implemented by child class.")