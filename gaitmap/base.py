"""Base class for all algorithms."""

import inspect
from typing import Callable, Dict, TypeVar, Type, Any, List

import numpy as np
import pandas as pd

from gaitmap.utils.dataset_helper import Dataset, SingleSensorDataset, StrideList

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

    def __getattr__(self, item):
        """Add helpful info for certain missing attributes."""
        if item.endswith("_") and not item.startswith("__") and not self._action_is_applied:
            raise AttributeError(
                "`{}` appears to be a result. This means you need to call `{}` before accessing it.".format(
                    item, self._action_method
                )
            )
        super_getter = getattr(super(BaseAlgorithm, self), "__getattr__", None)
        if super_getter:
            return super_getter(item)  # pylint: disable=not-callable
        raise AttributeError(item)

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
        attrs = {v: getattr(self, v) for v in vars(self) if v.endswith("_") and not v.startswith("__")}
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

    def detect(self: BaseType, data: Dataset, sampling_rate_hz: float, segmented_stride_list: pd.DataFrame) -> BaseType:
        """Find gait events in data within strides provided by stride_list."""
        raise NotImplementedError("Needs to be implemented by child class.")


class BaseOrientationEstimation(BaseAlgorithm):
    """Base class for all algorithms that estimate an orientation from measured sensor signals."""

    estimated_orientations_: Rotation

    def estimate(self, data: SingleSensorDataset, sampling_rate_hz: float):
        """Estimates orientation of the sensor for all samples in sensor data based on the given initial orientation.

        Parameters
        ----------
        data : pandas.DataFrame
            Contains at least gyroscope data, optionally also acceleration data of one or several sensors.
        sampling_rate_hz : float
            Data with which gyroscope data was sampled in Hz.

        """
        raise NotImplementedError()

    # I would like to leave out get/set_parameters since this is not necessary for all methods (e.g. gyroscope
    # integration)


class BaseTemporalParameterCalculation(BaseAlgorithm):
    """Base class for temporal parameters calculation."""

    _action_method = "calculate"

    def calculate(self: BaseType, stride_event_list: StrideList, sampling_rate_hz: float) -> BaseType:
        """Find temporal parameters in in strides after segmentation and detecting events of each stride."""
        raise NotImplementedError("Needs to be implemented by child class.")
