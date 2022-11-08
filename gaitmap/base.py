"""Base class for all algorithms."""

import json
import warnings
from typing import Any, Dict, Type, TypeVar, Union

import numpy as np
import pandas as pd
import tpcp
from joblib import Memory
from pomegranate.hmm import HiddenMarkovModel
from scipy.spatial.transform import Rotation

from gaitmap.utils.consts import GF_ORI
from gaitmap.utils.datatype_helper import (
    OrientationList,
    PositionList,
    SensorData,
    SingleSensorData,
    SingleSensorOrientationList,
    StrideList,
    VelocityList,
)

BaseType = TypeVar("BaseType", bound="_BaseSerializable")  # noqa: invalid-name


def _hint_tuples(item):
    """Encode tuple values for json serialization.

    Modified based on: https://stackoverflow.com/questions/15721363/preserve-python-tuples-with-json
    """
    if isinstance(item, tuple):
        return dict(_obj_type="Tuple", tuple=item)
    if isinstance(item, list):
        return [_hint_tuples(e) for e in item]
    if isinstance(item, dict):
        return {key: _hint_tuples(value) for key, value in item.items()}
    return item


class _CustomEncoder(json.JSONEncoder):
    def encode(self, o: Any) -> str:
        return super().encode(_hint_tuples(o))

    def default(self, o):  # noqa: method-hidden
        if isinstance(o, _BaseSerializable):
            return o._to_json_dict()
        if isinstance(o, Rotation):
            return dict(_obj_type="Rotation", quat=o.as_quat().tolist())
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return dict(_obj_type="Array", array=o.tolist())
        if isinstance(o, pd.DataFrame):
            return dict(_obj_type="DataFrame", df=o.to_json(orient="split"))
        if isinstance(o, pd.Series):
            return dict(_obj_type="Series", df=o.to_json(orient="split"))
        if isinstance(o, HiddenMarkovModel):
            warnings.warn(
                "Exporting `pomegranate.hmm.HiddenMarkovModel` objects to json can sometimes not provide perfect "
                "round-trips. I.e. sometimes values (in particular weightings of distributions) might change slightly "
                "in the re-imported model due to rounding issue. "
                "This is a limitation of the underlying pomegrante library."
            )
            return dict(_obj_type="HiddenMarkovModel", hmm=json.loads(o.to_json()))
        if o is tpcp.NOTHING:
            return dict(_obj_type="EmptyDefault")
        if isinstance(o, Memory):
            warnings.warn(
                "Exporting `joblib.Memory` objects to json is not supported. "
                "The value will be replaced by `None` and caching needs to be reactivated after loading the "
                "object again. "
                "This can be using `instance.set_params(memory=Memory(...))`"
            )
            return None
        # Let the base class default method raise the TypeError
        return super().default(o)


def _custom_deserialize(json_obj):  # noqa: too-many-return-statements
    if "_gaitmap_obj" in json_obj:
        return _BaseSerializable._find_subclass(json_obj["_gaitmap_obj"])._from_json_dict(json_obj)
    if "_obj_type" in json_obj:
        if json_obj["_obj_type"] == "Rotation":
            return Rotation.from_quat(json_obj["quat"])
        if json_obj["_obj_type"] == "Array":
            return np.array(json_obj["array"])
        if json_obj["_obj_type"] in ["Series", "DataFrame"]:
            typ = "series" if json_obj["_obj_type"] == "Series" else "frame"
            return pd.read_json(json_obj["df"], orient="split", typ=typ)
        if json_obj["_obj_type"] == "HiddenMarkovModel":
            return HiddenMarkovModel.from_dict(json_obj["hmm"])
        if json_obj["_obj_type"] == "EmptyDefault":
            return tpcp.NOTHING
        if json_obj["_obj_type"] == "Tuple":
            return tuple(json_obj["tuple"])
        raise ValueError("Unknown object type found in serialization!")

    return json_obj


class _BaseSerializable(tpcp.BaseTpcpObject):
    @classmethod
    def _get_subclasses(cls: Type[BaseType]):
        for subclass in cls.__subclasses__():
            yield from subclass._get_subclasses()
            yield subclass

    @classmethod
    def _find_subclass(cls: Type[BaseType], name: str) -> Type[BaseType]:
        for subclass in _BaseSerializable._get_subclasses():
            if subclass.__name__ == name:
                return subclass
        raise ValueError("No algorithm class with name {} exists".format(name))

    @classmethod
    def _from_json_dict(cls: Type[BaseType], json_dict: Dict) -> BaseType:
        params = json_dict["params"]
        input_data = {k: params[k] for k in tpcp.get_param_names(cls) if k in params}
        instance = cls(**input_data)
        return instance

    def _to_json_dict(self) -> Dict[str, Any]:
        json_dict: Dict[str, Union[str, Dict[str, Any]]] = {
            "_gaitmap_obj": self.__class__.__name__,
            "params": self.get_params(deep=False),
        }
        return json_dict

    def to_json(self) -> str:
        """Export the current object parameters as json.

        For details have a look at the this :ref:`example <algo_serialize>`.

        You can use the `from_json` method of any gaitmap algorithm to load the object again.

        .. warning:: This will only export the Parameters of the instance, but **not** any results!

        """
        final_dict = self._to_json_dict()
        return json.dumps(final_dict, indent=4, cls=_CustomEncoder)

    @classmethod
    def from_json(cls: Type[BaseType], json_str: str) -> BaseType:
        """Import an gaitmap object from its json representation.

        For details have a look at the this :ref:`example <algo_serialize>`.

        You can use the `to_json` method of a class to export it as a compatible json string.

        Parameters
        ----------
        json_str
            json formatted string

        """
        instance = json.loads(json_str, object_hook=_custom_deserialize)
        return instance


class BaseAlgorithm(tpcp.Algorithm, _BaseSerializable):
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

    def align(self: BaseType, data: SensorData, **kwargs) -> BaseType:
        """Align sensor data."""
        raise NotImplementedError("Needs to be implemented by child class.")


class BaseStrideSegmentation(BaseAlgorithm):
    """Base class for all stride segmentation algorithms."""

    _action_methods = ("segment",)

    stride_list_: StrideList

    def segment(self: BaseType, data: SensorData, sampling_rate_hz: float, **kwargs) -> BaseType:
        """Find stride candidates in data."""
        raise NotImplementedError("Needs to be implemented by child class.")


class BaseEventDetection(BaseAlgorithm):
    """Base class for all event detection algorithms."""

    _action_methods = ("detect",)

    def detect(self: BaseType, data: SensorData, stride_list: StrideList, sampling_rate_hz: float) -> BaseType:
        """Find gait events in data within strides provided by roi_list."""
        raise NotImplementedError("Needs to be implemented by child class.")


class BaseOrientationMethod(BaseAlgorithm):
    """Base class for the individual Orientation estimation methods that work on pd.DataFrame data."""

    _action_methods = ("estimate",)
    orientation_object_: Rotation

    @property
    def orientation_(self) -> SingleSensorOrientationList:
        """Orientations as pd.DataFrame."""
        df = pd.DataFrame(self.orientation_object_.as_quat(), columns=GF_ORI)
        df.index.name = "sample"
        return df

    def estimate(self: BaseType, data: SingleSensorData, sampling_rate_hz: float) -> BaseType:
        """Estimate the orientation of the sensor based on the input data."""
        raise NotImplementedError("Needs to be implemented by child class.")


class BasePositionMethod(BaseAlgorithm):
    """Base class for the individual Position estimation methods that work on pd.DataFrame data."""

    _action_methods = ("estimate",)
    velocity_: VelocityList
    position_: PositionList

    def estimate(self: BaseType, data: SingleSensorData, sampling_rate_hz: float) -> BaseType:
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

    def calculate(self: BaseType, stride_event_list: StrideList, sampling_rate_hz: float) -> BaseType:
        """Find temporal parameters in strides after segmentation and detecting events of each stride."""
        raise NotImplementedError("Needs to be implemented by child class.")


class BaseSpatialParameterCalculation(BaseAlgorithm):
    """Base class for spatial parameters calculation."""

    _action_methods = ("calculate",)

    def calculate(
        self: BaseType,
        stride_event_list: StrideList,
        positions: PositionList,
        orientations: OrientationList,
        sampling_rate_hz: float,
    ) -> BaseType:
        """Find spatial parameters in strides after segmentation and detecting events of each stride."""
        raise NotImplementedError("Needs to be implemented by child class.")


class BaseGaitDetection(BaseAlgorithm):
    """Base class for all gait detection algorithms."""

    _action_methods = ("detect",)

    def detect(self: BaseType, data: SensorData, sampling_rate_hz: float) -> BaseType:
        """Find gait sequences or other regions of interest in data."""
        raise NotImplementedError("Needs to be implemented by child class.")


class BaseZuptDetector(BaseAlgorithm):
    """Base class for all detection algorithms."""

    _action_methods = ("detect",)

    zupts_: pd.DataFrame
    per_sample_zupts_: np.ndarray

    def detect(self: BaseType, data: SensorData, sampling_rate_hz: float, **kwargs) -> BaseType:
        """Find ZUPTs in data."""
        raise NotImplementedError("Needs to be implemented by child class.")
