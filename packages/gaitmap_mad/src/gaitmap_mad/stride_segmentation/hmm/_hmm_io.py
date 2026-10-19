"""Persistence adapters for native HMM parameters and legacy gaitmap model files."""

from __future__ import annotations

import json
from functools import partial
from io import StringIO
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tpcp import NOTHING, get_param_names

from gaitmap_mad.stride_segmentation.hmm._hmm_distributions import GaussianMixture
from gaitmap_mad.stride_segmentation.hmm._hmm_inference import HiddenMarkovModel

if TYPE_CHECKING:
    from gaitmap_mad.stride_segmentation.hmm._composite_model import CompositeHmm
    from gaitmap_mad.stride_segmentation.hmm._hmm_stride_segmentation import HmmStrideSegmentation
    from gaitmap_mad.stride_segmentation.hmm._segmentation_model import BaseSegmentationHmm
    from gaitmap_mad.stride_segmentation.hmm._simple_model import SimpleHmm


def load_legacy_hmm_json(serialized: str) -> SimpleHmm | CompositeHmm | BaseSegmentationHmm | HmmStrideSegmentation:
    """Load an HMM wrapper saved with the former algorithm JSON API.

    This read-only migration helper accepts existing ``SimpleHmm``,
    ``CompositeHmm``, segmentation HMM and ``HmmStrideSegmentation`` files,
    including their nested feature transforms and postprocessing settings.
    Custom classes used in the file must be imported before loading.
    Save the loaded object with ``joblib.dump`` for subsequent use.
    Numeric HMM files use :meth:`HiddenMarkovModel.from_json` instead.

    Parameters
    ----------
    serialized
        JSON text exported by the former HMM wrapper ``to_json()`` method.

    Returns
    -------
    model
        Restored HMM wrapper, including its saved parameters and feature transforms.
    """
    from gaitmap.base import BaseAlgorithm  # noqa: PLC0415
    from gaitmap_mad.stride_segmentation.hmm._composite_model import CompositeHmm  # noqa: PLC0415
    from gaitmap_mad.stride_segmentation.hmm._segmentation_model import BaseSegmentationHmm  # noqa: PLC0415
    from gaitmap_mad.stride_segmentation.hmm._simple_model import SimpleHmm  # noqa: PLC0415

    def subclasses(cls):
        for child in cls.__subclasses__():
            yield from subclasses(child)
        yield cls

    classes = {
        cls.__name__: cls
        for root in (BaseAlgorithm, SimpleHmm, CompositeHmm, BaseSegmentationHmm)
        for cls in subclasses(root)
    }

    model = json.loads(serialized, object_hook=partial(_decode_legacy_hmm_object, classes=classes))
    if isinstance(model, (SimpleHmm, CompositeHmm, BaseSegmentationHmm)):
        return model
    # The bundled model loads while HmmStrideSegmentation's defaults are being defined.
    from gaitmap_mad.stride_segmentation.hmm._hmm_stride_segmentation import HmmStrideSegmentation  # noqa: PLC0415

    if isinstance(model, HmmStrideSegmentation):
        return model
    raise TypeError("Expected a saved gaitmap HMM wrapper.")


def _decode_legacy_hmm_object(obj, *, classes):  # noqa: PLR0911
    if "_gaitmap_obj" in obj:
        cls = classes[obj["_gaitmap_obj"]]
        return cls(**{key: value for key, value in obj["params"].items() if key in get_param_names(cls)})
    kind = obj.get("_obj_type")
    if kind is None:
        return obj
    if kind == "HiddenMarkovModel":
        return HiddenMarkovModel.from_dict(obj["hmm"])
    if kind == "Tuple":
        return tuple(obj["tuple"])
    if kind == "Array":
        return np.array(obj["array"])
    if kind == "Rotation":
        return Rotation.from_quat(obj["quat"])
    if kind in ("Series", "DataFrame"):
        return pd.read_json(StringIO(obj["df"]), orient="split", typ="series" if kind == "Series" else "frame")
    if kind == "EmptyDefault":
        return NOTHING
    raise ValueError(f"Unknown object type in saved HMM: {kind}.")


def load_emission(data: dict) -> GaussianMixture:
    """Read native Gaussian mixtures or legacy Gaussian/GMM dictionaries."""
    if "means" in data:
        return GaussianMixture(
            data["means"],
            data["covariances"],
            data["weights"],
            frozen=data.get("frozen", False),
            component_frozen=data.get("component_frozen"),
        )
    if data.get("class") == "GeneralMixtureModel":
        components = [load_emission(component) for component in data["distributions"]]
        if any(component.n_components != 1 for component in components):
            raise ValueError("Nested Gaussian mixtures are not supported.")
        return GaussianMixture(
            np.concatenate([component.means for component in components]),
            np.concatenate([component.covariances for component in components]),
            data["weights"],
            frozen=data.get("frozen", False),
            component_frozen=[component.frozen for component in components],
        )
    if data.get("class") != "Distribution" or data.get("name") != "MultivariateGaussianDistribution":
        raise ValueError("Only full-covariance Gaussian emissions are supported.")
    mean, covariance = data["parameters"]
    return GaussianMixture([mean], [covariance], [1.0], frozen=data.get("frozen", False))


def serialize_emission(emission: GaussianMixture) -> dict:
    """Write native emission parameters without implementation caches."""
    return {
        "means": emission.means.tolist(),
        "covariances": emission.covariances.tolist(),
        "weights": emission.weights.tolist(),
        "frozen": emission.frozen,
        "component_frozen": emission.component_frozen.tolist(),
    }


def serialize_model(model: HiddenMarkovModel) -> dict:
    """Write the versioned native HMM format, containing only emitting states."""
    return {
        "format": "gaitmap-hmm",
        "version": 1,
        "name": model.name,
        "emissions": [serialize_emission(emission) for emission in model.emissions],
        "transitions": model.transitions.tolist(),
        "starts": model.starts.tolist(),
        "ends": None if model.ends is None else model.ends.tolist(),
        "transition_mask": model.transition_mask.tolist(),
        "start_mask": model.start_mask.tolist(),
        "end_mask": None if model.end_mask is None else model.end_mask.tolist(),
        "state_names": list(model.state_names),
        "state_weights": model.state_weights.tolist(),
    }


def _validate_probabilities(transitions, starts, ends, state_weights):
    for values in (transitions, starts, ends):
        if values is None:
            continue
        probabilities = np.asarray(values, dtype=float)
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("Saved probabilities must be finite values between zero and one.")
    if not np.isfinite(state_weights).all() or np.any(np.asarray(state_weights) < 0):
        raise ValueError("Saved state weights must be finite and nonnegative.")


def load_model(data: dict) -> HiddenMarkovModel:
    """Load native models or convert the supported legacy pomegranate subset."""
    if data.get("format") == "gaitmap-hmm":
        if data["version"] != 1:
            raise ValueError(f"Unsupported HMM format version: {data['version']}.")
        _validate_probabilities(data["transitions"], data["starts"], data["ends"], data["state_weights"])
        return HiddenMarkovModel(
            [load_emission(emission) for emission in data["emissions"]],
            data["transitions"],
            data["starts"],
            data["ends"],
            name=data["name"],
            state_names=data["state_names"],
            state_weights=data["state_weights"],
            transition_mask=data["transition_mask"],
            start_mask=data["start_mask"],
            end_mask=data["end_mask"],
        )
    return _load_legacy_model(data)


def _load_legacy_model(data: dict) -> HiddenMarkovModel:
    if data.get("class") != "HiddenMarkovModel":
        raise ValueError("Expected a native or legacy Gaussian HMM model.")
    if data.get("distribution ties") or any(edge[4] is not None for edge in data["edges"]):
        raise ValueError("Tied distributions and transitions are not supported.")
    n_states = data["silent_index"]
    start, end = data["start_index"], data["end_index"]
    states = data["states"]
    if n_states < 1 or {start, end} != {n_states, n_states + 1} or len(states) != n_states + 2:
        raise ValueError("Additional silent states are not supported.")
    if any(state["distribution"] is None for state in states[:n_states]) or any(
        state["distribution"] is not None for state in states[n_states:]
    ):
        raise ValueError("Only emitting states followed by silent start/end states are supported.")
    probabilities = np.zeros((n_states + 2, n_states + 2))
    allowed = np.zeros_like(probabilities, dtype=bool)
    finite = False
    for source, target, probability, _, _ in data["edges"]:
        if not 0 <= source < n_states + 2 or not 0 <= target < n_states + 2:
            raise ValueError("Transition indices must reference existing states.")
        if target == start or source == end:
            raise ValueError("Transitions into start or out of end are not supported.")
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Transition probabilities must be finite values between zero and one.")
        probabilities[source, target] = probability
        allowed[source, target] = True
        finite |= target == end
    state_weights = [state["weight"] for state in states[:n_states]]
    _validate_probabilities(probabilities, probabilities[start, :n_states], None, state_weights)
    return HiddenMarkovModel(
        [load_emission(state["distribution"]) for state in states[:n_states]],
        probabilities[:n_states, :n_states],
        probabilities[start, :n_states],
        probabilities[:n_states, end] if finite else None,
        name=data["name"],
        state_names=[state["name"] for state in states[:n_states]],
        state_weights=state_weights,
        transition_mask=allowed[:n_states, :n_states],
        start_mask=allowed[start, :n_states],
        end_mask=allowed[:n_states, end] if finite else None,
    )
