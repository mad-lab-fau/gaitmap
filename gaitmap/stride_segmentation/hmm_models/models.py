"""Dtw template base classes and helper."""
from importlib.resources import open_text
from typing import Optional, List
import json
from pomegranate import HiddenMarkovModel as pgHMM
import numpy as np
from gaitmap.base import _BaseSerializable


class HiddenMarkovModel(_BaseSerializable):
    """Wrap all required information about a pre-trained HMM.

    Parameters
    ----------
    model_file_name
        Path to a valid pre-trained and serialized model-json
    sampling_rate_hz_model
        The sampling rate of the data the model was trained with
    low_pass_cutoff_hz
        Cutoff frequency of low-pass filter for preprocessing
    low_pass_order
        Low-pass filter order
    axis
        List of sensor axis which will be used as model input
    features
        List of features which will be used as model input
    window_size_samples
        window size of moving centered window for feature extraction
    standardization
        Flag for feature standardization /  z-score normalization

    See Also
    --------
    TBD

    """

    model_file_name: Optional[str]
    sampling_rate_hz_model: Optional[float]
    low_pass_cutoff_hz: Optional[float]
    low_pass_order: Optional[int]
    axis: Optional[List[str]]
    features: Optional[List[str]]
    window_size_samples: Optional[int]
    standardization: Optional[bool]

    _model_combined: Optional[pgHMM]
    _model_stride: Optional[pgHMM]
    _model_transition: Optional[pgHMM]
    _n_states_stride: Optional[int]
    _n_states_transition: Optional[int]

    def __init__(
        self,
        model_file_name: Optional[str] = None,
        sampling_rate_hz_model: Optional[float] = None,
        low_pass_cutoff_hz: Optional[float] = None,
        low_pass_order: Optional[int] = None,
        axis: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        window_size_samples: Optional[int] = None,
        standardization: Optional[bool] = None,
    ):
        self.model_file_name = model_file_name
        self.sampling_rate_hz_model = sampling_rate_hz_model
        self.low_pass_cutoff_hz = low_pass_cutoff_hz
        self.low_pass_order = low_pass_order
        self.axis = axis
        self.features = features
        self.window_size_samples = window_size_samples
        self.standardization = standardization

        # try to load models
        with open_text("gaitmap.stride_segmentation.hmm_models", self.model_file_name) as test_data:
            with open(test_data.name) as f:
                models_dict = json.load(f)

        if (
            "combined_model" not in models_dict.keys()
            or "stride_model" not in models_dict.keys()
            or "transition_model" not in models_dict.keys()
        ):
            raise ValueError(
                "Invalid model-json! Keys within the model-json required are: 'combined_model', 'stride_model' and "
                "'transition_model'"
            )

        self._model_stride = pgHMM.from_json(models_dict["stride_model"])
        self._model_transition = pgHMM.from_json(models_dict["transition_model"])
        self._model_combined = pgHMM.from_json(models_dict["combined_model"])

        # we need to subtract the silent start and end state form the state count!
        self._n_states_stride = self._model_stride.state_count() - 2
        self._n_states_transition = self._model_transition.state_count() - 2

    def predict_hidden_states(self, feature_data, algorithm="viterbi"):
        """Perform prediction based on given data and given model."""
        feature_data = np.ascontiguousarray(feature_data.to_numpy())

        # need to check if memory layout of given data is
        # see related pomegranate issue: https://github.com/jmschrei/pomegranate/issues/717
        if not np.array(feature_data).flags["C_CONTIGUOUS"]:
            raise ValueError("Memory Layout of given input data is not contiguois! Consider using ")

        labels_predicted = np.asarray(self._model_combined.predict(feature_data, algorithm=algorithm))
        # pomegranate always adds an additional label for the start- and end-state, which can be ignored here!
        return np.asarray(labels_predicted[1:-1])

    @property
    def stride_states_(self):
        """Get list of possible stride state keys."""
        return np.arange(self._n_states_transition, self._n_states_stride + self._n_states_transition)

    @property
    def transition_states_(self):
        """Get list of possible transition state keys."""
        return np.arange(self._n_states_transition)


class HiddenMarkovModelStairs(HiddenMarkovModel):
    """Hidden Markov Model trained for stride segmentation including stair strides.

    Notes
    -----
    This is a pre-trained model aiming to segment also "stair strides" next to "normal strides"

    See Also
    --------
    TBD

    """

    model_file_name = "hmm_stairs.json"
    # preprocessing settings
    sampling_rate_hz_model = 51.2
    low_pass_cutoff_hz = 10.0
    low_pass_order = 4
    # feature settings
    axis = ["gyr_ml"]
    features = ["raw", "gradient"]
    window_size_samples = 11
    standardization = True

    def __init__(self):
        super().__init__(
            model_file_name=self.model_file_name,
            sampling_rate_hz_model=self.sampling_rate_hz_model,
            low_pass_cutoff_hz=self.low_pass_cutoff_hz,
            low_pass_order=self.low_pass_order,
            axis=self.axis,
            features=self.features,
            window_size_samples=self.window_size_samples,
            standardization=self.standardization,
        )
