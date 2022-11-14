"""Simple _model base classes and helper."""

import copy
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
import pomegranate as pg
from pomegranate import HiddenMarkovModel as pgHMM
from pomegranate.hmm import History
from tpcp import OptiPara, make_optimize_safe
from typing_extensions import Self

from gaitmap.base import _BaseSerializable
from gaitmap.utils.datatype_helper import SingleSensorData, SingleSensorStrideList
from gaitmap_mad.stride_segmentation.hmm._utils import (
    _clone_model,
    _HackyClonableHMMFix,
    create_transition_matrix_fully_connected,
    create_transition_matrix_left_right,
    fix_model_names,
    gmms_from_samples,
    predict,
)


def initialize_hmm(
    data_train_sequence: Sequence[np.ndarray],
    labels_initialization_sequence: Sequence[np.ndarray],
    *,
    n_states: int,
    n_gmm_components: int,
    architecture: Literal["left-right-strict", "left-right-loose", "fully-connected"],
    name: str = "untrained",
    verbose: bool = False,
):
    """Model Initialization.

    Parameters
    ----------
    data_train_sequence
        list of training sequences this might be e.g. a list of strides where each strides is represented by one
        np.ndarray (which might contain multiple dimensions)
    labels_initialization_sequence
        list of labels which are used to initialize the emission distributions. Note: These labels are just for
        initialization. Distributions will be optimized later in the training process of the model!
        Length of each np.ndarray in "labels_initialization_sequence" needs to match the length of each
        corresponding np.ndarray in "data_train_sequence"
    n_states
        number of hidden-states within the model
    n_gmm_components
        number of components for multivariate distributions
    architecture
        type of model architecture, for more details see below
    name
        The name of the final model object
    verbose
        Whether info should be printed to stdout


    Notes
    -----
    This function supports currently the following "architectures":
    TODO: Add this documentation also to the model level

    - "left-right-strict":
        This will result in a strictly left-right structure, with no self-transitions and
        start- and end-state bound to the first and last state, respectively.
        Example transition matrix for a 5-state model:
        transition_matrix: 1  1  0  0  0   starts: 1  0  0  0  0
                           0  1  1  0  0   stops:  0  0  0  0  1
                           0  0  1  1  0
                           0  0  0  1  1
                           0  0  0  0  1
    - "left-right-loose":
        This will result in a loose left-right structure, with allowed self-transitions and
        start- and end-state not specified initially.
        Example transition matrix for a 5-state model:
        transition_matrix: 1  1  0  0  0   starts: 1  1  1  1  1
                           0  1  1  0  0   stops:  1  1  1  1  1
                           0  0  1  1  0
                           0  0  0  1  1
                           1  0  0  0  1
    - "fully-connected":
        This will result in a fully connected structure where all existing edges are initialized with the same
        probability.
        Example transition matrix for a 5-state model:
        transition_matrix: 1  1  1  1  1   starts: 1  1  1  1  1
                           1  1  1  1  1   stops:  1  1  1  1  1
                           1  1  1  1  1
                           1  1  1  1  1
                           1  1  1  1  1

    """
    if architecture not in ["left-right-strict", "left-right-loose", "fully-connected"]:
        raise ValueError(
            'Invalid architecture given. Must be either "left-right-strict", "left-right-loose" or "fully-connected"'
        )

    # Note: In the past we used a fixed ransom state when generating the gmms.
    # Now we are using a different method of initialization, where this is not needed anymore.
    distributions, _ = gmms_from_samples(
        data_train_sequence,
        labels_initialization_sequence,
        n_gmm_components,
        verbose=verbose,
    )

    # if we force the model into a left-right architecture we know that stride borders should correspond to the point
    # where the model "loops" (aka state-0 and state-n) so we also will enforce the model to start with "state-0" and
    # always end with "state-n"
    if architecture == "left-right-strict":
        transition_matrix, start_probs, end_probs = create_transition_matrix_left_right(n_states, self_transition=False)

    # allow transition model to start and end in all states (as we do not have any specific information about
    # "transitions", this could be actually anything in the data which is no stride)
    elif architecture == "left-right-loose":
        transition_matrix, _, _ = create_transition_matrix_left_right(n_states, self_transition=True)

        start_probs = np.ones(n_states).astype(float)
        end_probs = np.ones(n_states).astype(float)

    # fully connected model with all transitions initialized equally. Allowing all possible transitions.
    else:  # architecture == "fully-connected"
        transition_matrix, start_probs, end_probs = create_transition_matrix_fully_connected(n_states)

    model = pg.HiddenMarkovModel.from_matrix(
        transition_probabilities=transition_matrix,
        distributions=copy.deepcopy(distributions),
        starts=start_probs,
        ends=end_probs,
        verbose=verbose,
    )

    # pomegranate seems to have a strange sorting bug where state names >= 10 (e.g. s10 get sorted in a bad order like
    # s0, s1, s10, s2 usw..)
    model = fix_model_names(model)
    # make sure that transition-matrix is normalized
    model.bake()
    model.name = name

    return model


def train_hmm(
    model_untrained: pgHMM,
    data_train_sequence: Sequence[np.ndarray],
    *,
    max_iterations: int,
    stop_threshold: float,
    algo_train: Literal["viterbi", "baum-welch"],
    name: str = "trained",
    verbose: bool = True,
    n_jobs: int = 1,
) -> Tuple[pgHMM, History]:
    """Train Model.

    Parameters
    ----------
    model_untrained (pomegranate.HiddenMarkovModel)
        pomegranate HiddenMarkovModel object with initialized distributions and transition matrix
    data_train_sequence (list of np.ndarrays)
        list of training sequences this might be e.g. a list of strides where each strides is represented by one
        np.ndarray (which might contain multiple dimensions)
    algo_train (str)
        algorithm for training, can be "viterbi", "baum-welch"
    stop_threshold (float)
        termination criteria for training improvement e.g. 1e-9
    max_iterations (int)
        termination criteria for training iteration number e.g. 1000

    """
    # check if all training sequences have a minimum length of n-states, smaller sequences or empty sequences can lead
    # to unexpected behaviour!
    length_of_training_sequences = np.array([len(data) for data in data_train_sequence])
    # -2 here is because the number of states contains an implicit start and end state which we don't need.
    if np.any(length_of_training_sequences < (len(model_untrained.states) - 2)):
        raise ValueError(
            "Length of all training sequences must be equal or larger than the number of states in the given _model!"
        )

    # make copy from untrained model, as pomegranate will just update parameters in the given model and not returning a
    # copy
    model_trained = _clone_model(model_untrained)

    history: History
    _, history = model_trained.fit(
        sequences=np.array(data_train_sequence, dtype=object).copy(),
        labels=None,
        algorithm=algo_train,
        stop_threshold=stop_threshold,
        max_iterations=max_iterations,
        return_history=True,
        verbose=verbose,
        n_jobs=n_jobs,
    )
    model_trained.name = name

    return model_trained, history


class SimpleHMM(_BaseSerializable, _HackyClonableHMMFix):
    """Wrap all required information to train a new HMM.

    Parameters
    ----------

    See Also
    --------
    TBD

    """

    n_states: int
    n_gmm_components: int
    algo_train: Literal["viterbi", "baum-welch"]
    stop_threshold: float
    max_iterations: Optional[int]
    architecture: Literal["left-right-strict", "left-right-loose", "fully-connected"]
    verbose: bool
    n_jobs: int
    name: Optional[str]
    model: OptiPara[Optional[pgHMM]]
    data_columns: OptiPara[Optional[Tuple[str, ...]]]

    def __init__(
        self,
        n_states: int,
        n_gmm_components: int = None,
        algo_train: Literal["viterbi", "baum-welch"] = "viterbi",
        stop_threshold: float = 1e-9,
        max_iterations: Optional[int] = None,
        architecture: Literal["left-right-strict", "left-right-loose", "fully-connected"] = "left-right-strict",
        verbose: bool = True,
        n_jobs: int = 1,
        name: str = "my_model",
        model: Optional[pgHMM] = None,
        data_columns: Optional[Tuple[str, ...]] = None,
    ):
        self.n_states = n_states
        self.n_gmm_components = n_gmm_components
        self.algo_train = algo_train
        self.stop_threshold = stop_threshold
        self.max_iterations = max_iterations
        self.architecture = architecture
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.name = name
        self.model = model
        self.data_columns = data_columns

    def predict_hidden_state_sequence(
        self, feature_data: SingleSensorData, algorithm: Literal["viterbi", "map"] = "viterbi"
    ) -> np.ndarray:
        """Perform prediction based on given data and given model."""
        # NOTE: We don't consider this method an "action method" by definition, as it requires the algorithm to be
        # specified and does not return self.
        # The reason for that is, that we regularly need to call this method with different algorithms on the same
        # model.
        # Hence, it felt more natural to do it that way.
        # However, as this means this model should always be wrapped in a `SimpleSegmentationHMM` to be used with a
        # standardized API.
        return predict(self.model, feature_data, expected_columns=self.data_columns, algorithm=algorithm)

    @make_optimize_safe
    def self_optimize(
        self, data_sequence: Sequence[SingleSensorData], labels_sequence: Sequence[SingleSensorStrideList]
    ) -> Self:
        return self.self_optimize_with_info(data_sequence, labels_sequence)[0]

    def self_optimize_with_info(
        self, data_sequence: Sequence[SingleSensorData], labels_sequence: Sequence[SingleSensorStrideList]
    ) -> Tuple[Self, History]:
        """Build model."""
        if len(data_sequence) != len(labels_sequence):
            raise ValueError(
                "The given training sequence and initial training labels do not match in their number of individual "
                f"sequences! len(data_train_sequence_list) = {len(data_sequence)} !=  {len(labels_sequence)} = len("
                "initial_hidden_states_sequence_list)"
            )

        for data in data_sequence:
            if len(data) < self.n_states:
                raise ValueError(
                    "Invalid Training Sequence! At least one training sequence has less samples than the specified "
                    f"value of states! n_states = {self.n_states} > {len(data)} = len(data)"
                )

        self.data_columns = tuple(data_sequence[0].columns)
        # you have to make always sure that the input data is in a correct format when using pomegranate, if not this
        # can lead to extremely strange behaviour! Unfortunately pomegranate will not tell if data has a bad format!
        # We also ensure that in all provided dataframes the same columns and column order exists
        data_sequence_train = [
            np.ascontiguousarray(dataset[list(self.data_columns)].to_numpy().copy()) for dataset in data_sequence
        ]

        # initialize model by naive equidistant labels
        model_untrained = initialize_hmm(
            data_sequence_train,
            labels_sequence,
            n_states=self.n_states,
            n_gmm_components=self.n_gmm_components,
            architecture=self.architecture,
            name=self.name + "-untrained",
        )

        # train model
        model_trained, history = train_hmm(
            model_untrained,
            data_sequence_train,
            max_iterations=self.max_iterations,
            stop_threshold=self.stop_threshold,
            algo_train=self.algo_train,
            name=self.name + "_trained",
            verbose=self.verbose,
            n_jobs=self.n_jobs,
        )
        self.model = model_trained

        return self, history
