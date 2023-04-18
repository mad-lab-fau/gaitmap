"""Simple model base classes and helper."""

import warnings
from typing import Literal, Optional, Sequence, Tuple, Union, List

import numpy as np
import pandas as pd
from pomegranate.distributions._distribution import Distribution
from pomegranate.hmm import DenseHMM as pgHMM
from tpcp import OptiPara, make_optimize_safe
from typing_extensions import Self

from gaitmap.base import _BaseSerializable
from gaitmap.utils.datatype_helper import SingleSensorData
from gaitmap_mad.stride_segmentation.hmm._utils import (
    ShortenedHMMPrint,
    create_transition_matrix_fully_connected,
    create_transition_matrix_left_right,
    gmms_from_samples,
    labels_to_prior,
    predict,
)


def initialize_distributions_and_transmat(
    data_train_sequence: Sequence[np.ndarray],
    labels_initialization_sequence: Sequence[np.ndarray],
    *,
    n_states: int,
    n_gmm_components: int,
    architecture: Literal["left-right-strict", "left-right-loose", "fully-connected"],
    verbose: bool = False,
) -> Tuple[List[Distribution], np.ndarray, np.ndarray, np.ndarray]:
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
    verbose
        Whether info should be printed to stdout


    Notes
    -----
    This function supports currently the following "architectures":

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

    # Note: In the past we used a fixed random state when generating the gmms.
    # Now we are using a different method of initialization, where this is not needed anymore.
    distributions, _ = gmms_from_samples(
        data_train_sequence,
        labels_initialization_sequence,
        n_gmm_components,
        n_states,
        verbose=verbose,
    )
    n_states = len(distributions)
    # if we force the model into a left-right architecture we know that stride borders should correspond to the point
    # where the model "loops" (aka state-0 and state-n) so we also will enforce the model to start with "state-0" and
    # always end with "state-n"
    if architecture == "left-right-strict":
        transition_matrix, start_probs, end_probs = create_transition_matrix_left_right(n_states, self_transition=False)

    # allow transition model to start and end in all states (as we do not have any specific information about
    # "transitions", this could be actually anything in the data which is no stride)
    elif architecture == "left-right-loose":
        transition_matrix, _, _ = create_transition_matrix_left_right(n_states, self_transition=True)

        start_probs = np.ones(n_states).astype(np.float32) / n_states
        end_probs = np.ones(n_states).astype(np.float32) / n_states

    # fully connected model with all transitions initialized equally. Allowing all possible transitions.
    else:  # architecture == "fully-connected"
        transition_matrix, start_probs, end_probs = create_transition_matrix_fully_connected(n_states)

    return distributions, transition_matrix.astype(np.float32), start_probs, end_probs


class SimpleHmm(_BaseSerializable, ShortenedHMMPrint):
    """Wrap all required information to train a new HMM.

    This is a thin wrapper around the pomegranate HiddenMarkovModel class and basically calls out the pomegranate for
    all core functionality.

    .. note:: This class is not intended to be used directly, but should be used as stride/transition model in the
              :class:`~gaitmap.stride_segmentation.hmm.SegmentationModel`.
              `SimpleHmm` therefore does not provide the same interface as other gaitmap algorithms.
              It does not have a dedicated action method, but only has a `predict_hidden_state_sequence` method that
              directly returns the hidden state sequence and does not store it on the object.
              The reason for that is, that we regularly need to call this method with different algorithms on the same
              model.
              Hence, it felt more natural to do it that way.

    Parameters
    ----------
    n_states
        The number of states in the model.
    n_gmm_components
        The number of components in the GMMs.
        Each state will be represented by its own GMM with this number of components.
    architecture
        The architecture of the model. Can be either "left-right-strict", "left-right-loose" or "fully-connected".
        See Notes for more information.
    algo_train
        The algorithm to use for training.
        Can be either "viterbi" or "baum-welch".
    stop_threshold
        The threshold for the training algorithm to stop.
    max_iterations
        The maximum number of iterations for the training algorithm.
    name
        The name of the model.
    verbose
        Whether to print progress information during training.
    n_jobs
        The number of jobs to use for training.
        If set to -1, all available cores will be used.
    model
        The actual pomegranate HMM model.
        This can be set to `None` initially.
        A model will then be created during the optimization step.
        If you want to use a pre-trained model, you can set this parameter to the respective model.
        However, we recommend to ideally export this entire class instead of just the model to make sure that things
        like the feature transform are also exported/stored.
    data_columns
        The expected columns of the input data in feature space.
        This will be automatically set based on the feature transform output during the optimization step.
        This does not affect the output, but is used as a sanity check to ensure that valid input data is provided
        and that the column order is correct.

    Notes
    -----
    This model supports currently the following "architectures":

    - "left-right-strict":
      This will result in a strictly left-right structure, with no self-transitions and
      start- and end-state bound to the first and last state, respectively.
      Example transition matrix for a 5-state model:

      .. code::

          transition_matrix: 1  1  0  0  0   starts: 1  0  0  0  0
                             0  1  1  0  0   stops:  0  0  0  0  1
                             0  0  1  1  0
                             0  0  0  1  1
                             0  0  0  0  1

    - "left-right-loose":
      This will result in a loose left-right structure, with allowed self-transitions and
      start- and end-state not specified initially.
      Example transition matrix for a 5-state model:

      .. code::

          transition_matrix: 1  1  0  0  0   starts: 1  1  1  1  1
                             0  1  1  0  0   stops:  1  1  1  1  1
                             0  0  1  1  0
                             0  0  0  1  1
                             1  0  0  0  1

    - "fully-connected":
      This will result in a fully connected structure where all existing edges are initialized with the same
      probability.
      Example transition matrix for a 5-state model:

      .. code::

          transition_matrix: 1  1  1  1  1   starts: 1  1  1  1  1
                             1  1  1  1  1   stops:  1  1  1  1  1
                             1  1  1  1  1
                             1  1  1  1  1
                             1  1  1  1  1

    See Also
    --------
    TBD

    """

    n_states: int
    n_gmm_components: int
    architecture: Literal["left-right-strict", "left-right-loose", "fully-connected"]
    stop_threshold: float
    max_iterations: int
    verbose: bool
    n_jobs: int
    model: OptiPara[Optional[pgHMM]]
    data_columns: OptiPara[Optional[Tuple[str, ...]]]

    def __init__(
        self,
        n_states: int,
        n_gmm_components: int,
        *,
        architecture: Literal["left-right-strict", "left-right-loose", "fully-connected"] = "left-right-strict",
        stop_threshold: float = 1e-9,
        max_iterations: int = 1e8,
        verbose: bool = True,
        n_jobs: int = 1,
        model: Optional[pgHMM] = None,
        data_columns: Optional[Tuple[str, ...]] = None,
    ):
        self.n_states = n_states
        self.n_gmm_components = n_gmm_components
        self.stop_threshold = stop_threshold
        self.max_iterations = max_iterations
        self.architecture = architecture
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.model = model
        self.data_columns = data_columns

    def predict_hidden_state_sequence(self, feature_data: SingleSensorData) -> np.ndarray:
        """Perform prediction based on given data and given model.

        Parameters
        ----------
        feature_data
            The data to predict the hidden state sequence for.
            Note, that the data must have at least the same columns as the data used for training.
            The order of the columns does not matter.
        algorithm
            The algorithm to use for prediction.
            Can be either "viterbi" or "map".

        Returns
        -------
        np.ndarray
            The predicted hidden state sequence.

        """
        # NOTE: We don't consider this method an "action method" by definition, as it requires the algorithm to be
        # specified and does not return self.
        # The reason for that is, that we regularly need to call this method with different algorithms on the same
        # model.
        # Hence, it felt more natural to do it that way.
        # However, as this means this model should always be wrapped in a `RothSegmentationHmm` to be used with a
        # standardized API.
        return predict(self.model, feature_data, expected_columns=self.data_columns)

    @make_optimize_safe
    def self_optimize(
        self,
        data_sequence: Sequence[SingleSensorData],
        labels_sequence: Sequence[Union[np.ndarray, pd.Series, pd.DataFrame]],
    ) -> Self:
        """Create and train the HMM model based on the given data and labels.

        Parameters
        ----------
        data_sequence
            Sequence of gaitmap sensordata objects.
        labels_sequence
            Sequence of gaitmap stride lists.
            The number of stride lists must match the number of sensordata objects (i.e. they must belong together).
            Each label sequence should only contain integers in the range [0, n_states - 1].
            The usage of the labels depends on the train algorithm.
            In case of `viterbi` and `baum-welch`, the labels are only used to identify the initial data clusters.

        Returns
        -------
        self
            The trained model instance.

        """
        return self.self_optimize_with_info(data_sequence, labels_sequence)[0]

    def self_optimize_with_info(
        self,
        data_sequence: Sequence[SingleSensorData],
        labels_sequence: Sequence[Union[np.ndarray, pd.Series, pd.DataFrame]],
    ) -> Tuple[Self, None]:
        """Create and train the HMM model based on the given data and labels.

        This is identical to `self_optimize`, but returns additional information about the training process.

        Parameters
        ----------
        data_sequence
            Sequence of gaitmap sensordata objects.
        labels_sequence
            Sequence of gaitmap stride lists.
            The number of stride lists must match the number of sensordata objects (i.e. they must belong together).
            Each label sequence should only contain integers in the range [0, n_states - 1].
            The usage of the labels depends on the train algorithm.
            In case of `viterbi` and `baum-welch`, the labels are only used to identify the intial data clusters.

        Returns
        -------
        self
            The trained model instance.
        history
            The history callback containing the training history.

        """
        if len(data_sequence) != len(labels_sequence):
            raise ValueError(
                "The given training sequence and initial training labels do not match in their number of individual "
                f"sequences! len(data_train_sequence_list) = {len(data_sequence)} !=  {len(labels_sequence)} = len("
                "initial_hidden_states_sequence_list)"
            )

        for i, (data, labels) in enumerate(zip(data_sequence, labels_sequence)):
            if len(data) < self.n_states:
                raise ValueError(
                    "Invalid training sequence! At least one training sequence has less samples than the specified "
                    "value of states! "
                    f"For sequence {i}: n_states = {self.n_states} > {len(data)} = len(data)"
                )
            # We allow None labels for some sequences, as this is also supported by pomegranate.
            if labels is not None:
                if len(data) != len(labels):
                    raise ValueError(
                        "Invalid training sequence! At least one training sequence has a different number of samples "
                        "than the corresponding label sequence! "
                        f"For sequence {i}: len(data) = {len(data)} != {len(labels)} = len(labels)"
                    )
                if not np.all(np.logical_and(labels >= 0, labels < self.n_states)):
                    raise ValueError(
                        "Invalid label sequence! At least one training sequence contains invalid state labels! "
                        f"For sequence {i}: labels not in [0, {self.n_states})"
                    )

        self.data_columns = tuple(data_sequence[0].columns)
        # you have to make always sure that the input data is in a correct format when using pomegranate, if not this
        # can lead to extremely strange behaviour! Unfortunately pomegranate will not tell if data has a bad format!
        # We also ensure that in all provided dataframes the same columns and column order exists
        data_sequence_train = [
            np.ascontiguousarray(dataset[list(self.data_columns)].to_numpy())
            for dataset in data_sequence
        ]
        labels_sequence_train = []
        for labels in labels_sequence:
            labels = labels.to_numpy().squeeze() if isinstance(labels, (pd.Series, pd.DataFrame)) else labels.squeeze()
            labels_sequence_train.append(np.ascontiguousarray(labels.copy()))

        if self.model is not None:
            warnings.warn("Model already exists. Overwriting existing model.")

        # initialize model by naive equidistant labels
        distributions, trans_mat, start_probs, end_probs = initialize_distributions_and_transmat(
            data_sequence_train,
            labels_sequence_train,
            n_states=self.n_states,
            n_gmm_components=self.n_gmm_components,
            architecture=self.architecture,
        )

        model = pgHMM(
            distributions,
            edges=trans_mat,
            starts=start_probs,
            ends=end_probs,
            verbose=self.verbose,
            max_iter=self.max_iterations,
            tol=self.stop_threshold,
        )

        labels_sequence_as_prior = labels_to_prior(labels_sequence_train, n_states=self.n_states)

        model.fit(
            data_sequence_train,
            priors=labels_sequence_as_prior,
        )

        self.model = model

        return self, None
