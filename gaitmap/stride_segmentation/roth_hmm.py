"""The msDTW based stride segmentation algorithm by Barth et al 2013."""
from typing import Optional, Union, Dict, List, Tuple
import numpy as np
import pandas as pd
from gaitmap.base import BaseAlgorithm, BaseType
from gaitmap.utils.dataset_helper import Dataset
from gaitmap.stride_segmentation.hmm_models import HiddenMarkovModel


class RothHMM(BaseAlgorithm):
    """Segment strides using a pre-trained Hidden Markov Model.

    TBD: short description of HMM

    Parameters
    ----------
    model
        The HMM class need a valid pre-trained model to segment strides
    snap_to_min
        Boolean flag to indicate if snap to minimum action shall be performend


    Attributes
    ----------
    stride_list_ : A stride list or dictionary with such values
        The same output as `matches_start_end_`, but as properly formatted pandas DataFrame that can be used as input to
        other algorithms.
        If `snap_to_min` is `True`, the start and end value might not match to the output of `hidden_state_sequence_`.
        Refer to `matches_start_end_original_` for the unmodified start and end values.
    matches_start_end_ : 2D array of shape (n_detected_strides x 2) or dictionary with such values
        The start (column 1) and end (column 2) of each detected stride.
    hidden_state_sequence_ : List of length n_detected_strides or dictionary with such values
        The cost value associated with each stride.
    matches_start_end_original_ : 2D array of shape (n_detected_strides x 2) or dictionary with such values
        Identical to `matches_start_end_` if `snap_to_min` is equal to `False`.
        Otherwise, it return the start and end values before the sanpping is applied.

    Other Parameters
    ----------------
    data
        The data passed to the `segment` method.
    sampling_rate_hz
        The sampling rate of the data

    Notes
    -----
    Post Processing
        This algorithm uses an optional post-processing step that "snaps" the stride borders to the closest local
        minimum in the raw data.
        However, this assumes that the start and the end of each match is marked by a clear minimum in one axis of the
        raw data.

    .. [1] ref to JBHI HMM paper


    See Also
    --------
    TBD
    """

    snap_to_min: Optional[bool]
    snap_to_min_axis: Optional[str]
    model: Optional[HiddenMarkovModel]

    data: Union[np.ndarray, Dataset]
    sampling_rate_hz: float

    def __init__(
        self,
        model: Optional[HiddenMarkovModel] = None,
        snap_to_min: Optional[bool] = True,
        snap_to_min_axis: Optional[str] = "gyr_ml",
    ):
        self.snap_to_min = snap_to_min
        self.snap_to_min_axis = snap_to_min_axis
        self.model = model

    @property
    def stride_list_(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Return start and end of each match as pd.DataFrame."""
        start_ends = self.matches_start_end_
        if isinstance(start_ends, dict):
            return {k: self._format_stride_list(v) for k, v in start_ends.items()}
        return self._format_stride_list(start_ends)

    @staticmethod
    def _format_stride_list(array: np.ndarray) -> pd.DataFrame:
        if len(array) == 0:
            array = None
        as_df = pd.DataFrame(array, columns=["start", "end"])
        # Add the s_id
        as_df.index.name = "s_id"
        return as_df

    def _postprocess_matches(
        self, data, paths: List, cost: np.ndarray, matches_start_end: np.ndarray, to_keep: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: implement

        # Apply snap to minimum
        if self.snap_to_min:
            return 0

        return 1

    def segment(self: BaseType, data: Union[np.ndarray, Dataset], sampling_rate_hz: float, **_) -> BaseType:
        """Find matches by warping the provided template to the data.

        Parameters
        ----------
        data : array, single-sensor dataframe, or multi-sensor dataset
            The input data.
            For details on the required datatypes review the class docstring.
        sampling_rate_hz
            The sampling rate of the data signal. This will be used to convert all parameters provided in seconds into
            a number of samples and it will be used to perform the required feature transformation`.

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # TODO do magic
        return self

    def _segment_single_dataset(self, dataset, template):
        # TODO do magic
        return 0

    def _predict(self, model, data, algorithm="viterbi"):
        """Perform prediction based on given data and given model."""
        # need to check if memory layout of given data is
        # see related pomegranate issue: https://github.com/jmschrei/pomegranate/issues/717
        if not np.array(data).flags["C_CONTIGUOUS"]:
            raise ValueError("Memory Layout of given input data is not contiguois! Consider using ")

        labels_predicted = np.asarray(model.predict(data, algorithm=algorithm))
        # pomegranate always adds an additional label for the start- and end-state, which can be ignored here!
        return np.asarray(labels_predicted[1:-1])

    def _hidden_states_to_stride_borders(self, data_to_snap_to, hidden_states_predicted, stride_states):
        """This function converts the output of a hmm prediction aka the resulting hidden-state sequence to meaningful stride borders.
            Therefore, potential stride-borders are derived form the hidden states and the actual border is snapped
            to the data minimum within these potential border windows.
            The potential border windows are derived from the stride-start / end-states plus the adjacent states, which might
            be e.g. transition-stride_start, stride_end-transition, stride_end-stride_start, stride_start-stride_end.

        - data_to_snap_to:
            1D array where the "snap to minimum" operation will be performed to find the actual stride border: This is usually
            the "gyr_ml" data!!

        - labels_predicted:
            Predicted hidden-state labels (this should be an array of some discrete values!)

        - stride_states:
            This is the actual list of states we are looking for so e.g. [5,6,7,8,9,10]

        Returns a list of strides with [start,stop] indices

        Example:
        stride_borders = hidden_states_to_stride_borders2(gyr_ml_data, predicted_labels, np.aragne(stride_start_state, stride_end_state))
        stride_borders...
        ... [[100,211],
             [211,346],
             [346,478]
             ...
             ]
        """

        if data_to_snap_to.ndim > 1:
            raise ValueError("Snap to minimum only allows 1D arrays as inputs")

        # get all existing label transitions
        transitions, _, _ = self._extract_transitions_starts_stops_from_hidden_state_sequence(hidden_states_predicted)

        if len(transitions) == 0:
            return []

        # START-Window
        adjacent_labels_starts = [a[0] for a in transitions if a[-1] == stride_states[0]]

        potential_start_window = []
        for l in adjacent_labels_starts:
            potential_start_window.extend(
                self.binary_array_to_start_stop_list(
                    np.logical_or(hidden_states_predicted == stride_states[0], hidden_states_predicted == l).astype(int)
                )
            )
        # remove windows where there is actually no stride-start-state label present
        potential_start_window = [
            a for a in potential_start_window if stride_states[0] in np.unique(hidden_states_predicted[a[0] : a[1] + 1])
        ]

        # melt all windows together
        bin_array_starts = np.zeros(len(hidden_states_predicted))
        for w in potential_start_window:
            bin_array_starts[w[0] : w[1] + 1] = 1

        start_windows = self.binary_array_to_start_stop_list(bin_array_starts)
        start_borders = [np.argmin(data_to_snap_to[w[0] : w[1] + 1]) + w[0] for w in start_windows]

        # END-Window
        adjacent_labels_ends = [a[-1] for a in transitions if a[0] == stride_states[-1]]

        potential_end_window = []
        for l in adjacent_labels_ends:
            potential_end_window.extend(
                self.binary_array_to_start_stop_list(
                    np.logical_or(hidden_states_predicted == stride_states[-1], hidden_states_predicted == l).astype(
                        int
                    )
                )
            )

        potential_end_window = [
            a for a in potential_end_window if stride_states[-1] in np.unique(hidden_states_predicted[a[0] : a[1] + 1])
        ]

        # melt all windows together
        bin_array_ends = np.zeros(len(hidden_states_predicted))
        for w in potential_end_window:
            bin_array_ends[w[0] : w[1] + 1] = 1

        end_windows = self._binary_array_to_start_stop_list(bin_array_ends)
        end_borders = [np.argmin(data_to_snap_to[w[0] : w[1] + 1]) + w[0] for w in end_windows]

        return np.array(list(zip(start_borders, end_borders)))

    def _extract_transitions_starts_stops_from_hidden_state_sequence(self, hidden_state_sequence):
        """This function will return a list of transitions as well as start and stop labels that can be found within the
        input sequences.

        input = [[1,1,1,1,1,3,3,3,3,2,2,2,2,4,4,4,4,5,5],
                 [0,0,1,1,1,3,3,3,3,2,2,2,6]]
        output_transitions = [[1,3],
                              [3,2],
                              [2,4],
                              [4,5],
                              [0,1],
                              [2,6]]

        output_starts = [1,0]
        output_stops = [5,6]
        """
        if not isinstance(hidden_state_sequence, list):
            hidden_state_sequence = [hidden_state_sequence]

        transitions = []
        starts = []
        ends = []
        for labels in hidden_state_sequence:
            starts.append(labels[0])
            ends.append(labels[-1])
            for idx in np.where(abs(np.diff(labels)) > 0)[0]:
                transitions.append([labels[idx], labels[idx + 1]])

        if len(transitions) > 0:
            transitions = np.unique(transitions, axis=0).astype(int)
        starts = np.unique(starts).astype(int)
        ends = np.unique(ends).astype(int)

        return [transitions, starts, ends]

    def _binary_array_to_start_stop_list(self, bin_array):
        starts = np.where(np.diff(bin_array) > 0)[0] + 1
        stops = np.where(np.diff(bin_array) < 0)[0]
        if bin_array[0] == 1:
            starts = np.insert(starts, 0, 0)
        if bin_array[-1] == 1:
            stops = np.append(stops, len(bin_array) - 1)

        return np.column_stack((starts, stops))
