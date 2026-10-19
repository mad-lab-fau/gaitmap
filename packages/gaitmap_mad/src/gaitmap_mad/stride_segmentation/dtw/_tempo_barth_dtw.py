"""Experimental stride segmentation with persistent positive local tempo."""

from typing import Optional, Union

import numpy as np
from joblib import Memory
from tpcp import CloneFactory, get_results
from typing_extensions import Literal

from gaitmap.base import BaseStrideSegmentation
from gaitmap.utils._types import _Hashable
from gaitmap_mad.stride_segmentation.dtw._barth_dtw import BarthDtw
from gaitmap_mad.stride_segmentation.dtw._base_dtw import _timeseries_reshape, find_matches_find_peaks
from gaitmap_mad.stride_segmentation.dtw._dtw_templates import BarthOriginalTemplate, BaseDtwTemplate
from gaitmap_mad.stride_segmentation.dtw._tempo_dtw import tempo_dtw_paths, tempo_regularized_subsequence_dtw


class ExperimentalTempoBarthDtw(BaseStrideSegmentation):
    """Experimental Barth-style segmentation on a regularized block-tempo grid.

    This is a separate algorithm, with no established gait-accuracy validation.
    It adapts second-order warp regularization to changes of log local tempo.
    Unlike consecutive-run constraints, tempo remains part of the DP state.
    See :ref:`tempo_regularized_dtw` for the exact discretized model, objective,
    duration constraints, complexity, and relationship to Deriso and Boyd.

    Parameters
    ----------
    template
        A DTW template, optionally per sensor. Physical template sampling rate is
        required even when resampling is disabled. Default is BarthOriginalTemplate.
    resample_template
        Resample the template to the signal rate using the existing DTW helper.
        Without resampling, both physical sampling rates are used directly.
    n_segments
        Number of equal template-time blocks. Default 20. Sample bins can cross
        block edges; every bin contributes its exact overlap. More blocks change
        both the tempo grid and regularizer, so reconsider the penalty coefficient.
    tempo_bounds
        Finite positive lower and upper tempo bounds. Tempo is signal seconds
        per template second, so values above one describe slower matches. Integer
        signal advances realize a subset of this interval, capped at 64 states.
    tempo_change_penalty
        Nonnegative coefficient of sum(diff(log(tempo)) ** 2). The first tempo
        is unbiased; every constant feasible tempo has zero change penalty.
        Default 0.1 is an experimental choice, without gait calibration.
    max_cost
        Threshold on sqrt(mean squared feature loss + tempo-change penalty).
        This normalized score differs from existing DTW; thresholds are not
        transferable. The default 1.0 is uncalibrated.
    min_match_length_s
        Optional strict lower duration bound. The numerical solver finds the best
        duration-valid path per endpoint by retaining alternative starts through
        exact bounded recomputation. Defaults to None.
    max_match_length_s
        Optional strict upper duration bound, default None. Snapping and later
        Barth postprocessing can still modify or reject the selected intervals.
    find_matches_method
        Existing "find_peaks" or "min_under_thres" endpoint selection.
    snap_to_min_win_ms
        Existing Barth snapping window, disabled by default. Costs and paths
        always describe the original, unsnapped intervals.
    snap_to_min_axis
        Data column used for snapping, default "gyr_ml".
    conflict_resolution
        Apply existing Barth duplicate-start conflict resolution, default True.
    memory
        Optional joblib.Memory cache for numerical alignment.

    Attributes
    ----------
    acc_cost_mat_
        K-by-N projected squared costs, with block rows and signal-sample columns.
        Only the final row enforces optional complete-match duration constraints.
    cost_function_
        Square root of the complete-template final cost row.
    costs_
        Square roots of the detected regularized objectives.
    paths_
        Selected integer sample-pair arrays, in familiar Barth result format.
    matches_start_end_
        Postprocessed start and exclusive end indices.
    matches_start_end_original_
        Original start and exclusive end indices, before snapping.
    stride_list_
        Detected intervals as a dataframe with s_id index.
    alignments_
        Selected TempoDtwAlignment objects containing path weights, block edges,
        actual tempos and separate data/tempo/total squared costs.
    path_weights_
        Positive sample-pair weights for each selected alignment, summing to one.
    block_edges_
        Exclusive signal block edges for each selected alignment.
    tempo_profiles_
        Actual physical block tempos for each selected alignment.
    tempo_alphabet_
        Actual positive tempo states allowed by integer signal advances.
    duration_fallback_count_
        Number of endpoint-specific duration recomputations. Many fallbacks can
        substantially increase runtime; state memory remains bounded.

    Other Parameters
    ----------------
    data
        Data passed to segment().
    sampling_rate_hz
        Signal sampling rate passed to segment().

    Notes
    -----
    Multi-sensor outputs are dictionaries, as for BarthDtw. At zero penalty this
    solves the positive block-tempo lattice, not ordinary unconstrained DTW.
    Existing DTW solvers, scores and public parameters are unchanged.
    """

    def __init__(
        self,
        template: Optional[Union[BaseDtwTemplate, dict[_Hashable, BaseDtwTemplate]]] = CloneFactory(
            BarthOriginalTemplate()
        ),
        resample_template: bool = True,
        n_segments: int = 20,
        tempo_bounds: tuple[float, float] = (0.5, 2.0),
        tempo_change_penalty: float = 0.1,
        find_matches_method: Literal["min_under_thres", "find_peaks"] = "find_peaks",
        max_cost: Optional[float] = 1.0,
        min_match_length_s: Optional[float] = None,
        max_match_length_s: Optional[float] = None,
        snap_to_min_win_ms: Optional[float] = None,
        snap_to_min_axis: Optional[str] = "gyr_ml",
        conflict_resolution: bool = True,
        memory: Optional[Memory] = None,
    ) -> None:
        self.n_segments = n_segments
        self.tempo_bounds = tempo_bounds
        self.tempo_change_penalty = tempo_change_penalty
        self.template = template
        self.resample_template = resample_template
        self.find_matches_method = find_matches_method
        self.max_cost = max_cost
        self.min_match_length_s = min_match_length_s
        self.max_match_length_s = max_match_length_s
        self.snap_to_min_win_ms = snap_to_min_win_ms
        self.snap_to_min_axis = snap_to_min_axis
        self.conflict_resolution = conflict_resolution
        self.memory = memory

    def segment(self, data, sampling_rate_hz: float, **_):
        """Find stride candidates; rates and sample bins have physical time units.

        Parameters
        ----------
        data
            Array, single-sensor dataframe or multi-sensor data, as for BarthDtw.
        sampling_rate_hz
            Positive sampling rate of the signal in Hz.

        Returns
        -------
        self
            The algorithm with results populated.
        """
        options = self.get_params(deep=False)
        tempo_options = {name: options.pop(name) for name in ("n_segments", "tempo_bounds", "tempo_change_penalty")}
        # The ephemeral adapter retains Barth's complete constructor contract.
        # Its numerical settings are passed explicitly to the cached pure solver;
        # the adapter itself is never cached, cloned or returned to callers.
        engine = _TempoBarthEngine(**options)
        engine._tempo_options = tempo_options
        engine.segment(data, sampling_rate_hz=sampling_rate_hz)
        self.data, self.sampling_rate_hz = data, sampling_rate_hz
        for name, value in get_results(engine).items():
            setattr(self, name, value)
        return self


class _TempoBarthEngine(BarthDtw):
    """Ephemeral adapter reusing Barth dispatch and stride postprocessing."""

    def _segment_single_dataset(self, dataset, template, memory):
        rate = getattr(template, "sampling_rate_hz", None)
        if rate is None:
            raise ValueError("Experimental tempo alignment requires template.sampling_rate_hz.")
        template_array, signal = self._extract_relevant_data_and_template(template, dataset, self.sampling_rate_hz)
        if self.resample_template and rate != self.sampling_rate_hz:
            template_array = self._resample_template(template_array, rate, self.sampling_rate_hz)
            rate = self.sampling_rate_hz
        x, y = _timeseries_reshape(template_array), _timeseries_reshape(signal)
        result = memory.cache(tempo_regularized_subsequence_dtw)(
            x,
            y,
            template_sampling_rate_hz=rate,
            signal_sampling_rate_hz=self.sampling_rate_hz,
            n_segments=self._tempo_options["n_segments"],
            tempo_bounds=self._tempo_options["tempo_bounds"],
            tempo_change_penalty=self._tempo_options["tempo_change_penalty"],
            min_duration_samples=None
            if self._min_sequence_length is None
            else int(np.floor(self._min_sequence_length)) + 1,
            max_duration_samples=None
            if self._max_sequence_length is None
            else int(np.ceil(self._max_sequence_length)) - 1,
        )
        method = self._allowed_methods_map[self.find_matches_method]
        matches = self._find_matches(
            result.costs,
            self.max_cost,
            self._min_sequence_length,
            method,
            include_boundaries=True,
        )
        matches = np.asarray(matches, dtype=int)
        matches = np.sort(matches[np.isfinite(result.costs[-1, matches])])
        alignments = tempo_dtw_paths(x, y, result, matches) if len(matches) else []
        paths = [a.path for a in alignments]
        intervals = np.asarray([[a.block_edges[0], a.block_edges[-1]] for a in alignments], dtype=int).reshape(-1, 2)
        costs = np.sqrt(result.costs[-1, matches])
        keep = np.ones(len(matches), dtype=bool)
        if len(matches):
            intervals, keep = self._postprocess_matches(
                data=dataset,
                paths=paths,
                cost=costs,
                matches_start_end=intervals,
                acc_cost_mat=result.costs,
                to_keep=keep,
                memory=memory,
            )
            if method is find_matches_find_peaks:
                keep = self._filter_constrained_peaks(result.costs, paths, keep)
            self._post_postprocess_check(intervals[keep])
        alignments = [a for a, retained in zip(alignments, keep) if retained]
        return {
            "acc_cost_mat": result.costs,
            "paths": [a.path for a in alignments],
            "costs": costs[keep],
            "matches_start_end": intervals[keep],
            "alignments": alignments,
            "path_weights": [a.weights for a in alignments],
            "block_edges": [a.block_edges for a in alignments],
            "tempo_profiles": [a.tempos for a in alignments],
            "tempo_alphabet": result.tempos,
            "duration_fallback_count": result.fallback_count,
        }
