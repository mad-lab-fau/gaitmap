"""Dtw template base classes and helper."""
import warnings
from importlib.resources import open_text
from typing import List, Optional, Tuple, Union, cast, Iterable

import numpy as np
import pandas as pd

from gaitmap.base import _BaseSerializable
from gaitmap.data_transform import BaseTransformer, FixedScaler, TrainableTransformerMixin
from gaitmap.utils._types import _Hashable
from gaitmap.utils.array_handling import multi_array_interpolation
from gaitmap.utils.datatype_helper import is_single_sensor_data, SingleSensorData, SingleSensorStrideList


class DtwTemplate(_BaseSerializable):
    """Wrap all required information about a dtw template.

    Parameters
    ----------
    data
        The actual data representing the template.
        If this should be a array or a dataframe might depend on your usecase.
        Usually it is a good idea to scale the data to a range from 0-1 and then use the scale parameter to downscale
        the signal.
        Do not use this attribute directly to access the data, but use the `get_data` method instead.
    template_file_name
        Alternative to providing a `template` you can provide the filename of any file stored in
        `gaitmap/stride_segmentation/dtw_templates`.
        If you want to use a template stored somewhere else, load it manually and then provide it as template.
    sampling_rate_hz
        The sampling rate that was used to record the template data
    scaling
        A multiplicative factor used to downscale the signal before the template is applied.
        The downscaled signal should then have have the same value range as the template signal.
        A large scale difference between data and template will result in mismatches.
        At the moment only homogeneous scaling of all axis is supported.
        Note that the actual use of the scaling depends on the DTW implementation and not all DTW class might use the
        scaling factor in the same way.
    use_cols
        The columns of the template that should actually be used.
        If the template is an array this must be a list of **int**, if it is a dataframe, the content of `use_cols`
        must match a subset of these columns.

    See Also
    --------
    gaitmap.stride_segmentation.BaseDtw: How to apply templates
    gaitmap.stride_segmentation.BarthDtw: How to apply templates for stride segmentation

    """

    sampling_rate_hz: Optional[float]
    template_file_name: Optional[str]
    use_cols: Optional[Tuple[Union[str, int], ...]]
    scaling: Optional[BaseTransformer]
    data: Optional[Union[np.ndarray, pd.DataFrame]]

    def __init__(
        self,
        *,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        template_file_name: Optional[str] = None,
        sampling_rate_hz: Optional[float] = None,
        scaling: Optional[BaseTransformer] = None,
        use_cols: Optional[Tuple[Union[str, int], ...]] = None,
    ):

        self.data = data
        self.template_file_name = template_file_name
        self.sampling_rate_hz = sampling_rate_hz
        if isinstance(scaling, float):
            warnings.warn(
                "Using a numeric value for scaling is deprecated."
                "Use `scaling=FixedScaler({})` instead.".format(scaling)
            )
        self.scaling = scaling
        self.use_cols = use_cols
        super().__init__()

    def create_template(
        self,
        data_sequences: List[SingleSensorData],
        label_sequences: List[SingleSensorStrideList],
        sampling_rate_hz: float,
        **kwargs,
    ):
        raise NotImplementedError(
            "Creating a new template array from data is not supported for the template base "
            "class or any classes with pre-calculated template arrays (e.g. "
            "`BarthOriginalTemplate`). "
            "If you want to create a custom template use on of the template subclasses that "
            "implements this method."
        )

    def train_scaler(self):
        """Adapt the scaler based on the current template data.

        .. note:: This will only have an effect, if the scaler is actually trainable.

        .. note:: Usually the scaler will be retrained when you create a template using the `create_template` method.
                  This means, you only need to use this method after manually modifying `data` attribute or changed
                  the scaler after creating the template.

        Returns
        -------
        self
            The instance of the template with the `self.scaling` attribute updated.

        """
        if isinstance(self.scaling, TrainableTransformer):
            # We train the transformer, but we don't apply the transformation to the template data.
            self.scaling = self.scaling.train(self.get_data(), self.sampling_rate_hz)
        return self

    def get_data(self) -> Union[np.ndarray, pd.DataFrame]:
        """Return the template of the dataset.

        If no dataset is registered yet, it will be read from the file path if provided.
        """
        if self.data is None and self.template_file_name is None:
            raise ValueError("Neither a template array nor a template file is provided.")
        data = self.data
        if data is None:
            with open_text(
                "gaitmap.stride_segmentation.dtw_templates", cast(str, self.template_file_name)
            ) as test_data:
                data = pd.read_csv(test_data, header=0)
        template = data

        if self.use_cols is None:
            return self._apply_scaling(template, self.sampling_rate_hz)
        use_cols = list(self.use_cols)
        if isinstance(template, np.ndarray):
            if template.ndim < 2:
                raise ValueError("The stored template is only 1D, but a 2D array is required to use `use_cols`")
            return np.squeeze(template[:, use_cols])
        return self._apply_scaling(template[use_cols], self.sampling_rate_hz)

    def _apply_scaling(self, data, sampling_rate_hz: float) -> SingleSensorData:
        if not self.scaling:
            return data
        if isinstance(data, np.ndarray):
            raise ValueError(
                "Data Transformations are only supported for dataframe templates at the moment."
                "Explicitly set `self.data_transform` to None."
            )
        if isinstance(self.scaling, float):
            # TODO: Remove once Deprecation is done
            return FixedScaler(self.scaling, 0).transform(data, sampling_rate_hz)
        return self.scaling.transform(data, sampling_rate_hz)

    def transform_data(self, data: SingleSensorData, sampling_rate_hz: float) -> SingleSensorData:
        return self._apply_scaling(data, sampling_rate_hz)


class InterpolatedDtwTemplate(DtwTemplate):
    def __init__(
        self,
        *,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        sampling_rate_hz: Optional[float] = None,
        scaling: Optional[BaseTransformer] = None,
        interpolation_method: str = "linear",
        n_samples: Optional[int] = None,
        use_cols: Optional[Tuple[Union[str, int], ...]] = None,
    ):
        self.interpolation_method = interpolation_method
        self.n_samples = n_samples
        super().__init__(
            data=data,
            template_file_name=None,
            sampling_rate_hz=sampling_rate_hz,
            scaling=scaling,
            use_cols=use_cols,
        )

    def create_template(
        self,
        data_sequences: List[SingleSensorData],
        label_sequences: List[SingleSensorStrideList],
        sampling_rate_hz: float,
        *,
        columns: Optional[List[_Hashable]] = None,
        **kwargs,
    ):
        template_df, effective_sampling_rate = _create_interpolated_dtw_template(
            data_sequences,
            label_sequences,
            sampling_rate_hz,
            kind=self.interpolation_method,
            n_samples=self.n_samples,
            columns=columns,
        )
        self.sampling_rate_hz = effective_sampling_rate
        if isinstance(self.scaling, TrainableTransformerMixin):
            # We train the transformer, but we don't apply the transformation to the template data.
            self.scaling = self.scaling.train(template_df, self.sampling_rate_hz)
        self.data = template_df
        self.train_scaler()
        return self


class BarthOriginalTemplate(DtwTemplate):
    """Template used for stride segmentation by Barth et al.

    Parameters
    ----------
    scaling
        A multiplicative factor used to downscale the signal before the template is applied.
        The downscaled signal should then have have the same value range as the template signal.
        A large scale difference between data and template will result in mismatches.
        At the moment only homogeneous scaling of all axis is supported.
        Note that the actual use of the scaling depends on the DTW implementation and not all DTW class might use the
        scaling factor in the same way.
        For this template the default value is 500, which is adapted for data that has a max-gyro peak of approx.
        500 deg/s in `gyr_ml` during the swing phase.
        This is appropriate for most walking styles.
    use_cols
        The columns of the template that should actually be used.
        The default (all gyro axis) should work well, but will not match turning stride.
        Note, that this template only consists of gyro data (i.e. you can only select one of
        :obj:`~gaitmap.utils.consts.BF_GYR`)

    Notes
    -----
    As this template was generated by interpolating multiple strides, it does not really have a single sampling rate.
    The original were all recorded at 102.4 Hz, but as the template is interpolated to 200 samples, its is closer to an
    effective sampling rate of 200 Hz (a normal stride is around 0.8-1.5s).
    This template reports a sampling rate of 204.8 Hz, to prevent resampling for this very common sampling rate.

    See Also
    --------
    gaitmap.stride_segmentation.DtwTemplate: Base class for templates
    gaitmap.stride_segmentation.BarthDtw: How to apply templates for stride segmentation

    """

    # TODO: Change Barth Orignal template and multiply it by the scaler.
    template_file_name = "barth_original_template.csv"
    sampling_rate_hz = 204.8

    def __init__(self, *, scaling=FixedScaler(scale=500.0), use_cols: Optional[Tuple[Union[str, int], ...]] = None):
        super().__init__(
            use_cols=use_cols,
            template_file_name=self.template_file_name,
            sampling_rate_hz=self.sampling_rate_hz,
            scaling=scaling,
        )


def create_dtw_template(
    template: Union[np.ndarray, pd.DataFrame],
    sampling_rate_hz: Optional[float] = None,
    scaling: Optional[float] = None,
    use_cols: Optional[Tuple[Union[str, int]]] = None,
) -> DtwTemplate:
    """Create a DtwTemplate from custom input data.

    Parameters
    ----------
    template
        The actual data representing the template.
        If this should be a array or a dataframe might depend on your usecase.
        Note, that complex dataframe structures might not be preserved exactly when the template object is converted
        to json.
    sampling_rate_hz
        The sampling rate that was used to record the template data
    scaling
        A multiplicative factor used to downscale the signal before the template is applied.
        The downscaled signal should then have have the same value range as the template signal.
        A large scale difference between data and template will result in mismatches.
        At the moment only homogeneous scaling of all axis is supported.
        Note that the actual use of the scaling depends on the DTW implementation and not all DTW class might use the
        scaling factor in the same way.
    use_cols
        The columns of the template that should actually be used.
        If the template is an array this must be a list of **int**, if it is a dataframe, the content of `use_cols`
        must match a subset of these columns.

    See Also
    --------
    gaitmap.stride_segmentation.BaseDtw: How to apply templates
    gaitmap.stride_segmentation.BarthDtw: How to apply templates for stride segmentation
    gaitmap.stride_segmentation.DtwTemplate: Template base class

    """
    warnings.warn(
        "`create_dtw_template` is deprecated. Use the `DtwTemplate` constructor directly.", DeprecationWarning
    )
    template_instance = DtwTemplate(
        data=template, sampling_rate_hz=sampling_rate_hz, scaling=scaling, use_cols=use_cols
    )

    return template_instance


def create_interpolated_dtw_template(
    signal_sequence: Union[pd.DataFrame, List[pd.DataFrame]],
    kind: str = "linear",
    n_samples: Optional[int] = None,
    sampling_rate_hz: Optional[float] = None,
    scaling: Optional[float] = None,
    use_cols: Optional[Tuple[Union[str, int]]] = None,
) -> DtwTemplate:
    """Create a DtwTemplate by interpolation. If multiple sequences are given use mean to combine them.

    This function can be used to generate a DtwTemplate from multiple input signal sequences, all sequences will be
    interpolated to the same length, and combined by calculating their mean. Interpolation and mean calculation will
    be performed over all given input axis.

    Parameters
    ----------
    signal_sequence
        Either a single dataframe or a list of dataframes which shall be used for template generation. Each dataframe
        should therefore full fill the gaitmap body frame convention
    kind
        Interpolation function. Please refer to :py:class:`scipy.interpolate.interp1d`.
    n_samples
        Number of samples to which the data will be interpolated. If None, the number of samples will be the mean
        length of all given input sequences.
    sampling_rate_hz
        The sampling rate that was used to record the template data
    scaling
        A multiplicative factor used to downscale the signal before the template is applied.
        The downscaled signal should then have have the same value range as the template signal.
        A large scale difference between data and template will result in mismatches.
        At the moment only homogeneous scaling of all axis is supported.
        Note that the actual use of the scaling depends on the DTW implementation and not all DTW class might use the
        scaling factor in the same way.
    use_cols
        The columns of the template that should actually be used.
        If the template is an array this must be a list of **int**, if it is a dataframe, the content of `use_cols`
        must match a subset of these columns.

    See Also
    --------
    gaitmap.stride_segmentation.create_dtw_template: Helper function to create instance of template class
    gaitmap.stride_segmentation.BaseDtw: How to apply templates
    gaitmap.stride_segmentation.BarthDtw: How to apply templates for stride segmentation
    gaitmap.stride_segmentation.DtwTemplate: Template base class

    """
    warnings.warn(
        "`create_interpolated_dtw_template` is deprecated. Use `InterpolatedDtwTemplate.create_template` instead.",
        DeprecationWarning,
    )
    if not isinstance(signal_sequence, list):
        signal_sequence = [signal_sequence]
    # TODO: Deprecate method
    fake_stride_list = [pd.DataFrame([[0, len(df)]], columns=["start", "end"]) for df in signal_sequence]
    template_df, _ = _create_interpolated_dtw_template(
        signal_sequence, fake_stride_list, sampling_rate_hz, kind, n_samples
    )
    return create_dtw_template(template_df, sampling_rate_hz, scaling, use_cols)


def _create_interpolated_dtw_template(
    signal_sequence: List[SingleSensorData],
    label_sequences: List[SingleSensorStrideList],
    sampling_rate_hz: float,
    kind: str = "linear",
    n_samples: Optional[int] = None,
    columns: Optional[List[_Hashable]] = None,
) -> Tuple[pd.DataFrame, float]:
    for df in signal_sequence:
        is_single_sensor_data(df, check_acc=False, check_gyr=False, frame="any", raise_exception=True)

    # We need to ensure that the columns of all dfs have the right order before
    # exporting to numpy.
    expected_col_order = columns or signal_sequence[0].columns
    arrays = list(_cut_strides_from_labels(signal_sequence, label_sequences, expected_col_order))
    # get mean stride length over given strides
    mean_stride_samples = int(np.rint(np.mean([len(df) for df in arrays])))
    n_samples = n_samples or mean_stride_samples
    resampled_sequences_df_list = multi_array_interpolation(arrays, n_samples, kind=kind)

    template = np.mean(resampled_sequences_df_list, axis=0)
    template_df = pd.DataFrame(template.T, columns=expected_col_order)

    # When we interpolate all templates to a fixed number of samples, the effective sampling rate changes.
    # We approximate the sampling rate using the average stride length in the provided data.
    effective_sampling_rate = sampling_rate_hz
    if n_samples and sampling_rate_hz:
        effective_sampling_rate = n_samples / (mean_stride_samples / sampling_rate_hz)

    return template_df, effective_sampling_rate


def _cut_strides_from_labels(
    signal_sequence: Iterable[SingleSensorData], label_sequences: Iterable[SingleSensorStrideList], expected_col_order
):
    for df, labels in zip(signal_sequence, label_sequences):
        df = df.reindex(columns=expected_col_order).to_numpy()
        for (_, s, e) in labels[["start", "end"]].itertuples():
            yield df[s:e]
