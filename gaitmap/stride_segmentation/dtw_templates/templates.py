"""Dtw template base classes and helper."""
from importlib.resources import open_text
from typing import Optional, Union, Tuple, List, cast

import numpy as np
import pandas as pd

from gaitmap.base import _BaseSerializable
from gaitmap.utils.array_handling import multi_array_interpolation
from gaitmap.utils.datatype_helper import is_single_sensor_data


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
    scaling: Optional[float]
    data: Optional[Union[np.ndarray, pd.DataFrame]]

    def __init__(
        self,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        template_file_name: Optional[str] = None,
        sampling_rate_hz: Optional[float] = None,
        scaling: Optional[float] = None,
        use_cols: Optional[Tuple[Union[str, int], ...]] = None,
    ):
        self.data = data
        self.template_file_name = template_file_name
        self.sampling_rate_hz = sampling_rate_hz
        self.scaling = scaling
        self.use_cols = use_cols

    def get_data(self) -> Union[np.ndarray, pd.DataFrame]:
        """Return the template of the dataset.

        If no dataset is registered yet, it will be read from the file path if provided.
        """
        if self.data is None and self.template_file_name is None:
            raise ValueError("Neither a template array nor a template file is provided.")
        if self.data is None:
            with open_text(
                "gaitmap.stride_segmentation.dtw_templates", cast(str, self.template_file_name)
            ) as test_data:
                self.data = pd.read_csv(test_data, header=0)
        template = self.data

        if self.use_cols is None:
            return template
        use_cols = list(self.use_cols)
        if isinstance(template, np.ndarray):
            if template.ndim < 2:
                raise ValueError("The stored template is only 1D, but a 2D array is required to use `use_cols`")
            return np.squeeze(template[:, use_cols])
        return template[use_cols]


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

    template_file_name = "barth_original_template.csv"
    sampling_rate_hz = 204.8

    def __init__(self, scaling=500.0, use_cols: Optional[Tuple[Union[str, int], ...]] = None):
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
    if not isinstance(signal_sequence, list):
        signal_sequence = [signal_sequence]

    for df in signal_sequence:
        is_single_sensor_data(df, check_acc=False, check_gyr=False, frame="any", raise_exception=True)

    if n_samples is None:
        # get mean stride length over given strides
        n_samples = int(np.rint(np.mean([len(df) for df in signal_sequence])))

    # We need to ensure that the columns of all dfs have the right order before
    # exporting to numpy.
    expected_col_order = signal_sequence[0].columns
    arrays = [df.reindex(columns=expected_col_order).to_numpy() for df in signal_sequence]
    resampled_sequences_df_list = multi_array_interpolation(arrays, n_samples, kind=kind)

    template = np.mean(resampled_sequences_df_list, axis=0)
    template_df = pd.DataFrame(template.T, columns=expected_col_order)

    return create_dtw_template(template_df, sampling_rate_hz, scaling, use_cols)
