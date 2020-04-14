"""Dtw template base classes and helper."""
from importlib.resources import open_text
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd


class DtwTemplate:
    """Wrap all required information about a dtw template.

    Parameters
    ----------
    data
        The actual data representing the template.
        If this should be a array or a dataframe might depend on your usecase.
    template_file_name
        Alternative to providing a `template` you can provide the filename of any file stored in
        `gaitmap/stride_segmentation/dtw_templates`.
        If you want to use a template stored somewhere else, load it manually and then provide it as template.
    sampling_rate_hz
        The sampling rate that was used to record the template data
    scaling
        A multiplicative factor multiplied onto the template to adapt for another signal range.
        Usually the scaled template should have the same value range as the data signal.
        A large scale difference between data and template will result in mismatches.
    use_cols
        The columns of the template that should actually be used.
        If the template is an array this must be a list of **int**, if it is a dataframe, the content of `use_cols`
        must match a subset of these columns.

    See Also
    --------
    gaitmap.stride_segmentation.base_dtw.BaseDtw: How to apply templates
    gaitmap.stride_segmentation.barth_dtw.BarthDtw: How to apply templates for stride segmentation

    """

    sampling_rate_hz: Optional[float]
    template_file_name: Optional[str]
    use_cols: Optional[Tuple[Union[str, int]]]
    scaling: Optional[float]

    _data: Optional[Union[np.ndarray, pd.DataFrame]]

    def __init__(
        self,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        template_file_name: Optional[str] = None,
        sampling_rate_hz: Optional[float] = None,
        scaling: Optional[float] = None,
        use_cols: Optional[Tuple[Union[str, int]]] = None,
    ):
        self._data = data
        self.template_file_name = template_file_name
        self.sampling_rate_hz = sampling_rate_hz
        self.scaling = scaling
        self.use_cols = use_cols

    @property
    def data(self) -> Union[np.ndarray, pd.DataFrame]:
        """Return the template of the dataset.

        If no dataset is registered yet, it will be read from the file path if provided.
        """
        if self._data is None and self.template_file_name is None:
            raise AttributeError("Neither a template array nor a template file is provided.")
        if self._data is None:
            with open_text("gaitmap.stride_segmentation.dtw_templates", self.template_file_name) as test_data:
                self._data = pd.read_csv(test_data, header=0)
        scaling = getattr(self, "scaling", None) or 1
        template = self._data * scaling

        if getattr(self, "use_cols", None) is None:
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
        A multiplicative factor multiplied onto the template to adapt for another signal range.
        For this template the default value is 500, which loosly scales the template to match a signal that has a
        a max-gyro peak of approx. 500 deg/s in `gyr_ml` during the swing phase.
    use_cols
        The columns of the template that should actually be used.
        The default (all gyro axis) should work well, but will not match turning stride.
        Note, that this template only consists of gyro data (i.e. you can only select one of
        :obj:`~gaitmap.utils.consts.BF_GYR`)

    See Also
    --------
    gaitmap.stride_segmentation.dtw_templates.templates.DtwTemplate: Base class for templates
    gaitmap.stride_segmentation.barth_dtw.BarthDtw: How to apply templates for stride segmentation

    """

    template_file_name = "barth_original_template.csv"
    sampling_rate_hz = 204.8

    def __init__(self, scaling=500.0, use_cols: Optional[Tuple[Union[str, int]]] = None):
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
    sampling_rate_hz
        The sampling rate that was used to record the template data
    scaling
        A multiplicative factor multiplied onto the template to adapt for another signal range
    use_cols
        The columns of the template that should actualle be used.
        If the template is an array this must be a list of **int**, if it is a dataframe, the content of `use_cols`
        must match a subset of these columns.

    See Also
    --------
    gaitmap.stride_segmentation.base_dtw.BaseDtw: How to apply templates
    gaitmap.stride_segmentation.barth_dtw.BarthDtw: How to apply templates for stride segmentation
    gaitmap.stride_segmentation.dtw_templates.templates.DtwTemplate: Template base class

    """
    template_instance = DtwTemplate(
        data=template, sampling_rate_hz=sampling_rate_hz, scaling=scaling, use_cols=use_cols
    )

    return template_instance