"""Dtw template base classes and helper."""
from importlib.resources import open_text
from typing import Optional, Union, List

import numpy as np
import pandas as pd


class DtwTemplate:
    """Wrap all required information about a dtw template.

    Parameters
    ----------
    template
        The actual data representing the template.
        If this should be a array or a dataframe might depend on your usecase.
    template_file_name
        Alternative to providing a `template` you can provide the filename of any file stored in
        `gaitmap/stride_segmentation/dtw_templates`.
        If you want to use a template stored somewhere else, load it manually and then provide it as template.
    sampling_rate_hz
        The sampling rate that was used to record the template data
    use_cols
        The columns of the template that should actualle be used.
        If the template is an array this must be a list of **int**s, if it is a dataframe, the content of `use_cols`
        must match a subset of these columns.

    See Also
    --------
    gaitmap.stride_segmentation.base_dtw.BaseDtw: How to apply templates
    gaitmap.stride_segmentation.barth_dtw.BarthDtw: How to apply templates for stride segmentation

    """

    sampling_rate_hz: Optional[float]
    template_file_name: Optional[str]
    use_cols: Optional[List[Union[str, int]]]

    _template: Optional[Union[np.ndarray, pd.DataFrame]]

    def __init__(
        self,
        template: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        template_file_name: Optional[str] = None,
        sampling_rate_hz: Optional[float] = None,
        use_cols: Optional[List[Union[str, int]]] = None,
    ):
        self._template = template
        self.template_file_name = template_file_name
        self.sampling_rate_hz = sampling_rate_hz
        self.use_cols = use_cols

    @property
    def template(self) -> Union[np.ndarray, pd.DataFrame]:
        """Return the template of the dataset.

        If not dataset is registered yet, it will be read from the file path if provided.
        """
        if self._template is None and self.template_file_name is None:
            raise AttributeError("Neither a template array nor a template file is provided.")
        if self._template is None:
            with open_text("gaitmap.stride_segmentation.dtw_templates", self.template_file_name) as test_data:
                self._template = pd.read_csv(test_data, header=0)
        template = self._template

        if getattr(self, "use_cols", None) is None:
            return template

        if isinstance(template, np.ndarray):
            if template.ndim < 2:
                raise ValueError("The stored template is only 1D, but a 2D array is required to use `use_cols`")
            return np.squeeze(template[:, self.use_cols])
        return template[self.use_cols]


class BarthOriginalTemplate(DtwTemplate):
    """Template used for stride segmentation by Barth et al."""

    template_file_name = "barth_original_template.csv"
    sampling_rate_hz = 204.8

    def __init__(
        self, use_cols: Optional[List[Union[str, int]]] = None,
    ):
        super().__init__(
            use_cols=use_cols, template_file_name=self.template_file_name, sampling_rate_hz=self.sampling_rate_hz
        )


def create_dtw_template(
    template: Union[np.ndarray, pd.DataFrame],
    sampling_rate_hz: Optional[float],
    use_cols: Optional[List[Union[str, int]]] = None,
) -> DtwTemplate:
    """Create a DtwTemplate from custom input data.

    Parameters
    ----------
    template
        The actual data representing the template.
        If this should be a array or a dataframe might depend on your usecase.
    sampling_rate_hz
        The sampling rate that was used to record the template data
    use_cols
        The columns of the template that should actualle be used.
        If the template is an array this must be a list of **int**s, if it is a dataframe, the content of `use_cols`
        must match a subset of these columns.

    See Also
    --------
    gaitmap.stride_segmentation.base_dtw.BaseDtw: How to apply templates
    gaitmap.stride_segmentation.barth_dtw.BarthDtw: How to apply templates for stride segmentation
    gaitmap.stride_segmentation.dtw_templates.templates.DtwTemplate: Template base class

    """
    template_instance = DtwTemplate(template=template, sampling_rate_hz=sampling_rate_hz, use_cols=use_cols)

    return template_instance
