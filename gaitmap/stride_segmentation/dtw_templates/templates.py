from importlib.resources import open_text
from pathlib import Path
from typing import Optional, Union, List
import numpy as np
import pandas as pd

from gaitmap.stride_segmentation import dtw_templates


class DtwTemplate:
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
            with open_text(dtw_templates, self.template_file_name) as test_data:
                self._template = pd.read_csv(test_data, header=0)
        template = self._template

        if getattr(self, "use_cols", None) is None:
            return template

        if isinstance(template, np.ndarray):
            if template.ndim < 2:
                raise ValueError("The stored template is only 1D, but a 2D array is required to use `use_cols`")
            return np.squeeze(template[:, self.use_cols])
        return template[self.use_cols]


def create_dtw_template(
    template: Union[np.ndarray, pd.DataFrame],
    sampling_rate_hz: Optional[float],
    use_cols: Optional[List[Union[str, int]]] = None,
) -> DtwTemplate:
    template_instance = DtwTemplate(template=template, sampling_rate_hz=sampling_rate_hz, use_cols=use_cols)

    return template_instance
