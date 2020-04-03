from importlib.resources import open_text
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd

from gaitmap.stride_segmentation import dtw_templates


class DtwTemplate:
    sampling_rate_hz: Optional[float]
    template_file_name: Optional[str]

    _template: Optional[Union[np.ndarray, pd.DataFrame]]

    @property
    def template(self) -> Union[np.ndarray, pd.DataFrame]:
        """Return the template of the dataset.

        If not dataset is registered yet, it will be read from the file path if provided.
        """
        try:
            return self._template
        except AttributeError:
            if not self.template_file_name:
                raise AttributeError("Neither a template array nor a template file is provided.")
            with open_text(dtw_templates, self.template_file_name) as test_data:
                self._template = pd.read_csv(test_data, header=0)
            return self._template


def create_dtw_template(template: Union[np.ndarray, pd.DataFrame], sampling_rate_hz: Optional[float]) -> DtwTemplate:
    template_instance = DtwTemplate()
    template_instance._template = template
    template_instance.sampling_rate_hz = sampling_rate_hz

    return template_instance
