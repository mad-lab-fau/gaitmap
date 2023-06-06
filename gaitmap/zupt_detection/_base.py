import numpy as np
import pandas as pd

from gaitmap.utils.array_handling import bool_array_to_start_end_array, start_end_array_to_bool_array
from gaitmap.utils.datatype_helper import SingleSensorData


class PerSampleZuptDetectorMixin:
    """A mixin for ZUPT detectors that internally detect ZUPTs per samples.

    It automatically provides the `zupts_` property that converts the `per_sample_zupts_` to a dataframe of start/end
    values.
    """

    per_sample_zupts_: np.ndarray

    @property
    def zupts_(self) -> pd.DataFrame:
        """Get the start and end values of all zupts."""
        start_ends = bool_array_to_start_end_array(self.per_sample_zupts_)
        if len(start_ends) > 0:
            return pd.DataFrame(start_ends, columns=["start", "end"], dtype="Int64")
        return pd.DataFrame(columns=["start", "end"], dtype="Int64")


class RegionZuptDetectorMixin:
    """A mixin for ZUPT detectors that internally detect ZUPTs per regions.

    If a detector detects ZUPTs per region (i.e. start and end values), it can use this mixin to automatically provide
    the `per_sample_zupts_` property.

    Note, that for this to work the `data` attribute must be set as part of the detect method, as we need the length of
    the data to create the `per_sample_zupts_` array.
    """

    data: SingleSensorData
    zupts_: pd.DataFrame

    @property
    def per_sample_zupts_(self) -> np.ndarray:
        """Get a bool array of length data with all Zupts as True."""
        return start_end_array_to_bool_array(self.zupts_.to_numpy(), self.data.shape[0])
