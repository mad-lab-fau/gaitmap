from copy import copy

import pandas as pd
from scipy import signal
from typing_extensions import Self

from gaitmap.data_transform._base import BaseTransformer
from gaitmap.utils.datatype_helper import SingleSensorData


class Decimate(BaseTransformer):
    downsampling_factor: int
    def __init__(self, downsampling_factor: int):
        self.downsampling_factor = downsampling_factor

    def transform(self, data: SingleSensorData, **kwargs) -> Self:
        if self.downsampling_factor < 1:
            raise ValueError("Downsampling factor must be >=1!")

        if self.downsampling_factor == 1:
            self.transformed_data_ = copy(data)
            return self

        data_decimated = signal.decimate(
            data,
            self.downsampling_factor,
            n=None,
            ftype="iir",
            axis=0,
            zero_phase=True,
        )
        if isinstance(data, pd.DataFrame):
            data_decimated = pd.DataFrame(data_decimated, columns=data.columns)
        elif isinstance(data, pd.Series):
            data_decimated = pd.Series(data_decimated, name=data.name)

        self.transformed_data_ = data_decimated

        return self
