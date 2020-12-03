from copy import deepcopy
from typing import Union, Dict

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from gaitmap.base import BaseStrideSegmentation, BaseType
from gaitmap.stride_segmentation import create_dtw_template, BarthDtw
from gaitmap.stride_segmentation.roi_stride_segmentation import RoiStrideSegmentation
from gaitmap.utils.dataset_helper import (
    Dataset,
    is_multi_sensor_dataset,
    get_multi_sensor_dataset_names,
    is_single_sensor_stride_list,
    is_multi_sensor_stride_list,
)
from gaitmap.utils.exceptions import ValidationError
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin


class TestMetaFunctionality(TestAlgorithmMixin):
    algorithm_class = RoiStrideSegmentation
    __test__ = True

    @pytest.fixture()
    def after_action_instance(self) -> RoiStrideSegmentation:
        # We use a simple dtw to create the instance
        template = create_dtw_template(pd.DataFrame([0, 1.0, 0]), sampling_rate_hz=100.0)
        dtw = BarthDtw(template=template, max_cost=0.5, min_match_length_s=None,)
        data = pd.DataFrame(np.array([0, 1.0, 0]))
        instance = RoiStrideSegmentation(segmentation_algorithm=dtw)
        instance.segment(
            data, sampling_rate_hz=100, regions_of_interest=pd.DataFrame([[0, 3, 0]], columns=["start", "end", "gs_id"])
        )
        return instance


def create_dummy_single_sensor_roi():
    cols = ["start", "end", "gs_id"]
    return pd.DataFrame(columns=cols)


def create_dummy_multi_sensor_roi():
    cols = ["start", "end", "gs_id"]
    return {"sensor_1": pd.DataFrame(columns=cols)}


class TestParameterValidation:
    @pytest.fixture(autouse=True)
    def _create_instance(self):
        instance = RoiStrideSegmentation(BarthDtw())
        self.instance = instance

    def test_empty_algorithm_raises_error(self):
        instance = RoiStrideSegmentation()
        with pytest.raises(ValueError) as e:
            instance.segment(pd.DataFrame([0, 0, 0]), 1, pd.DataFrame([[0, 3, 0]], columns=["start", "end", "gs_id"]))

        assert "`segmentation_algorithm` must be a valid instance of a StrideSegmentation algorithm"

    @pytest.mark.parametrize("data", (pd.DataFrame, [], None))
    def test_unsuitable_datatype(self, data):
        """No proper Dataset provided."""
        with pytest.raises(ValidationError) as e:
            self.instance.segment(
                data=data,
                sampling_rate_hz=100,
                regions_of_interest=pd.DataFrame([[0, 3, 0]], columns=["start", "end", "gs_id"]),
            )

        assert "neither a single- or a multi-sensor dataset" in str(e)

    @pytest.mark.parametrize(
        "roi", (pd.DataFrame(), None),
    )
    def test_invalid_roi_single_dataset(self, roi):
        """Test that an error is raised if an invalid roi is provided."""
        with pytest.raises(ValidationError) as e:
            # call segment with invalid ROI
            self.instance.segment(pd.DataFrame(), sampling_rate_hz=10.0, regions_of_interest=roi)

        assert "neither a single- or a multi-sensor regions of interest list" in str(e)

    def test_multi_roi_single_sensor(self):
        with pytest.raises(ValidationError) as e:
            # call segment with invalid ROI
            self.instance.segment(
                pd.DataFrame(), sampling_rate_hz=10.0, regions_of_interest=create_dummy_multi_sensor_roi()
            )

        assert "multi-sensor regions of interest list with a single sensor" in str(e)

    def test_invalid_roi_multiple_dataset(self):
        """Test that an error is raised if an invalid roi is provided."""
        with pytest.raises(ValidationError) as e:
            # call segment with invalid ROI
            self.instance.segment({"sensor": pd.DataFrame()}, sampling_rate_hz=10.0, regions_of_interest=pd.DataFrame())

        assert "neither a single- or a multi-sensor regions of interest list" in str(e)

    def test_single_roi_unsync_multi(self):
        with pytest.raises(ValidationError) as e:
            # call segment with invalid ROI
            # Note, that the empty dataframe as data is actually valid data object and will not raise a validation
            # error.
            self.instance.segment(
                {"sensor": pd.DataFrame()}, sampling_rate_hz=10.0, regions_of_interest=create_dummy_single_sensor_roi()
            )

        assert "single-sensor regions of interest list with an unsynchronised" in str(e)

    def test_invalid_stride_id_naming(self):
        self.instance.set_params(s_id_naming="wrong")

        with pytest.raises(ValueError) as e:
            # Note, that the empty dataframe as data is actually valid data object and will not raise a validation
            # error.
            self.instance.segment(
                pd.DataFrame(), sampling_rate_hz=10.0, regions_of_interest=create_dummy_single_sensor_roi()
            )
        assert "s_id_naming" in str(e)

    def test_additional_sensors_in_roi(self):
        with pytest.raises(KeyError) as e:
            # Note, that the empty dataframe as data is actually valid data object and will not raise a validation
            # error.
            self.instance.segment(
                {"sensor": pd.DataFrame()}, sampling_rate_hz=10.0, regions_of_interest=create_dummy_multi_sensor_roi()
            )

        assert "The regions of interest list contains information for a sensor" in str(e)


class MockStrideSegmentation(BaseStrideSegmentation):
    """A Mock stride segmentation class for testing."""

    def __init__(self, n=3):
        self.n = 3

    def segment(self: BaseType, data: Dataset, sampling_rate_hz: float, **kwargs) -> BaseType:
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        # For testing we will save the kwargs
        self._kwargs = kwargs
        # We will just detect a n-strides in the data
        if is_multi_sensor_dataset(data, check_gyr=False, check_acc=False):
            stride_list = {}
            for sensor in get_multi_sensor_dataset_names(data):
                tmp = np.linspace(0, len(data[sensor]), self.n + 1).astype(int)
                stride_list[sensor] = pd.DataFrame({"s_id": np.arange(len(tmp) - 1), "start": tmp[:-1], "end": tmp[1:]})
            self._stride_list_ = stride_list
        else:
            tmp = np.linspace(0, len(data), self.n + 1).astype(int)
            self._stride_list_ = pd.DataFrame({"s_id": np.arange(len(tmp) - 1), "start": tmp[:-1], "end": tmp[1:]})

        return self

    @property
    def stride_list_(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        return deepcopy(self._stride_list_)


class TestCombinedStridelist:
    """Test the actual ROI stuff."""

    @pytest.fixture(autouse=True, params=["replace", "prefix"])
    def _s_id_naming(self, request):
        self.s_id_naming = request.param

    def test_single_sensor(self):
        roi_seg = RoiStrideSegmentation(MockStrideSegmentation(), self.s_id_naming)
        data = pd.DataFrame(np.ones(27))
        roi = pd.DataFrame(np.array([[0, 1, 3], [0, 9, 18], [8, 17, 26]]).T, columns=["roi_id", "start", "end"])

        roi_seg.segment(data, sampling_rate_hz=100, regions_of_interest=roi)
        assert len(roi_seg.stride_list_) == len(roi) * roi_seg.segmentation_algorithm.n
        assert is_single_sensor_stride_list(roi_seg.stride_list_)

        assert len(roi_seg.instances_per_roi_) == len(roi)
        assert all([isinstance(o, MockStrideSegmentation) for o in roi_seg.instances_per_roi_.values()])

        if self.s_id_naming == "replace":
            assert_array_equal(roi_seg.stride_list_.index, list(range(len(roi_seg.stride_list_))))
            assert roi_seg.stride_list_.index.name == "s_id"
        else:
            assert roi_seg.stride_list_.index[0] == "0_0"
            assert roi_seg.stride_list_.index.name == "s_id"

        # test if the stride starts are actually greater or equal to the roi starts
        for r in roi.iterrows():
            for stride in roi_seg.stride_list_.iterrows():
                if r[1]["roi_id"] == stride[1]["roi_id"]:
                    assert stride[1]["start"] >= r[1]["start"]

    def test_multi_sensor(self):
        roi_seg = RoiStrideSegmentation(MockStrideSegmentation(), self.s_id_naming)
        data = {"s1": pd.DataFrame(np.ones(27)), "s2": pd.DataFrame(np.zeros(27))}
        roi = pd.DataFrame(np.array([[0, 1, 3], [0, 9, 18], [8, 17, 26]]).T, columns=["roi_id", "start", "end"])
        roi = {"s1": roi, "s2": roi.copy().iloc[:2]}

        roi_seg.segment(data, sampling_rate_hz=100, regions_of_interest=roi)
        assert is_multi_sensor_stride_list(roi_seg.stride_list_)
        assert len(roi_seg.instances_per_roi_) == len(roi)
        assert all([isinstance(o, dict) for o in roi_seg.instances_per_roi_.values()])

        for sensor in ["s1", "s2"]:
            assert all([isinstance(o, MockStrideSegmentation) for o in roi_seg.instances_per_roi_[sensor].values()])
            assert len(roi_seg.stride_list_[sensor]) == len(roi[sensor]) * roi_seg.segmentation_algorithm.n
            if self.s_id_naming == "replace":
                assert_array_equal(roi_seg.stride_list_[sensor].index, list(range(len(roi_seg.stride_list_[sensor]))))
                assert roi_seg.stride_list_[sensor].index.name == "s_id"
            else:
                assert roi_seg.stride_list_[sensor].index[0] == "0_0"
                assert roi_seg.stride_list_[sensor].index.name == "s_id"

        # test if the stride starts are actually greater or equal to the roi starts
        for sensor in get_multi_sensor_dataset_names(data):
            for r in roi[sensor].iterrows():
                for stride in roi_seg.stride_list_[sensor].iterrows():
                    if r[1]["roi_id"] == stride[1]["roi_id"]:
                        assert stride[1]["start"] >= r[1]["start"]

    def test_multi_sensor_sync(self):
        roi_seg = RoiStrideSegmentation(MockStrideSegmentation(), self.s_id_naming)
        data = pd.concat({"s1": pd.DataFrame(np.ones(27)), "s2": pd.DataFrame(np.zeros(27))}, axis=1)
        roi = pd.DataFrame(np.array([[0, 1, 3], [0, 9, 18], [8, 17, 26]]).T, columns=["roi_id", "start", "end"])

        roi_seg.segment(data, sampling_rate_hz=100, regions_of_interest=roi)
        assert is_multi_sensor_stride_list(roi_seg.stride_list_)
        assert len(roi_seg.instances_per_roi_) == len(roi)
        assert all([isinstance(o, MockStrideSegmentation) for o in roi_seg.instances_per_roi_.values()])

        for sensor in ["s1", "s2"]:
            assert len(roi_seg.stride_list_[sensor]) == len(roi) * roi_seg.segmentation_algorithm.n
            if self.s_id_naming == "replace":
                assert_array_equal(roi_seg.stride_list_[sensor].index, list(range(len(roi_seg.stride_list_[sensor]))))
                assert roi_seg.stride_list_[sensor].index.name == "s_id"
            else:
                assert roi_seg.stride_list_[sensor].index[0] == "0_0"
                assert roi_seg.stride_list_[sensor].index.name == "s_id"

        # test if the stride starts are actually greater or equal to the roi starts
        for sensor in get_multi_sensor_dataset_names(data):
            for r in roi.iterrows():
                for stride in roi_seg.stride_list_[sensor].iterrows():
                    if r[1]["roi_id"] == stride[1]["roi_id"]:
                        assert stride[1]["start"] >= r[1]["start"]
