import numpy as np
import pandas as pd
import pytest

from gaitmap.stride_segmentation import create_dtw_template, BarthDtw
from gaitmap.stride_segmentation.roi_stride_segmentation import RoiStrideSegmentation
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin


class TestMetaFunctionality(TestAlgorithmMixin):
    algorithm_class = RoiStrideSegmentation
    __test__ = True

    @pytest.fixture()
    def after_action_instance(self) -> RoiStrideSegmentation:
        # We use a simple dtw to create the instance
        template = create_dtw_template(np.array([0, 1.0, 0]), sampling_rate_hz=100.0)
        dtw = BarthDtw(template=template, max_cost=0.5, min_match_length_s=None,)
        data = np.array([0, 1.0, 0])
        instance = RoiStrideSegmentation(segmentation_algorithms=dtw)
        instance.segment(
            data, sampling_rate_hz=100, regions_of_interest=pd.DataFrame([0, 3, 0], columns=["start", "end", "gs_id"])
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
        with pytest.raises(ValueError) as e:
            self.instance.segment(
                data=data,
                sampling_rate_hz=100,
                regions_of_interest=pd.DataFrame([[0, 3, 0]], columns=["start", "end", "gs_id"]),
            )

        assert "Invalid data object passed." in str(e)

    @pytest.mark.parametrize(
        "roi", (pd.DataFrame(), None),
    )
    def test_invalid_roi_single_dataset(self, roi):
        """Test that an error is raised if an invalid roi is provided."""
        with pytest.raises(ValueError) as e:
            # call segment with invalid ROI
            self.instance.segment(pd.DataFrame(), sampling_rate_hz=10.0, regions_of_interest=roi)

        assert "Invalid object passed for `regions_of_interest`" in str(e)

    def test_multi_roi_single_sensor(self):
        with pytest.raises(ValueError) as e:
            # call segment with invalid ROI
            self.instance.segment(
                pd.DataFrame(), sampling_rate_hz=10.0, regions_of_interest=create_dummy_multi_sensor_roi()
            )

        assert "multi-sensor regions of interest list with a single sensor" in str(e)

    def test_invalid_roi_multiple_dataset(self):
        """Test that an error is raised if an invalid roi is provided."""
        with pytest.raises(ValueError) as e:
            # call segment with invalid ROI
            self.instance.segment({"sensor": pd.DataFrame()}, sampling_rate_hz=10.0, regions_of_interest=pd.DataFrame())

        assert "Invalid object passed for `regions_of_interest`" in str(e)

    def test_single_roi_unsync_multi(self):
        with pytest.raises(ValueError) as e:
            # call segment with invalid ROI
            self.instance.segment(
                {"sensor": pd.DataFrame()}, sampling_rate_hz=10.0, regions_of_interest=create_dummy_single_sensor_roi()
            )

        assert "single-sensor regions of interest list with an unsynchronised" in str(e)
