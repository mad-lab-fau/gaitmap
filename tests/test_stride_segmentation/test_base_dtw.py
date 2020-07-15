"""Test the basic DTW implementation.

Notes
-----
- The min_match_length_s parameter is not really tested, as it gets passed down to a scipy function, which hopefully
does
the right thing.
- The same is True for the threshold/max_cost

"""
from typing import Union, Dict

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from gaitmap.base import BaseType
from gaitmap.stride_segmentation import BarthOriginalTemplate
from gaitmap.stride_segmentation.base_dtw import BaseDtw
from gaitmap.stride_segmentation.dtw_templates import create_dtw_template
from gaitmap.utils.dataset_helper import get_multi_sensor_dataset_names
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin


class TestMetaFunctionality(TestAlgorithmMixin):
    algorithm_class = BaseDtw
    __test__ = True

    @pytest.fixture()
    def after_action_instance(self) -> BaseType:
        template = create_dtw_template(np.array([0, 1.0, 0]), sampling_rate_hz=100.0)
        dtw = self.algorithm_class(template=template, max_cost=0.5, min_match_length_s=None,)
        data = np.array([0, 1.0, 0])
        dtw.segment(data, sampling_rate_hz=100)
        return dtw


class DtwTestBase:
    def init_dtw(self, template, **kwargs):
        defaults = dict(max_cost=0.5, min_match_length_s=None, find_matches_method="min_under_thres")
        kwargs = {**defaults, **kwargs}
        return BaseDtw(template=template, **kwargs)


class TestIOErrors(DtwTestBase):
    """Test that the correct errors are raised if wrong parameter values are provided."""

    def test_no_template_provided(self):
        with pytest.raises(ValueError) as e:
            dtw = self.init_dtw(template=None)
            dtw.segment(None, None)

        assert "`template` must be specified" in str(e)

    @pytest.mark.parametrize("data", (pd.DataFrame, [], None))
    def test_unsuitable_datatype(self, data):
        """No proper Dataset provided."""
        template = create_dtw_template(np.array([0, 1.0, 0]), sampling_rate_hz=100.0)
        with pytest.raises(ValueError) as e:
            dtw = self.init_dtw(template=template)
            dtw.segment(data=data, sampling_rate_hz=100)

        assert "The type or shape of the provided dataset is not supported." in str(e)

    def test_invalid_template_combination(self):
        """Invalid combinations of template and dataset format"""
        template = create_dtw_template(np.array([0, 1.0, 0]), sampling_rate_hz=100.0)
        with pytest.raises(ValueError) as e:
            dtw = self.init_dtw(template=template)
            dtw.segment(data=pd.DataFrame(columns=["col1"]), sampling_rate_hz=100)

        assert "Invalid combination of data and template" in str(e)

    def test_multi_sensor_dataset_without_proper_template(self):
        """Invalid combination of template and multisensor dataset."""
        # This template can not be used with multi sensor dataframes.
        template = create_dtw_template(np.array([0, 1.0, 0]), sampling_rate_hz=100.0)
        sensor1 = np.array([*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2])
        sensor1 = pd.DataFrame(sensor1, columns=["col1"])
        sensor2 = np.array([*np.ones(2) * 2, 0, 1.0, 0, *np.ones(8) * 2])
        sensor2 = pd.DataFrame(sensor2, columns=["col1"])
        data = {"sensor1": sensor1, "sensor2": sensor2}

        with pytest.raises(ValueError) as e:
            dtw = self.init_dtw(template=template)
            dtw.segment(data=data, sampling_rate_hz=100)

        assert "template must either be of type `Dict[str, DtwTemplate]`" in str(e)

    def test_invalid_find_matches_method(self):
        template = create_dtw_template(np.array([0, 1.0, 0]), sampling_rate_hz=100.0)
        with pytest.raises(ValueError) as e:
            dtw = self.init_dtw(template=template, find_matches_method="invalid_name")
            dtw.segment(data=np.array([]), sampling_rate_hz=100)

        assert "find_matches_method" in str(e)


class TestSimpleSegment(DtwTestBase):
    """Simple Tests with toy examples for matching."""

    template = create_dtw_template(np.array([0, 1.0, 0]), sampling_rate_hz=100.0)

    @pytest.fixture(params=list(BaseDtw._allowed_methods_map.keys()), autouse=True)
    def _create_instance(self, request):
        dtw = self.init_dtw(template=self.template, find_matches_method=request.param,)
        self.dtw = dtw

    def test_sdtw_simple_match(self):
        """Test dtw with single match and hand calculated outcomes."""
        sequence = [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]

        dtw = self.dtw.segment(np.array(sequence), sampling_rate_hz=100.0,)

        np.testing.assert_array_equal(dtw.paths_, [[(0, 5), (1, 6), (2, 7)]])
        assert dtw.costs_ == [0.0]
        np.testing.assert_array_equal(dtw.matches_start_end_, [[5, 7]])
        np.testing.assert_array_equal(
            dtw.acc_cost_mat_,
            [
                [4.0, 4.0, 4.0, 4.0, 4.0, 0.0, 1.0, 0.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [9.0, 9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ],
        )

        np.testing.assert_array_equal(dtw.data, sequence)

    def test_sdtw_multi_match(self):
        """Test dtw with multiple matches and hand calculated outcomes."""
        sequence = 2 * [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]

        dtw = self.dtw.segment(np.array(sequence), sampling_rate_hz=100.0,)

        np.testing.assert_array_equal(dtw.paths_, [[(0, 5), (1, 6), (2, 7)], [(0, 18), (1, 19), (2, 20)]])
        np.testing.assert_array_equal(dtw.matches_start_end_, [[5, 7], [18, 20]])
        np.testing.assert_array_equal(dtw.costs_, [0.0, 0.0])

        np.testing.assert_array_equal(dtw.data, sequence)


class TestRoiSegment(DtwTestBase):
    """Simple Tests that test the use of roi."""

    template = create_dtw_template(pd.DataFrame(np.array([0, 1.0, 0]), columns=["col1"]), sampling_rate_hz=100.0)

    @pytest.fixture(params=list(BaseDtw._allowed_methods_map.keys()), autouse=True)
    def _create_instance(self, request):
        dtw = self.init_dtw(template=self.template, find_matches_method=request.param,)
        self.dtw = dtw

    def test_simple_roi(self):
        """Test dtw where the center match should be ignored, as it is outside the ROI."""
        sequence = [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]
        roi = pd.DataFrame(
            np.array([[0, len(sequence)], [2 * len(sequence), 3 * len(sequence)]]), columns=["start", "end"]
        )
        roi.index.name = "roi_id"
        roi = roi.reset_index()

        sequence = pd.DataFrame(sequence * 3, columns=["col1"])

        dtw = self.dtw.segment(sequence, sampling_rate_hz=100.0, regions_of_interest=roi)

        np.testing.assert_array_equal(dtw.paths_, [[(0, 5), (1, 6), (2, 7)], [[0, 31], [1, 32], [2, 33]]])
        np.testing.assert_array_equal(dtw.costs_, [0.0, 0.0])
        np.testing.assert_array_equal(dtw.matches_start_end_, [[5, 7], [31, 33]])
        np.testing.assert_array_equal(
            dtw.acc_cost_mat_[0],
            [
                [4.0, 4.0, 4.0, 4.0, 4.0, 0.0, 1.0, 0.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [9.0, 9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ],
        )
        np.testing.assert_array_equal(
            dtw.acc_cost_mat_[1],
            [
                [4.0, 4.0, 4.0, 4.0, 4.0, 0.0, 1.0, 0.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [9.0, 9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ],
        )

        np.testing.assert_array_equal(dtw.data, sequence)

    def test_multi_sensor_roi(self):
        sequence = [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]
        roi = pd.DataFrame(
            np.array([[0, len(sequence)], [2 * len(sequence), 3 * len(sequence)]]), columns=["start", "end"]
        )
        roi.index.name = "roi_id"
        roi = roi.reset_index()

        sequence = sequence * 3
        sensor1 = np.array(sequence)
        sensor1 = pd.DataFrame(sensor1, columns=["col1"])
        sensor2 = np.array(sequence)
        sensor2 = pd.DataFrame(sensor2, columns=["col1"])
        data = {"sensor1": sensor1, "sensor2": sensor2}

        dtw = self.dtw.segment(data, sampling_rate_hz=100.0, regions_of_interest=roi)

        correct_paths = [[(0, 5), (1, 6), (2, 7)], [[0, 31], [1, 32], [2, 33]]]
        for sensor in ["sensor1", "sensor2"]:
            np.testing.assert_array_equal(dtw.paths_[sensor], correct_paths)
            np.testing.assert_array_equal(dtw.costs_[sensor], [0.0, 0.0])
            np.testing.assert_array_equal(dtw.matches_start_end_[sensor], [[5, 7], [31, 33]])
            np.testing.assert_array_equal(
                dtw.acc_cost_mat_[sensor][0],
                [
                    [4.0, 4.0, 4.0, 4.0, 4.0, 0.0, 1.0, 0.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    [9.0, 9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                ],
            )
            np.testing.assert_array_equal(
                dtw.acc_cost_mat_[sensor][1],
                [
                    [4.0, 4.0, 4.0, 4.0, 4.0, 0.0, 1.0, 0.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    [9.0, 9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                ],
            )
            np.testing.assert_array_equal(
                dtw.cost_function_[sensor][0],
                np.sqrt([9.0, 9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            )
            np.testing.assert_array_equal(
                dtw.cost_function_[sensor][1],
                np.sqrt([9.0, 9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            )

            assert_frame_equal(dtw.data[sensor], data[sensor])


class TestMultiDimensionalArrayInputs(DtwTestBase):
    @pytest.mark.parametrize(
        "template, data",
        (
            (np.array([[0, 1.0, 0]]), np.array(2 * [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2])),
            (np.array([0, 1.0, 0]), np.array([2 * [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]])),
            (np.array([[0, 1.0, 0]]), np.array([2 * [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]])),
        ),
    )
    def test_pseudo_2d_inputs(self, template, data):
        template = create_dtw_template(template, sampling_rate_hz=100.0)

        dtw = self.init_dtw(template=template)
        dtw = dtw.segment(data, sampling_rate_hz=100.0)

        np.testing.assert_array_equal(dtw.matches_start_end_, [[5, 7], [18, 20]])

    def test_1d_dataframe_inputs(self):
        template = pd.DataFrame(np.array([0, 1.0, 0]), columns=["col1"])
        data = pd.DataFrame(np.array(2 * [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]), columns=["col1"])
        template = create_dtw_template(template, sampling_rate_hz=100.0)

        dtw = self.init_dtw(template=template)
        dtw = dtw.segment(data, sampling_rate_hz=100.0)

        np.testing.assert_array_equal(dtw.matches_start_end_, [[5, 7], [18, 20]])

    def test_no_matches_found(self):
        """Test that no errors are raised when no matches are found."""
        template = pd.DataFrame(np.array([0, 1.0, 0]), columns=["col1"])
        data = pd.DataFrame(np.ones(10), columns=["col1"])
        template = create_dtw_template(template, sampling_rate_hz=100.0)

        dtw = self.init_dtw(template=template)
        dtw = dtw.segment(data, sampling_rate_hz=100.0)

        np.testing.assert_array_equal(dtw.matches_start_end_, [])
        np.testing.assert_array_equal(dtw.paths_, [])
        np.testing.assert_array_equal(dtw.costs_, [])

    @pytest.mark.parametrize("m_cols", (2, 3))
    @pytest.mark.parametrize("input_type", (np.array, pd.DataFrame))
    def test_valid_multi_d_input(self, m_cols, input_type):
        """Test if we get the same results with simple multi D inputs.

        Data and template are repeated to have the shape (n, m_cols), where n is the number of samples
        """
        data = np.repeat([2 * [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]], m_cols, axis=0).T
        data = input_type(data)
        if isinstance(data, pd.DataFrame):
            data.columns = ["col" + str(i + 1) for i in range(m_cols)]

        template = np.repeat([[0, 1.0, 0]], m_cols, axis=0).T
        template = input_type(template)
        if isinstance(template, pd.DataFrame):
            template.columns = ["col" + str(i + 1) for i in range(m_cols)]
        template = create_dtw_template(template, sampling_rate_hz=100.0)

        dtw = self.init_dtw(template=template)
        dtw = dtw.segment(data, sampling_rate_hz=100.0)

        np.testing.assert_array_equal(dtw.matches_start_end_, [[5, 7], [18, 20]])

    @pytest.mark.parametrize("input_type", (np.array, pd.DataFrame))
    def test_data_has_more_cols_than_template(self, input_type):
        """In case the data has more cols, only the first m cols of the data is used.

        Note that this does not really tests if this work, but just that it doesn't throw an error.
        """
        n_cols_template = 2
        n_cols_data = 3
        data = np.repeat([2 * [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]], n_cols_data, axis=0).T
        data = input_type(data)
        if isinstance(data, pd.DataFrame):
            data.columns = ["col" + str(i + 1) for i in range(n_cols_data)]

        template = np.repeat([[0, 1.0, 0]], n_cols_template, axis=0).T
        template = input_type(template)
        if isinstance(template, pd.DataFrame):
            template.columns = ["col" + str(i + 1) for i in range(n_cols_template)]
        template = create_dtw_template(template, sampling_rate_hz=100.0)

        dtw = self.init_dtw(template=template)
        dtw = dtw.segment(data, sampling_rate_hz=100.0)

        np.testing.assert_array_equal(dtw.matches_start_end_, [[5, 7], [18, 20]])

    def test_data_has_less_cols_than_template_array(self):
        """In case the data has more cols, only the first m cols of the data is used."""
        n_cols_template = 3
        n_cols_data = 2
        template = create_dtw_template(np.repeat([[0, 1.0, 0]], n_cols_template, axis=0).T, sampling_rate_hz=100.0)

        data = np.repeat([2 * [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]], n_cols_data, axis=0).T

        dtw = self.init_dtw(template=template)

        with pytest.raises(ValueError) as e:
            dtw.segment(data, sampling_rate_hz=100.0)

        assert "less columns" in str(e)

    def test_data_has_wrong_cols_than_template_df(self):
        """An error should be raised, if the template has columns that are not in the df."""
        n_cols_template = 3
        n_cols_data = 2
        data = np.repeat([2 * [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]], n_cols_data, axis=0).T
        data = pd.DataFrame(data, columns=["col" + str(i + 1) for i in range(n_cols_data)])
        template = np.repeat([[0, 1.0, 0]], n_cols_template, axis=0).T
        template = pd.DataFrame(template, columns=["col" + str(i + 1) for i in range(n_cols_template)])
        template = create_dtw_template(template, sampling_rate_hz=100.0)

        dtw = self.init_dtw(template=template)

        with pytest.raises(KeyError) as e:
            dtw.segment(data, sampling_rate_hz=100.0)

        assert str(["col3"]) in str(e)

    def test_no_sampling_rate_for_resample(self):
        """Error is raised when resample is True, but no sampling rate provided."""
        template = create_dtw_template(np.ndarray([]))

        dtw = self.init_dtw(template=template)
        with pytest.raises(ValueError) as e:
            dtw.segment(np.ndarray([]), sampling_rate_hz=100.0)

        assert "sampling_rate_hz" in str(e)

    def test_sampling_rate_mismatch_warning(self):
        """Test if warning is raised, when template and data do not have the same sampling rate and resample is False"""
        template = pd.DataFrame(np.array([0, 1.0, 0]), columns=["col1"])
        data = pd.DataFrame(np.array(2 * [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]), columns=["col1"])
        template = create_dtw_template(template, sampling_rate_hz=140.0)  # sampling rate different than data.

        dtw = self.init_dtw(template=template, resample_template=False)
        with pytest.warns(UserWarning) as w:
            dtw.segment(data, sampling_rate_hz=100.0)

        assert "140.0" in str(w[0])
        assert "100.0" in str(w[0])


class TestMultiSensorInputs(DtwTestBase):
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]

    @pytest.fixture(params=("dict", "frame"), autouse=True)
    def multi_sensor_dataset(self, request):
        sensor1 = np.array([*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2])
        sensor1 = pd.DataFrame(sensor1, columns=["col1"])
        sensor2 = np.array([*np.ones(2) * 2, 0, 1.0, 0, *np.ones(8) * 2])
        sensor2 = pd.DataFrame(sensor2, columns=["col1"])
        data = {"sensor1": sensor1, "sensor2": sensor2}
        if request.param == "dict":
            self.data = data
        elif request.param == "frame":
            self.data = pd.concat(data, axis=1)

    def test_single_template_multi_sensors(self):
        """In case a single template and multiple sensors are provided, the template is applied to all sensors."""
        template = [0, 1.0, 0]
        template = pd.DataFrame(template, columns=["col1"])
        template = create_dtw_template(template, sampling_rate_hz=100.0)
        dtw = self.init_dtw(template=template)

        dtw = dtw.segment(data=self.data, sampling_rate_hz=100)

        assert {"sensor1", "sensor2"} == set(dtw.paths_.keys())
        assert {"sensor1", "sensor2"} == set(dtw.matches_start_end_.keys())
        assert {"sensor1", "sensor2"} == set(dtw.costs_.keys())
        assert {"sensor1", "sensor2"} == set(dtw.acc_cost_mat_.keys())
        assert {"sensor1", "sensor2"} == set(dtw.cost_function_.keys())

        # Test output for sensor 1
        sensor = "sensor1"
        np.testing.assert_array_equal(dtw.paths_[sensor], [[(0, 5), (1, 6), (2, 7)]])
        assert dtw.costs_[sensor] == [0.0]
        np.testing.assert_array_equal(dtw.matches_start_end_[sensor], [[5, 7]])
        np.testing.assert_array_equal(
            dtw.acc_cost_mat_[sensor],
            [
                [4.0, 4.0, 4.0, 4.0, 4.0, 0.0, 1.0, 0.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [9.0, 9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ],
        )

        # Test output for sensor 2
        sensor = "sensor2"
        np.testing.assert_array_equal(dtw.paths_[sensor], [[(0, 2), (1, 3), (2, 4)]])
        assert dtw.costs_[sensor] == [0.0]
        np.testing.assert_array_equal(dtw.matches_start_end_[sensor], [[2, 4]])
        np.testing.assert_array_equal(
            dtw.acc_cost_mat_[sensor],
            [
                [4.0, 4.0, 0.0, 1.0, 0.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                [5.0, 5.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0],
                [9.0, 9.0, 1.0, 1.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0, 9.0],
            ],
        )

    def test_no_matches_found_multiple(self):
        """Test postprocessing still works, even when there are no matches."""
        template = [0, 1.0, 0]
        template = pd.DataFrame(template, columns=["col1"])
        template = create_dtw_template(template, sampling_rate_hz=100.0)
        dtw = self.init_dtw(template=template)
        data = self.data
        for c in get_multi_sensor_dataset_names(data):
            if isinstance(data, pd.DataFrame):
                data.loc[:, c] = np.ones(13)[:, None]
            else:
                data[c][:] = np.ones(13)[:, None]
        dtw = dtw.segment(data, sampling_rate_hz=100.0)

        for s in ["sensor1", "sensor2"]:
            np.testing.assert_array_equal(dtw.matches_start_end_[s], [])
            np.testing.assert_array_equal(dtw.paths_[s], [])
            np.testing.assert_array_equal(dtw.costs_[s], [])

    def test_multi_template_multi_sensors(self):
        """Test multiple templates with multiple sensors.

        In case a multiple template and multiple sensors are provided, each template is applied to data with the same
        name sensors.
        """
        template1 = [0, 1.0, 0]
        template1 = pd.DataFrame(template1, columns=["col1"])
        template1 = create_dtw_template(template1, sampling_rate_hz=100.0)
        template2 = [0, 0, 1.0]
        template2 = pd.DataFrame(template2, columns=["col1"])
        template2 = create_dtw_template(template2, sampling_rate_hz=100.0)
        dtw = self.init_dtw(template={"sensor1": template1, "sensor2": template2})

        dtw = dtw.segment(data=self.data, sampling_rate_hz=100)

        assert {"sensor1", "sensor2"} == set(dtw.paths_.keys())
        assert {"sensor1", "sensor2"} == set(dtw.matches_start_end_.keys())
        assert {"sensor1", "sensor2"} == set(dtw.costs_.keys())
        assert {"sensor1", "sensor2"} == set(dtw.acc_cost_mat_.keys())
        assert {"sensor1", "sensor2"} == set(dtw.cost_function_.keys())

        # Test output for sensor 1
        sensor = "sensor1"
        np.testing.assert_array_equal(dtw.paths_[sensor], [[(0, 5), (1, 6), (2, 7)]])
        assert dtw.costs_[sensor] == [0.0]
        np.testing.assert_array_equal(dtw.matches_start_end_[sensor], [[5, 7]])
        np.testing.assert_array_equal(
            dtw.acc_cost_mat_[sensor],
            [
                [4.0, 4.0, 4.0, 4.0, 4.0, 0.0, 1.0, 0.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [9.0, 9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ],
        )

        # Test output for sensor 2
        sensor = "sensor2"
        np.testing.assert_array_equal(dtw.paths_[sensor], [[(0, 2), (1, 2), (2, 3)]])
        assert dtw.costs_[sensor] == [0.0]
        np.testing.assert_array_equal(dtw.matches_start_end_[sensor], [[2, 3]])
        np.testing.assert_array_equal(
            dtw.acc_cost_mat_[sensor],
            [
                [4.0, 4.0, 0.0, 1.0, 0.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                [8.0, 8.0, 0.0, 1.0, 0.0, 4.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
                [9.0, 9.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ],
        )


class TestTemplateResampling(DtwTestBase):
    def test_resample_length(self):
        """Unittest the resample func."""
        test_template = np.ones(100)
        resampled_template = BaseDtw._resample_template(test_template, 100, 200)
        assert len(resampled_template) == 200

    def test_resample_real_example(self):
        """Toy example with hand calculated outcomes."""
        # Test that this works in general
        template = [0, 1, 2, 3, 4, 3, 2, 1, 0]
        test_data = np.array([0, 0, *template, 0, 0, *template, 0, 0])

        template = create_dtw_template(np.array(template), sampling_rate_hz=20)
        dtw = self.init_dtw(template=template)

        dtw.segment(test_data, 20)

        assert len(dtw.matches_start_end_) == 2

        # Test with template that is shorter, but resample it
        short_template = [0, 2, 4, 2, 0]

        template = create_dtw_template(np.array(short_template), sampling_rate_hz=20 * 5 / 9)
        dtw = self.init_dtw(template=template)

        dtw.segment(test_data, 20)

        assert len(dtw.matches_start_end_) == 2

        # Test with template that is shorter, but dont't resample it
        short_template = [0, 2, 4, 2, 0]

        template = create_dtw_template(np.array(short_template), sampling_rate_hz=20 * 5 / 9)
        dtw = self.init_dtw(template=template, resample_template=False)

        dtw.segment(test_data, 20)

        assert len(dtw.matches_start_end_) == 0

    @pytest.mark.parametrize("sampling_rate", (102.4, 204.8, 100, 500, 409.6))
    def test_resample_decimal_values(self, sampling_rate):
        """As a test, we gonna try to resample the Barth Template to various sampling rates."""
        test_template = BarthOriginalTemplate().data
        resampled_template = BaseDtw._resample_template(
            test_template, BarthOriginalTemplate().sampling_rate_hz, sampling_rate
        )

        # We just gonna check the length of the resampled template
        assert len(resampled_template) == int(200 * sampling_rate / 204.8)

    def test_bug_missing_if(self):
        """Test that no error is thrown if sampling rates are different, but the template has no sampling rate."""

        template = [0, 1, 2, 3, 4, 3, 2, 1, 0]
        test_data = np.array([0, 0, *template, 0, 0, *template, 0, 0])

        template = create_dtw_template(np.array(template), sampling_rate_hz=None)
        dtw = self.init_dtw(template=template, resample_template=False)

        try:
            dtw.segment(test_data, 20)
        except Exception as e:
            pytest.fail("Raised unexpected error: {}".format(e), pytrace=True)

    def test_error_if_not_template_sampling_rate(self):
        """If the template should be resampled, but the template has no sampling rate, raise an error."""
        template = [0, 1, 2, 3, 4, 3, 2, 1, 0]
        test_data = np.array([0, 0, *template, 0, 0, *template, 0, 0])

        template = create_dtw_template(np.array(template), sampling_rate_hz=None)
        dtw = self.init_dtw(template=template, resample_template=True)

        with pytest.raises(ValueError) as e:
            dtw.segment(test_data, 20)

        assert "resample the template" in str(e)

    def test_warning_when_no_resample_but_different_sampling_rate(self):
        """In case the template and the data have different sampling rates, but resample is false, the user will be
        warned."""

        template_arr = [0, 1, 2, 3, 4, 3, 2, 1, 0]
        test_data = np.array([0, 0, *template_arr, 0, 0, *template_arr, 0, 0])

        template = create_dtw_template(np.array(template_arr), sampling_rate_hz=10)
        dtw = self.init_dtw(template=template, resample_template=False)

        with pytest.warns(UserWarning) as w:
            dtw.segment(test_data, 20)

        assert len(w) == 1
        assert "sampling rate are different" in str(w[0])

        # The warning should not be raised in case their is no template sampling rate
        template = create_dtw_template(np.array(template_arr), sampling_rate_hz=False)
        dtw = self.init_dtw(template=template, resample_template=False)

        with pytest.warns(None) as w:
            dtw.segment(test_data, 20)

        assert len(w) == 0
