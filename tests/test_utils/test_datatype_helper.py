"""Test the dataset helpers."""
import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from gaitmap.utils.consts import (
    BF_ACC,
    BF_COLS,
    BF_GYR,
    GF_ORI,
    GF_POS,
    GF_VEL,
    SF_ACC,
    SF_COLS,
    SF_GYR,
    TRAJ_TYPE_COLS,
)
from gaitmap.utils.datatype_helper import (
    get_multi_sensor_names,
    is_multi_sensor_data,
    is_multi_sensor_orientation_list,
    is_multi_sensor_position_list,
    is_multi_sensor_regions_of_interest_list,
    is_multi_sensor_stride_list,
    is_multi_sensor_velocity_list,
    is_orientation_list,
    is_position_list,
    is_regions_of_interest_list,
    is_sensor_data,
    is_single_sensor_data,
    is_single_sensor_orientation_list,
    is_single_sensor_position_list,
    is_single_sensor_regions_of_interest_list,
    is_single_sensor_stride_list,
    is_single_sensor_velocity_list,
    is_stride_list,
    is_velocity_list,
    set_correct_index,
    to_dict_multi_sensor_data,
)
from gaitmap.utils.exceptions import ValidationError


def _create_test_multiindex():
    return pd.MultiIndex.from_product([list("abc"), list("123")])


@pytest.fixture(params=(("both", True, True), ("acc", True, False), ("gyr", False, True)))
def combinations(request):
    return request.param


@pytest.fixture(params=("any", "body", "sensor"))
def frame(request):
    return request.param


@pytest.fixture(params=("any", "min_vel", "ic", "segmented"))
def stride_types(request):
    return request.param


@pytest.fixture(params=("any", "gs", "roi"))
def roi_types(request):
    return request.param


@pytest.fixture(params=(True, False))
def as_index(request):
    return request.param


class TestIsSingleSensorDataset:
    @pytest.mark.parametrize(
        "value",
        ({"test": pd.DataFrame}, list(range(6)), "test", np.arange(6), pd.DataFrame(columns=_create_test_multiindex())),
    )
    def test_wrong_datatype(self, value):
        assert not is_single_sensor_data(value, check_acc=False, check_gyr=False)

    def test_correct_datatype(self):
        assert is_single_sensor_data(pd.DataFrame(), check_acc=False, check_gyr=False)

    @pytest.mark.parametrize(
        "cols, frame_valid, col_check_valid",
        (
            (SF_COLS, "sensor", "both"),
            (BF_COLS, "body", "both"),
            (BF_GYR, "body", "gyr"),
            (BF_ACC, "body", "acc"),
            (SF_GYR, "sensor", "gyr"),
            (SF_ACC, "sensor", "acc"),
        ),
    )
    def test_correct_columns(self, cols, frame_valid, col_check_valid, combinations, frame):
        """Test all possible combinations of inputs."""
        col_check, check_acc, check_gyro = combinations
        output = is_single_sensor_data(
            pd.DataFrame(columns=cols), check_acc=check_acc, check_gyr=check_gyro, frame=frame
        )

        valid_frame = (frame_valid == frame) or (frame == "any")
        valid_cols = (col_check == col_check_valid) or (col_check_valid == "both")
        expected_outcome = valid_cols and valid_frame

        assert output == expected_outcome

    def test_invalid_frame_argument(self):
        with pytest.raises(ValueError):
            is_single_sensor_data(pd.DataFrame(), frame="invalid_value")

    def test_error_raising(self):
        with pytest.raises(ValidationError) as e:
            is_single_sensor_data(pd.DataFrame(), frame="body", check_acc=True, check_gyr=False, raise_exception=True)

        assert "The passed object does not seem to be SingleSensorData." in str(e)
        assert str(BF_ACC) in str(e.value)


class TestIsMultiSensorDataset:
    @pytest.mark.parametrize(
        "value",
        (list(range(6)), "test", np.arange(6), {}, pd.DataFrame(), pd.DataFrame(columns=[*range(3)])),
    )
    def test_wrong_datatype(self, value):
        assert not is_multi_sensor_data(value, check_acc=False, check_gyr=False)

    def test_correct_datatype(self):
        assert is_multi_sensor_data(
            pd.DataFrame([[*range(9)]], columns=_create_test_multiindex()), check_acc=False, check_gyr=False
        )

    @pytest.mark.parametrize(
        "cols, frame_valid, col_check_valid",
        (
            (SF_COLS, "sensor", "both"),
            (BF_COLS, "body", "both"),
            (BF_GYR, "body", "gyr"),
            (BF_ACC, "body", "acc"),
            (SF_GYR, "sensor", "gyr"),
            (SF_ACC, "sensor", "acc"),
        ),
    )
    def test_correct_columns(self, cols, frame_valid, col_check_valid, combinations, frame):
        """Test all possible combinations of inputs."""
        col_check, check_acc, check_gyro = combinations
        output = is_multi_sensor_data(
            pd.DataFrame([[*range(len(cols) * 2)]], columns=pd.MultiIndex.from_product((("a", "b"), cols))),
            check_acc=check_acc,
            check_gyr=check_gyro,
            frame=frame,
        )

        valid_frame = (frame_valid == frame) or (frame == "any")
        valid_cols = (col_check == col_check_valid) or (col_check_valid == "both")
        expected_outcome = valid_cols and valid_frame

        assert output == expected_outcome

    def test_invalid_frame_argument(self):
        with pytest.raises(ValueError):
            is_multi_sensor_data(pd.DataFrame([[*range(9)]], columns=_create_test_multiindex()), frame="invalid_value")

    def test_error_raising(self):
        with pytest.raises(ValidationError) as e:
            is_multi_sensor_data(pd.DataFrame(), raise_exception=True)

        assert "The passed object does not seem to be MultiSensorData." in str(e)
        assert "MultiIndex" in str(e)

    def test_nested_error_raising(self):
        with pytest.raises(ValidationError) as e:
            is_multi_sensor_data(
                {"s1": pd.DataFrame()}, frame="body", check_acc=True, check_gyr=False, raise_exception=True
            )

        assert "The passed object appears to be MultiSensorData" in str(e.value)
        assert 'for the sensor with the name "s1"' in str(e.value)
        assert str(BF_ACC) in str(e.value)


class TestIsDataset:
    def test_raises_error_correctly(self):
        with pytest.raises(ValidationError) as e:
            is_sensor_data(pd.DataFrame(), frame="body", check_acc=True, check_gyr=False)

        assert "The passed object appears to be neither single- or multi-sensor data." in str(e)
        assert str(BF_ACC) in str(e.value)
        assert "MultiIndex" in str(e.value)

    @pytest.mark.parametrize(("obj", "out"), ((pd.DataFrame(), "single"), ({"s1": pd.DataFrame()}, "multi")))
    def test_basic_function(self, obj, out):
        assert is_sensor_data(obj, check_gyr=False, check_acc=False) == out


class TestGetMultiSensorDatasetNames:
    @pytest.mark.parametrize("obj", ({"a": [], "b": [], "c": []}, pd.DataFrame(columns=_create_test_multiindex())))
    def test_names_simple(self, obj):
        assert set(get_multi_sensor_names(obj)) == {"a", "b", "c"}


class TestIsSingleSensorStrideList:
    @pytest.mark.parametrize(
        "value",
        (
            list(range(6)),
            "test",
            np.arange(6),
            {},
            pd.DataFrame(),
            pd.DataFrame(columns=[*range(3)]),
            pd.DataFrame([[*range(9)]], columns=_create_test_multiindex()),
        ),
    )
    def test_wrong_datatype(self, value):
        assert not is_single_sensor_stride_list(value)

    @pytest.mark.parametrize(
        "cols, stride_types_valid",
        (
            (["s_id", "start", "end", "gsd_id"], ["any"]),
            (["s_id", "start", "end", "gsd_id", "something_extra"], ["any"]),
            (["s_id", "start", "end", "gsd_id", "pre_ic", "ic", "min_vel", "tc"], ["segmented", "min_vel", "ic"]),
            (
                ["s_id", "start", "end", "gsd_id", "pre_ic", "ic", "min_vel", "tc", "something_extra"],
                ["segmented", "min_vel", "ic"],
            ),
            (["s_id", "start", "end", "gsd_id", "ic", "min_vel", "tc"], ["ic", "segmented"]),
            (["s_id", "start", "end", "gsd_id", "ic", "min_vel", "tc", "something_extra"], ["ic", "segmented"]),
        ),
    )
    def test_valid_versions(self, cols, stride_types_valid, stride_types, as_index):
        expected_outcome = stride_types in stride_types_valid or stride_types == "any"
        df = pd.DataFrame(columns=cols)
        if as_index:
            df = df.set_index("s_id")

        out = is_single_sensor_stride_list(df, stride_type=stride_types)

        assert expected_outcome == out

    @pytest.mark.parametrize("check_additional_cols", (True, False, ("ic",)))
    def test_check_additional_columns(self, check_additional_cols):
        # We construct a df that only has the minimal columns for min_vel
        df = pd.DataFrame(columns=["s_id", "start", "end", "min_vel"])

        if check_additional_cols is not False:
            with pytest.raises(ValidationError):
                is_single_sensor_stride_list(
                    df, stride_type="min_vel", check_additional_cols=check_additional_cols, raise_exception=True
                )
        else:
            is_single_sensor_stride_list(
                df, stride_type="min_vel", check_additional_cols=check_additional_cols, raise_exception=True
            )

    @pytest.mark.parametrize(
        "start, min_vel, expected",
        ((np.arange(10), np.arange(10), True), (np.arange(10), np.arange(10) + 1, False), ([], [], True)),
    )
    def test_columns_same_min_vel(self, start, min_vel, expected):
        """Test that the column equals check for min_vel_strides work."""
        min_vel_cols = ["s_id", "start", "end", "gsd_id", "pre_ic", "ic", "min_vel", "tc"]
        stride_list = pd.DataFrame(columns=min_vel_cols)
        stride_list["s_id"] = start
        stride_list["start"] = start
        stride_list["min_vel"] = min_vel

        out = is_single_sensor_stride_list(stride_list, stride_type="min_vel")

        assert out == expected

    @pytest.mark.parametrize(
        "start, ic, expected",
        ((np.arange(10), np.arange(10), True), (np.arange(10), np.arange(10) + 1, False), ([], [], True)),
    )
    def test_columns_same_ic(self, start, ic, expected):
        """Test that the column equals check for ic_strides work."""
        min_vel_cols = ["s_id", "start", "end", "gsd_id", "ic", "min_vel", "tc"]
        stride_list = pd.DataFrame(columns=min_vel_cols)
        stride_list["s_id"] = start
        stride_list["start"] = start
        stride_list["ic"] = ic

        out = is_single_sensor_stride_list(stride_list, stride_type="ic")

        assert out == expected

    def test_invalid_stride_type_argument(self):
        valid_cols = ["s_id", "start", "end", "gsd_id"]
        valid = pd.DataFrame(columns=valid_cols)

        with pytest.raises(ValueError):
            is_single_sensor_stride_list(valid, stride_type="invalid_value")

    def test_identical_stride_ids(self):
        """Test that the search for identical stride ids works."""
        min_vel_cols = ["s_id", "start", "end"]
        stride_list = pd.DataFrame(columns=min_vel_cols)
        stride_list["s_id"] = np.array([1, 2, 2])
        expected_outcome = False

        out = is_single_sensor_stride_list(stride_list, stride_type="any")

        assert expected_outcome == out

    def test_error_raising(self):
        with pytest.raises(ValidationError) as e:
            is_single_sensor_stride_list(pd.DataFrame(), raise_exception=True)

        assert "The passed object does not seem to be a SingleSensorStrideList." in str(e)
        assert str(["s_id"]) in str(e.value)


class TestIsMultiSensorStrideList:
    @pytest.mark.parametrize(
        "value",
        (list(range(6)), "test", np.arange(6), {}, pd.DataFrame(), pd.DataFrame(columns=[*range(3)])),
    )
    def test_wrong_datatype(self, value):
        assert not is_multi_sensor_stride_list(value)

    @pytest.mark.parametrize(
        "cols, stride_types_valid",
        (
            (["s_id", "start", "end", "gsd_id"], ["any"]),
            (["s_id", "start", "end", "gsd_id", "something_extra"], ["any"]),
            (["s_id", "start", "end", "gsd_id", "pre_ic", "ic", "min_vel", "tc"], ["segmented", "min_vel", "ic"]),
            (
                ["s_id", "start", "end", "gsd_id", "pre_ic", "ic", "min_vel", "tc", "something_extra"],
                ["segmented", "min_vel", "ic"],
            ),
            (["s_id", "start", "end", "gsd_id", "ic", "min_vel", "tc"], ["ic", "segmented"]),
            (["s_id", "start", "end", "gsd_id", "ic", "min_vel", "tc", "something_extra"], ["ic", "segmented"]),
        ),
    )
    def test_valid_versions(self, cols, stride_types_valid, stride_types, as_index):
        expected_outcome = stride_types in stride_types_valid or stride_types == "any"
        df = pd.DataFrame(columns=cols)
        if as_index:
            df = df.set_index("s_id")

        out = is_multi_sensor_stride_list({"s1": df}, stride_type=stride_types)

        assert expected_outcome == out

    def test_only_one_invalid(self):
        valid_cols = ["s_id", "start", "end", "gsd_id"]
        invalid_cols = ["start", "end", "gsd_id"]
        valid = {"s1": pd.DataFrame(columns=valid_cols)}
        invalid = {"s2": pd.DataFrame(columns=invalid_cols), **valid}

        assert is_multi_sensor_stride_list(valid)
        assert not is_multi_sensor_stride_list(invalid)

    def test_invalid_stride_type_argument(self):
        valid_cols = ["s_id", "start", "end", "gsd_id"]
        valid = {"s1": pd.DataFrame(columns=valid_cols)}

        with pytest.raises(ValueError):
            is_multi_sensor_stride_list(valid, stride_type="invalid_value")

    def test_nested_error_raising(self):
        with pytest.raises(ValidationError) as e:
            is_multi_sensor_stride_list({"s1": pd.DataFrame()}, raise_exception=True)

        assert "The passed object appears to be a MultiSensorStrideList" in str(e.value)
        assert 'for the sensor with the name "s1"' in str(e.value)
        assert str(["s_id"]) in str(e.value)


class TestIsStrideList:
    def test_raises_error_correctly(self):
        with pytest.raises(ValidationError) as e:
            is_stride_list(pd.DataFrame())

        assert "The passed object appears to be neither a single- or a multi-sensor stride list." in str(e)
        assert "s_id" in str(e.value)
        assert "'dict'" in str(e.value)

    @pytest.mark.parametrize(
        ("obj", "out"),
        (
            (pd.DataFrame(columns=["s_id", "start", "end", "gsd_id"]), "single"),
            ({"s1": pd.DataFrame(columns=["s_id", "start", "end", "gsd_id"])}, "multi"),
        ),
    )
    def test_basic_function(self, obj, out):
        assert is_stride_list(obj) == out


class TestIsSingleSensorTrajLikeList:
    @pytest.fixture(
        autouse=True,
        params=(
            [is_single_sensor_position_list, "SingleSensorPositionList", GF_POS],
            [is_single_sensor_velocity_list, "SingleSensorVelocityList", GF_VEL],
            [is_single_sensor_orientation_list, "SingleSensorOrientationList", GF_ORI],
        ),
        ids=("pos", "vel", "ori"),
    )
    def traj_like_lists(self, request):
        self.func, self.dtype, self.valid_cols = request.param

    @pytest.mark.parametrize(
        "value",
        (
            list(range(6)),
            "test",
            np.arange(6),
            {},
            pd.DataFrame(),
            pd.DataFrame(columns=[*range(3)]),
            pd.DataFrame(columns=["s_id", "sample", "wrong1", "wrong2"]),
        ),
    )
    def test_wrong_datatype(self, value):
        assert not self.func(value)

    @pytest.mark.parametrize(
        "cols, index",
        (
            (["s_id", "sample"], []),
            (["s_id", "sample", "something_else"], []),
            (["sample"], ["s_id"]),
            ([], ["s_id", "sample"]),
            (["something_else"], ["s_id", "sample"]),
        ),
    )
    def test_valid_versions(self, cols, index):
        df = pd.DataFrame(columns=[*self.valid_cols, *cols, *index])
        if index:
            df = df.set_index(index)

        assert self.func(df, "stride")

    @pytest.mark.parametrize(
        "cols, index, both",
        (
            (["s_id", "sample"], [], True),
            (["sample"], [], False),
            ([], ["s_id", "sample"], True),
            ([], ["sample"], False),
        ),
    )
    def test_valid_versions_without_s_id(self, cols, index, both):

        df = pd.DataFrame(columns=[*self.valid_cols, *cols, *index])
        if index:
            df = df.set_index(index)

        assert self.func(df, "stride") == both
        assert self.func(df) is True

    @pytest.mark.parametrize("list_type, index", TRAJ_TYPE_COLS.items())
    def test_different_list_types(self, list_type, index):
        valid_cols = [index, "sample", *self.valid_cols]
        df = pd.DataFrame(columns=valid_cols)
        for k in TRAJ_TYPE_COLS:
            assert self.func(df, k) == (k == list_type)

    @pytest.mark.parametrize("list_type, index", TRAJ_TYPE_COLS.items())
    def test_any_roi_list_type(self, list_type, index):
        valid_cols = [index, "sample", *self.valid_cols]
        df = pd.DataFrame(columns=valid_cols)
        assert self.func(df, "any_roi") == (list_type in ["roi", "gs"])

    def test_error_raising(self):
        with pytest.raises(ValidationError) as e:
            self.func(pd.DataFrame(), raise_exception=True)

        assert "The passed object does not seem to be a {}.".format(self.dtype) in str(e)
        assert str(["sample"]) in str(e.value)


class TestIsMultiSensorTrajLikeList:
    @pytest.fixture(
        autouse=True,
        params=(
            [is_multi_sensor_position_list, "MultiSensorPositionList", GF_POS],
            [is_multi_sensor_velocity_list, "MultiSensorVelocityList", GF_VEL],
            [is_multi_sensor_orientation_list, "MultiSensorOrientationList", GF_ORI],
        ),
        ids=("pos", "vel", "ori"),
    )
    def traj_like_lists(self, request):
        self.func, self.dtype, self.valid_cols = request.param

    @pytest.mark.parametrize(
        "value",
        (list(range(6)), "test", np.arange(6), {}, pd.DataFrame(), pd.DataFrame(columns=[*range(3)])),
    )
    def test_wrong_datatype(self, value):
        assert not self.func(value)

    @pytest.mark.parametrize(
        "cols, index",
        (
            (["s_id", "sample"], []),
            (["s_id", "sample", "something_else"], []),
            (["sample"], ["s_id"]),
            ([], ["s_id", "sample"]),
            (["something_else"], ["s_id", "sample"]),
        ),
    )
    def test_valid_versions(self, cols, index):
        df = pd.DataFrame(columns=[*self.valid_cols, *cols, *index])
        if index:
            df = df.set_index(index)

        assert self.func({"s1": df}, "stride")

    def test_only_one_invalid(self):
        valid_cols = ["s_id", "sample", *self.valid_cols]
        invalid_cols = ["sample", *self.valid_cols]
        valid = {"s1": pd.DataFrame(columns=valid_cols)}
        invalid = {"s2": pd.DataFrame(columns=invalid_cols), **valid}

        assert self.func(valid, "stride")
        assert not self.func(invalid, "stride")

    def test_nested_error_raising(self):
        with pytest.raises(ValidationError) as e:
            self.func({"s1": pd.DataFrame()}, raise_exception=True)

        assert "The passed object appears to be a {}".format(self.dtype) in str(e.value)
        assert 'for the sensor with the name "s1"' in str(e.value)
        assert str(["sample"]) in str(e.value)


class TestIsTrajLikeList:
    @pytest.fixture(
        autouse=True,
        params=(
            [is_position_list, "position", GF_POS],
            [is_velocity_list, "velocity", GF_VEL],
            [is_orientation_list, "orientation", GF_ORI],
        ),
        ids=("pos", "vel", "ori"),
    )
    def traj_like_lists(self, request):
        self.func, self.dtype, self.valid_cols = request.param

    def test_raises_error_correctly(self):
        with pytest.raises(ValidationError) as e:
            self.func(pd.DataFrame())

        assert "The passed object appears to be neither a single- or a multi-sensor {} list.".format(self.dtype) in str(
            e
        )
        assert "sample" in str(e.value)
        assert "'dict'" in str(e.value)

    def test_basic_function(self):
        valid_cols = ["s_id", "sample", *self.valid_cols]
        obj = pd.DataFrame(columns=valid_cols)
        assert self.func(obj) == "single"
        assert self.func({"s1": obj}) == "multi"


class TestSetCorrectIndex:
    def test_no_change_needed(self):
        index_names = ["t1", "t2"]
        test = _create_test_multiindex()
        test = test.rename(index_names)
        df = pd.DataFrame(range(9), index=test, columns=["c"])

        assert_frame_equal(df, set_correct_index(df, index_names))

    @pytest.mark.parametrize("level", (0, 1, [0, 1]))
    def test_cols_to_index(self, level):
        """Test what happens if one or multiple of the expected index cols are normal cols."""
        index_names = ["t1", "t2"]
        test = _create_test_multiindex()
        test = test.rename(index_names)
        df = pd.DataFrame(range(9), index=test, columns=["c"])

        reset_df = df.reset_index(level=level)

        out = set_correct_index(reset_df, index_names)

        assert out.index.names == index_names
        # Nothing was changed besides setting the index
        assert_frame_equal(df, out)

    def test_col_does_not_exist(self):
        index_names = ["t1", "t2"]
        test = _create_test_multiindex()
        test = test.rename(index_names)
        df = pd.DataFrame(range(9), index=test, columns=["c"])

        with pytest.raises(ValidationError):
            set_correct_index(df, ["does_not_exist", *index_names])

    @pytest.mark.parametrize("drop_additional", (True, False))
    def test_additional_index_col(self, drop_additional):
        index_names = ["t1", "t2"]
        test = _create_test_multiindex()
        test = test.rename(index_names)
        df = pd.DataFrame(range(9), index=test, columns=["c"])

        expected = ["t1", "c"]
        out = set_correct_index(df, expected, drop_false_index_cols=drop_additional)

        assert out.index.names == expected
        assert ("t2" in out.columns) is not drop_additional


class TestIsSingleRegionsOfInterestList:
    @pytest.mark.parametrize(
        "value",
        (
            list(range(6)),
            "test",
            np.arange(6),
            {},
            pd.DataFrame(),
            pd.DataFrame(columns=[*range(3)]),
            pd.DataFrame([[*range(9)]], columns=_create_test_multiindex()),
        ),
    )
    def test_wrong_datatype(self, value):
        assert not is_single_sensor_regions_of_interest_list(value)

    @pytest.mark.parametrize(
        "cols, roi_type_valid",
        (
            (["start", "end", "gs_id"], "gs"),
            (["start", "end", "gs_id", "something_extra"], "gs"),
            (["start", "end", "roi_id"], "roi"),
            (["start", "end", "roi_id", "something_extra"], "roi"),
        ),
    )
    def test_valid_versions(self, cols, roi_type_valid, roi_types):
        expected_outcome = roi_types in roi_type_valid or roi_types == "any"

        out = is_single_sensor_regions_of_interest_list(pd.DataFrame(columns=cols), region_type=roi_types)

        assert expected_outcome == out

    def test_invalid_region_type_argument(self):
        valid_cols = ["start", "end", "gs_id"]
        valid = pd.DataFrame(columns=valid_cols)

        with pytest.raises(ValueError):
            is_single_sensor_regions_of_interest_list(valid, region_type="invalid_value")

    @pytest.mark.parametrize("col_name", ("gs_id", "roi_id"))
    def test_identical_region_ids(self, col_name):
        """Test that the search for identical region ids works."""
        cols = [col_name, "start", "end"]
        roi_list = pd.DataFrame(columns=cols)
        roi_list[col_name] = np.array([1, 2, 2])
        expected_outcome = False

        out = is_single_sensor_regions_of_interest_list(roi_list, region_type="any")

        assert expected_outcome == out

    @pytest.mark.parametrize("col_name", ("gs_id", "roi_id"))
    def test_id_col_as_index(self, col_name):
        """Test that the id col can either be the index or a column."""
        cols = [col_name, "start", "end"]
        roi_list = pd.DataFrame(columns=cols)
        roi_list = roi_list.set_index(col_name)

        out = is_single_sensor_regions_of_interest_list(roi_list, region_type="any")

        assert out is True

    def test_error_raising(self):
        with pytest.raises(ValidationError) as e:
            is_single_sensor_regions_of_interest_list(pd.DataFrame(), raise_exception=True)

        assert "The passed object does not seem to be a SingleSensorRegionsOfInterestList." in str(e)
        assert str(["roi_id", "gs_id"]) in str(e.value)


class TestIsMultiSensorRegionsOfInterestList:
    @pytest.mark.parametrize(
        "value",
        (list(range(6)), "test", np.arange(6), {}, pd.DataFrame(), pd.DataFrame(columns=[*range(3)])),
    )
    def test_wrong_datatype(self, value):
        assert not is_multi_sensor_regions_of_interest_list(value)

    @pytest.mark.parametrize(
        "cols, roi_type_valid",
        (
            (["start", "end", "gs_id"], "gs"),
            (["start", "end", "gs_id", "something_extra"], "gs"),
            (["start", "end", "roi_id"], "roi"),
            (["start", "end", "roi_id", "something_extra"], "roi"),
        ),
    )
    def test_valid_versions(self, cols, roi_type_valid, roi_types):
        expected_outcome = roi_types in roi_type_valid or roi_types == "any"

        out = is_multi_sensor_regions_of_interest_list({"s1": pd.DataFrame(columns=cols)}, region_type=roi_types)

        assert expected_outcome == out

    def test_only_one_invalid(self):
        valid_cols = ["gs_id", "start", "end"]
        invalid_cols = ["start", "end"]
        valid = {"s1": pd.DataFrame(columns=valid_cols)}
        invalid = {"s2": pd.DataFrame(columns=invalid_cols), **valid}

        assert is_multi_sensor_regions_of_interest_list(valid)
        assert not is_multi_sensor_regions_of_interest_list(invalid)

    def test_invalid_region_type_argument(self):
        valid_cols = ["start", "end", "gs_id"]
        valid = pd.DataFrame(columns=valid_cols)

        with pytest.raises(ValueError):
            is_multi_sensor_regions_of_interest_list({"si": valid}, region_type="invalid_value")

    def test_nested_error_raising(self):
        with pytest.raises(ValidationError) as e:
            is_multi_sensor_regions_of_interest_list({"s1": pd.DataFrame()}, raise_exception=True)

        assert "The passed object appears to be a MultiSensorRegionsOfInterestList" in str(e.value)
        assert 'for the sensor with the name "s1"' in str(e.value)
        assert str(["roi_id", "gs_id"]) in str(e.value)


class TestIsRegionsOfInterestList:
    def test_raises_error_correctly(self):
        with pytest.raises(ValidationError) as e:
            is_regions_of_interest_list(pd.DataFrame())

        assert "The passed object appears to be neither a single- or a multi-sensor regions of interest list." in str(e)
        assert "gs_id" in str(e.value)
        assert "'dict'" in str(e.value)

    @pytest.mark.parametrize(
        ("obj", "out"),
        (
            (pd.DataFrame(columns=["gs_id", "start", "end"]), "single"),
            (pd.DataFrame(columns=["roi_id", "start", "end"]), "single"),
            ({"s1": pd.DataFrame(columns=["roi_id", "start", "end"])}, "multi"),
            ({"s1": pd.DataFrame(columns=["gs_id", "start", "end"])}, "multi"),
        ),
    )
    def test_basic_function(self, obj, out):
        assert is_regions_of_interest_list(obj) == out


class TestToDictMultiSensorData:
    def test_convert_simple(self):
        data = pd.DataFrame(np.ones((10, 3)), columns=BF_GYR)
        data = pd.concat([data, data], axis=1, keys=["s1", "s2"])

        out = to_dict_multi_sensor_data(data)

        assert isinstance(out, dict)
        assert len(out) == 2
        assert list(out.keys()) == ["s1", "s2"]
        assert out["s1"].shape == (10, 3)
        assert out["s2"].shape == (10, 3)

    def test_dict_is_just_returned(self):
        data = pd.DataFrame(np.ones((10, 3)), columns=BF_GYR)
        data = {"s1": data, "s2": data}

        out = to_dict_multi_sensor_data(data)
        assert out is data
