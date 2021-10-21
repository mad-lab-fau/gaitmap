import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA

from gaitmap.preprocessing.sensor_alignment import PcaAlignment
from gaitmap.utils.datatype_helper import MultiSensorData, get_multi_sensor_names


class TestPcaAlignment:
    """Test the pca alignment class `PcaAlignment`."""

    sample_sensor_data: pd.DataFrame
    sample_sensor_dataset: MultiSensorData

    def test_single_sensor_input(self, healthy_example_imu_data):
        """Dummy test to see if the algorithm is generally working on the example data"""
        data = healthy_example_imu_data["left_sensor"]

        pca_align = PcaAlignment(pca_plane_axis=("gyr_x", "gyr_y"))
        pca_align = pca_align.align(data=data)

        assert len(pca_align.aligned_data_) == len(data)

        assert isinstance(pca_align.aligned_data_, pd.DataFrame)
        assert isinstance(pca_align.rotation_, Rotation)
        assert isinstance(pca_align.pca_, PCA)

    def test_multi_sensor_input(self, healthy_example_imu_data):
        """Dummy test to see if the algorithm is generally working on the example data"""
        data = healthy_example_imu_data

        pca_align = PcaAlignment(pca_plane_axis=("gyr_x", "gyr_y"))
        pca_align = pca_align.align(data=data)

        for sensor in get_multi_sensor_names(data):
            assert len(pca_align.aligned_data_[sensor]) == len(data[sensor])

            assert isinstance(pca_align.aligned_data_[sensor], pd.DataFrame)
            assert isinstance(pca_align.rotation_[sensor], Rotation)
            assert isinstance(pca_align.pca_[sensor], PCA)

    def test_multi_sensor_input(self, healthy_example_imu_data):
        """Dummy test to see if the algorithm is generally working on the example data"""
        data = healthy_example_imu_data

        pca_align = PcaAlignment(target_axis="y", pca_plane_axis=("gyr_x", "gyr_y"))
        pca_align = pca_align.align(data=data)

        for sensor in get_multi_sensor_names(data):
            assert len(pca_align.aligned_data_[sensor]) == len(data[sensor])

            assert isinstance(pca_align.aligned_data_[sensor], pd.DataFrame)
            assert isinstance(pca_align.rotation_[sensor], Rotation)
            assert isinstance(pca_align.pca_[sensor], PCA)

    def test_invalid_pca_plane_axis(self, healthy_example_imu_data):
        """Test if value error is raised correctly if invalid axis for the search plane are defined."""
        data = healthy_example_imu_data

        with pytest.raises(ValueError, match=r".*Invalid axis for pca plane *"):
            PcaAlignment(target_axis="y", pca_plane_axis=("abc", "gyr_y")).align(data)

        with pytest.raises(ValueError, match=r".*Invalid axis for pca plane *"):
            PcaAlignment(target_axis="y", pca_plane_axis=("acc_x", "gyr_y")).align(data)

        with pytest.raises(ValueError, match=r".*Invalid axis for pca plane *"):
            PcaAlignment(target_axis="y", pca_plane_axis=("acc_x")).align(data)

    def test_invalid_target_axis(self, healthy_example_imu_data):
        """Test if value error is raised correctly if invalid axis for the search plane are defined."""
        data = healthy_example_imu_data

        with pytest.raises(ValueError, match=r".*Invalid target aixs *"):
            PcaAlignment(target_axis="z", pca_plane_axis=("gyr_x", "gyr_y")).align(data)

        with pytest.raises(ValueError, match=r".*Invalid target aixs *"):
            PcaAlignment(target_axis="a", pca_plane_axis=("gyr_x", "gyr_y")).align(data)

    @pytest.mark.parametrize(
        "axis,rot",
        (
            ("x", np.array([[0.28177506, 0.95948049, 0.0], [-0.95948049, 0.28177506, -0.0], [-0.0, 0.0, 1.0]])),
            ("y", np.array([[0.95948049, -0.28177506, 0.0], [0.28177506, 0.95948049, 0.0], [0.0, 0.0, 1.0]])),
        ),
    )
    def test_correct_rotation_regression(self, healthy_example_imu_data, snapshot, axis, rot):
        """Test if the alignment actually returns the expected rotation matrix on real imu data."""
        data = healthy_example_imu_data["left_sensor"]

        pca_align = PcaAlignment(target_axis=axis, pca_plane_axis=("gyr_x", "gyr_y"))
        pca_align = pca_align.align(data=data)

        assert_almost_equal(rot, pca_align.rotation_.as_matrix())
        snapshot.assert_match(pca_align.aligned_data_, check_names=False)

    def test_correct_rotation_complementary(self, healthy_example_imu_data):
        """Test if the alignment actually returns the expected rotation matrix on real imu data."""
        data = healthy_example_imu_data["left_sensor"]

        pca_align_y = PcaAlignment(target_axis="y", pca_plane_axis=("gyr_x", "gyr_y")).align(data=data)
        pca_align_x = PcaAlignment(target_axis="x", pca_plane_axis=("gyr_x", "gyr_y")).align(data=data)

        assert_almost_equal(
            pca_align_y.aligned_data_["gyr_y"].to_numpy(), pca_align_x.aligned_data_["gyr_x"].to_numpy()
        )
        assert_almost_equal(
            pca_align_y.aligned_data_["acc_y"].to_numpy(), pca_align_x.aligned_data_["acc_x"].to_numpy()
        )

        assert_almost_equal(
            pca_align_y.aligned_data_["acc_x"].to_numpy(), -pca_align_x.aligned_data_["acc_y"].to_numpy()
        )
        assert_almost_equal(
            pca_align_y.aligned_data_["gyr_x"].to_numpy(), -pca_align_x.aligned_data_["gyr_y"].to_numpy()
        )

    @pytest.mark.parametrize("axis", ("x", "y"))
    def test_is_righthanded_rotation(self, healthy_example_imu_data, axis):
        """Test if the resulting rotation object is a valid righthanded rotation."""
        data = healthy_example_imu_data["left_sensor"]

        pca_align = PcaAlignment(target_axis=axis, pca_plane_axis=("gyr_x", "gyr_y"))
        pca_align = pca_align.align(data=data)

        rot_matrix = pca_align.rotation_.as_matrix()

        # a valid right handed rotation matrix must have a determinant of 1
        assert_almost_equal(1.0, np.linalg.det(rot_matrix))
