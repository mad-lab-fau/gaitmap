import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from gaitmap.trajectory_reconstruction.position_methods import PieceWiseLinearDedriftedIntegration
from gaitmap.utils.array_handling import bool_array_to_start_end_array


class TestPieceWiseLinearDedriftedIntegration:
    """Test the position estimation class `PieceWiseLinearDedriftedIntegration`."""

    def test_drift_model_simple(self):
        """Run a simple example and estimate its drift model"""
        data = np.array(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
                5,
                5,
                5,
                5,
                5,
                6,
                6,
                6,
                6,
                6,
                6,
                6,
                6,
                6,
                7,
                7,
                7,
                7,
                9,
                9,
                9,
                9,
                9,
                9,
                9,
                9,
                9,
            ]
        )
        zupt = np.array(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ]
        )
        expected_output = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.15384615,
                1.30769231,
                1.46153846,
                1.61538462,
                1.76923077,
                1.92307692,
                2.07692308,
                2.23076923,
                2.38461538,
                2.53846154,
                2.69230769,
                2.84615385,
                3.0,
                3.0,
                3.0,
                3.0,
                3.0,
                3.4,
                3.8,
                4.2,
                4.6,
                5.0,
                5.0,
                5.0,
                5.0,
                5.0,
                5.28571429,
                5.57142857,
                5.85714286,
                6.14285714,
                6.42857143,
                6.71428571,
                7.0,
                7.28571429,
                7.57142857,
                7.85714286,
                8.14285714,
                8.42857143,
                8.71428571,
                9.0,
                9.0,
                9.0,
                9.0,
                9.0,
                9.0,
                9.0,
                9.0,
                9.0,
            ]
        )

        estimated_drift_model = PieceWiseLinearDedriftedIntegration()._estimate_piece_wise_linear_drift_model(
            data, bool_array_to_start_end_array(zupt.astype(bool))
        )
        assert_almost_equal(estimated_drift_model, expected_output)

        estimated_drift_model = PieceWiseLinearDedriftedIntegration()._estimate_piece_wise_linear_drift_model(
            np.repeat(data[None, :], 3, axis=0), bool_array_to_start_end_array(zupt.astype(bool))
        )
        assert_almost_equal(estimated_drift_model, np.repeat(expected_output[None, :], 3, axis=0))

    def test_drift_model(self):
        """Test drift model on simple slope with different zupt edge conditions"""
        data = np.arange(20)
        zupt = np.repeat(False, 20)
        zupt[11:15] = True
        zupt[5:8] = True

        estimated_drift_model = PieceWiseLinearDedriftedIntegration()._estimate_piece_wise_linear_drift_model(
            data, bool_array_to_start_end_array(zupt.astype(bool))
        )
        assert_almost_equal(data, estimated_drift_model)

        estimated_drift_model = PieceWiseLinearDedriftedIntegration()._estimate_piece_wise_linear_drift_model(
            data, bool_array_to_start_end_array(~zupt.astype(bool))
        )
        assert_almost_equal(data, estimated_drift_model)

    def test_all_zupt_data(self):
        """Test drift model all zupt."""
        data = np.arange(20)
        zupt = np.repeat(True, 20)

        estimated_drift_model = PieceWiseLinearDedriftedIntegration()._estimate_piece_wise_linear_drift_model(
            data, bool_array_to_start_end_array(zupt.astype(bool))
        )
        assert_almost_equal(data, estimated_drift_model)

    def test_no_zupt_data(self):
        """Test drift model no zupts available."""
        data = np.arange(20)
        zupt = np.repeat(False, 20)
        with pytest.raises(ValueError, match=r".*No valid zupt regions available*"):
            PieceWiseLinearDedriftedIntegration()._estimate_piece_wise_linear_drift_model(
                data, bool_array_to_start_end_array(zupt.astype(bool))
            )
