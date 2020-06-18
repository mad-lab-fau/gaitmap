import matplotlib
import numpy as np

# This is needed to avoid plots to open
from tests.conftest import compare_algo_objects

matplotlib.use("Agg")


def test_base_dtw_generic(snapshot):
    from examples.base_dtw_generic import dtw

    assert len(dtw.matches_start_end_) == 5
    snapshot.assert_match(dtw.matches_start_end_)


def test_barth_dtw_example(snapshot):
    from examples.barth_dtw_stride_segmentation import dtw

    assert len(dtw.matches_start_end_["left_sensor"]) == 28
    snapshot.assert_match(dtw.matches_start_end_["left_sensor"])


def test_preprocessing_example(snapshot):
    import numpy as np
    from examples.preprocessing_example import dataset_sf_aligned_to_gravity
    from gaitmap.utils.consts import SF_ACC

    desired_acc_vec = np.array([0.0, 0.0, 9.81])

    # check if at least first 5 samples of left- and right-sensor are correctly aligned to gravity
    left_acc = dataset_sf_aligned_to_gravity["left_sensor"][SF_ACC].to_numpy()[:5, :]
    right_acc = dataset_sf_aligned_to_gravity["right_sensor"][SF_ACC].to_numpy()[:5, :]

    for acc_l, acc_r in zip(left_acc, right_acc):
        np.testing.assert_almost_equal(acc_l, desired_acc_vec, decimal=0)
        np.testing.assert_almost_equal(acc_r, desired_acc_vec, decimal=0)

    # just check first 1000 rows to make sure that snapshot stays in a kB range
    snapshot.assert_match(dataset_sf_aligned_to_gravity.to_numpy()[:1000])


def test_temporal_parameters(snapshot):
    from examples.temporal_parameters import p

    snapshot.assert_match(p.parameters_["left_sensor"])


def test_spatial_parameters(snapshot):
    from examples.spatial_parameters import p

    snapshot.assert_match(p.parameters_["left_sensor"])


def test_rampp_event_detection(snapshot):
    from examples.rampp_event_detection import ed

    assert len(ed.min_vel_event_list_["left_sensor"]) == 26
    assert len(ed.min_vel_event_list_["right_sensor"]) == 29
    snapshot.assert_match(ed.min_vel_event_list_["left_sensor"])
    snapshot.assert_match(ed.min_vel_event_list_["right_sensor"])


def test_json_example(snapshot):
    from examples.algo_serialize import json_str, slt, loaded_slt

    snapshot.assert_match(json_str)

    compare_algo_objects(slt, loaded_slt)


def test_trajectory_reconstruction(snapshot):
    from examples.trajectory_reconstruction import trajectory

    # just look at last values to see if final result is correct and save runtime
    snapshot.assert_match(trajectory.position_["left_sensor"].tail(20))
    snapshot.assert_match(trajectory.orientation_["left_sensor"].tail(20))


def test_mad_pipeline(snapshot):
    from examples.mad_gait_pipeline import ed, spatial_paras, temporal_paras

    snapshot.assert_match(ed.min_vel_event_list_["left_sensor"], "strides_left")
    snapshot.assert_match(ed.min_vel_event_list_["right_sensor"], "strides_right")
    snapshot.assert_match(spatial_paras.parameters_pretty_["right_sensor"], "spatial_paras_right")
    snapshot.assert_match(temporal_paras.parameters_pretty_["right_sensor"], "temporal_paras_right")
    snapshot.assert_match(spatial_paras.parameters_pretty_["left_sensor"], "spatial_paras_left")
    snapshot.assert_match(temporal_paras.parameters_pretty_["left_sensor"], "temporal_paras_left")


def test_multi_process():
    """Test the multiprocess example.

    We do not test the multi process example.
    Unfortunately, it is somehow not possible to execute something that uses multiprocessing in pytest.
    """
