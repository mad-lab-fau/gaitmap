import matplotlib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from gaitmap.utils.consts import SF_ACC
from tests.conftest import compare_algo_objects

# This is needed to avoid plots to open
matplotlib.use("Agg")


def test_base_dtw_generic(snapshot):
    from examples.generic_algorithms.base_dtw_generic import dtw

    assert len(dtw.matches_start_end_) == 5
    snapshot.assert_match(dtw.matches_start_end_)


def test_barth_dtw_example(snapshot):
    from examples.stride_segmentation.barth_dtw_stride_segmentation import dtw

    assert len(dtw.matches_start_end_["left_sensor"]) == 28
    snapshot.assert_match(dtw.matches_start_end_["left_sensor"])


def test_constrained_barth_dtw_example(snapshot):
    from examples.stride_segmentation.constrained_barth_dtw_stride_segmentation import cdtw, default_cdtw, dtw

    assert len(dtw.matches_start_end_["left_sensor"]) == 74
    snapshot.assert_match(dtw.matches_start_end_["left_sensor"], "dtw")

    assert len(cdtw.matches_start_end_["left_sensor"]) == 75
    snapshot.assert_match(cdtw.matches_start_end_["left_sensor"], "cdtw")

    assert len(default_cdtw.matches_start_end_["left_sensor"]) == 75
    snapshot.assert_match(default_cdtw.matches_start_end_["left_sensor"], "default_cdtw")


def test_roi(snapshot):
    from examples.stride_segmentation.barth_dtw_stride_segmentation_roi import roi_seg

    snapshot.assert_match(roi_seg.stride_list_["left_sensor"])


def test_preprocessing_example(snapshot):
    from examples.preprocessing.manual_sensor_alignment import dataset_sf_aligned_to_gravity

    desired_acc_vec = np.array([0.0, 0.0, 9.81])

    # check if at least first 5 samples of left- and right-sensor are correctly aligned to gravity
    left_acc = dataset_sf_aligned_to_gravity["left_sensor"][SF_ACC].to_numpy()[:5, :]
    right_acc = dataset_sf_aligned_to_gravity["right_sensor"][SF_ACC].to_numpy()[:5, :]

    for acc_l, acc_r in zip(left_acc, right_acc):
        np.testing.assert_almost_equal(acc_l, desired_acc_vec, decimal=0)
        np.testing.assert_almost_equal(acc_r, desired_acc_vec, decimal=0)

    # just check first 1000 rows to make sure that snapshot stays in a kB range
    snapshot.assert_match(dataset_sf_aligned_to_gravity.to_numpy()[:1000])


def test_sensor_alignment_detailed_example(snapshot):
    from examples.preprocessing.automatic_sensor_alignment_details import (
        forward_aligned_data,
        gravity_aligned_data,
        pca_aligned_data,
    )

    desired_acc_vec = np.array([0.0, 0.0, 9.81])

    # check if at least first 5 samples of left- and right-sensor are correctly aligned to gravity
    left_acc = pca_aligned_data["left_sensor"][SF_ACC].to_numpy()[:5, :]
    right_acc = pca_aligned_data["right_sensor"][SF_ACC].to_numpy()[:5, :]

    for acc_l, acc_r in zip(left_acc, right_acc):
        np.testing.assert_almost_equal(acc_l, desired_acc_vec, decimal=0)
        np.testing.assert_almost_equal(acc_r, desired_acc_vec, decimal=0)

    # just check first 1000 rows to make sure that snapshot stays in a kB range
    for sensor in ["left_sensor", "right_sensor"]:
        snapshot.assert_match(gravity_aligned_data[sensor].to_numpy()[:1000])
        snapshot.assert_match(pca_aligned_data[sensor].to_numpy()[:1000])
        snapshot.assert_match(forward_aligned_data[sensor].to_numpy()[:1000])


def test_sensor_alignment_detailed_simple(snapshot):
    from examples.preprocessing.automatic_sensor_alignment_details import (
        forward_aligned_data,
        gravity_aligned_data,
        pca_aligned_data,
    )

    # just check first 1000 rows to make sure that snapshot stays in a kB range
    for sensor in ["left_sensor", "right_sensor"]:
        snapshot.assert_match(gravity_aligned_data[sensor].to_numpy()[:1000])
        snapshot.assert_match(pca_aligned_data[sensor].to_numpy()[:1000])
        snapshot.assert_match(forward_aligned_data[sensor].to_numpy()[:1000])


def test_temporal_parameters(snapshot):
    from examples.parameters.temporal_parameters import p

    snapshot.assert_match(p.parameters_["left_sensor"])


def test_spatial_parameters(snapshot):
    from examples.parameters.spatial_parameters import p

    snapshot.assert_match(p.parameters_["left_sensor"])


def test_rampp_event_detection(snapshot):
    from examples.event_detection.rampp_event_detection import ed

    assert len(ed.min_vel_event_list_["left_sensor"]) == 26
    assert len(ed.min_vel_event_list_["right_sensor"]) == 29
    snapshot.assert_match(ed.min_vel_event_list_["left_sensor"])
    snapshot.assert_match(ed.min_vel_event_list_["right_sensor"])


def test_json_example(snapshot):
    from examples.advanced_features.algo_serialize import json_str, loaded_slt, slt

    snapshot.assert_match(json_str)

    compare_algo_objects(slt, loaded_slt)


def test_trajectory_reconstruction(snapshot):
    from examples.trajectory_reconstruction.trajectory_reconstruction import trajectory

    # just look at last values to see if final result is correct and save runtime
    snapshot.assert_match(trajectory.position_["left_sensor"].tail(20))
    snapshot.assert_match(trajectory.orientation_["left_sensor"].tail(20))


def test_region_trajectory_reconstruction(snapshot):
    from examples.trajectory_reconstruction.trajectory_reconstruction_region import (
        trajectory_full,
        trajectory_per_stride,
    )

    # look at some random values in the center to test
    snapshot.assert_match(trajectory_full.position_["left_sensor"].iloc[5000:5020])
    snapshot.assert_match(trajectory_full.orientation_["left_sensor"].iloc[5000:5020])

    snapshot.assert_match(trajectory_per_stride.position_["left_sensor"].loc[4].tail(20))
    snapshot.assert_match(trajectory_per_stride.orientation_["left_sensor"].loc[4].tail(20))


def test_mad_pipeline(snapshot):
    from examples.full_pipelines.mad_gait_pipeline import ed, spatial_paras, temporal_paras

    snapshot.assert_match(ed.min_vel_event_list_["left_sensor"], "strides_left")
    snapshot.assert_match(ed.min_vel_event_list_["right_sensor"], "strides_right")
    snapshot.assert_match(spatial_paras.parameters_pretty_["right_sensor"], "spatial_paras_right")
    snapshot.assert_match(temporal_paras.parameters_pretty_["right_sensor"], "temporal_paras_right")
    snapshot.assert_match(spatial_paras.parameters_pretty_["left_sensor"], "spatial_paras_left")
    snapshot.assert_match(temporal_paras.parameters_pretty_["left_sensor"], "temporal_paras_left")


def test_ullrich_gait_sequence_detection(snapshot):
    from examples.gait_detection.ullrich_gait_sequence_detection import gsd

    assert len(gsd.gait_sequences_) == 2
    snapshot.assert_match(gsd.gait_sequences_.astype(np.int64))


def test_caching(snapshot):
    from examples.advanced_features.caching import first_call_results, second_call_results

    # We will not store the actual ouputs, but just check if they are actually idential
    for sensor, s_list in first_call_results.stride_list_.items():
        assert_frame_equal(s_list, second_call_results.stride_list_[sensor])


def test_custom_dataset():
    # There is not really anything specific, we want to test here, so we just run everything and check that there are
    # no errors.
    import examples.datasets_and_pipelines.custom_dataset  # noqa


def test_grid_search(snapshot):
    from examples.datasets_and_pipelines.gridsearch import results, segmented_stride_list

    snapshot.assert_match(segmented_stride_list, check_dtype=False)
    snapshot.assert_match(pd.DataFrame(results).drop("score_time", axis=1), check_dtype=False)


def test_optimizable_pipelines(snapshot):
    from examples.datasets_and_pipelines.optimizable_pipelines import optimized_results, results

    snapshot.assert_match(results.segmented_stride_list_, check_dtype=False)
    snapshot.assert_match(optimized_results.segmented_stride_list_, check_dtype=False)
    snapshot.assert_match(optimized_results.template.get_data())


def test_cross_validation(snapshot):
    from examples.datasets_and_pipelines.cross_validation import result_df

    result_df = result_df.drop(["score_time", "optimize_time", "optimizer"], axis=1)
    snapshot.assert_match(result_df, check_dtype=False)


def test_gridsearch_cv(snapshot):
    from examples.datasets_and_pipelines.gridsearch_cv import cached_results, results_df

    ignore_cols = ["mean_score_time", "mean_optimize_time", "std_optimize_time", "std_score_time"]

    results_df = results_df.drop(ignore_cols, axis=1)
    cached_results = pd.DataFrame(cached_results)
    cached_results = cached_results.drop(ignore_cols, axis=1)
    pd.testing.assert_frame_equal(cached_results, results_df)

    snapshot.assert_match(results_df, check_dtype=False)


def test_multi_process():
    """Test the multiprocess example.

    We do not test the multi process example.
    Unfortunately, it is somehow not possible to execute something that uses multiprocessing in pytest.
    """
