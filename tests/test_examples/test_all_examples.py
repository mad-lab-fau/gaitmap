import matplotlib

# This is needed to avoid plots to open
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
