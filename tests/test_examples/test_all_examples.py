import matplotlib

# This is needed to avoid plots to open
matplotlib.use("Agg")


def test_base_dtw_generic(snapshot):
    from examples.base_dtw_generic import dtw

    assert len(dtw.matches_start_end_) == 5
    # snapshot.assert_match(dtw.matches_start_end_)
