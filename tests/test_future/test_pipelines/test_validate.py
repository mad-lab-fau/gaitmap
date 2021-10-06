from unittest.mock import call, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from gaitmap.future.pipelines import Optimize, cross_validate
from tests.test_future.test_pipelines.conftest import DummyDataset, DummyPipeline, dummy_single_score_func


class TestCrossValidate:
    def test_optimize_called(self):
        """Test that optimize of the pipeline is called correctly."""
        ds = DummyDataset()
        pipeline = DummyPipeline()

        # The we use len(ds) splits, effectively a leave one our CV for testing.
        cv = KFold(n_splits=len(ds))
        train, test = zip(*cv.split(ds))
        with patch.object(DummyPipeline, "self_optimize", return_value=pipeline) as mock:
            cross_validate(Optimize(pipeline), ds, cv=cv, scoring=lambda x, y: 1)

        assert mock.call_count == len(train)
        for expected, actual in zip(train, mock.call_args_list):
            pd.testing.assert_frame_equal(ds[expected].index, actual[0][0].index)

    def test_run_called(self):
        """Test that optimize of the pipeline is called correctly."""

        def scoring(pipe, ds):
            pipe.run(ds)
            return 1

        ds = DummyDataset()
        pipeline = DummyPipeline()

        # We want to have two datapoints in the test set sometimes
        cv = KFold(n_splits=len(ds) // 2)
        train, test = zip(*cv.split(ds))
        with patch.object(DummyPipeline, "run", return_value=pipeline) as mock:
            cross_validate(Optimize(pipeline), ds, cv=cv, scoring=scoring)

        test_flat = [t for split in test for t in split]
        assert mock.call_count == len(test_flat)
        for expected, actual in zip(test_flat, mock.call_args_list):
            pd.testing.assert_frame_equal(ds[expected].index, actual[0][0].index)

    def test_single_score(self):
        ds = DummyDataset()
        # The we use len(ds) splits, effectively a leave one our CV for testing.
        cv = KFold(n_splits=len(ds))

        results = cross_validate(
            Optimize(DummyPipeline()), ds, scoring=dummy_single_score_func, cv=cv, return_train_score=True
        )
        results_df = pd.DataFrame(results)

        assert len(results_df) == 5  # n folds
        assert set(results.keys()) == {
            "train_data_labels",
            "test_data_labels",
            "test_score",
            "test_single_score",
            "train_score",
            "train_single_score",
            "score_time",
            "optimize_time",
        }
        assert all(len(v) == len(ds) - 1 for v in results_df["train_data_labels"])
        assert all(len(v) == len(ds) - 1 for v in results_df["train_single_score"])
        assert all(len(v) == 1 for v in results_df["test_data_labels"])
        assert all(len(v) == 1 for v in results_df["test_single_score"])
        # The dummy scorer is returning the dataset group id -> The datapoint id is also the result
        for i, r in results_df.iterrows():
            all_ids = ds.groups
            assert r["test_data_labels"] == [i]
            assert r["test_data_labels"] == r["test_single_score"]
            assert r["test_score"] == i
            all_ids.remove(i)
            assert r["train_data_labels"] == all_ids
            assert all(r["train_data_labels"] == r["train_single_score"])
            assert r["train_score"] == np.mean(all_ids)

    @pytest.mark.parametrize(
        "kwargs,expected",
        (
            ({"return_optimizer": True}, ("optimizer",)),
            ({"return_train_score": True}, ("train_score", "train_single_score")),
        ),
    )
    def test_return_elements(self, kwargs, expected):
        results = cross_validate(Optimize(DummyPipeline()), DummyDataset(), scoring=dummy_single_score_func)
        results_additionally = cross_validate(
            Optimize(DummyPipeline()), DummyDataset(), scoring=dummy_single_score_func, **kwargs
        )

        assert set(results_additionally.keys()) - set(results.keys()) == set(expected)
