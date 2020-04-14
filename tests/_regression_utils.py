"""Utils to perform snapshot tests easily.

This is inspired by github.com/syrusakbary/snapshottest.
Note that it can not be used in combination with this module!
"""
import re
from pathlib import Path
import pandas as pd


import pytest
from pandas._testing import assert_frame_equal


def pytest_addoption(parser):
    group = parser.getgroup("snapshottest")
    group.addoption(
        "--snapshot-update", action="store_true", default=False, dest="snapshot_update", help="Update the snapshots."
    )


class SnapshotNotFound(Exception):
    pass


class PyTestSnapshotTest:
    def __init__(self, request=None):
        self.request = request
        self.curr_snapshot_number = 0
        super(PyTestSnapshotTest, self).__init__()

    @property
    def update(self):
        return self.request.config.option.snapshot_update

    @property
    def module(self):
        return Path(self.request.node.fspath.strpath).parent

    @property
    def snapshot_folder(self):
        return self.module / "snapshot"

    @property
    def file_name(self):
        return self.snapshot_folder / "{}.json".format(self.test_name)

    @property
    def test_name(self):
        cls_name = getattr(self.request.node.cls, "__name__", "")
        flattened_node_name = re.sub(r"\s+", " ", self.request.node.name.replace(r"\n", " "))
        return "{}{}_{}".format("{}.".format(cls_name) if cls_name else "", flattened_node_name, self.curr_snapshot)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def store(self, value):
        self.snapshot_folder.mkdir(parents=True, exist_ok=True)
        if isinstance(value, pd.DataFrame):
            value.to_json(self.file_name, indent=4)

    def retrieve(self, dtype):
        if not self.file_name.is_file():
            raise SnapshotNotFound()

        if dtype == pd.DataFrame:
            return pd.read_json(self.file_name)
        else:
            raise ValueError("The dtype {} is not supported for snapshot testing".format(dtype))

    def assert_match(self, value, name=""):
        self.curr_snapshot = name or self.curr_snapshot_number
        if self.update:
            self.store(value)
        else:
            value_dtype = type(value)
            try:
                prev_snapshot = self.retrieve(value_dtype)
            except SnapshotNotFound:
                self.store(value)  # first time this test has been seen
            except:
                raise
            else:
                if value_dtype == pd.DataFrame:
                    assert_frame_equal(value, prev_snapshot)
        self.curr_snapshot_number += 1
