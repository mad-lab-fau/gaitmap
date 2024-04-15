"""Utils to perform snapshot tests easily.

This is inspired by github.com/syrusakbary/snapshottest.
Note that it can not be used in combination with this module!
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal


def pytest_addoption(parser) -> None:
    group = parser.getgroup("snapshottest")
    group.addoption(
        "--snapshot-update", action="store_true", default=False, dest="snapshot_update", help="Update the snapshots."
    )


class SnapshotNotFound(Exception):
    pass


class PyTestSnapshotTest:
    def __init__(self, request=None) -> None:
        self.request = request
        self.curr_snapshot_number = 0
        super().__init__()

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
    def file_name_json(self):
        return self.snapshot_folder / f"{self.test_name}.json"

    @property
    def file_name_csv(self):
        return self.snapshot_folder / f"{self.test_name}.csv"

    @property
    def file_name_txt(self):
        return self.snapshot_folder / f"{self.test_name}.txt"

    @property
    def test_name(self):
        cls_name = getattr(self.request.node.cls, "__name__", "")
        flattened_node_name = re.sub(r"\s+", " ", self.request.node.name.replace(r"\n", " "))
        return "{}{}_{}".format(f"{cls_name}." if cls_name else "", flattened_node_name, self.curr_snapshot)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def store(self, value) -> None:
        self.snapshot_folder.mkdir(parents=True, exist_ok=True)
        if isinstance(value, pd.DataFrame):
            value.to_json(self.file_name_json, indent=4, orient="table")
        elif isinstance(value, np.ndarray):
            np.savetxt(self.file_name_csv, value, delimiter=",")
        elif isinstance(value, str):
            with open(self.file_name_txt, "w") as f:
                f.write(value)
        else:
            raise TypeError(f"The dtype {type(value)} is not supported for snapshot testing")

    def retrieve(self, dtype):
        if dtype == pd.DataFrame:
            filename = self.file_name_json
            if not filename.is_file():
                raise SnapshotNotFound()
            return pd.read_json(filename, orient="table")
        elif dtype == np.ndarray:
            filename = self.file_name_csv
            if not filename.is_file():
                raise SnapshotNotFound()
            return np.genfromtxt(filename, delimiter=",")
        elif dtype == str:
            filename = self.file_name_txt
            if not filename.is_file():
                raise SnapshotNotFound()
            with open(self.file_name_txt) as f:
                value = f.read()
            return value
        else:
            raise ValueError(f"The dtype {dtype} is not supported for snapshot testing")

    def assert_match(self, value, name="", **kwargs) -> None:
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
                    assert_frame_equal(value, prev_snapshot, **kwargs)
                elif value_dtype == np.ndarray:
                    np.testing.assert_array_almost_equal(value, prev_snapshot, **kwargs)
                elif value_dtype == str:
                    # Display the string diff line by line as part of error message using difflib
                    import difflib

                    diff = difflib.ndiff(value.splitlines(keepends=True), prev_snapshot.splitlines(keepends=True))
                    diff = "".join(diff)
                    assert value == prev_snapshot, diff
                else:
                    raise ValueError(f"The dtype {value_dtype} is not supported for snapshot testing")

        self.curr_snapshot_number += 1
