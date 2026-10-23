"""Shared interface for dataset-specific cross-validation challenges."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

from tpcp import BaseTpcpObject, Dataset
from tpcp.optimize import BaseOptimize


@dataclass(repr=False)
class BaseChallenge(BaseTpcpObject):
    """Run an optimizer on a prescribed dataset, split scheme, and scorer.

    Subclasses define the dataset and scoring protocol. Cross-validation results
    are available in ``cv_results_`` after calling ``run``. No global configuration
    or result storage is required.
    """

    VERSION: ClassVar[str]
    dataset_: Dataset = field(init=False)
    optimizer: BaseOptimize = field(init=False)
    cv_results_: dict = field(init=False)
    runtime_: float = field(init=False)
    run_start_datetime_: str = field(init=False)
    end_start_datetime_: str = field(init=False)
    run_start_datetime_utc_timestamp_: float = field(init=False)
    end_start_datetime_utc_timestamp_: float = field(init=False)

    @contextmanager
    def _measure_time(self):
        self.run_start_datetime_utc_timestamp_ = datetime.now(UTC).timestamp()
        self.run_start_datetime_ = datetime.now().astimezone().isoformat()
        start = time.perf_counter()
        yield
        self.end_start_datetime_utc_timestamp_ = datetime.now(UTC).timestamp()
        self.end_start_datetime_ = datetime.now().astimezone().isoformat()
        self.runtime_ = time.perf_counter() - start

    def _get_cv_iterator(self):
        if isinstance(self.cv_iterator, Iterator):
            return list(self.cv_iterator)
        return self.cv_iterator

    def run(self, optimizer: BaseOptimize):
        """Execute the challenge and store cross-validation results."""
        raise NotImplementedError


def resolve_dataset(dataset, dataset_type):
    """Accept a dataset instance or construct its loader from a data directory."""
    if isinstance(dataset, (str, Path)):
        return dataset_type(data_folder=Path(dataset))
    if isinstance(dataset, dataset_type):
        return dataset
    raise TypeError(f"dataset must be a path or an instance of {dataset_type.__name__}.")
