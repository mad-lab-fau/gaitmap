from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def import_legacy_hmm_backend():
    return pytest.importorskip("gaitmap_mad.stride_segmentation.hmm.legacy")


def import_modern_hmm_backend():
    return pytest.importorskip("gaitmap_mad.stride_segmentation.hmm.modern")


def require_trainable_hmm_backend():
    try:
        return import_modern_hmm_backend()
    except pytest.skip.Exception:
        return import_legacy_hmm_backend()


def get_pretrained_inference_snapshot_path(sensor: str) -> Path:
    return (
        Path(__file__).resolve().parent
        / "test_examples"
        / "snapshot"
        / f"test_roth_hmm_stride_segmentation_{sensor}.json"
    )


def load_pretrained_inference_stride_list_snapshot(sensor: str) -> pd.DataFrame:
    return pd.read_json(get_pretrained_inference_snapshot_path(sensor), orient="table")
