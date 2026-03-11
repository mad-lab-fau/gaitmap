import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from gaitmap_mad.stride_segmentation.hmm import _backend_base as backend_base
from gaitmap_mad.stride_segmentation.hmm import _segmentation_model as segmentation_model_module
from gaitmap_mad.stride_segmentation.hmm.legacy import PomegranateLegacyHmmBackend
from gaitmap_mad.stride_segmentation.hmm.modern import PomegranateModernHmmBackend
from gaitmap_mad.stride_segmentation.hmm.scipy import ScipyHmmInferenceBackend


@pytest.fixture(autouse=True)
def _restore_segmentation_model():
    yield
    importlib.reload(segmentation_model_module)


class _FakeLegacyBackend(PomegranateLegacyHmmBackend):
    def __init__(self) -> None:
        backend_base.BaseHmmBackend.__init__(self, backend_id="pomegranate-legacy")


class _FakeModernBackend(PomegranateModernHmmBackend):
    def __init__(self) -> None:
        backend_base.BaseHmmBackend.__init__(self, backend_id="pomegranate-modern")
        self.inference_implementation = "native"


class _FakeScipyBackend(ScipyHmmInferenceBackend):
    def __init__(self) -> None:
        backend_base.BaseHmmBackend.__init__(self, backend_id="scipy-inference")


def _reload_segmentation_model(monkeypatch, *, modern_available: bool, legacy_available: bool):
    def _fake_import_module(module_name: str):
        if module_name == "gaitmap_mad.stride_segmentation.hmm.modern":
            if not modern_available:
                raise ImportError("modern backend unavailable")
            return SimpleNamespace(PomegranateModernHmmBackend=_FakeModernBackend)
        if module_name == "gaitmap_mad.stride_segmentation.hmm.legacy":
            if not legacy_available:
                raise ImportError("legacy backend unavailable")
            return SimpleNamespace(PomegranateLegacyHmmBackend=_FakeLegacyBackend)
        if module_name == "gaitmap_mad.stride_segmentation.hmm.scipy":
            return SimpleNamespace(ScipyHmmInferenceBackend=_FakeScipyBackend)
        return importlib.import_module(module_name)

    monkeypatch.setattr(backend_base, "import_module", _fake_import_module)
    return importlib.reload(segmentation_model_module)


def test_default_backend_without_pomegranate(monkeypatch) -> None:
    segmentation_model = _reload_segmentation_model(monkeypatch, modern_available=False, legacy_available=False)

    assert isinstance(segmentation_model.DEFAULT_HMM_BACKEND, ScipyHmmInferenceBackend)
    assert isinstance(segmentation_model.RothSegmentationHmm().backend, ScipyHmmInferenceBackend)


def test_default_backend_with_legacy_pomegranate(monkeypatch) -> None:
    segmentation_model = _reload_segmentation_model(monkeypatch, modern_available=False, legacy_available=True)

    assert isinstance(segmentation_model.DEFAULT_HMM_BACKEND, PomegranateLegacyHmmBackend)
    assert isinstance(segmentation_model.RothSegmentationHmm().backend, PomegranateLegacyHmmBackend)


def test_default_backend_with_modern_pomegranate(monkeypatch) -> None:
    segmentation_model = _reload_segmentation_model(monkeypatch, modern_available=True, legacy_available=True)

    assert isinstance(segmentation_model.DEFAULT_HMM_BACKEND, PomegranateModernHmmBackend)
    assert isinstance(segmentation_model.RothSegmentationHmm().backend, PomegranateModernHmmBackend)


def test_packaged_pretrained_model_uses_migrated_state_format() -> None:
    model_json = Path(
        "packages/gaitmap_mad/src/gaitmap_mad/stride_segmentation/hmm/_pre_trained_models/fallriskpd_at_lab_model.json"
    ).read_text(encoding="utf8")

    assert "SimpleHmm" not in model_json
    assert '"stride_model"' not in model_json
    assert '"transition_model"' not in model_json
