import importlib
from pathlib import Path
from types import SimpleNamespace

from gaitmap_mad.stride_segmentation.hmm import _backend_base as backend_base


class _FakeLegacyBackend(backend_base.BaseHmmBackend):
    def __init__(self) -> None:
        backend_base.BaseHmmBackend.__init__(self, backend_id="pomegranate-legacy")


class _FakeModernBackend(backend_base.BaseHmmBackend):
    def __init__(self) -> None:
        backend_base.BaseHmmBackend.__init__(self, backend_id="pomegranate-modern")
        self.inference_implementation = "native"


class _FakeScipyBackend(backend_base.BaseHmmBackend):
    def __init__(self) -> None:
        backend_base.BaseHmmBackend.__init__(self, backend_id="scipy-inference")


def _get_default_backend(monkeypatch, *, modern_available: bool, legacy_available: bool):
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
    return backend_base.get_default_hmm_backend()


def test_default_backend_without_pomegranate(monkeypatch) -> None:
    backend = _get_default_backend(monkeypatch, modern_available=False, legacy_available=False)

    assert isinstance(backend, _FakeScipyBackend)


def test_default_backend_with_legacy_pomegranate(monkeypatch) -> None:
    backend = _get_default_backend(monkeypatch, modern_available=False, legacy_available=True)

    assert isinstance(backend, _FakeLegacyBackend)


def test_default_backend_with_modern_pomegranate(monkeypatch) -> None:
    backend = _get_default_backend(monkeypatch, modern_available=True, legacy_available=True)

    assert isinstance(backend, _FakeModernBackend)


def test_packaged_pretrained_model_uses_migrated_state_format() -> None:
    model_json = Path(
        "packages/gaitmap_mad/src/gaitmap_mad/stride_segmentation/hmm/_pre_trained_models/fallriskpd_at_lab_model.json"
    ).read_text(encoding="utf8")

    assert "SimpleHmm" not in model_json
    assert '"stride_model"' not in model_json
    assert '"transition_model"' not in model_json
