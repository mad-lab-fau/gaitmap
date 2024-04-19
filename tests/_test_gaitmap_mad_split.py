"""Testing the dynamic import of the gaitmap_mad module.

For some reason, importing this file in a normal testrun creates some issue with Pickling other classes.
Not sure why this is happening, but it seems to be related to the fact that I am wildly reloading and modifying
sys.modules.

Hence, these tests are excluded (Leading "_" in filename), but can be run manually.
"""

import importlib
import sys

import pytest

from gaitmap.utils.exceptions import GaitmapMadImportError


@pytest.fixture(scope="module")
def _gaitmap_mad_sys_modifier():
    # For the purpose of this test, we can simulate that gaitmap_mad is not installed bz setting its sys.modules
    # entry to None.

    # This import will force gaitmap_mad to be in sys.modules.
    import gaitmap_mad

    sys.modules["gaitmap_mad"] = None
    yield
    sys.modules.pop("gaitmap_mad")
    import gaitmap_mad  # noqa: F401, F811

    # We just go overboard to be save and reimport all gaitmap modules after the cleanup.
    modules_to_reload = []
    for module in sys.modules.values():
        if (spec := getattr(module, "__spec__", None)) and spec.name.startswith("gaitmap"):
            modules_to_reload.append(module)
    # We do that in two steps to avoid changing sys.modules while iterating over it.
    for module in modules_to_reload:
        importlib.reload(module)


@pytest.fixture(scope="module")
def _gaitmap_mad_change_version():
    import gaitmap_mad

    real_version = gaitmap_mad.__version__
    gaitmap_mad.__version__ = "0.0.0"
    yield
    gaitmap_mad.__version__ = real_version
    importlib.reload(gaitmap_mad)


def test_raises_error_gaitmap_mad_not_installed(_gaitmap_mad_sys_modifier) -> None:
    # First we need to remove gaitmap_mad from sys.modules, so that it is not imported.
    # We need to make sure that this is a fresh import:
    sys.modules.pop("gaitmap.stride_segmentation", None)
    with pytest.raises(GaitmapMadImportError) as e:
        from gaitmap.stride_segmentation import BarthDtw  # noqa: F401

    assert e.value.object_name == "BarthDtw"
    assert e.value.module_name == "gaitmap.stride_segmentation"
    assert "BarthDtw" in str(e.value)
    assert "gaitmap.stride_segmentation" in str(e.value)


def test_error_not_raised_when_importing_ori_methods(_gaitmap_mad_sys_modifier) -> None:
    # First we need to remove gaitmap_mad from sys.modules, so that it is not imported.
    # We need to make sure that this is a fresh import:
    sys.modules.pop("gaitmap.trajectory_reconstruction.orientation_methods", None)
    sys.modules.pop("gaitmap.trajectory_reconstruction", None)
    from gaitmap.trajectory_reconstruction import MadgwickAHRS  # noqa: F401


def test_gaitmap_mad_version_mismatch(_gaitmap_mad_change_version) -> None:
    # We need to make sure that this is a fresh import:
    sys.modules.pop("gaitmap.stride_segmentation", None)
    with pytest.raises(AssertionError):
        from gaitmap.stride_segmentation import BarthDtw  # noqa: F401


def test_raises_no_error_gaitmap_mad_installed() -> None:
    # We need to make sure that this is a fresh import:
    sys.modules.pop("gaitmap.stride_segmentation", None)
    from gaitmap.stride_segmentation import BarthDtw  # noqa: F401

    assert True
