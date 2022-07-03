import importlib
import sys

import pytest

from gaitmap.utils.exceptions import GaitmapMadImportError


@pytest.fixture()
def _gaitmap_mad_sys_modifier():
    # For the purpose of this test, we can simulate that gaitmap_mad is not installed bz setting its sys.modules
    # entry to None.

    # This import will force gaitmap_mad to be in sys.modules.
    import gaitmap_mad  # noqa: unused-import

    sys.modules["gaitmap_mad"] = None
    yield
    sys.modules.pop("gaitmap_mad")
    import gaitmap_mad  # noqa: unused-import

    # We just go overboard to be save and reimport all gaitmap modules after the cleanup.
    modules_to_reload = []
    for module in sys.modules.values():
        if (spec := getattr(module, "__spec__", None)) and spec.name.startswith("gaitmap"):
            modules_to_reload.append(module)
    # We do that in two steps to avoid changing sys.modules while iterating over it.
    for module in modules_to_reload:
        importlib.reload(module)


def test_raises_error_gaitmap_mad_not_installed(_gaitmap_mad_sys_modifier):
    # First we need to remove gaitmap_mad from sys.modules, so that it is not imported.

    with pytest.raises(GaitmapMadImportError) as e:
        from gaitmap.stride_segmentation import BarthDtw  # noqa: unused-import


    assert e.value.object_name == "BarthDtw"
    assert e.value.module_name == "gaitmap.stride_segmentation"
    assert "BarthDtw" in str(e.value)
    assert "gaitmap.stride_segmentation" in str(e.value)


def test_raises_no_error_gaitmap_mad_installed():
    from gaitmap.stride_segmentation import BarthDtw  # noqa: unused-import

    assert True
