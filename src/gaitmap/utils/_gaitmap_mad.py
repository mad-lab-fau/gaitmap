"""Helper functions to handle the gaitmap/gaitmap_mad split."""

from importlib import import_module

import gaitmap
from gaitmap.utils.exceptions import GaitmapMadImportError


def patch_gaitmap_mad_import(_gaitmap_mad_modules, current_module_name):
    """Check if the gaitmap_mad module is available and return a patched getattr method if not."""
    try:
        gaitmap_mad = import_module("gaitmap_mad")
    except ImportError:
        gaitmap_mad = None

    if gaitmap_mad is not None:
        if (gm_version := gaitmap_mad.__version__) != (g_version := gaitmap.__version__):
            raise ImportError(
                "We only support using the exact same version of `gaitmap` and `gaitmap_mad`. "
                f"Currently you have the versions `gaitmap`: v{g_version} and `gaitmap_mad`: v{gm_version}. "
                "Update the `gaitmap` and `gaitmap_mad` packages to the same version (likely you just forgot to update "
                "`gaitmap_mad` when you updated `gaitmap`)."
            )
        return None

    def new_getattr(name: str):
        if name in _gaitmap_mad_modules:
            raise GaitmapMadImportError(name, current_module_name)
        return globals()[name]

    return new_getattr
