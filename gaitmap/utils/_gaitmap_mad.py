"""Helper functions to handle the gaitmap/gaitmap_mad split."""
from importlib.util import find_spec

import gaitmap


def patch_gaitmap_mad_import(_gaitmap_mad_modules, current_module_name):
    """Check if the gaitmap_mad module is available and return a patched getattr method if not."""
    if find_spec("gaitmap_mad"):
        import gaitmap_mad  # pylint: disable=import-outside-toplevel

        assert (gm_version := gaitmap_mad.__version__) == (g_version := gaitmap.__version__), (
            "We only support using the exact same version of `gaitmap` and `gaitmap_mad`. "
            f"Currently you have the versions `gaitmap`: v{g_version} and `gaitmap_mad`: v{gm_version}."
        )
        return None
    from gaitmap.utils.exceptions import GaitmapMadImportError  # pylint: disable=import-outside-toplevel

    def new_getattr(name: str):
        if name in _gaitmap_mad_modules:
            raise GaitmapMadImportError(name, current_module_name)
        return globals()[name]

    return new_getattr
