from importlib.util import find_spec


def patch_gaitmap_mad_import(_gaitmap_mad_modules):
    if bool(find_spec("gaitmap_mad")):
        return None
    from gaitmap.utils.exceptions import GaitmapMadImportError
    def new_getattr(name: str):
        if name in _gaitmap_mad_modules:
            raise GaitmapMadImportError(name, __name__)
        return globals()[name]

    return new_getattr
