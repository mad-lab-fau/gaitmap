"""A set of custom exceptions."""


class ValidationError(Exception):
    """An error indicating that data-object does not comply with the guidelines."""


class GaitmapMadImportError(ImportError):
    """An error indicating that the algorithm is implemented in gaitmap-mad and not gaitmap."""

    def __init__(self, object_name: str, module_name: str) -> None:
        self.object_name = object_name
        self.module_name = module_name
        super().__init__()

    def __str__(self) -> str:
        """Return a string representation of the error."""
        return (
            f"You are trying to import {self.object_name} from {self.module_name}."
            "\n\n"
            "This class/function is only available via the `gaitmap_mad` package that needs to be installed "
            "separately.\n"
            "Install it using TODO: Update instructions.\n\n"
            "Warning: `gaitmap_mad` has a different license (AGPL3) than gaitmap. "
            "Make sure you understand what this means, before you proceed! "
            "See the README of gaitmap for more info."
        )
