"""A set of custom exceptions."""


class ValidationError(Exception):
    """An error indicating that data-object does not comply with the guidelines."""


class PotentialUserErrorWarning(UserWarning):
    """A warning indicating that the user might not use certain gaitmap features correctly."""
