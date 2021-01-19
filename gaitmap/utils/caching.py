"""Some helper to avoid making repeated calculations by using `joblib` to cache algorithm results."""
from joblib import Memory

from gaitmap.base import BaseAlgorithm


def cached_call_method(_algo: BaseAlgorithm, _method_name: str, _memory: Memory, *args, **kwargs):
    """Call a method on the algo object and cache the output.

    Repeated calls to this function with the same algorithm and the same args, and kwargs, will return cached results
    saved on disk.

    .. warning ::
        This method will clone the algorithm object before calling the method.
        This ensures that the cache is not invalidated because of results stored on the object.

    Parameters
    ----------
    _algo
        The algorithm instance to use
    _method_name
        The name of the method to call
    _memory
        A instance of `joblib.memory` used for caching
    args
        Positional arguments passed to the called method
    kwargs
        Keyword arguments passed to the called method.

    Returns
    -------
    method_return
        The return value of the called methods either calculated or cached.

    See Also
    --------
    gaitmap.utils.caching.cached_call_action

    """

    def _call_method(_algo, _method_name, *args, **kwargs):
        return getattr(_algo, _method_name)(*args, **kwargs)

    _algo = _algo.clone()
    return _memory.cache(_call_method)(_algo, _method_name, *args, **kwargs)


def cached_call_action(_algo: BaseAlgorithm, _memory: Memory, *args, **kwargs):
    """Call the action method on the algo object and cache the output.

    Repeated calls to this function with the same algorithm and the same args, and kwargs, will return cached results
    saved on disk.

    .. warning ::
        This method will clone the algorithm object before calling the method.
        This ensures that the cache is not invalidated because of results stored on the object.

    Parameters
    ----------
    _algo
        The algorithm instance to use
    _method_name
        The name of the method to call
    _memory
        A instance of `joblib.memory` used for caching
    args
        Positional arguments passed to the called method
    kwargs
        Keyword arguments passed to the called method.

    Returns
    -------
    method_return
        The return value of the called methods either calculated or cached.

    Notes
    -----
    In case the algorithm object has multiple action methods, use :func:`~gaitmap.utils.caching.cached_call_method`
    to explicitly call the one you want.

    See Also
    --------
    gaitmap.utils.caching.cached_call_method

    """
    return cached_call_method(_algo, _algo._action_method, _memory, *args, **kwargs)
