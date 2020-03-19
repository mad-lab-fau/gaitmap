import inspect
from typing import Callable, Dict, TypeVar, Type, Any, List

BaseType = TypeVar("BaseType", bound="BaseAlgorithms")


class BaseAlgorithm:
    _action_method: str

    @property
    def _action_is_applied(self) -> bool:
        if len(self.get_attributes()) == 0:
            return True
        return False

    def _get_action_method(self) -> Callable:
        return getattr(self, self._action_method)

    @classmethod
    def _get_param_names(cls) -> List[str]:
        """Get parameter names for the estimator.

        The parameters of an algorithm are defined based on its `__init__` method.
        All parameters of this method are considered parameters of the algorithm.

        Notes
        -----
        Adopted based on `sklearn BaseEstimator._get_param_names`.

        Returns
        -------
        param_names
            List of parameter names of the algorithm

        """
        # fetch the constructor or the original constructor before deprecation wrapping if any
        init = cls.__init__
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values() if p.name != "self" and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "gaitmap-algorithms should always specify their parameters in the signature of their "
                    "__init__ (no  varargs). {} with constructor {} doesn't  follow this convention.".format(
                        cls, init_signature
                    )
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self) -> Dict[str, Any]:
        """Get parameters for this algorithm.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.

        """
        return {k: getattr(self, k) for k in self._get_param_names()}

    def set_params(self: BaseType, **params: Dict[str, Any]) -> Type[BaseType]:
        pass

    def get_other_params(self) -> Dict[str, Any]:
        params = self.get_params()
        attrs = {
            v: getattr(self, v)
            for v in vars(self)
            if not v.endswith("_") and not v.startswith("_") and v not in params
        }
        return attrs

    def get_attributes(self) -> Dict[str, Any]:
        attrs = {v: getattr(self, v) for v in vars(self) if v.endswith("_") and not v.startswith("__")}
        return attrs
