from typing import Callable, Dict, TypeVar, Type, Any

BaseType = TypeVar("BaseType", bound="BaseAlgorithms")


class BaseAlgorithm:
    _action_method: str

    def _get_action_method(self) -> Callable:
        return getattr(self, self._action_method)

    def get_params(self) -> Dict[str, Any]:
        pass

    def set_params(self: BaseType, **params: Dict[str, Any]) -> Type[BaseType]:
        pass

    def get_other_params(self) -> Dict[str, Any]:
        pass

    def get_attributes(self) -> Dict[str, Any]:
        attrs = {v: getattr(self, v) for v in vars(self) if v.endswith("_") and not v.startswith("__")}
        return attrs

    @property
    def _action_is_applied(self) -> bool:
        if len(self.get_attributes()) == 0:
            return True
        return False
