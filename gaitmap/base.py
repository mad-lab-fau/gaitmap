from typing import Callable, Dict, TypeVar, Type, Any

BASE_TYPE = TypeVar("BASE_TYPE", bound="BaseAlgorithms")


class BaseAlgorithm:
    _action_method: str

    def _get_action_method(self) -> Callable:
        return getattr(self, self._action_method)

    def get_params(self) -> Dict[str, Any]:
        pass

    def set_params(self: BASE_TYPE, **params: Dict[str, Any]) -> Type[BASE_TYPE]:
        pass

    def get_other_params(self) -> Dict[str, Any]:
        pass

    def get_attributes(self) -> Dict[str, Any]:
        pass

    @property
    def _action_is_applied(self) -> bool:
        if len(self.get_attributes()):
            return True
        return False
