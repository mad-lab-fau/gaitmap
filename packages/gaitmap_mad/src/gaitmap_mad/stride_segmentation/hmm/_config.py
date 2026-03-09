"""Configuration objects for composite HMM segmentation models."""

from typing import Union

from tpcp import cf
from typing_extensions import Literal

from gaitmap.base import _BaseSerializable


def _default_modules() -> tuple["HmmSubModelConfig", ...]:
    return (
        HmmSubModelConfig(
            name="transition",
            role="transition",
            n_states=5,
            n_gmm_components=3,
            architecture="left-right-loose",
            algo_train="baum-welch",
            stop_threshold=1e-9,
            max_iterations=10,
        ),
        HmmSubModelConfig(
            name="stride",
            role="stride",
            n_states=20,
            n_gmm_components=6,
            architecture="left-right-strict",
            algo_train="baum-welch",
            stop_threshold=1e-9,
            max_iterations=10,
        ),
    )


class HmmSubModelConfig(_BaseSerializable):
    """Configuration of a single trainable HMM submodule."""

    name: str
    role: Union[Literal["transition", "stride"], str]
    n_states: int
    n_gmm_components: int
    architecture: Literal["left-right-strict", "left-right-loose", "fully-connected"]
    algo_train: Literal["viterbi", "baum-welch", "labeled"]
    stop_threshold: float
    max_iterations: int
    verbose: bool
    n_jobs: int

    def __init__(
        self,
        name: str,
        role: Union[Literal["transition", "stride"], str],
        n_states: int,
        n_gmm_components: int,
        *,
        architecture: Literal["left-right-strict", "left-right-loose", "fully-connected"] = "left-right-strict",
        algo_train: Literal["viterbi", "baum-welch", "labeled"] = "baum-welch",
        stop_threshold: float = 1e-9,
        max_iterations: int = 10,
        verbose: bool = True,
        n_jobs: int = 1,
    ) -> None:
        self.name = name
        self.role = role
        self.n_states = n_states
        self.n_gmm_components = n_gmm_components
        self.architecture = architecture
        self.algo_train = algo_train
        self.stop_threshold = stop_threshold
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.n_jobs = n_jobs


class CompositeHmmConfig(_BaseSerializable):
    """Configuration of a composite HMM with named submodules."""

    modules: tuple[HmmSubModelConfig, ...]
    transition_model_name: str

    def __init__(
        self,
        modules: tuple[HmmSubModelConfig, ...] = cf(_default_modules()),
        *,
        transition_model_name: str = "transition",
    ) -> None:
        self.modules = modules
        self.transition_model_name = transition_model_name

    @property
    def _module_configs_by_name(self) -> dict[str, HmmSubModelConfig]:
        modules_by_name = {module.name: module for module in self.modules}
        if len(modules_by_name) != len(self.modules):
            raise ValueError("All configured HMM submodule names must be unique.")
        return modules_by_name

    def get_module(self, module_name: str) -> HmmSubModelConfig:
        """Return the config of a single named submodule."""
        try:
            return self._module_configs_by_name[module_name]
        except KeyError as e:
            raise ValueError(f"No HMM submodule with the name `{module_name}` exists in the model config.") from e

    @property
    def explicit_region_model_names(self) -> tuple[str, ...]:
        """Return all explicit region model names except the implicit transition module."""
        return tuple(module.name for module in self.modules if module.name != self.transition_model_name)

    @property
    def transition_model(self) -> HmmSubModelConfig:
        """Return the configuration of the implicit transition module."""
        transition_model = self.get_module(self.transition_model_name)
        if transition_model.role != "transition":
            raise ValueError(
                "The configured transition model is expected to have the role `transition`, "
                f"but `{transition_model.name}` has the role `{transition_model.role}`."
            )
        return transition_model

    @property
    def stride_model_names(self) -> tuple[str, ...]:
        """Return the names of all modules that should be interpreted as stride-like states."""
        return tuple(module.name for module in self.modules if module.role == "stride")

    @property
    def custom_model_names(self) -> tuple[str, ...]:
        """Return the names of all modules with custom non-built-in roles."""
        return tuple(module.name for module in self.modules if module.role not in {"transition", "stride"})

    def get_module_role(self, module_name: str) -> str:
        """Return the configured role of a named module."""
        return self.get_module(module_name).role
