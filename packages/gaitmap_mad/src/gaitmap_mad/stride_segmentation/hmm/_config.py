"""Configuration objects for composite HMM segmentation models."""

from typing import Literal

from tpcp import cf

from gaitmap.base import _BaseSerializable


def _default_modules() -> dict[str, "HmmSubModelConfig"]:
    return {
        "transition": HmmSubModelConfig(
            name="transition",
            role="transition",
            n_states=5,
            n_gmm_components=3,
            architecture="left-right-loose",
            algo_train="baum-welch",
            stop_threshold=1e-9,
            max_iterations=10,
        ),
        "stride": HmmSubModelConfig(
            name="stride",
            role="stride",
            n_states=20,
            n_gmm_components=6,
            architecture="left-right-strict",
            algo_train="baum-welch",
            stop_threshold=1e-9,
            max_iterations=10,
        ),
    }


class HmmSubModelConfig(_BaseSerializable):
    """Configuration of a single trainable HMM submodule."""

    name: str
    role: Literal["transition", "stride", "other"]
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
        role: Literal["transition", "stride", "other"],
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HmmSubModelConfig):
            return False
        return self.get_params(deep=False) == other.get_params(deep=False)


class CompositeHmmConfig(_BaseSerializable):
    """Configuration of a composite HMM with named submodules."""

    modules: dict[str, HmmSubModelConfig]
    transition_model_name: str

    def __init__(
        self,
        modules: dict[str, HmmSubModelConfig] = cf(_default_modules()),
        *,
        transition_model_name: str = "transition",
    ) -> None:
        self.modules = modules
        self.transition_model_name = transition_model_name

    @property
    def explicit_region_model_names(self) -> tuple[str, ...]:
        """Return all module names except the implicit transition module."""
        return tuple(name for name in self.modules if name != self.transition_model_name)

    @property
    def transition_model(self) -> HmmSubModelConfig:
        """Return the configuration of the implicit transition module."""
        return self.modules[self.transition_model_name]

    @property
    def stride_model_names(self) -> tuple[str, ...]:
        """Return module names that should be interpreted as stride-like states."""
        return tuple(name for name, module in self.modules.items() if module.role == "stride")
