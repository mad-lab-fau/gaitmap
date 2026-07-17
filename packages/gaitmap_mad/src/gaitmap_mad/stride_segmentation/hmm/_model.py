"""Backend-neutral hidden Markov model definitions and fitted parameters."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from tpcp import OptiPara

from gaitmap.base import _BaseSerializable

StateId = tuple[str, ...]


class HmmComposition(_BaseSerializable):
    """The named parts and routes used to construct a composite HMM."""

    _composite_params = ("parts",)

    parts: list[tuple[str, HmmModel]]
    routes: list[tuple[str, tuple[str, ...]]]
    starts: tuple[str, ...]
    ends: tuple[str, ...]

    def __init__(
        self,
        parts: list[tuple[str, HmmModel]],
        routes: list[tuple[str, tuple[str, ...]]],
        starts: tuple[str, ...],
        ends: tuple[str, ...],
    ) -> None:
        self.parts = parts
        self.routes = routes
        self.starts = starts
        self.ends = ends


class HmmParameters(_BaseSerializable):
    """Numerical parameters learned for an :class:`HmmModel` topology."""

    transition_probabilities: np.ndarray
    start_probabilities: np.ndarray
    end_probabilities: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    weights: np.ndarray
    data_columns: tuple[str, ...]

    def __init__(
        self,
        transition_probabilities: np.ndarray,
        start_probabilities: np.ndarray,
        end_probabilities: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        weights: np.ndarray,
        data_columns: tuple[str, ...],
    ) -> None:
        self.transition_probabilities = transition_probabilities
        self.start_probabilities = start_probabilities
        self.end_probabilities = end_probabilities
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.data_columns = data_columns


class HmmModel(_BaseSerializable):
    """An HMM topology with optional backend-neutral fitted parameters."""

    allowed_transitions: np.ndarray
    allowed_starts: np.ndarray
    allowed_ends: np.ndarray
    n_gmm_components: np.ndarray
    state_ids: tuple[StateId, ...]
    state_groups: dict[str, tuple[StateId, ...]]
    composition: HmmComposition | None
    parameters: OptiPara[Optional[HmmParameters]]  # noqa: UP045

    def __init__(
        self,
        allowed_transitions: np.ndarray,
        allowed_starts: np.ndarray,
        allowed_ends: np.ndarray,
        n_gmm_components: np.ndarray,
        state_ids: tuple[StateId, ...],
        state_groups: dict[str, tuple[StateId, ...]],
        composition: HmmComposition | None = None,
        parameters: HmmParameters | None = None,
    ) -> None:
        self.allowed_transitions = allowed_transitions
        self.allowed_starts = allowed_starts
        self.allowed_ends = allowed_ends
        self.n_gmm_components = n_gmm_components
        self.state_ids = state_ids
        self.state_groups = state_groups
        self.composition = composition
        self.parameters = parameters

    @classmethod
    def left_right(
        cls,
        n_states: int,
        *,
        n_gmm_components: int,
        cycle: bool = False,
        starts: Literal["first", "all"] = "first",
        ends: Literal["last", "all"] = "last",
    ) -> HmmModel:
        """Construct an unfitted left-right HMM definition."""
        if n_states <= 0 or n_gmm_components <= 0:
            raise ValueError("The number of states and GMM components must be positive.")
        if starts not in ("first", "all"):
            raise ValueError("`starts` must be either 'first' or 'all'.")
        if ends not in ("last", "all"):
            raise ValueError("`ends` must be either 'last' or 'all'.")
        allowed_transitions = np.eye(n_states, dtype=bool)
        allowed_transitions[np.arange(n_states - 1), np.arange(1, n_states)] = True
        if cycle:
            allowed_transitions[-1, 0] = True
        allowed_starts = np.ones(n_states, dtype=bool) if starts == "all" else np.arange(n_states) == 0
        allowed_ends = np.ones(n_states, dtype=bool) if ends == "all" else np.arange(n_states) == n_states - 1
        return cls.from_matrix(
            allowed_transitions=allowed_transitions,
            allowed_starts=allowed_starts,
            allowed_ends=allowed_ends,
            n_gmm_components=np.full(n_states, n_gmm_components, dtype=int),
            state_ids=tuple((f"state_{state}",) for state in range(n_states)),
            state_groups={},
        )

    @classmethod
    def from_matrix(  # noqa: C901, PLR0912
        cls,
        *,
        allowed_transitions: np.ndarray,
        allowed_starts: np.ndarray,
        allowed_ends: np.ndarray,
        n_gmm_components: np.ndarray,
        state_ids: tuple[StateId, ...] | None = None,
        state_groups: dict[str, tuple[StateId, ...]] | None = None,
    ) -> HmmModel:
        """Construct a validated unfitted HMM from an arbitrary state graph."""
        transitions = np.asarray(allowed_transitions, dtype=bool)
        if transitions.ndim != 2 or transitions.shape[0] != transitions.shape[1] or transitions.shape[0] == 0:
            raise ValueError("`allowed_transitions` must be a non-empty square matrix.")

        n_states = len(transitions)
        starts = np.asarray(allowed_starts, dtype=bool)
        ends = np.asarray(allowed_ends, dtype=bool)
        raw_component_counts = np.asarray(n_gmm_components)
        if starts.shape != (n_states,) or ends.shape != (n_states,) or raw_component_counts.shape != (n_states,):
            raise ValueError("Start, end, and GMM-component definitions must contain one value per state.")
        try:
            numeric_component_counts = raw_component_counts.astype(float)
        except (TypeError, ValueError) as e:
            raise ValueError("GMM-component counts must be finite, positive, integer-valued numbers.") from e
        if (
            np.any(~np.isfinite(numeric_component_counts))
            or np.any(numeric_component_counts <= 0)
            or np.any(numeric_component_counts != np.floor(numeric_component_counts))
        ):
            raise ValueError("GMM-component counts must be finite, positive, integer-valued numbers.")
        component_counts = numeric_component_counts.astype(int)
        if not np.any(starts) or not np.any(ends):
            raise ValueError("At least one start state and one end state are required.")

        identities = state_ids or tuple((f"state_{state}",) for state in range(n_states))
        if len(identities) != n_states or len(set(identities)) != n_states:
            raise ValueError("State identities must be unique and contain one path per state.")
        groups = state_groups or {}
        unknown_group_states = {state for states in groups.values() for state in states} - set(identities)
        if unknown_group_states:
            raise ValueError(f"State groups reference unknown identities: {tuple(sorted(unknown_group_states))}.")

        reachable = starts.copy()
        while True:
            expanded = reachable | np.any(transitions[reachable], axis=0)
            if np.array_equal(expanded, reachable):
                break
            reachable = expanded
        if not np.all(reachable):
            raise ValueError(f"States {tuple(np.flatnonzero(~reachable))} are unreachable from every allowed start.")

        can_reach_end = ends.copy()
        while True:
            expanded = can_reach_end | np.any(transitions[:, can_reach_end], axis=1)
            if np.array_equal(expanded, can_reach_end):
                break
            can_reach_end = expanded
        if not np.all(can_reach_end):
            raise ValueError(f"States {tuple(np.flatnonzero(~can_reach_end))} cannot reach an allowed end.")

        return cls(
            allowed_transitions=transitions,
            allowed_starts=starts,
            allowed_ends=ends,
            n_gmm_components=component_counts,
            state_ids=identities,
            state_groups=groups,
        )

    @classmethod
    def compose(  # noqa: C901
        cls,
        parts: dict[str, HmmModel],
        *,
        routes: dict[str, tuple[str, ...]],
        starts: tuple[str, ...],
        ends: tuple[str, ...],
    ) -> HmmModel:
        """Compose named HMMs through routes from source exits to target entries."""
        if not parts:
            raise ValueError("At least one named HMM is required for composition.")
        if invalid_names := tuple(name for name in parts if "." in name):
            raise ValueError(f"Composition part names must not contain dots: {invalid_names}.")
        fitted_parts = [part.is_fitted for part in parts.values()]
        if any(fitted_parts) and not all(fitted_parts):
            raise ValueError("Composition requires either all fitted or all unfitted model parts.")

        names = set(parts)
        referenced_names = {*starts, *ends, *routes}
        referenced_names.update(target for targets in routes.values() for target in targets)
        if unknown := referenced_names - names:
            raise ValueError(f"Composition references unknown model parts: {tuple(sorted(unknown))}.")

        offsets = {}
        next_offset = 0
        for name, part in parts.items():
            offsets[name] = next_offset
            next_offset += part.n_states

        allowed_transitions = np.zeros((next_offset, next_offset), dtype=bool)
        allowed_starts = np.zeros(next_offset, dtype=bool)
        allowed_ends = np.zeros(next_offset, dtype=bool)
        state_ids = []
        state_groups = {}
        component_counts = []
        for name, part in parts.items():
            offset = offsets[name]
            part_slice = slice(offset, offset + part.n_states)
            allowed_transitions[part_slice, part_slice] = part.allowed_transitions
            if name in starts:
                allowed_starts[part_slice] = part.allowed_starts
            if name in ends:
                allowed_ends[part_slice] = part.allowed_ends

            prefixed_ids = tuple((name, *state_id) for state_id in part.state_ids)
            state_ids.extend(prefixed_ids)
            state_groups[name] = prefixed_ids
            state_groups.update(
                {
                    f"{name}.{group}": tuple((name, *state_id) for state_id in group_states)
                    for group, group_states in part.state_groups.items()
                }
            )
            component_counts.append(part.n_gmm_components)

        for source, targets in routes.items():
            source_offset = offsets[source]
            source_exits = np.flatnonzero(parts[source].allowed_ends) + source_offset
            for target in targets:
                target_offset = offsets[target]
                target_entries = np.flatnonzero(parts[target].allowed_starts) + target_offset
                allowed_transitions[np.ix_(source_exits, target_entries)] = True

        parameters = cls._compose_fitted_parameters(parts, routes, starts, ends, offsets) if all(fitted_parts) else None
        return cls(
            allowed_transitions=allowed_transitions,
            allowed_starts=allowed_starts,
            allowed_ends=allowed_ends,
            n_gmm_components=np.concatenate(component_counts),
            state_ids=tuple(state_ids),
            state_groups=state_groups,
            composition=HmmComposition(
                [(name, part._without_parameters()) for name, part in parts.items()],
                list(routes.items()),
                starts,
                ends,
            ),
            parameters=parameters,
        )

    def _without_parameters(self) -> HmmModel:
        return HmmModel(
            allowed_transitions=self.allowed_transitions,
            allowed_starts=self.allowed_starts,
            allowed_ends=self.allowed_ends,
            n_gmm_components=self.n_gmm_components,
            state_ids=self.state_ids,
            state_groups=self.state_groups,
            composition=self.composition,
        )

    def set_params(self, **params):
        """Set tpcp parameters and recompile flattened composition metadata when required."""
        result = super().set_params(**params)
        if self.composition is not None and any(name.startswith("composition__") for name in params):
            composition = self.composition
            rebuilt = type(self).compose(
                dict(composition.parts),
                routes=dict(composition.routes),
                starts=composition.starts,
                ends=composition.ends,
            )
            self.__dict__.update(rebuilt.__dict__)
        return result

    def with_parameters(
        self,
        *,
        transition_probabilities: np.ndarray,
        start_probabilities: np.ndarray,
        end_probabilities: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        weights: np.ndarray,
        data_columns: tuple[str, ...],
    ) -> HmmModel:
        """Return this topology with fitted backend-neutral parameters."""
        return HmmModel(
            allowed_transitions=self.allowed_transitions,
            allowed_starts=self.allowed_starts,
            allowed_ends=self.allowed_ends,
            n_gmm_components=self.n_gmm_components,
            state_ids=self.state_ids,
            state_groups=self.state_groups,
            composition=self.composition,
            parameters=HmmParameters(
                transition_probabilities=transition_probabilities,
                start_probabilities=start_probabilities,
                end_probabilities=end_probabilities,
                means=means,
                covariances=covariances,
                weights=weights,
                data_columns=data_columns,
            ),
        )

    @staticmethod
    def _compose_fitted_parameters(  # noqa: C901
        parts: dict[str, HmmModel],
        routes: dict[str, tuple[str, ...]],
        starts: tuple[str, ...],
        ends: tuple[str, ...],
        offsets: dict[str, int],
    ) -> HmmParameters:
        data_columns = {part.data_columns for part in parts.values()}
        if len(data_columns) != 1:
            raise ValueError("All fitted model parts must use identical feature columns.")

        n_states = sum(part.n_states for part in parts.values())
        n_features = len(next(iter(data_columns)))
        max_components = max(int(part.n_gmm_components.max()) for part in parts.values())
        transitions = np.zeros((n_states, n_states))
        start_probabilities = np.zeros(n_states)
        end_probabilities = np.zeros(n_states)
        means = np.zeros((n_states, max_components, n_features))
        covariances = np.zeros((n_states, max_components, n_features, n_features))
        weights = np.zeros((n_states, max_components))

        def normalized_entries(name: str) -> np.ndarray:
            probabilities = parts[name].start_probabilities
            total = probabilities.sum()
            if not np.all(np.isfinite(probabilities)) or not np.isfinite(total) or total <= 0:
                raise ValueError(f"Fitted model part {name!r} must have positive entry probability mass.")
            return probabilities / total

        for name, part in parts.items():
            offset = offsets[name]
            part_slice = slice(offset, offset + part.n_states)
            transitions[part_slice, part_slice] = part.transition_probabilities
            means[part_slice, : part.means.shape[1]] = part.means
            covariances[part_slice, : part.covariances.shape[1]] = part.covariances
            weights[part_slice, : part.weights.shape[1]] = part.weights

        for name in starts:
            part = parts[name]
            entry_probabilities = normalized_entries(name)
            offset = offsets[name]
            start_probabilities[offset : offset + part.n_states] = entry_probabilities / len(starts)

        for source, part in parts.items():
            destinations = [*routes.get(source, ()), *([None] if source in ends else [])]
            if not destinations and np.any(part.end_probabilities):
                raise ValueError(f"Fitted model part {source!r} has exit mass but no route or global end.")
            if not destinations:
                continue

            source_offset = offsets[source]
            source_slice = slice(source_offset, source_offset + part.n_states)
            route_share = 1 / len(destinations)
            for target in destinations:
                if target is None:
                    end_probabilities[source_slice] += part.end_probabilities * route_share
                    continue
                target_part = parts[target]
                entry_probabilities = normalized_entries(target)
                target_offset = offsets[target]
                target_slice = slice(target_offset, target_offset + target_part.n_states)
                transitions[source_slice, target_slice] += (
                    np.outer(part.end_probabilities, entry_probabilities) * route_share
                )

        return HmmParameters(
            transition_probabilities=transitions,
            start_probabilities=start_probabilities,
            end_probabilities=end_probabilities,
            means=means,
            covariances=covariances,
            weights=weights,
            data_columns=next(iter(data_columns)),
        )

    @classmethod
    def from_parameters(
        cls,
        *,
        transition_probabilities: np.ndarray,
        start_probabilities: np.ndarray,
        end_probabilities: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        weights: np.ndarray,
        n_gmm_components: np.ndarray,
        data_columns: tuple[str, ...],
        state_groups: dict[str, tuple[int, ...]] | None = None,
    ) -> HmmModel:
        """Construct a fitted model from compiled numerical parameters."""
        state_ids = tuple((f"state_{state}",) for state in range(len(start_probabilities)))
        groups = {name: tuple(state_ids[state] for state in states) for name, states in (state_groups or {}).items()}
        return cls(
            allowed_transitions=transition_probabilities > 0,
            allowed_starts=start_probabilities > 0,
            allowed_ends=end_probabilities > 0,
            n_gmm_components=n_gmm_components,
            state_ids=state_ids,
            state_groups=groups,
            parameters=HmmParameters(
                transition_probabilities=transition_probabilities,
                start_probabilities=start_probabilities,
                end_probabilities=end_probabilities,
                means=means,
                covariances=covariances,
                weights=weights,
                data_columns=data_columns,
            ),
        )

    @property
    def is_fitted(self) -> bool:
        """Whether numerical parameters are available for inference."""
        return self.parameters is not None

    @property
    def n_states(self) -> int:
        """Number of hidden states."""
        return len(self.state_ids)

    def states(self, group: str) -> tuple[StateId, ...]:
        """Return the stable identities belonging to a named state group."""
        try:
            return self.state_groups[group]
        except KeyError as e:
            raise KeyError(f"Unknown state group {group!r}. Available groups: {tuple(self.state_groups)}") from e

    def state_indices(self, group: str) -> tuple[int, ...]:
        """Return compiled integer positions belonging to a named state group."""
        positions = {state_id: position for position, state_id in enumerate(self.state_ids)}
        return tuple(positions[state_id] for state_id in self.states(group))

    @property
    def stride_states(self) -> tuple[int, ...]:
        """Return the compiled positions of the legacy ``stride`` group."""
        return self.state_indices("stride")

    def _require_parameters(self) -> HmmParameters:
        if self.parameters is None:
            raise ValueError("The HMM is not fitted. Train it or load fitted parameters before inference.")
        return self.parameters

    @property
    def transition_probabilities(self) -> np.ndarray:
        return self._require_parameters().transition_probabilities

    @property
    def start_probabilities(self) -> np.ndarray:
        return self._require_parameters().start_probabilities

    @property
    def end_probabilities(self) -> np.ndarray:
        return self._require_parameters().end_probabilities

    @property
    def means(self) -> np.ndarray:
        return self._require_parameters().means

    @property
    def covariances(self) -> np.ndarray:
        return self._require_parameters().covariances

    @property
    def weights(self) -> np.ndarray:
        return self._require_parameters().weights

    @property
    def n_components(self) -> np.ndarray:
        return self.n_gmm_components

    @property
    def data_columns(self) -> tuple[str, ...]:
        return self._require_parameters().data_columns

    @classmethod
    def from_legacy_pomegranate(
        cls,
        payload: dict,
        *,
        data_columns: tuple[str, ...],
        stride_states: tuple[int, ...],
    ) -> HmmModel:
        """Compile a pomegranate 0.14 ``HiddenMarkovModel`` dictionary."""
        emission_indices = [index for index, state in enumerate(payload["states"]) if state["distribution"] is not None]
        state_index = {legacy_index: index for index, legacy_index in enumerate(emission_indices)}
        n_states = len(emission_indices)

        components = []
        component_weights = []
        for legacy_index in emission_indices:
            distribution = payload["states"][legacy_index]["distribution"]
            if distribution.get("class") == "GeneralMixtureModel":
                components.append(distribution["distributions"])
                component_weights.append(distribution["weights"])
            else:
                components.append([distribution])
                component_weights.append([1.0])

        n_components = np.array([len(state_components) for state_components in components])
        n_features = len(components[0][0]["parameters"][0])
        means = np.zeros((n_states, int(n_components.max()), n_features))
        covariances = np.zeros((n_states, int(n_components.max()), n_features, n_features))
        weights = np.zeros((n_states, int(n_components.max())))
        for state, (state_components, state_weights) in enumerate(zip(components, component_weights)):
            weights[state, : len(state_weights)] = state_weights
            for component, distribution in enumerate(state_components):
                means[state, component] = distribution["parameters"][0]
                covariances[state, component] = distribution["parameters"][1]

        transitions = np.zeros((n_states, n_states))
        starts = np.zeros(n_states)
        ends = np.zeros(n_states)
        for source, target, probability, *_ in payload["edges"]:
            if source == payload["start_index"] and target in state_index:
                starts[state_index[target]] = probability
            elif target == payload["end_index"] and source in state_index:
                ends[state_index[source]] = probability
            elif source in state_index and target in state_index:
                transitions[state_index[source], state_index[target]] = probability

        return cls.from_parameters(
            transition_probabilities=transitions,
            start_probabilities=starts,
            end_probabilities=ends,
            means=means,
            covariances=covariances,
            weights=weights,
            n_gmm_components=n_components,
            data_columns=data_columns,
            state_groups={"stride": stride_states},
        )
