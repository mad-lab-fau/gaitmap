"""Behavioral tests for backend-neutral HMM definitions and fitted parameters."""

import numpy as np
import pytest

from gaitmap.stride_segmentation.hmm import HmmModel


def _fitted_one_state_model(*, mean: float, self_transition: float, end: float) -> HmmModel:
    return HmmModel.from_parameters(
        transition_probabilities=np.array([[self_transition]]),
        start_probabilities=np.array([1.0]),
        end_probabilities=np.array([end]),
        means=np.array([[[mean]]]),
        covariances=np.array([[[[0.1]]]]),
        weights=np.array([[1.0]]),
        n_gmm_components=np.array([1]),
        data_columns=("feature",),
    )


def test_left_right_hmm_is_a_complete_unfitted_model_definition() -> None:
    """Topology and stable state identity exist before numerical fitting."""
    model = HmmModel.left_right(
        n_states=3,
        n_gmm_components=2,
        cycle=False,
        starts="first",
        ends="last",
    )

    assert model.is_fitted is False
    assert model.state_ids == (("state_0",), ("state_1",), ("state_2",))
    np.testing.assert_array_equal(
        model.allowed_transitions,
        [[True, True, False], [False, True, True], [False, False, True]],
    )
    np.testing.assert_array_equal(model.allowed_starts, [True, False, False])
    np.testing.assert_array_equal(model.allowed_ends, [False, False, True])
    np.testing.assert_array_equal(model.n_gmm_components, [2, 2, 2])


@pytest.mark.parametrize(("argument", "value"), [("starts", "last"), ("ends", "first")])
def test_left_right_hmm_rejects_unknown_boundary_modes(argument, value) -> None:
    """Typos in topology modes must not silently select the default."""
    with pytest.raises(ValueError, match=argument):
        HmmModel.left_right(2, n_gmm_components=1, **{argument: value})


@pytest.mark.parametrize("component_count", [1.5, np.inf])
def test_left_right_hmm_rejects_non_integer_component_counts(component_count) -> None:
    """Convenience construction applies the same component validation as matrix construction."""
    with pytest.raises(ValueError, match="integer-valued"):
        HmmModel.left_right(2, n_gmm_components=component_count)


def test_matrix_topology_supports_arbitrary_reachable_state_graphs() -> None:
    """Custom domains can define graphs without implementing another HMM class."""
    model = HmmModel.from_matrix(
        allowed_transitions=np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=bool),
        allowed_starts=np.array([1, 0, 0], dtype=bool),
        allowed_ends=np.array([0, 0, 1], dtype=bool),
        n_gmm_components=np.array([1, 2, 1]),
        state_ids=(("idle",), ("ascending",), ("descending",)),
        state_groups={"stairs": (("ascending",), ("descending",))},
    )

    assert model.state_ids == (("idle",), ("ascending",), ("descending",))
    assert model.state_indices("stairs") == (1, 2)
    assert not model.is_fitted


@pytest.mark.parametrize(
    ("transitions", "components", "error"),
    [
        (np.ones((2, 3)), np.ones(2), "square matrix"),
        (np.eye(2), np.ones(1), "one value per state"),
        (np.eye(2), np.ones(2), "unreachable"),
        (np.ones((2, 2)), np.array([1.5, 1]), "integer-valued"),
    ],
)
def test_matrix_topology_rejects_incomplete_or_unreachable_definitions(transitions, components, error) -> None:
    """Invalid graphs fail while they still carry useful topology context."""
    with pytest.raises(ValueError, match=error):
        HmmModel.from_matrix(
            allowed_transitions=transitions,
            allowed_starts=np.array([1, 0]),
            allowed_ends=np.array([0, 1]),
            n_gmm_components=components,
        )


def test_matrix_topology_requires_every_state_to_reach_an_end() -> None:
    """A topology accepted for training has a legal completion from every state."""
    with pytest.raises(ValueError, match="reach an allowed end"):
        HmmModel.from_matrix(
            allowed_transitions=np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]),
            allowed_starts=np.array([1, 0, 1]),
            allowed_ends=np.array([0, 1, 0]),
            n_gmm_components=np.ones(3),
        )


def test_composition_flattens_any_number_of_named_models_without_losing_identity() -> None:
    """Named model routes expand into a flat topology with addressable groups."""
    model = HmmModel.compose(
        parts={
            "transition": HmmModel.left_right(2, n_gmm_components=1, starts="all", ends="all"),
            "walking": HmmModel.left_right(2, n_gmm_components=2),
            "running": HmmModel.left_right(3, n_gmm_components=3),
        },
        routes={
            "transition": ("walking", "running"),
            "walking": ("transition",),
            "running": ("transition",),
        },
        starts=("transition",),
        ends=("transition",),
    )

    assert model.state_ids == (
        ("transition", "state_0"),
        ("transition", "state_1"),
        ("walking", "state_0"),
        ("walking", "state_1"),
        ("running", "state_0"),
        ("running", "state_1"),
        ("running", "state_2"),
    )
    assert model.states("walking") == (("walking", "state_0"), ("walking", "state_1"))
    assert model.get_params()["composition__parts__running__n_gmm_components"][0] == 3
    assert model.allowed_transitions[1, 2]
    assert model.allowed_transitions[1, 4]
    assert model.allowed_transitions[3, 0]
    assert model.allowed_transitions[6, 0]
    assert not model.allowed_transitions[3, 4]


def test_nested_tpcp_updates_recompile_a_composite_topology() -> None:
    """Changing a named part never leaves cached flattened arrays stale."""
    model = HmmModel.compose(
        parts={"a": HmmModel.left_right(1, n_gmm_components=1), "b": HmmModel.left_right(1, n_gmm_components=1)},
        routes={"a": ("b",)},
        starts=("a",),
        ends=("b",),
    )

    model.set_params(composition__parts__a__n_gmm_components=np.array([3]))

    np.testing.assert_array_equal(model.n_gmm_components, [3, 1])


def test_composition_rejects_ambiguous_part_names() -> None:
    """Dots remain reserved for generated hierarchical group names."""
    with pytest.raises(ValueError, match="must not contain dots"):
        HmmModel.compose(
            parts={"activity.walk": HmmModel.left_right(1, n_gmm_components=1)},
            routes={},
            starts=("activity.walk",),
            ends=("activity.walk",),
        )


def test_composition_lifts_fitted_exit_mass_into_macro_routes() -> None:
    """Fitted submodels compose without losing emissions or probability mass."""
    model = HmmModel.compose(
        parts={
            "background": _fitted_one_state_model(mean=0.0, self_transition=0.8, end=0.2),
            "event": _fitted_one_state_model(mean=5.0, self_transition=0.7, end=0.3),
        },
        routes={"background": ("event",), "event": ("background",)},
        starts=("background",),
        ends=("background",),
    )

    assert model.is_fitted
    np.testing.assert_allclose(model.transition_probabilities, [[0.8, 0.1], [0.3, 0.7]])
    np.testing.assert_allclose(model.end_probabilities, [0.1, 0.0])
    np.testing.assert_allclose(model.start_probabilities, [1.0, 0.0])
    np.testing.assert_allclose(model.means[:, 0, 0], [0.0, 5.0])
    np.testing.assert_allclose(model.transition_probabilities.sum(axis=1) + model.end_probabilities, 1.0)
    assert all(not part.is_fitted for _, part in model.composition.parts)


def test_fitted_composition_rejects_route_targets_without_entry_mass() -> None:
    """Invalid fitted entry distributions fail instead of producing NaNs."""
    source = _fitted_one_state_model(mean=0.0, self_transition=0.8, end=0.2)
    target = _fitted_one_state_model(mean=5.0, self_transition=1.0, end=0.0)
    target.parameters.start_probabilities[:] = 0

    with pytest.raises(ValueError, match="positive entry probability"):
        HmmModel.compose(
            parts={"source": source, "target": target},
            routes={"source": ("target",)},
            starts=("source",),
            ends=("target",),
        )
