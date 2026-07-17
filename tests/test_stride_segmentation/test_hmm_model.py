"""Behavioral tests for backend-neutral HMM definitions and fitted parameters."""

import numpy as np

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
