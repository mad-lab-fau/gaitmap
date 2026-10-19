.. _hmm_models:

Training and composing HMMs
==========================

Use :class:`~gaitmap.stride_segmentation.hmm.SimpleHmm` to train a Gaussian or
Gaussian-mixture HMM from feature matrices and initial internal-state labels.
The :ref:`HMM training example <example_segmentation_hmm_training>` shows how to train the Roth stride
segmentation model. The :ref:`custom hierarchical HMM example
<example_stance_swing_transition_hmm>` trains three submodels and combines them.

Training Gaussian emissions
---------------------------

Observation matrices have shape ``(n_observations, n_features)``. All sequences
must use the same features. ``SimpleHmm.min_covar`` sets the minimum covariance
eigenvalue during training and defaults to ``1e-6`` in squared feature units.
Increase it when mixture components collapse around a few observations. You can
set it separately on each submodel when their feature scales differ.
Loading an existing model preserves its saved covariance matrices.

For models with explicitly chosen probabilities, use
:class:`~gaitmap.stride_segmentation.hmm.HiddenMarkovModel` and
:class:`~gaitmap.stride_segmentation.hmm.GaussianMixture` directly::

    from gaitmap.stride_segmentation.hmm import GaussianMixture, HiddenMarkovModel

    emission = GaussianMixture(means=[[0.0]], covariances=[[[1.0]]], weights=[1.0])
    model = HiddenMarkovModel([emission], transitions=[[1.0]], starts=[1.0])
    trained, history = model.fit([[[0.0], [0.2], [-0.1]]], max_iterations=10)

Fitting returns a new model and its training history. ``history.log_likelihoods``
contains the initial likelihood and the likelihood after each parameter update.
Its final entry describes the returned model. A likelihood decrease stops
training with ``history.converged=False``; this can occur with Viterbi training.
Pass ``update_emissions=False`` to fit transitions with fixed emissions.

Combining trained submodels
---------------------------

Use :class:`~gaitmap.stride_segmentation.hmm.CompositeHmm` to combine trained
numeric models by name. Supply one annotation table per feature sequence, with
``start``, exclusive ``end``, and ``label`` columns. Labels identify submodels;
non-overlapping intervals must cover the complete sequence. Keep consecutive
instances of the same submodel in separate rows, for example two adjacent strides.

.. code-block:: python

    combined, history = CompositeHmm({"rest": rest_model, "stride": stride_model}).fit_transitions(
        sequences=feature_sequences, labels=annotation_tables
    )
    movement_labels = combined.predict_labels(test_features)
    internal_states = combined.predict_hidden_state_sequence(test_features)

Submodels must use the same feature columns in the same order. Composition
infers internal states within each annotated instance, then estimates transition
and initial-state probabilities from their counts. Emissions remain fixed.
Connections are derived from data only. The recording may end in any internal
state, including partway through a movement instance.

Saved models
------------

Existing gaitmap HMM JSON files remain loadable. Newly exported models require
the updated HMM implementation. Arbitrary pomegranate graphs and emission types
other than Gaussian mixtures are not supported.
