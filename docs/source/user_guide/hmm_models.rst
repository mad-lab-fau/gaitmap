.. _hmm_models:

Training and composing HMMs
=============================

Use :class:`~gaitmap.stride_segmentation.hmm.SimpleHmm` to train a Gaussian or
Gaussian-mixture HMM from feature matrices and initial internal-state labels.
The :ref:`HMM training example <example_segmentation_hmm_training>` shows how to train the Roth stride
segmentation model. The :ref:`custom hierarchical HMM example
<example_custom_hierarchical_hmms>` trains three submodels and combines them.

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

Save complete model configurations, including preprocessing and feature columns,
with joblib::

    import joblib

    joblib.dump(model.clone(), "model.joblib")
    restored = joblib.load("model.joblib")

Cloning first stores parameters and trained models without action results or
input recordings. Record the gaitmap and dependency versions alongside the file.
Only load joblib files from trusted sources.

Algorithm wrappers no longer provide ``to_json()`` or ``from_json()``. To migrate
an existing gaitmap HMM wrapper JSON file::

    from pathlib import Path
    from gaitmap.stride_segmentation.hmm import load_legacy_hmm_json

    restored = load_legacy_hmm_json(Path("old_model.json").read_text())
    joblib.dump(restored, "model.joblib")

Import any custom classes used in the saved configuration before loading.
The numeric :class:`~gaitmap.stride_segmentation.hmm.HiddenMarkovModel` retains
its dedicated JSON format and ``to_json()``/``from_json()`` methods. Newly exported
numeric models require the updated HMM implementation. Arbitrary pomegranate
graphs and emission types other than Gaussian mixtures are not supported.
