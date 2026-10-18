gaitmap.stride_segmentation: Algorithms to find stride candidates
=================================================================

.. automodule:: gaitmap.stride_segmentation
    :no-members:
    :no-inherited-members:



Dtw Stride Segmentation
***********************
.. currentmodule:: gaitmap.stride_segmentation

Classes
-------

.. autosummary::
   :toctree: generated/stride_segmentation
   :template: class.rst

    BaseDtw
    BarthDtw
    ConstrainedBarthDtw
    RoiStrideSegmentation
    BaseDtwTemplate
    DtwTemplate
    BarthOriginalTemplate
    InterpolatedDtwTemplate

Functions
---------

.. autosummary::
   :toctree: generated/stride_segmentation
   :template: function.rst

    find_matches_find_peaks
    find_matches_min_under_threshold


HMM Stride Segmentation
***********************
.. currentmodule:: gaitmap.stride_segmentation.hmm

HMMs support Gaussian and Gaussian-mixture emissions. ``SimpleHmm.min_covar``
bounds each estimated covariance eigenvalue, preventing mixture components from
collapsing around a few observations. It defaults to ``1e-6`` in squared feature
units. Set it on the stride and transition submodels separately when their feature
scales require different regularization.
Loading an existing model and training with frozen emissions preserve its original
covariance matrices.

To define a custom model, provide its emission and transition probabilities.
A single Gaussian is a mixture with one component. For example::

    from gaitmap.stride_segmentation.hmm import GaussianMixture, HiddenMarkovModel

    emission = GaussianMixture(means=[[0.0]], covariances=[[[1.0]]], weights=[1.0])
    model = HiddenMarkovModel([emission], transitions=[[1.0]], starts=[1.0])
    trained, history = model.fit([[[0.0], [0.2], [-0.1]]], max_iterations=10)

``model.transitions`` contains only emitting-state probabilities. ``model.starts``
and optional ``model.ends`` hold start and termination probabilities separately.
For finite models, each transition row plus its termination probability sums to
one. Start probabilities sum to one. Constructors copy parameter arrays and make
them read-only. Fitting returns a new model and its history, leaving the input
model unchanged. Pass ``update_emissions=False`` to fit transitions while keeping
emission parameters fixed.

Observation matrices have shape ``(n_observations, n_features)``. Both prediction
algorithms return one state index per observation, including paths shorter than
the number of states when the model topology permits them. The gaitmap wrappers
preserve this count and map the complete label sequence back to sensor samples.

``max_iterations`` counts parameter updates. Zero returns an unchanged model.
The returned ``TrainingHistory`` contains the initial log likelihood and the
likelihood after every update in ``log_likelihoods``. Its final entry describes
the returned model. ``n_iterations`` counts completed updates; ``improvements``
contains their likelihood differences. ``converged`` is true when a nonnegative
improvement meets ``stop_threshold``. A likelihood decrease stops training with
``converged=False`` and remains visible in the history. A decrease can occur with
Viterbi training and does not imply invalid model parameters.

When migrating a run that previously reached its iteration limit, increase the
limit by one to retain the same training budget.

Existing gaitmap HMM JSON files remain loadable. Newly exported models require
the updated HMM implementation. Arbitrary pomegranate graphs and emission types
other than Gaussian mixtures are not supported.

.. autosummary::
   :toctree: generated/stride_segmentation/hmm
   :template: class.rst

    HmmStrideSegmentation
    RothSegmentationHmm
    BaseSegmentationHmm
    SimpleHmm
    RothHmmFeatureTransformer
    PreTrainedRothSegmentationModel
    BaseHmmFeatureTransformer
    HiddenMarkovModel
    GaussianMixture
    TrainingHistory
