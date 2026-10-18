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

Training uses a local NumPy/Numba backend. Gaussian covariance estimates use
centered differences and weights normalized to remove dependence on their absolute
scale. Each estimate floors its eigenvalues at ``SimpleHmm.min_covar``, which
defaults to ``1e-6`` in squared feature units. The floor prevents mixture components
from collapsing around a few observations. Set it on the stride and transition
submodels separately when their feature scales require different regularization.
Loading an existing model and training with frozen emissions preserve its original
covariance matrices.

The local model uses explicit numeric parameters. A single Gaussian is a mixture
with one component. For example::

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

``max_iterations`` counts parameter updates. Zero returns an unchanged model.
The returned ``TrainingHistory`` contains the initial log likelihood and the
likelihood after every update in ``log_likelihoods``. Its final entry describes
the returned model. ``n_iterations`` counts completed updates; ``improvements``
contains their likelihood differences. ``converged`` is true when a nonnegative
improvement meets ``stop_threshold``. A likelihood decrease stops training with
``converged=False`` and remains visible in the history. A decrease can occur with
Viterbi training and does not imply invalid model parameters.

The old pomegranate iteration limit could perform one extra update and omit that
update from its history. To reproduce the amount of work in an old run that hit
its iteration limit, increase that limit by one.

New model exports use the versioned ``gaitmap-hmm`` JSON format. The loader also
accepts existing gaitmap files containing pomegranate Gaussian models. Legacy
silent states and edge dictionaries are converted by the loader and do not enter
the inference or training implementation.

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
