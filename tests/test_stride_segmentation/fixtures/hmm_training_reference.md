# HMM training references

`hmm_legacy_training_example.npz` captures the complete
`examples.stride_segmentation.segmentation_hmm_training` example from revision
`070843ccfd0ff6b85e41b9e33750238b28347be8`, whose training still used pomegranate.
The environment was CPython 3.9.25, pomegranate 0.14.6, NumPy 1.26.4, SciPy 1.13.1,
and pandas 2.3.3. The example sets its own random seed and uses both feet of the
healthy recording at 204.8 Hz.

To regenerate it, install that revision and those dependencies in a separate
environment, then run `generate_hmm_legacy_training_example.py` with that
environment's Python. Never regenerate the reference using the local trainer.

The reference contains state labels and snapped/original stride boundaries. Its
28 left and 30 right strides, including the original boundaries before snapping,
and all stride-versus-transition classifications must remain unchanged.

Fitted mixture parameters and individual null-state labels are not stable golden
targets for this example. Two initialization clusters, stride state 11 and
transition state 4, develop nearly singular covariance matrices. The implementations
agree near machine precision until rounding determines which iteration requires
covariance repair. That changes the subsequent EM trajectory. The old fitter
itself ends these initialization fits with negative likelihood improvements of
about 21 and 20. The existing example test already avoids fitted-parameter
snapshots because training varies across machines.

The focused training tests separately compare weighted Gaussian and mixture
updates, deterministic clustering, transition probabilities, likelihoods,
histories, iteration limits, and convergence against stable legacy references.
`hmm_distributions_legacy.json` includes three updates of the real six-component
stride-state-11 mixture before covariance collapse. `hmm_legacy_training.json`
covers tiny finite and unrestricted HMMs with multiple training algorithms and
iteration limits. Both include their generation provenance.
