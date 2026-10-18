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
transition state 4, develop nearly singular covariance matrices. Pomegranate
accepts arbitrarily small positive-definite covariance matrices, then adds a
diagonal shift only after Cholesky fails. Rounding determines which iteration
needs repair and changes the subsequent EM trajectory. The old fitter itself ends
these initialization fits with negative likelihood improvements of about 21 and 20.

The local estimator deliberately fixes this policy. It uses centered weighted
covariance, normalizes positive weights, and floors covariance eigenvalues at
`min_covar`, default `1e-6`, on every estimate. Scaling all positive weights no
longer changes whether an update takes place. Only zero-support and frozen
distributions retain their previous parameters. A machine-precision relative
floor applies when the requested absolute floor is too small to produce a
representably positive-definite matrix.

With the default floor on CPython 3.10 / NumPy 1.26.4, 258 of 7928 left-foot labels
and 174 of 7928 right-foot labels differ from the old reference. All stride-versus-
transition classifications and all 58 stride boundaries remain exact. Repeated
updates of the real stride-state-11 mixture have finite nondecreasing likelihood
and bounded covariance eigenvalues. Tests retain the original reference unchanged
and check these stability properties independently.

The focused training tests separately compare weighted Gaussian and mixture
updates, deterministic clustering, transition probabilities, likelihoods,
histories, iteration limits, and convergence against stable legacy references.
`hmm_distributions_legacy.json` includes three updates of the real six-component
stride-state-11 mixture before covariance collapse. `hmm_legacy_training.json`
covers tiny finite and unrestricted HMMs with multiple training algorithms and
iteration limits. Both include their generation provenance.
