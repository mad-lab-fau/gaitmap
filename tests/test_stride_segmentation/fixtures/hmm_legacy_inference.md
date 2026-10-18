# HMM inference reference

`hmm_legacy_inference.npz` was generated before replacing pomegranate, from
revision `7c521a49d4e856ed2f0241532d351a5208a14bce`, using:

- CPython 3.9.25
- pomegranate 0.14.6
- NumPy 1.26.4
- SciPy 1.13.1
- pandas 2.3.3

The input is the complete `get_healthy_example_imu_data()` recording, converted
with `convert_to_fbf(..., left_like="left_", right_like="right_")`. Its sampling
rate is 204.8 Hz. The model is the unchanged bundled
`fallriskpd_at_lab_model.json`. Segmentation uses a 300 ms minimum search window,
as in the pretrained model example.

Each sensor and decoding algorithm has feature-space states, states resampled to
input data, snapped stride boundaries, original stride boundaries, and the
sequence log likelihood. Feature matrices are included to compare inference
independently of feature preprocessing.

For this model, legacy Viterbi returns 1981 feature states for 1982 feature
samples. Pomegranate omits the end-state label for models without explicit
termination, while gaitmap removes the first and last labels. The initial replacement preserved this behavior; the native-interface follow-up
corrects it as described below. MAP returns all 1982 feature labels.

The reference has 29 left and 31 right Viterbi strides; MAP has 29 left and 30
right strides. Likelihoods are 2139.6616989141808 and 2449.2080755522943 for the
left and right sensor, respectively. Legacy likelihood inputs are explicitly made
C-contiguous, as required by the old backend. Passing pandas
`DataFrame.to_numpy()` directly produced incorrect likelihoods because the
feature arrays use Fortran order and pomegranate reads their unchecked memory
layout. The local backend also receives the original array in the regression
test, so it must handle this layout correctly.

To regenerate, install the stated revision in a separate Python 3.9 environment
with pomegranate 0.14.6 and NumPy below 2, then run
`generate_hmm_legacy_inference.py` with that environment's Python. Do not generate
a replacement reference using the new backend.

`hmm_legacy_finite.json` is a two-state, one-dimensional `SimpleHmm` saved by
the same legacy backend. Its means are -2 and 2, variances are 1, emitting
transitions are `[[0.6, 0.4], [0, 0.5]]`, and the second state terminates with
probability 0.5. Both algorithms return `[0, 0, 1]` for observations
`[-2, -1, 2]`. Unlike the unrestricted pretrained model, legacy Viterbi includes
both the start and end label here, so the public result keeps every observation.
## Complete decoding in the native interface

The native interface returns one emitting label per observation. The old gaitmap
Viterbi wrapper discarded the last label for unrestricted models, leaving 1981
labels for 1982 feature observations in these recordings. The reference file
remains unchanged. The regression tests append the known final null-state label
and check the corrected exact 4:1 interpolation grid independently using repeated
labels with earlier-label midpoint ties. Original boundaries are mapped from the
old stretched grid to that complete grid. Likelihoods, all preceding Viterbi
labels, MAP predictions, and the 300 ms snapped stride lists remain exact.
