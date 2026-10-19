"""
.. _example_experimental_tempo_stride_segmentation:

Experimental segmentation with persistent local tempo
=====================================================

This synthetic example compares two fits on the same positive-tempo block grid.
It demonstrates the objective, not gait accuracy. Existing BarthDtw scores and
thresholds are different and remain unchanged.

Run this file directly to also measure warm numerical runtime and whole-process
peak resident memory in one worker. The bounded adversarial duration cases force
recomputation at every duration-feasible endpoint. No recording download or
hyperparameter search is involved.
"""

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from gaitmap_mad.stride_segmentation.dtw import tempo_regularized_subsequence_dtw

from gaitmap.stride_segmentation import DtwTemplate, ExperimentalTempoBarthDtw

# %%
# A closer sample fit can require sudden compression followed by expansion.
# Both algorithms optimize the same six-bin duration, including every sample.

template = DtwTemplate(data=np.array([0.0, 1.0, 1.0, 0.0]), sampling_rate_hz=4)
signal = np.array([1.0, 3.0, 3.0, 0.0, 0.0, 0.0])
segmentations = []
for penalty in (0.0, 2.0):
    segmentation = ExperimentalTempoBarthDtw(
        template=template,
        resample_template=False,
        n_segments=4,
        tempo_bounds=(1.0, 2.0),
        tempo_change_penalty=penalty,
        min_match_length_s=5.5 / 4,
        max_match_length_s=6.5 / 4,
        max_cost=10,
        find_matches_method="min_under_thres",
    ).segment(signal, sampling_rate_hz=4)
    alignment = segmentation.alignments_[0]
    print(
        f"penalty={penalty}: edges={alignment.block_edges.tolist()}, "
        f"data={alignment.data_cost:.6f}, regularizer={alignment.tempo_cost:.6f}"
    )
    segmentations.append(segmentation)

# %%
# Paths contain integer sample pairs. Their positive weights sum to one; costs
# are the square root of weighted feature loss plus the log-tempo regularizer.
# Block edges and actual physical tempos remain available for inspection.

print(segmentations[1].stride_list_)
print("weights sum:", segmentations[1].path_weights_[0].sum())


def measure_bounded_cases():
    """Measure one worker, including the adverse duration-fallback time cost."""
    x = np.zeros(100)
    options = {
        "template_sampling_rate_hz": 100,
        "signal_sampling_rate_hz": 100,
        "n_segments": 10,
        "tempo_bounds": (0.5, 2.0),
        "tempo_change_penalty": 0.1,
    }
    start = perf_counter()
    tempo_regularized_subsequence_dtw(x, np.zeros(200), **options)
    first_call = perf_counter() - start
    records = []
    for n in (500, 1000):
        for constrained in (False, True):
            duration = {"min_duration_samples": 191, "max_duration_samples": 191} if constrained else {}
            start = perf_counter()
            result = tempo_regularized_subsequence_dtw(x, np.zeros(n), **options, **duration)
            elapsed = perf_counter() - start
            records.append(
                {
                    "samples": n,
                    "duration_samples": 191 if constrained else None,
                    "seconds": elapsed,
                    "fallbacks": result.fallback_count,
                    "tempo_states": len(result.advances),
                    "projection_bytes": result.costs.nbytes,
                }
            )
            del result
    # Linux ru_maxrss can retain an inherited pre-exec high-water value. VmHWM
    # belongs to this process image, so use that kernel-recorded peak on Linux.
    # The raw resource reading is reported too, rather than hiding the mismatch.
    peak_mib, rusage_mib = None, None
    if sys.platform in ("linux", "darwin"):
        import resource  # noqa: PLC0415  # Optional Unix-only measurement support.

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rusage_mib = peak / (1024**2 if sys.platform == "darwin" else 1024)
        peak_mib = rusage_mib
    if sys.platform == "linux":
        fields = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
        peak_mib = int(fields["VmHWM"].split()[0]) / 1024
    print(
        json.dumps(
            {
                "first_measurement_setup_call_seconds": first_call,
                "warm_cases": records,
                "whole_process_peak_rss_mib": peak_mib,
                "raw_rusage_peak_rss_mib": rusage_mib,
                "note": (
                    "One worker; imports/JIT/cache loading contribute to peak RSS. "
                    "Setup timing is not a cold-process time. RSS is unavailable on unsupported platforms."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    measure_bounded_cases()
