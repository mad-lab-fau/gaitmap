"""Reproduce issue 93 before/after comparisons on the supplied example recordings.

Run from the repository root::

    MPLBACKEND=Agg uv run python scripts/compare_constrained_dtw.py

The old numerical function is read from the specified git revision and compiled
without modifying the checkout. It returns the old cost-only result, selecting
the legacy traceback/peak-selection branch of BaseDtw. Both versions otherwise
use identical parameters, data, preprocessing and postprocessing. The plotting
function is loaded from the existing constrained Barth tutorial without running
the tutorial's top-level computations. Outputs include CSV/JSON summaries and
paired plots only for sensor/configuration cases with changed final stride borders.
"""

# Standalone developer command, not an importable package.
# ruff: noqa: INP001

import argparse
import ast
import json
import select
import subprocess
import sys
from html import escape
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from gaitmap_mad.stride_segmentation.dtw._constrained_dtw import constrained_subsequence_dtw
from gaitmap_mad.stride_segmentation.dtw._vendored_tslearn import _local_squared_dist
from numba import njit

from gaitmap.evaluation_utils import evaluate_segmented_stride_list
from gaitmap.example_data import (
    get_healthy_example_imu_data,
    get_healthy_example_imu_data_ic_stride,
    get_healthy_example_imu_data_stair_down,
    get_healthy_example_imu_data_stair_up,
    get_healthy_example_stride_borders,
    get_ms_example_imu_data,
)
from gaitmap.stride_segmentation import BarthDtw, BarthOriginalTemplate, ConstrainedBarthDtw
from gaitmap.utils.coordinate_conversion import convert_to_fbf

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "7c521a49d4e856ed2f0241532d351a5208a14bce"
COST_FUNCTION = "gaitmap_mad.stride_segmentation.dtw._base_dtw.subsequence_cost_matrix_with_constrains"


def _load_function(source, name, namespace):
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), f"comparison-{name}.py", "exec"), namespace)
    return namespace[name]


def legacy_calculator(revision):
    """Compile the unmodified historical forward calculation and return its cost layer."""
    path = "packages/gaitmap_mad/src/gaitmap_mad/stride_segmentation/dtw/_base_dtw.py"
    source = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True)
    function = njit(
        _load_function(
            source,
            "_subsequence_cost_matrix_with_constrains",
            {
                "np": np,
                "_local_squared_dist": _local_squared_dist,
            },
        )
    )

    def calculate(subseq, longseq, max_subseq_steps, max_longseq_steps, **_):
        # Always return the old cost-only representation to select ordinary
        # traceback and the original peak-selection order in the current caller.
        return function(subseq, longseq, max_subseq_steps, max_longseq_steps)[..., 0]

    return calculate


def _datasets():
    return [
        ("healthy", get_healthy_example_imu_data, 204.8),
        ("ms", get_ms_example_imu_data, 102.4),
        ("ic-stride", get_healthy_example_imu_data_ic_stride, 102.4),
        ("stair-up", get_healthy_example_imu_data_stair_up, 204.8),
        ("stair-down", get_healthy_example_imu_data_stair_down, 204.8),
    ]


def _variants(name):
    result = [("default", ConstrainedBarthDtw, {})]
    if name == "ms":
        result += [
            (
                "example-110",
                BarthDtw,
                {"max_template_stretch_ms": 110, "max_signal_stretch_ms": None, "min_match_length_s": None},
            ),
            ("example-120", ConstrainedBarthDtw, {"max_signal_stretch_ms": None, "min_match_length_s": None}),
        ]
    if name in ("healthy", "ms"):
        result += [("unconstrained", BarthDtw, {})]
    return result


def _path_checks(dtw, sensor):
    template, signal = dtw._extract_relevant_data_and_template(dtw.template, dtw.data[sensor], dtw.sampling_rate_hz)
    if dtw.resample_template and dtw.template.sampling_rate_hz != dtw.sampling_rate_hz:
        template = dtw._resample_template(template, dtw.template.sampling_rate_hz, dtw.sampling_rate_hz)
    vertical, horizontal = dtw._calculate_constrains(dtw.template)
    checks = []
    for path, reported in zip(dtw.paths_[sensor], dtw.costs_[sensor]):
        runs = [0, 0]
        maximum = [0, 0]
        for move in map(tuple, np.diff(path, axis=0)):
            assert move in ((1, 1), (1, 0), (0, 1))
            runs = [runs[0] + 1 if move == (1, 0) else 0, runs[1] + 1 if move == (0, 1) else 0]
            maximum = list(map(max, zip(maximum, runs)))
        actual = np.sqrt(np.sum((template[path[:, 0]] - signal[path[:, 1]]) ** 2))
        checks.append(
            {
                "start": int(path[0, 1]),
                "end": int(path[-1, 1] + 1),
                "vertical_run": maximum[0],
                "horizontal_run": maximum[1],
                "illegal": bool(maximum[0] > vertical or maximum[1] > horizontal),
                "reported_cost": float(reported),
                "actual_cost": float(actual),
                "cost_mismatch": not bool(np.isclose(reported, actual, rtol=1e-10, atol=1e-12)),
            }
        )
    return checks


def _intervals(dtw, sensor):
    return {tuple(map(int, row)) for row in dtw.matches_start_end_[sensor]}


def _reference_scores(before, after, sensor):
    """Evaluate the healthy recording against its compatible, hand-labeled borders."""
    reference = get_healthy_example_stride_borders()[sensor].set_index("s_id")
    scores = {}
    for label, dtw in (("before", before), ("after", after)):
        matches = evaluate_segmented_stride_list(
            ground_truth=reference, segmented_stride_list=dtw.stride_list_[sensor], tolerance=0
        )
        counts = matches["match_type"].value_counts()
        scores.update({f"{label}_{kind}": int(counts.get(kind, 0)) for kind in ("tp", "fp", "fn")})
    return scores


def _compare(before, after, sensor, name, variant, seconds):
    before_checks, after_checks = _path_checks(before, sensor), _path_checks(after, sensor)
    assert not any(item["illegal"] or item["cost_mismatch"] for item in after_checks)
    old, new = _intervals(before, sensor), _intervals(after, sensor)
    old_paths = {int(p[-1, 1]): p for p in before.paths_[sensor]}
    changed_paths = [
        p
        for p in after.paths_[sensor]
        if int(p[-1, 1]) not in old_paths or not np.array_equal(p, old_paths[int(p[-1, 1])])
    ]
    before_curve, after_curve = before.cost_function_[sensor], after.cost_function_[sensor]
    assert np.all(after_curve <= before_curve + 1e-9)
    row = {
        "dataset": name,
        "variant": variant,
        "sensor": sensor,
        "before_matches": len(before.paths_[sensor]),
        "after_matches": len(after.paths_[sensor]),
        "removed_intervals": len(old - new),
        "added_intervals": len(new - old),
        "changed_after_paths": len(changed_paths),
        "before_illegal_paths": sum(item["illegal"] for item in before_checks),
        "after_illegal_paths": sum(item["illegal"] for item in after_checks),
        "before_cost_mismatches": sum(item["cost_mismatch"] for item in before_checks),
        "after_cost_mismatches": sum(item["cost_mismatch"] for item in after_checks),
        "changed_endpoint_costs": int(np.count_nonzero(~np.isclose(before_curve, after_curve, rtol=1e-10, atol=1e-12))),
        "before_seconds_both_sensors": seconds[0],
        "after_seconds_both_sensors": seconds[1],
    }
    if name == "healthy":
        row.update(_reference_scores(before, after, sensor))
    details = {
        "summary": row,
        "before_paths": before_checks,
        "after_paths": after_checks,
        "removed_intervals": sorted(old - new),
        "added_intervals": sorted(new - old),
    }
    candidates = sorted(old ^ new)
    return row, details, candidates


def _select_intervals(before, after, sensor):
    old = sorted(_intervals(before, sensor) - _intervals(after, sensor))
    new = sorted(_intervals(after, sensor) - _intervals(before, sensor))
    paired = [(a, b) for a in old for b in new if a[0] == b[0] or a[1] == b[1]]
    if paired:
        selected = max(paired, key=lambda pair: sum(abs(a - b) for a, b in zip(*pair)))
    elif old:
        selected = (max(old, key=lambda interval: interval[1] - interval[0]), None)
    else:
        selected = (None, max(new, key=lambda interval: interval[1] - interval[0]))
    return selected


def _highlight_interval(axes, dtw, sensor, interval):
    # Dim all ordinary paths, then emphasize the path belonging to the
    # selected final interval. Raw path borders may differ after snapping.
    for line in axes[2].lines:
        line.set_alpha(0.25)
        line.set_color("white")
    if interval:
        axes[0].axvspan(*interval, color="tab:orange", alpha=0.2)
        for ax in axes:
            for border in interval:
                ax.axvline(border, color="tab:orange", linestyle="--", linewidth=1.5)
        for path, final in zip(dtw.paths_[sensor], dtw.matches_start_end_[sensor]):
            if tuple(final) == interval:
                axes[2].plot(path[:, 1], path[:, 0], color="tab:orange", linewidth=2.5)


def _plot(plot_dtw, before, after, sensor, row, output):
    selected = _select_intervals(before, after, sensor)
    start = min(interval[0] for interval in selected if interval)
    end = max(interval[1] for interval in selected if interval)
    margin = max(15, (end - start) // 3)
    window = (max(0, start - margin), min(len(before.data[sensor]) - 1, end + margin))
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(14, 8))
    for col, (label, dtw) in enumerate((("Before", before), ("After", after))):
        plot_dtw(dtw, sensor=sensor, axs=axes[:, col], xlim=window, show=False)
        count = row[f"{'before' if col == 0 else 'after'}_matches"]
        interval = selected[col]
        border_label = f"Final interval [{interval[0]}, {interval[1]})" if interval else "Selected interval absent"
        axes[0, col].set_title(f"{label} · {count} matches in full recording\n{border_label}")
        axes[0, col].set_xlabel("")
        axes[2, col].set_xlabel("Signal sample")
        _highlight_interval(axes[:, col], dtw, sensor, interval)
        # Overlay the other version's cost curve to expose localized changes.
        other = after if col == 0 else before
        axes[1, col].plot(other.cost_function_[sensor], color="tab:gray", linestyle=":", alpha=0.7)
    upper = max(np.max(dtw.cost_function_[sensor][slice(*window)]) for dtw in (before, after))
    for ax in axes[1]:
        ax.set_ylim(0, max(upper * 1.05, before.max_cost * 1.1))
    matrices = [dtw.acc_cost_mat_[sensor][:, slice(*window)] for dtw in (before, after)]
    vmax = max(float(np.quantile(matrix[np.isfinite(matrix)], 0.95)) for matrix in matrices)
    for ax in axes[2]:
        ax.images[0].set_clim(0, vmax)
    if selected[0] and selected[1]:
        shift = (selected[1][0] - selected[0][0], selected[1][1] - selected[0][1])
        change = f"Final border shift: start {shift[0]:+d}, end {shift[1]:+d} samples"
    else:
        change = "Selected final interval removed" if selected[0] else "Selected final interval added"
    fig.suptitle(
        f"{row['dataset']} · {sensor} · {row['variant']} · {change}\n"
        "Orange: selected final borders and associated raw path. Dotted gray: other version's cost curve."
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    filename = f"{row['dataset']}-{row['variant']}-{sensor}.png"
    fig.savefig(output / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    row["plot"] = filename
    row["plot_start"], row["plot_end"] = window


def _benchmark_engine(engine, revision):
    """Measure one warmed numerical calculation in an isolated process."""
    import resource  # noqa: PLC0415 (POSIX-only, imported only for the optional benchmark)

    function = legacy_calculator(revision) if engine == "before" else constrained_subsequence_dtw
    function(np.zeros((3, 3)), np.zeros((5, 3)), 20, 20)
    rng = np.random.default_rng(93)
    template, signal = rng.normal(size=(200, 3)), rng.normal(size=(20_000, 3))
    print("READY", flush=True)
    sys.stdin.readline()
    start = perf_counter()
    if engine == "before":
        result = function(template, signal, 20, 20).copy()
        retained = result.nbytes
    else:
        result = function(template, signal, 20, 20)
        retained = result.costs.nbytes + result.predecessors.nbytes + result.endpoint_states.nbytes
    print(
        json.dumps(
            {
                "engine": engine,
                "seconds": perf_counter() - start,
                "retained_bytes": retained,
                "process_high_water_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
            }
        ),
        flush=True,
    )
    # Keep results alive until the parent has sampled their retained allocation.
    sys.stdin.readline()


def _benchmarks(revision):
    results = []
    for engine in ("before", "after"):
        with subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--benchmark-engine",
                engine,
                "--baseline-revision",
                revision,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            cwd=ROOT,
        ) as child:
            assert child.stdout.readline().strip() == "READY"
            process = psutil.Process(child.pid)
            baseline = peak = process.memory_info().rss
            child.stdin.write("go\n")
            child.stdin.flush()
            # Read completion from a nonblocking pipe while sampling the child's
            # RSS independently of the GIL held by its numerical kernel.
            while not select.select([child.stdout], [], [], 0.002)[0]:
                peak = max(peak, process.memory_info().rss)
            peak = max(peak, process.memory_info().rss)
            result = json.loads(child.stdout.readline())
            peak = max(peak, result["process_high_water_bytes"])
            result.update(
                {"baseline_rss_bytes": baseline, "peak_rss_bytes": peak, "incremental_peak_bytes": peak - baseline}
            )
            child.stdin.write("done\n")
            child.stdin.flush()
            child.wait()
            results.append(result)
    return results


def _verify_replay(before, path):
    saved = joblib.load(path)["dtw"]
    for sensor in before.paths_:
        np.testing.assert_array_equal(before.acc_cost_mat_[sensor], saved.acc_cost_mat_[sensor])
        np.testing.assert_array_equal(before.matches_start_end_[sensor], saved.matches_start_end_[sensor])
        for a, b in zip(before.paths_[sensor], saved.paths_[sensor]):
            np.testing.assert_array_equal(a, b)


def _write_gallery(rows, output):
    sections = []
    for row in rows:
        if not row.get("plot"):
            continue
        label = escape(f"{row['dataset']} · {row['variant']} · {row['sensor']}")
        sections.append(
            f"<section><h2>{label}</h2><p>Full-recording matches: {row['before_matches']} → {row['after_matches']}. "
            f"Illegal paths: {row['before_illegal_paths']} → {row['after_illegal_paths']}.</p>"
            f'<img src="{escape(row["plot"])}" alt="Before and after DTW plots for {label}" loading="lazy"></section>'
        )
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        "<title>Constrained DTW before and after</title><style>body{font:16px/1.5 system-ui,sans-serif;"
        "max-width:1400px;margin:auto;padding:24px;color:#17212b;background:white}img{width:100%;height:auto}"
        "section{margin:48px 0}h1,h2{font-weight:600}</style><h1>Constrained DTW before and after</h1>"
        '<p><a href="summary.csv">All numerical comparisons</a> · <a href="details.json">Per-path details</a></p>'
        "<p>Each figure reuses the existing DTW plot tool: signal, endpoint cost and accumulated matrix/path. "
        "Only cases with changed final stride borders are shown. "
        "Orange marks selected final borders and their raw path. "
        "Additional detections have not been adjudicated against clinical ground truth.</p>"
        + "".join(sections)
        + "</html>",
        encoding="utf-8",
    )


def _parse_args():
    """Parse command options, keeping POSIX imports out of ordinary comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-revision", default=BASELINE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work/issue-93/comparison")
    parser.add_argument("--saved-baseline", type=Path, help="Optional pre-edit captures to verify historical replay")
    parser.add_argument("--benchmark-engine", choices=("before", "after"), help=argparse.SUPPRESS)
    parser.add_argument(
        "--benchmark-only", action="store_true", help="Measure the two numerical engines in fresh processes"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.benchmark_engine or args.benchmark_only) and sys.platform not in ("linux", "darwin"):
        parser.error("The optional RSS benchmark requires Linux or macOS; ordinary comparisons support Windows.")
    return args


def main():
    """Compare both engines and write numerical evidence and paired DTW plots."""
    args = _parse_args()
    if args.benchmark_engine:
        _benchmark_engine(args.benchmark_engine, args.baseline_revision)
        return
    if args.benchmark_only:
        results = _benchmarks(args.baseline_revision)
        (args.output_dir / "benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results), flush=True)
        return
    legacy = legacy_calculator(args.baseline_revision)
    zero_template, zero_signal = np.zeros((3, 2)), np.zeros((5, 2))
    legacy(zero_template, zero_signal, 1, 1)
    constrained_subsequence_dtw(zero_template, zero_signal, 1, 1)
    plot_source = ROOT / "examples/stride_segmentation/constrained_barth_dtw_stride_segmentation.py"
    plot_dtw = _load_function(plot_source.read_text(encoding="utf-8"), "plot_dtw", {"plt": plt})
    rows, details = [], []
    for name, loader, rate in _datasets():
        data = convert_to_fbf(loader(), left_like="left_", right_like="right_")
        for variant, cls, kwargs in _variants(name):
            prototype = cls(template=BarthOriginalTemplate(use_cols=("gyr_ml", "gyr_si")), **kwargs)
            elapsed = [[], []]
            for _ in range(3):
                with patch(COST_FUNCTION, legacy):
                    start = perf_counter()
                    before = prototype.clone().segment(data, sampling_rate_hz=rate)
                    elapsed[0].append(perf_counter() - start)
                start = perf_counter()
                after = prototype.clone().segment(data, sampling_rate_hz=rate)
                elapsed[1].append(perf_counter() - start)
            if args.saved_baseline:
                _verify_replay(before, args.saved_baseline / f"{name}-{variant}.joblib")
            for sensor in before.paths_:
                row, detail, candidates = _compare(before, after, sensor, name, variant, list(map(median, elapsed)))
                if candidates:
                    _plot(plot_dtw, before, after, sensor, row, args.output_dir)
                if variant == "unconstrained":
                    np.testing.assert_array_equal(before.acc_cost_mat_[sensor], after.acc_cost_mat_[sensor])
                    np.testing.assert_array_equal(before.costs_[sensor], after.costs_[sensor])
                    np.testing.assert_array_equal(before.matches_start_end_[sensor], after.matches_start_end_[sensor])
                    assert row["changed_after_paths"] == row["removed_intervals"] == row["added_intervals"] == 0
                    assert row["changed_endpoint_costs"] == 0
                rows.append(row)
                details.append(detail)
                print(json.dumps(row), flush=True)
            # Write after every configuration so partial evidence survives interruptions.
            pd.DataFrame(rows).to_csv(args.output_dir / "summary.csv", index=False)
            (args.output_dir / "details.json").write_text(
                json.dumps(
                    {
                        "baseline_revision": args.baseline_revision,
                        "head_revision": subprocess.check_output(
                            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                        ).strip(),
                        "cases": details,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    _write_gallery(rows, args.output_dir)


if __name__ == "__main__":
    main()
