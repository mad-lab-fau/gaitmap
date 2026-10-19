"""Paired issue-93 accuracy evaluation using gaitmap-bench's instep challenge.

Install gaitmap-datasets, then run with --bench-repo and --dataset-path.
Fixed defaults require no fitting: each participant is evaluated once, then the
five default KFold test partitions are aggregated from those same predictions.
"""

# Standalone developer command.
# ruff: noqa: INP001

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from compare_constrained_dtw import (
    BASELINE,
    COST_FUNCTION,
    ROOT,
    _highlight_interval,
    _load_function,
    _path_checks,
    legacy_calculator,
)
from sklearn.model_selection import KFold

from gaitmap.evaluation_utils import precision_recall_f1_score
from gaitmap.stride_segmentation import ConstrainedBarthDtw
from gaitmap.utils.coordinate_conversion import convert_to_fbf


def _load_challenge(repo):
    sys.path.insert(0, str(repo / "gaitmap-challenges"))
    from gaitmap_challenges.stride_segmentation import sensor_position_comparison_instep  # noqa: PLC0415

    return sensor_position_comparison_instep


def _score(challenge, dtw, datapoint):
    # Modern tpcp calls aggregators as instances. Keep the official scorer's
    # raw table directly; its matching and scalar metrics remain unmodified.
    with patch.object(challenge, "SingleValuePrecisionRecallF1", lambda table: table):
        scores = challenge.final_scorer(SimpleNamespace(safe_run=lambda _: dtw), datapoint)
    table = scores.pop("per_sample")
    assert scores == precision_recall_f1_score(table)
    return scores, table


def _counts(table):
    counts = table["match_type"].value_counts()
    return {kind: int(counts.get(kind, 0)) for kind in ("tp", "fp", "fn")}


def _paired(before, after, old, new):
    old_tp = set(old.loc[old.match_type == "tp", "s_id_ground_truth"].astype(int))
    new_tp = set(new.loc[new.match_type == "tp", "s_id_ground_truth"].astype(int))
    old_intervals = {tuple(map(int, x)) for x in before[["start", "end"]].to_numpy()}
    new_intervals = {tuple(map(int, x)) for x in after[["start", "end"]].to_numpy()}
    return {
        "gained_reference_ids": sorted(new_tp - old_tp),
        "lost_reference_ids": sorted(old_tp - new_tp),
        "removed_intervals": sorted(old_intervals - new_intervals),
        "added_intervals": sorted(new_intervals - old_intervals),
    }


def _plot_case(plot_dtw, before, after, reference, sensor, ref_id, label, participant, output):
    start, end = reference.loc[ref_id, ["start", "end"]].astype(int)
    margin = max(40, (end - start) // 3)
    window = (max(0, start - margin), min(len(before.data[sensor]) - 1, end + margin))
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(14, 8))
    for col, (name, dtw) in enumerate((("Before", before), ("After", after))):
        plot_dtw(dtw, sensor=sensor, axs=axes[:, col], xlim=window, show=False)
        # Render only the inspected window of these long recordings.
        image = axes[2, col].images[0]
        matrix = dtw.acc_cost_mat_[sensor]
        image.set_data(matrix[:, slice(*window)])
        image.set_extent((window[0] - 0.5, window[1] - 0.5, len(matrix) - 0.5, -0.5))
        axes[2, col].set_xlim(*window)
        intervals = dtw.matches_start_end_[sensor]
        interval = None
        if len(intervals):
            distance = np.max(np.abs(intervals - [start, end]), axis=1)
            nearest = int(np.argmin(distance))
            if np.max([intervals[nearest, 0], start]) < np.min([intervals[nearest, 1], end]):
                interval = tuple(intervals[nearest])
        _highlight_interval(axes[:, col], dtw, sensor, interval)
        for ax in axes[:, col]:
            for border in (start, end):
                ax.axvline(border, color="tab:blue", linestyle=":", linewidth=2)
        text = f"[{interval[0]}, {interval[1]})" if interval else "no overlapping detection"
        axes[0, col].set_title(f"{name}: nearest final interval {text}")
        axes[0, col].set_xlabel("")
        axes[2, col].set_xlabel("Signal sample")
    for row in (0, 1):
        limits = [axes[row, col].get_ylim() for col in (0, 1)]
        for ax in axes[row]:
            ax.set_ylim(min(x[0] for x in limits), max(x[1] for x in limits))
    matrices = [dtw.acc_cost_mat_[sensor][:, slice(*window)] for dtw in (before, after)]
    vmax = max(float(np.quantile(m[np.isfinite(m)], 0.95)) for m in matrices)
    for ax in axes[2]:
        ax.images[0].set_clim(0, vmax)
    fig.suptitle(
        f"{participant} · {sensor} · {label} · reference [{start}, {end})\n"
        "Blue dotted: expert reference borders. Orange dashed: nearest final detection and its raw path."
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _aggregate(rows, tables, participants):
    summary = {}
    frame = pd.DataFrame(rows)
    for protocol in frame.protocol.unique():
        arms = {}
        for engine in ("before", "after"):
            selected = [tables[protocol, engine, p] for p in participants]
            table = pd.concat(selected)
            per_participant = [precision_recall_f1_score(t) for t in selected]
            folds = []
            for _, test in KFold(n_splits=5).split(participants):
                fold = pd.concat([selected[i] for i in test])
                folds.append({**precision_recall_f1_score(fold), **_counts(fold)})
            arms[engine] = {
                "micro": {**precision_recall_f1_score(table), **_counts(table)},
                "participant_macro": pd.DataFrame(per_participant).mean().to_dict(),
                "five_fold_micro": folds,
                "fold_mean": pd.DataFrame(folds).mean().to_dict(),
                "segmentation_seconds": float(
                    frame.loc[(frame.protocol == protocol) & (frame.engine == engine), "seconds"].sum()
                ),
            }
        rng = np.random.default_rng(93)
        sample = rng.integers(0, len(participants), size=(2000, len(participants)))
        bootstraps = []
        for engine in ("before", "after"):
            counts = np.array([list(_counts(tables[protocol, engine, p]).values()) for p in participants])
            tp, fp, fn = counts[sample].sum(axis=1).T
            bootstraps.append(2 * tp / (2 * tp + fp + fn))
        arms["paired_micro_f1_delta_bootstrap_95_percentile"] = np.quantile(
            bootstraps[1] - bootstraps[0], [0.025, 0.975]
        ).tolist()
        summary[protocol] = arms
    return summary


def _run_participant(challenge, datapoint, legacy, plot_dtw, args, rows, tables, changes, best):
    participant = datapoint.group_label.participant
    raw = challenge.Challenge.get_imu_data(datapoint)
    reference = challenge.Challenge.get_reference_stride_list(datapoint)
    for protocol in ("explicit_feet", "historical_entry"):
        feet = (
            {"left": ["left_sensor"], "right": ["right_sensor"]}
            if protocol == "explicit_feet"
            else {"left_like": "l", "right_like": "r"}
        )
        data = convert_to_fbf(raw, **feet)
        engines, matches = {}, {}
        for engine in ("before", "after"):
            context = patch(COST_FUNCTION, legacy) if engine == "before" else nullcontext()
            with context:
                start = perf_counter()
                dtw = ConstrainedBarthDtw().segment(data, sampling_rate_hz=datapoint.sampling_rate_hz)
                seconds = perf_counter() - start
            _, table = _score(challenge, dtw, datapoint)
            tables[protocol, engine, participant] = table
            table.to_csv(args.output_dir / f"{participant}-{protocol}-{engine}-matches.csv")
            for sensor in dtw.paths_:
                checks = _path_checks(dtw, sensor)
                if engine == "after":
                    assert not any(c["illegal"] or c["cost_mismatch"] for c in checks)
                matching = table.loc[sensor]
                row = {
                    "participant": participant,
                    "protocol": protocol,
                    "engine": engine,
                    "sensor": sensor,
                    **_counts(matching),
                    **precision_recall_f1_score(matching),
                    "seconds": seconds / 2,
                    "returned_paths": len(checks),
                    "illegal_paths": sum(c["illegal"] for c in checks),
                    "cost_mismatches": sum(c["cost_mismatch"] for c in checks),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
            engines[engine], matches[engine] = dtw, table
        for sensor in reference:
            before, after = engines["before"], engines["after"]
            change = _paired(
                before.stride_list_[sensor],
                after.stride_list_[sensor],
                matches["before"].loc[sensor],
                matches["after"].loc[sensor],
            )
            changes.append({"participant": participant, "protocol": protocol, "sensor": sensor, **change})
            if protocol != "explicit_feet":
                continue
            for label, key in (
                ("gained true detection", "gained_reference_ids"),
                ("lost true detection", "lost_reference_ids"),
            ):
                count = len(change[key])
                if count > best.get(key, 0):
                    best[key] = count
                    _plot_case(
                        plot_dtw,
                        before,
                        after,
                        reference[sensor],
                        sensor,
                        change[key][0],
                        label,
                        participant,
                        args.output_dir / f"{key}.png",
                    )


def main():
    """Evaluate fixed defaults with both correct and historical benchmark foot mapping."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-repo", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--baseline-revision", default=BASELINE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work/issue-93/benchmark")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    challenge = _load_challenge(args.bench_repo)
    dataset = challenge.ChallengeDataset(data_folder=args.dataset_path)
    participants = dataset.index.participant.tolist()
    legacy = legacy_calculator(args.baseline_revision)
    zero = np.zeros((5, 3))
    from gaitmap_mad.stride_segmentation.dtw._constrained_dtw import constrained_subsequence_dtw  # noqa: PLC0415

    for layout in (np.ascontiguousarray, np.asfortranarray):
        legacy(layout(zero), layout(zero), 24.0, 24.0)
        constrained_subsequence_dtw(layout(zero), layout(zero), 24.0, 24.0)
    plot_source = ROOT / "examples/stride_segmentation/constrained_barth_dtw_stride_segmentation.py"
    plot_dtw = _load_function(plot_source.read_text(encoding="utf-8"), "plot_dtw", {"plt": plt})
    rows, tables, changes, best = [], {}, [], {}
    for datapoint in dataset:
        _run_participant(challenge, datapoint, legacy, plot_dtw, args, rows, tables, changes, best)
        pd.DataFrame(rows).to_csv(args.output_dir / "per_sensor.csv", index=False)
        (args.output_dir / "changes.json").write_text(json.dumps(changes, indent=2), encoding="utf-8")
    result = {
        "baseline_revision": args.baseline_revision,
        "corrected_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "benchmark_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.bench_repo, text=True
        ).strip(),
        "participants": participants,
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("gaitmap", "gaitmap-mad", "gaitmap-datasets", "numpy", "pandas", "numba", "tpcp")
        },
        "reference_file_sha256": {
            str(p.relative_to(args.dataset_path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.dataset_path.rglob("manual_stride_border.csv"))
        },
        "match_tolerance_samples": int(challenge.Challenge.match_tolerance_s * dataset.sampling_rate_hz),
        "parameters": json.loads(ConstrainedBarthDtw().to_json()),
        "results": _aggregate(rows, tables, participants),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
