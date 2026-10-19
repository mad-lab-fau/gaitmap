"""Participant-held-out tempo-DTW search and isolated inference measurements.

Reuse corrected-foot payloads from optimize_constrained_dtw.py. Candidate caches
contain compact summaries, not DP histories. Every threshold is checked through
public segment(); selected held-out predictions are recomputed without patches.
"""

# Standalone scientific experiment command.
# ruff: noqa: INP001

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from benchmark_constrained_dtw import _counts, _load_challenge
from gaitmap_mad.stride_segmentation.dtw import _tempo_barth_dtw as engine
from optimize_constrained_dtw import _prediction, _score_prediction
from sklearn.model_selection import KFold

from gaitmap.stride_segmentation import BarthOriginalTemplate, ConstrainedBarthDtw, ExperimentalTempoBarthDtw

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6)
GRID = [
    {"n_segments": segments, "penalty_at_20_blocks": penalty}
    for segments in (10, 20)
    for penalty in (0.0, 0.01, 0.1, 1.0)
]
SENSORS = ("left_sensor", "right_sensor")


def _json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _rss():
    values = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
    return int(values["VmHWM"].split()[0]) / 1024


def _model(config, threshold):
    return ExperimentalTempoBarthDtw(
        template=BarthOriginalTemplate(use_cols=("gyr_ml",)),
        n_segments=config["n_segments"],
        tempo_change_penalty=config["penalty_at_20_blocks"] * config["n_segments"] / 20,
        tempo_bounds=(0.65, 2.0),
        max_cost=threshold,
        min_match_length_s=0.6,
        max_match_length_s=3.0,
        snap_to_min_win_ms=300,
    )


def _baseline(params):
    return ConstrainedBarthDtw(
        template=BarthOriginalTemplate(use_cols=tuple(ast.literal_eval(params["dtw__template__use_cols"]))),
        max_cost=params["dtw__max_cost"],
        max_template_stretch_ms=params["dtw__max_template_stretch_ms"],
    )


def _metrics(counts):
    tp, fp, fn = (int(counts[k]) for k in ("tp", "fp", "fn"))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1_score": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def _table(challenge, dtw, payload):
    return _score_prediction(challenge, dtw.stride_list_, payload)


def _summary(dtw):
    return {
        **{f"{s}_intervals": dtw.matches_start_end_[s] for s in SENSORS},
        **{f"{s}_costs": dtw.costs_[s] for s in SENSORS},
    }


def _generate(prepared, config, output):
    """Replay real numerical results and paths with plain, cycle-free callables."""
    payload = joblib.load(prepared)
    captured, alignments = [], {}
    original_calculate, original_paths = engine.tempo_regularized_subsequence_dtw, engine.tempo_dtw_paths

    def calculate(*args, **kwargs):
        result = original_calculate(*args, **kwargs)
        captured.append(result)
        return result

    def paths(x, y, result, endpoints):
        value = original_paths(x, y, result, endpoints)
        alignments[id(result)] = {int(a.block_edges[-1] - 1): a for a in value}
        return value

    start = perf_counter()
    with (
        patch.object(engine, "tempo_regularized_subsequence_dtw", new=calculate),
        patch.object(engine, "tempo_dtw_paths", new=paths),
    ):
        dtw = _model(config, max(THRESHOLDS)).segment(payload["data"], sampling_rate_hz=payload["rate"])
    calculation_seconds = perf_counter() - start
    summary = _summary(dtw)

    def replay(*_args, **_kwargs):
        return next(results)

    def replay_paths(_x, _y, result, endpoints):
        return [alignments[id(result)][int(end)] for end in endpoints]

    # Raising a cost ceiling cannot suppress a lower-cost peak/conflict winner.
    # Verify the resulting compact-threshold equivalence at EVERY search value.
    for threshold in THRESHOLDS:
        results = iter(captured)
        with (
            patch.object(engine, "tempo_regularized_subsequence_dtw", new=replay),
            patch.object(engine, "tempo_dtw_paths", new=replay_paths),
        ):
            direct = _model(config, threshold).segment(payload["data"], sampling_rate_hz=payload["rate"])
        predicted = _prediction(summary, threshold)
        for sensor in SENSORS:
            pd.testing.assert_frame_equal(predicted[sensor], direct.stride_list_[sensor], check_dtype=False)
            np.testing.assert_array_equal(
                summary[f"{sensor}_costs"][summary[f"{sensor}_costs"] <= threshold], direct.costs_[sensor]
            )
    record = {
        "summary": summary,
        "calculation_seconds": calculation_seconds,
        "verification_seconds": perf_counter() - start - calculation_seconds,
        "peak_rss_mib": _rss(),
        "fallbacks": dtw.duration_fallback_count_,
        "tempo_alphabet": {s: dtw.tempo_alphabet_[s].tolist() for s in SENSORS},
    }
    temporary = output.with_suffix(".tmp")
    joblib.dump(record, temporary)
    temporary.replace(output)
    return {
        "participant": prepared.stem,
        "config": config,
        **{k: record[k] for k in ("calculation_seconds", "verification_seconds", "peak_rss_mib", "fallbacks")},
    }


def _fresh(prepared, config, threshold, output):
    payload = joblib.load(prepared)
    start = perf_counter()
    dtw = _model(config, threshold).segment(payload["data"], sampling_rate_hz=payload["rate"])
    joblib.dump(dtw.stride_list_, output)
    return {"participant": prepared.stem, "seconds": perf_counter() - start, "fallbacks": dtw.duration_fallback_count_}


def _select(frame, participants, baseline):
    selections = []
    ids = list(KFold(n_splits=5).split(participants))
    for fold, (train, test) in enumerate(ids):
        train_ids, test_ids = [participants[i] for i in train], [participants[i] for i in test]
        assert train_ids == baseline["folds"][fold]["train"]
        assert test_ids == baseline["folds"][fold]["test"]
        training = (
            frame[frame.participant.isin(train_ids)]
            .groupby(["config", "threshold"], sort=True)[["tp", "fp", "fn"]]
            .sum()
        )
        ranked = [
            {"config": int(config), "threshold": float(threshold), **_metrics(counts)}
            for (config, threshold), counts in training.iterrows()
        ]
        # Stable ties use lower config index, then lower threshold. GRID orders
        # the cheaper/coarser grid first. Zero penalty remains a valid winner.
        for arm, choices in (
            ("tempo", ranked),
            ("zero_penalty", [r for r in ranked if GRID[r["config"]]["penalty_at_20_blocks"] == 0]),
        ):
            best = max(choices, key=lambda r: r["f1_score"])
            selections.append({"fold": fold, "arm": arm, "train": train_ids, "test": test_ids, **best})
    return selections


def _score_grid(challenge, participants, prepared, out):
    records = []
    for participant in participants:
        payload = joblib.load(prepared / f"{participant}.joblib")
        for index, _config in enumerate(GRID):
            summary = joblib.load(out / "cache" / f"{participant}-{index}.joblib")["summary"]
            for threshold in THRESHOLDS:
                table = _score_prediction(challenge, _prediction(summary, threshold), payload)
                records.append({"participant": participant, "config": index, "threshold": threshold, **_counts(table)})
    return pd.DataFrame(records)


def _rescore_baseline(challenge, participants, args, baseline):
    rows = []
    for participant in participants:
        strides = joblib.load(args.baseline_dir / f"{participant}-heldout-strides.joblib")
        payload = joblib.load(args.prepared_dir / f"{participant}.joblib")
        table = _score_prediction(challenge, strides, payload)
        table.to_csv(args.output_dir / f"{participant}-constrained-matches.csv")
        fold = next(f["fold"] for f in baseline["folds"] if participant in f["test"])
        rows.append({"participant": participant, "fold": fold, "arm": "constrained", **_metrics(_counts(table))})
    return rows


def _search(args):
    start = perf_counter()
    out = args.output_dir
    prepared = args.prepared_dir
    participants = [p.stem for p in sorted(prepared.glob("*.joblib"))]
    baseline = json.loads((args.baseline_dir / "results.json").read_text())
    assert len(participants) == 14
    challenge = _load_challenge(args.bench_repo)
    out.joinpath("cache").mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for participant in participants:
            for index, config in enumerate(GRID):
                destination = out / "cache" / f"{participant}-{index}.joblib"
                if not destination.exists():
                    futures.append(pool.submit(_generate, prepared / f"{participant}.joblib", config, destination))
        for future in as_completed(futures):
            record = future.result()
            print(json.dumps({"generated": record}), flush=True)
            with (out / "computations.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        frame = _score_grid(challenge, participants, prepared, out)
        frame.to_csv(out / "search-counts.csv", index=False)
        selections = _select(frame, participants, baseline)
        _json(out / "selections.json", selections)
        jobs = []
        for selected in selections:
            for participant in selected["test"]:
                dest = out / f"{participant}-{selected['arm']}-strides.joblib"
                future = pool.submit(
                    _fresh, prepared / f"{participant}.joblib", GRID[selected["config"]], selected["threshold"], dest
                )
                jobs.append((selected, participant, dest, future))
        heldout = []
        for selected, participant, destination, future in jobs:
            record = future.result()
            strides = joblib.load(destination)
            cached = joblib.load(out / "cache" / f"{participant}-{selected['config']}.joblib")["summary"]
            expected = _prediction(cached, selected["threshold"])
            for sensor in SENSORS:
                pd.testing.assert_frame_equal(strides[sensor], expected[sensor], check_dtype=False)
            table = _score_prediction(challenge, strides, joblib.load(prepared / f"{participant}.joblib"))
            table.to_csv(out / f"{participant}-{selected['arm']}-matches.csv")
            heldout.append(
                {
                    "participant": participant,
                    "fold": selected["fold"],
                    "arm": selected["arm"],
                    **_metrics(_counts(table)),
                    **record,
                }
            )
        # Reuse predictions, never old counts: all arms use the current official
        # scorer and the same prepared references, even across scorer revisions.
        heldout.extend(_rescore_baseline(challenge, participants, args, baseline))
        evaluated = pd.DataFrame(heldout)
        evaluated.to_csv(out / "heldout-per-participant.csv", index=False)
        totals = {
            arm: {
                "pooled": _metrics(data[["tp", "fp", "fn"]].sum()),
                "participant_macro": data[["precision", "recall", "f1_score"]].mean().to_dict(),
            }
            for arm, data in evaluated.groupby("arm")
        }
        _json(
            out / "accuracy.json",
            {"arms": totals, "search_wall_seconds_this_invocation": perf_counter() - start, "selections": selections},
        )
        print(json.dumps(totals, indent=2), flush=True)


def _probe(args):
    spec = json.loads(args.probe_spec.read_text())
    payload = joblib.load(args.prepared_dir / f"{spec['participant']}.joblib")
    model = (
        _baseline(spec["baseline_params"])
        if spec["arm"] == "constrained"
        else _model(GRID[spec["config"]], spec["threshold"])
    )
    # Full-recording warmup covers cost and traceback specializations and is
    # explicitly excluded from warm timings. Every timed call remains uncached.
    start = perf_counter()
    model.segment(payload["data"], sampling_rate_hz=payload["rate"])
    warmup = perf_counter() - start
    samples = []
    for _ in range(args.repeats):
        del model
        model = (
            _baseline(spec["baseline_params"])
            if spec["arm"] == "constrained"
            else _model(GRID[spec["config"]], spec["threshold"])
        )
        start = perf_counter()
        model.segment(payload["data"], sampling_rate_hz=payload["rate"])
        samples.append(perf_counter() - start)
    reference = joblib.load(spec["expected_strides"])
    for sensor in SENSORS:
        pd.testing.assert_frame_equal(reference[sensor], model.stride_list_[sensor], check_dtype=False)
    record = {
        **spec,
        "warmup_seconds": warmup,
        "warm_seconds": samples,
        "warm_median_seconds": float(np.median(samples)),
        "peak_rss_mib": _rss(),
        "signal_seconds_both_feet": sum(len(v) for v in payload["data"].values()) / payload["rate"],
        "numerical_projection_bytes": sum(v.nbytes for v in model.acc_cost_mat_.values()),
        "retained_array_bytes_in_paths": sum(p.nbytes for values in model.paths_.values() for p in values),
        "fallbacks": getattr(model, "duration_fallback_count_", None),
    }
    _json(args.output_dir / f"timing-{spec['participant']}-{spec['arm']}.json", record)
    print(json.dumps(record), flush=True)


def _timing(args):
    selections = json.loads((args.output_dir / "selections.json").read_text())
    baseline = json.loads((args.baseline_dir / "results.json").read_text())
    # Fixed before inspecting labels/scores: first, middle and last participant.
    participants = ["4d91", "80b8", "e54d"]
    requests = []
    for participant in participants:
        fold = next(f for f in baseline["folds"] if participant in f["test"])
        requests.append(
            {
                "participant": participant,
                "arm": "constrained",
                "baseline_params": fold["best_params"],
                "expected_strides": str(args.baseline_dir / f"{participant}-heldout-strides.joblib"),
            }
        )
        for arm in ("tempo", "zero_penalty"):
            choice = next(s for s in selections if participant in s["test"] and s["arm"] == arm)
            requests.append(
                {
                    "participant": participant,
                    "arm": arm,
                    "config": choice["config"],
                    "threshold": choice["threshold"],
                    "expected_strides": str(args.output_dir / f"{participant}-{arm}-strides.joblib"),
                }
            )
    np.random.default_rng(93).shuffle(requests)
    # Sequential isolated processes: parallel search throughput is NOT inference
    # latency. Same thread limits, preprocessing, full warmup and repeats per arm.
    for request in requests:
        spec = args.output_dir / "probe-spec.json"
        _json(spec, request)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--stage",
            "probe",
            "--probe-spec",
            str(spec),
            "--prepared-dir",
            str(args.prepared_dir),
            "--output-dir",
            str(args.output_dir),
            "--repeats",
            str(args.repeats),
        ]
        subprocess.run(command, check=True)
    records = [json.loads(p.read_text()) for p in args.output_dir.glob("timing-*.json")]
    _json(args.output_dir / "timing.json", records)


def main():
    """Run the predeclared search or isolated timing stage, preserving caches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-repo", type=Path)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work/tempo-dtw-benchmark")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--stage", choices=("all", "search", "timing", "probe"), default="all")
    parser.add_argument("--probe-spec", type=Path)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.prepared_dir = args.prepared_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", message="Zero division happened.*")
    if args.stage == "probe":
        _probe(args)
        return
    args.baseline_dir = args.baseline_dir.resolve()
    manifest = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "grid": GRID,
        "thresholds": THRESHOLDS,
        "tempo_bounds": [0.65, 2.0],
        "axes": ["gyr_ml"],
        "min_match_length_s": 0.6,
        "max_match_length_s": 3.0,
        "snap_to_min_win_ms": 300,
        "threads": {
            k: os.environ.get(k)
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__).resolve(),
                ROOT / "packages/gaitmap_mad/src/gaitmap_mad/stride_segmentation/dtw/_tempo_dtw.py",
                ROOT / "packages/gaitmap_mad/src/gaitmap_mad/stride_segmentation/dtw/_tempo_barth_dtw.py",
            ]
        },
        "prepared_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.prepared_dir.glob("*.joblib")
        },
        "prepared_source_manifest": json.loads((args.prepared_dir.parent / "manifest.json").read_text()),
        "baseline_directory": str(args.baseline_dir),
        "benchmark_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.bench_repo, text=True
        ).strip(),
    }
    previous = args.output_dir / "manifest.json"
    if previous.exists():
        saved = json.loads(previous.read_text())
        assert all(
            saved[k] == json.loads(json.dumps(manifest[k]))
            for k in ("grid", "thresholds", "source_sha256", "prepared_sha256", "benchmark_revision")
        )
    else:
        _json(previous, manifest)
    for name, expected in manifest["prepared_source_manifest"]["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    if args.stage in ("all", "search"):
        _search(args)
    if args.stage in ("all", "timing"):
        _timing(args)


if __name__ == "__main__":
    main()
