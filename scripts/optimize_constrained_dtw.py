"""Rerun the five-fold gaitmap-bench constrained-DTW search with correct feet.

Uses the imported gaitmap checkout, including its pending numerical correction.
Optuna 3.2.0, 100 sequential trials per fold, and TPESampler(seed=42) are used.
Compact threshold-independent candidate summaries avoid retaining cost matrices.
Every summary is checked against public segment() calls at three thresholds;
held-out predictions are also checked by fresh, uncached public calculations.
"""

# Standalone scientific experiment command.
# ruff: noqa: INP001

import argparse
import ast
import hashlib
import importlib.metadata
import json
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import optuna
import pandas as pd
from benchmark_constrained_dtw import _counts, _load_challenge, _score
from compare_constrained_dtw import COST_FUNCTION, ROOT
from sklearn.model_selection import KFold

from gaitmap.evaluation_utils import precision_recall_f1_score
from gaitmap.stride_segmentation import BarthOriginalTemplate, ConstrainedBarthDtw
from gaitmap.utils.coordinate_conversion import convert_to_fbf

AXES = [("gyr_ml", "gyr_si", "gyr_pa"), ("gyr_ml", "gyr_si"), ("gyr_ml",)]
CHOICES = ['("gyr_ml", "gyr_si", "gyr_pa")', '("gyr_ml", "gyr_si")', '("gyr_ml",)']
CEILING = 3.5


def _atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _pipeline(axis, vertical_ms, max_cost):
    return ConstrainedBarthDtw(
        template=BarthOriginalTemplate(use_cols=AXES[axis]),
        max_template_stretch_ms=vertical_ms,
        max_cost=max_cost,
    )


def _prediction(summary, threshold):
    result = {}
    for sensor in ("left_sensor", "right_sensor"):
        intervals = summary[f"{sensor}_intervals"][summary[f"{sensor}_costs"] <= threshold]
        result[sensor] = pd.DataFrame(intervals, columns=["start", "end"]).rename_axis("s_id")
    return result


def _build_summary(prepared, axis, steps, destination):
    """Calculate fixed-template candidates without consulting reference labels."""
    payload = joblib.load(prepared)
    rate, data = payload["rate"], payload["data"]
    captured = []
    from gaitmap_mad.stride_segmentation.dtw._base_dtw import (  # noqa: PLC0415
        subsequence_cost_matrix_with_constrains,
    )

    def calculate(*args, **kwargs):
        result = subsequence_cost_matrix_with_constrains(*args, **kwargs)
        captured.append(result)
        return result

    start = perf_counter()
    with patch(COST_FUNCTION, side_effect=calculate):
        dtw = _pipeline(axis, steps * 1000 / rate, CEILING).segment(data, sampling_rate_hz=rate)
    summary = {}
    for sensor in dtw.costs_:
        intervals = dtw.matches_start_end_[sensor]
        summary[f"{sensor}_intervals"] = np.asarray(intervals, dtype=int).reshape(-1, 2)
        summary[f"{sensor}_costs"] = np.asarray(dtw.costs_[sensor])
    # Forward DP is independent of max_cost. Exercise the complete public peak
    # selection, snapping, duration and conflict processing with that same DP.
    for threshold in (2.0, 2.75, CEILING):
        with patch(COST_FUNCTION, side_effect=iter(captured)):
            direct = _pipeline(axis, steps * 1000 / rate, threshold).segment(data, sampling_rate_hz=rate)
        predicted = _prediction(summary, threshold)
        for sensor in predicted:
            pd.testing.assert_frame_equal(predicted[sensor], direct.stride_list_[sensor], check_dtype=False)
            costs = summary[f"{sensor}_costs"]
            np.testing.assert_array_equal(costs[costs <= threshold], direct.costs_[sensor])
    temporary = Path(destination).with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **summary)
    temporary.replace(destination)
    return {"file": str(destination), "seconds": perf_counter() - start}


def _direct_check(prepared, params, cache_file, output):
    payload = joblib.load(prepared)
    axis = AXES.index(tuple(ast.literal_eval(params["dtw__template__use_cols"])))
    dtw = _pipeline(axis, params["dtw__max_template_stretch_ms"], params["dtw__max_cost"]).segment(
        payload["data"], sampling_rate_hz=payload["rate"]
    )
    with np.load(cache_file) as summary:
        predicted = _prediction(summary, params["dtw__max_cost"])
    for sensor in predicted:
        pd.testing.assert_frame_equal(predicted[sensor], dtw.stride_list_[sensor], check_dtype=False)
    joblib.dump(dtw.stride_list_, output)
    return str(output)


def _prepare(challenge, dataset_path, output):
    folder = output / "prepared"
    folder.mkdir(exist_ok=True)
    dataset = challenge.ChallengeDataset(data_folder=dataset_path)
    payloads = {}
    for point in dataset:
        participant = point.group_label.participant
        path = folder / f"{participant}.joblib"
        if not path.exists():
            data = convert_to_fbf(challenge.Challenge.get_imu_data(point), left=["left_sensor"], right=["right_sensor"])
            references = challenge.Challenge.get_reference_stride_list(point)
            joblib.dump({"data": data, "references": references, "rate": point.sampling_rate_hz}, path)
        payloads[participant] = joblib.load(path)
        print(json.dumps({"prepared": participant}), flush=True)
    return payloads


def _score_prediction(challenge, strides, payload):
    point = SimpleNamespace(
        sampling_rate_hz=payload["rate"],
        segmented_stride_list_per_sensor_={
            "l_instep": payload["references"]["left_sensor"],
            "r_instep": payload["references"]["right_sensor"],
        },
    )
    return _score(challenge, SimpleNamespace(stride_list_=strides), point)[1]


def _key(participant, params, payloads, output):
    axis = AXES.index(tuple(ast.literal_eval(params["dtw__template__use_cols"])))
    steps = int(np.round(params["dtw__max_template_stretch_ms"] / 1000 * payloads[participant]["rate"]))
    path = output / "candidates" / f"{participant}-axes{axis}-vertical{steps}.npz"
    return participant, axis, steps, path


def _ensure_candidates(pool, requests, output):
    unique = {str(key[3]): key for key in requests if not key[3].exists()}
    futures = {
        pool.submit(_build_summary, output / "prepared" / f"{p}.joblib", axis, steps, path): str(path)
        for p, axis, steps, path in unique.values()
    }
    for future in as_completed(futures):
        record = future.result()
        with (output / "candidate-computations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")


def _trial_score(challenge, participants, params, payloads, output):
    tables = []
    for participant in participants:
        path = _key(participant, params, payloads, output)[3]
        with np.load(path) as summary:
            strides = _prediction(summary, params["dtw__max_cost"])
        tables.append(_score_prediction(challenge, strides, payloads[participant]))
    combined = pd.concat(tables)
    return {**precision_recall_f1_score(combined), **_counts(combined)}


def _sampler_checkpoint(study, path):
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(pickle.dumps(study.sampler))
    temporary.replace(path)


def _studies(output, splits, participants):
    studies = []
    for fold, (train, test) in enumerate(splits):
        path = output / f"fold-{fold}-sampler.pkl"
        sampler = pickle.loads(path.read_bytes()) if path.exists() else optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            study_name=f"corrected-constrained-dtw-fold-{fold}",
            storage=f"sqlite:///{output / f'fold-{fold}.sqlite3'}",
            load_if_exists=True,
            sampler=sampler,
            direction="maximize",
        )
        study.set_user_attr("training_participants", [participants[i] for i in train])
        study.set_user_attr("test_participants", [participants[i] for i in test])
        studies.append(study)
    return studies


def _next_trial(study):
    running = study.get_trials(states=(optuna.trial.TrialState.RUNNING,))
    if running:
        assert len(running) == 1
        return running[0]
    trial = study.ask()
    trial.suggest_float("dtw__max_cost", 2.0, CEILING)
    trial.suggest_float("dtw__max_template_stretch_ms", 100.0, 300.0)
    trial.suggest_categorical("dtw__template__use_cols", CHOICES)
    trial.set_user_attr("training_participants", study.user_attrs["training_participants"])
    return trial


def _optimize(studies, splits, participants, payloads, challenge, pool, args):
    rounds = 0
    while any(len(s.get_trials(states=(optuna.trial.TrialState.COMPLETE,))) < 100 for s in studies):
        if args.round_limit is not None and rounds >= args.round_limit:
            return False
        pending, requests = [], []
        for fold, study in enumerate(studies):
            if len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))) >= 100:
                continue
            trial = _next_trial(study)
            _sampler_checkpoint(study, args.output_dir / f"fold-{fold}-sampler.pkl")
            train = [participants[i] for i in splits[fold][0]]
            assert not set(train) & set(study.user_attrs["test_participants"])
            requests.extend(_key(p, trial.params, payloads, args.output_dir) for p in train)
            pending.append((fold, trial, train))
        _ensure_candidates(pool, requests, args.output_dir)
        for fold, trial, train in pending:
            scores = _trial_score(challenge, train, trial.params, payloads, args.output_dir)
            studies[fold].tell(trial.number, scores["f1_score"])
            _sampler_checkpoint(studies[fold], args.output_dir / f"fold-{fold}-sampler.pkl")
            studies[fold].trials_dataframe().to_csv(args.output_dir / f"fold-{fold}-trials.csv", index=False)
            print(
                json.dumps({"fold": fold, "trial": trial.number, "params": trial.params, "training": scores}),
                flush=True,
            )
        rounds += 1
    return True


def _evaluate(studies, splits, participants, payloads, challenge, pool, output):
    rows, tables, folds, jobs = [], [], [], []
    requests = []
    for fold, study in enumerate(studies):
        for i in splits[fold][1]:
            requests.append(_key(participants[i], study.best_params, payloads, output))
    _ensure_candidates(pool, requests, output)
    for fold, study in enumerate(studies):
        selected = [participants[i] for i in splits[fold][1]]
        for participant in selected:
            path = _key(participant, study.best_params, payloads, output)[3]
            jobs.append(
                (
                    fold,
                    participant,
                    pool.submit(
                        _direct_check,
                        output / "prepared" / f"{participant}.joblib",
                        study.best_params,
                        path,
                        output / f"{participant}-heldout-strides.joblib",
                    ),
                )
            )
        folds.append(
            {
                "fold": fold,
                "train": study.user_attrs["training_participants"],
                "test": selected,
                "best_params": study.best_params,
                "training_f1": study.best_value,
            }
        )
    for fold, participant, future in jobs:
        strides = joblib.load(future.result())
        table = _score_prediction(challenge, strides, payloads[participant])
        table.to_csv(output / f"{participant}-heldout-matches.csv")
        tables.append(table)
        rows.append({"fold": fold, "participant": participant, **precision_recall_f1_score(table), **_counts(table)})
    frame = pd.DataFrame(rows).sort_values(["fold", "participant"])
    frame.to_csv(output / "heldout-per-participant.csv", index=False)
    for fold in folds:
        counts = frame.loc[frame.fold == fold["fold"], ["tp", "fp", "fn"]].sum()
        tp, fp, fn = counts
        fold["heldout"] = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": tp / (tp + fp),
            "recall": tp / (tp + fn),
            "f1_score": 2 * tp / (2 * tp + fp + fn),
        }
    combined = pd.concat(tables)
    assert set(frame.participant) == set(participants)
    assert len(frame) == 14
    result = {
        "folds": folds,
        "pooled_heldout": {**precision_recall_f1_score(combined), **_counts(combined)},
        "participant_macro": frame[["precision", "recall", "f1_score"]].mean().to_dict(),
    }
    _atomic_json(output / "results.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    """Run or resume the complete outer-fold optimization experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-repo", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work/issue-93/optimized-rerun")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--round-limit", type=int, help="Checkpoint after this many rounds; full target remains100/fold"
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidates").mkdir(exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "gaitmap_revision": revision,
        "benchmark_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.bench_repo, text=True
        ).strip(),
        "data_folder": str(args.dataset_path.resolve()),
        "trials_per_fold": 100,
        "sampler_seed": 42,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["gaitmap", "gaitmap-mad", "gaitmap-datasets", "tpcp", "optuna", "numpy", "pandas", "numba"]
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / "packages/gaitmap_mad/src/gaitmap_mad/stride_segmentation/dtw/_constrained_dtw.py",
                ROOT / "packages/gaitmap_mad/src/gaitmap_mad/stride_segmentation/dtw/_base_dtw.py",
                ROOT / "packages/gaitmap_mad/src/gaitmap_mad/stride_segmentation/dtw/_barth_dtw.py",
            ]
        },
    }
    path = args.output_dir / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        for key in (
            "source_sha256",
            "versions",
            "data_folder",
            "benchmark_revision",
            "sampler_seed",
            "trials_per_fold",
        ):
            assert previous[key] == manifest[key]
    else:
        _atomic_json(path, manifest)
    challenge = _load_challenge(args.bench_repo)
    payloads = _prepare(challenge, args.dataset_path, args.output_dir)
    participants = list(payloads)
    splits = list(KFold(n_splits=5).split(participants))
    _atomic_json(args.output_dir / "participants.json", participants)
    studies = _studies(args.output_dir, splits, participants)
    # Processes, rather than threads, isolate the temporary public-calculator
    # patches used solely for cache-equivalence verification.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if _optimize(studies, splits, participants, payloads, challenge, pool, args):
            _evaluate(studies, splits, participants, payloads, challenge, pool, args.output_dir)
        else:
            print("Checkpoint saved; rerun without --round-limit to complete all500trials.", flush=True)


if __name__ == "__main__":
    main()
