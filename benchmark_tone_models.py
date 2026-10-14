"""Leakage-safe comparison of tone classifiers on the existing dialogue corpus.

This script deliberately consumes the saved forced-alignment cache. It never
loads WhisperX or rebuilds alignments, so model comparison does not compete
with the running API for memory.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import joblib
import numpy as np
import opencc
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import capt_core as core


DEFAULT_DATA_DIR = r"D:\hautran\Lab\mandarin-speaking data\Recorded Data\lessons_5_8_FINAL_content_verified_v3_finallllll"
DEFAULT_CACHE_PATH = "alignment_cache_v2.json"
DEFAULT_OUTPUT_DIR = "tone_benchmark"
MODEL_PATH = "tone_model_v2.joblib"
NEUTRAL_TONE = 5
TONE_LABELS = [1, 2, 3, 4, 5]
OUTER_FOLDS = 5
INNER_FOLDS = 3
MIN_NON_NEUTRAL_GAIN = 0.02
MAX_MACRO_RECALL_DROP = 0.015
MAX_PER_TONE_RECALL_DROP = 0.05


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nested group-CV benchmark using the current dialogue data only."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--outer-folds", type=int, default=OUTER_FOLDS)
    parser.add_argument(
        "--raw-pitch-focus",
        action="store_true",
        help="Compare current HGB, neutral-filtered HGB, and two absolute-pitch variants.",
    )
    return parser.parse_args()


def _feature_indices(variant):
    if variant == "all":
        return list(range(core.FEATURE_DIM))
    if variant == "all_plus_raw_pitch":
        return list(range(core.FEATURE_DIM + 5))
    if variant == "no_context":
        return list(range(14))
    if variant == "no_contour":
        return list(range(9)) + list(range(14, core.FEATURE_DIM))
    raise ValueError(f"Unknown feature variant: {variant}")


def _candidate_specs():
    """Small, predeclared model set; all selection happens inside each fold."""
    specs = [
        {
            "name": "hgb_current_all",
            "family": "HistGradientBoosting",
            "feature_variant": "all",
            "params": {"max_iter": 200, "learning_rate": 0.05},
        },
        {
            "name": "hgb_shallow_all",
            "family": "HistGradientBoosting",
            "feature_variant": "all",
            "params": {
                "max_iter": 200,
                "learning_rate": 0.05,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 20,
            },
        },
        {
            "name": "hgb_regularized_all",
            "family": "HistGradientBoosting",
            "feature_variant": "all",
            "params": {
                "max_iter": 200,
                "learning_rate": 0.05,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 30,
                "l2_regularization": 1.0,
            },
        },
        {
            "name": "hgb_current_no_context",
            "family": "HistGradientBoosting",
            "feature_variant": "no_context",
            "params": {"max_iter": 200, "learning_rate": 0.05},
        },
        {
            "name": "hgb_current_no_contour",
            "family": "HistGradientBoosting",
            "feature_variant": "no_contour",
            "params": {"max_iter": 200, "learning_rate": 0.05},
        },
        {
            "name": "hgb_current_exclude_api_neutrals",
            "family": "HistGradientBoosting",
            "feature_variant": "all",
            "training_filter": "exclude_api_neutral_examples",
            "params": {"max_iter": 200, "learning_rate": 0.05},
        },
        {
            "name": "hgb_regularized_exclude_api_neutrals",
            "family": "HistGradientBoosting",
            "feature_variant": "all",
            "training_filter": "exclude_api_neutral_examples",
            "params": {
                "max_iter": 200,
                "learning_rate": 0.05,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 30,
                "l2_regularization": 1.0,
            },
        },
        {
            "name": "hgb_current_exclude_api_neutrals_raw_pitch",
            "family": "HistGradientBoosting",
            "feature_variant": "all_plus_raw_pitch",
            "training_filter": "exclude_api_neutral_examples",
            "params": {"max_iter": 200, "learning_rate": 0.05},
        },
        {
            "name": "hgb_regularized_exclude_api_neutrals_raw_pitch",
            "family": "HistGradientBoosting",
            "feature_variant": "all_plus_raw_pitch",
            "training_filter": "exclude_api_neutral_examples",
            "params": {
                "max_iter": 200,
                "learning_rate": 0.05,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 30,
                "l2_regularization": 1.0,
            },
        },
    ]

    for min_leaf in (2, 5):
        for max_features in (0.7, 1.0):
            specs.append(
                {
                    "name": f"rf_leaf{min_leaf}_features{max_features}",
                    "family": "RandomForest",
                    "feature_variant": "all",
                    "params": {
                        "n_estimators": 150,
                        "min_samples_leaf": min_leaf,
                        "max_features": max_features,
                        "n_jobs": 1,
                    },
                }
            )

    for c_value in (0.3, 1.0, 3.0, 10.0):
        specs.append(
            {
                "name": f"svc_rbf_c{c_value:g}",
                "family": "RBF SVC",
                "feature_variant": "all",
                "params": {"C": c_value},
            }
        )
    return specs


def _build_estimator(spec):
    params = dict(spec["params"])
    if spec["family"] == "HistGradientBoosting":
        return HistGradientBoostingClassifier(random_state=42, **params)
    if spec["family"] == "RandomForest":
        return RandomForestClassifier(random_state=42, **params)
    if spec["family"] == "RBF SVC":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", **params))
    raise ValueError(f"Unknown family: {spec['family']}")


def _collect_data(data_dir, cache_path):
    data_dir = Path(data_dir)
    csv_path = data_dir / "FULL_audio_content_audit.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")
    if not Path(cache_path).is_file():
        raise FileNotFoundError(
            f"Alignment cache not found: {cache_path}. This benchmark will not load WhisperX."
        )

    df = pd.read_csv(csv_path)
    cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    s2twp = opencc.OpenCC("s2twp")
    records = []
    feature_rows = []

    for row in df.itertuples(index=False):
        filename = str(row.filename).strip()
        script = str(row.script).strip()
        audio_path = data_dir / filename
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not core.alignment_cache_is_current(
            cache.get(filename), script, audio_path, cache_path
        ):
            raise ValueError(
                f"Alignment cache is missing or stale for {filename}; rebuild it separately."
            )

        clauses = core.prepare_target(s2twp.convert(script))
        chars = list("".join(clauses))
        tones = core.expected_tones(clauses)
        align = cache[filename]
        if len(chars) != len(tones) or len(chars) != len(align["starts"]):
            raise ValueError(
                f"Character/label/alignment count mismatch for {filename}: "
                f"{len(chars)}/{len(tones)}/{len(align['starts'])}"
            )

        samples, sample_rate = core.load_audio_samples(str(audio_path))
        items = [
            {"start": start, "end": end, "interpolated": False}
            for start, end in zip(align["starts"], align["ends"])
        ]
        core.fill_missing_timestamps(items, len(samples) / sample_rate)
        features, windows, (utterance_pitch_mean, utterance_pitch_std) = core.utterance_features(
            samples,
            sample_rate,
            [item["start"] for item in items],
            tones,
            [len(clause) for clause in clauses],
            return_pitch_stats=True,
        )

        offsets = []
        total = 0
        for clause in clauses:
            offsets.append((total, total + len(clause), clause))
            total += len(clause)

        for index, (char, tone, feature, item, window) in enumerate(
            zip(chars, tones, features, items, windows)
        ):
            clause_text = ""
            clause_index = 0
            local_index = index
            for start, end, text in offsets:
                if start <= index < end:
                    clause_text = text
                    clause_index = start
                    local_index = index - start
                    break

            align_duration = None
            if (
                index != len(chars) - 1
                and item["start"] is not None
                and item["end"] is not None
            ):
                align_duration = item["end"] - item["start"]

            low_confidence = core.is_low_confidence(feature, align_duration)
            record = {
                "filename": filename,
                "group": f"{row.lesson}-{row.story}",
                "char_index": index,
                "char": char,
                "clause": clause_text,
                "clause_char_index": local_index,
                "expected_tone": tone,
                "has_features": feature is not None,
                "low_confidence": bool(low_confidence),
                "align_duration_sec": align_duration,
                "f0_window_sec": list(window) if window is not None else None,
            }
            records.append(record)
            if tone is not None and feature is not None:
                # Keep the deployed 19 features intact and append absolute
                # semitone register features for this benchmark variant.
                raw_pitch_features = [
                    feature[4] * utterance_pitch_std + utterance_pitch_mean,
                    feature[3] * utterance_pitch_std,
                    feature[8] * utterance_pitch_std,
                    utterance_pitch_mean,
                    utterance_pitch_std,
                ]
                feature_rows.append(
                    {**record, "features": list(feature) + raw_pitch_features}
                )

    feature_frame = pd.DataFrame(feature_rows)
    if feature_frame.empty:
        raise ValueError("No characters with both a tone label and measurable F0.")

    X = np.asarray(feature_frame["features"].tolist(), dtype=float)
    y = feature_frame["expected_tone"].to_numpy(dtype=int)
    groups = feature_frame["group"].to_numpy(dtype=str)
    expected_feature_dim = core.FEATURE_DIM + 5
    if X.shape[1] != expected_feature_dim:
        raise ValueError(f"Expected {expected_feature_dim} features, got {X.shape[1]}.")
    return pd.DataFrame(records), feature_frame.drop(columns="features"), X, y, groups


def _metrics(y_true, y_pred):
    non_neutral = y_true != NEUTRAL_TONE
    recalls = recall_score(
        y_true, y_pred, labels=TONE_LABELS, average=None, zero_division=0
    )
    return {
        "accuracy_all_assessable": float(accuracy_score(y_true, y_pred)),
        "accuracy_non_neutral": float(accuracy_score(y_true[non_neutral], y_pred[non_neutral])),
        "macro_recall": float(np.mean(recalls)),
        "recall_by_tone": {str(tone): float(value) for tone, value in zip(TONE_LABELS, recalls)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=TONE_LABELS).tolist(),
        "n_assessable": int(len(y_true)),
        "n_non_neutral": int(non_neutral.sum()),
    }


def _apply_api_neutral_rule(prediction, chars):
    prediction = np.asarray(prediction).copy()
    mask = np.isin(chars, list(core.NEUTRAL_TONE_CHARS))
    prediction[mask] = NEUTRAL_TONE
    return prediction


def _fit_and_predict(spec, X_train, y_train, chars_train, X_valid, chars_valid):
    train_mask = np.ones(len(y_train), dtype=bool)
    if spec.get("training_filter") == "exclude_api_neutral_examples":
        # Match runtime behavior exactly: the API short-circuits by character,
        # regardless of the tone label attached to that example.
        train_mask &= ~np.isin(chars_train, list(core.NEUTRAL_TONE_CHARS))
    indices = _feature_indices(spec["feature_variant"])
    estimator = _build_estimator(spec)
    estimator.fit(X_train[train_mask][:, indices], y_train[train_mask])
    raw_prediction = estimator.predict(X_valid[:, indices])
    return estimator, raw_prediction, _apply_api_neutral_rule(raw_prediction, chars_valid)


def _inner_score(spec, X, y, groups, chars, n_splits=INNER_FOLDS):
    indices = _feature_indices(spec["feature_variant"])
    prediction = np.zeros_like(y)
    cv = GroupKFold(n_splits=n_splits)
    for train_index, valid_index in cv.split(X, y, groups):
        _, _, prediction[valid_index] = _fit_and_predict(
            spec,
            X[train_index],
            y[train_index],
            chars[train_index],
            X[valid_index],
            chars[valid_index],
        )
    return _metrics(y, prediction)


def _select_spec(X, y, groups, chars, specs, n_splits=INNER_FOLDS):
    scores = []
    for spec in specs:
        metrics = _inner_score(spec, X, y, groups, chars, n_splits=n_splits)
        scores.append((spec, metrics))
    scores.sort(
        key=lambda item: (
            item[1]["accuracy_non_neutral"],
            item[1]["accuracy_all_assessable"],
            item[1]["macro_recall"],
            item[0]["name"],
        ),
        reverse=True,
    )
    return scores[0][0], scores[0][1], scores


def _cross_validate(X, y, groups, chars, outer_folds, raw_pitch_focus=False):
    if len(np.unique(groups)) != 12:
        raise ValueError(
            f"Expected the existing 12 lesson-story groups; found {len(np.unique(groups))}."
        )
    if outer_folds != 5:
        raise ValueError("Use the same 5 GroupKFold splits as the reported 85.1% baseline.")

    specs = _candidate_specs()
    if raw_pitch_focus:
        focused_names = {
            "hgb_current_all",
            "hgb_current_exclude_api_neutrals",
            "hgb_current_exclude_api_neutrals_raw_pitch",
            "hgb_regularized_exclude_api_neutrals_raw_pitch",
        }
        specs = [spec for spec in specs if spec["name"] in focused_names]
    baseline_spec = specs[0]
    baseline_raw_prediction = np.zeros_like(y)
    selected_raw_prediction = np.zeros_like(y)
    fold_records = []
    inner_records = []
    outer_cv = GroupKFold(n_splits=outer_folds)

    for fold_number, (train_index, test_index) in enumerate(
        outer_cv.split(X, y, groups), start=1
    ):
        train_groups = groups[train_index]
        if set(train_groups) & set(groups[test_index]):
            raise AssertionError("Group leakage detected in outer fold.")
        baseline = _build_estimator(baseline_spec)
        base_indices = _feature_indices(baseline_spec["feature_variant"])
        baseline.fit(X[train_index][:, base_indices], y[train_index])
        baseline_raw_prediction[test_index] = baseline.predict(X[test_index][:, base_indices])

        # Inner validation uses only groups belonging to this outer training set.
        inner_splits = min(INNER_FOLDS, len(np.unique(train_groups)))
        best_spec, best_inner_metrics, all_inner = _select_spec(
            X[train_index],
            y[train_index],
            train_groups,
            chars[train_index],
            specs,
            n_splits=inner_splits,
        )
        _, selected_raw, _ = _fit_and_predict(
            best_spec,
            X[train_index],
            y[train_index],
            chars[train_index],
            X[test_index],
            chars[test_index],
        )
        selected_raw_prediction[test_index] = selected_raw

        fold_records.append(
            {
                "outer_fold": fold_number,
                "test_groups": sorted(set(groups[test_index].tolist())),
                "selected_candidate": best_spec["name"],
                "inner_accuracy_non_neutral": best_inner_metrics["accuracy_non_neutral"],
            }
        )
        for spec, metric in all_inner:
            inner_records.append(
                {
                    "outer_fold": fold_number,
                    "candidate": spec["name"],
                    "inner_accuracy_non_neutral": metric["accuracy_non_neutral"],
                    "inner_accuracy_all_assessable": metric["accuracy_all_assessable"],
                    "inner_macro_recall": metric["macro_recall"],
                }
            )
        print(
            f"Outer fold {fold_number}/{outer_folds}: chose {best_spec['name']} "
            f"(inner non-neutral {best_inner_metrics['accuracy_non_neutral']:.3f})"
        )

    return (
        baseline_raw_prediction,
        selected_raw_prediction,
        fold_records,
        inner_records,
        specs,
    )


def _per_group_metrics(
    feature_frame,
    baseline_raw_prediction,
    baseline_prediction,
    selected_raw_prediction,
    selected_prediction,
):
    data = feature_frame.copy()
    data["baseline_raw_prediction"] = baseline_raw_prediction
    data["baseline_prediction"] = baseline_prediction
    data["selected_raw_prediction"] = selected_raw_prediction
    data["selected_prediction"] = selected_prediction
    rows = []
    for group, subset in data.groupby("group", sort=True):
        y = subset["expected_tone"].to_numpy(dtype=int)
        non_neutral = y != NEUTRAL_TONE
        rows.append(
            {
                "group": group,
                "n": len(subset),
                "non_neutral_n": int(non_neutral.sum()),
                "baseline_classifier_accuracy": float(
                    np.mean(subset["baseline_raw_prediction"].to_numpy() == y)
                ),
                "baseline_accuracy": float(
                    np.mean(subset["baseline_prediction"].to_numpy() == y)
                ),
                "selected_accuracy": float(
                    np.mean(subset["selected_prediction"].to_numpy() == y)
                ),
                "baseline_non_neutral_accuracy": float(
                    np.mean(subset.loc[non_neutral, "baseline_prediction"].to_numpy() == y[non_neutral])
                ),
                "selected_non_neutral_accuracy": float(
                    np.mean(subset.loc[non_neutral, "selected_prediction"].to_numpy() == y[non_neutral])
                ),
            }
        )
    return pd.DataFrame(rows)


def _save_outputs(
    output_dir,
    all_characters,
    feature_frame,
    baseline_raw_prediction,
    baseline_prediction,
    selected_raw_prediction,
    selected_prediction,
    fold_records,
    inner_records,
    baseline_raw_metrics,
    baseline_api_metrics,
    selected_metrics,
    group_metrics,
    counts,
    selected_spec,
    acceptance,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assessable = feature_frame.copy()
    assessable["baseline_raw_prediction"] = baseline_raw_prediction
    assessable["baseline_prediction"] = baseline_prediction
    assessable["selected_raw_prediction"] = selected_raw_prediction
    assessable["selected_prediction"] = selected_prediction
    assessable["baseline_correct"] = baseline_prediction == assessable["expected_tone"].to_numpy()
    assessable["selected_correct"] = selected_prediction == assessable["expected_tone"].to_numpy()
    assessable["fold_error"] = ~assessable["selected_correct"]
    assessable.to_csv(output_dir / "tone_oof_predictions.csv", index=False, encoding="utf-8-sig")

    every_character = all_characters.copy()
    every_character["baseline_prediction"] = pd.NA
    every_character["selected_prediction"] = pd.NA
    every_character["baseline_correct"] = pd.NA
    every_character["selected_correct"] = pd.NA
    lookup = assessable.set_index(["filename", "char_index"])
    for column in (
        "baseline_prediction",
        "selected_prediction",
        "baseline_correct",
        "selected_correct",
    ):
        keys = list(zip(every_character["filename"], every_character["char_index"]))
        every_character[column] = [
            lookup.at[key, column] if key in lookup.index else pd.NA for key in keys
        ]
    every_character["status"] = np.where(
        every_character["has_features"], "scored", "unassessable_no_f0"
    )
    every_character.to_csv(
        output_dir / "tone_all_character_oof.csv", index=False, encoding="utf-8-sig"
    )

    errors = assessable.loc[~assessable["selected_correct"]].copy()
    error_patterns = (
        errors.groupby(
            ["expected_tone", "selected_prediction", "char", "low_confidence"],
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["count", "expected_tone"], ascending=[False, True])
    )
    error_patterns.to_csv(
        output_dir / "tone_error_patterns.csv", index=False, encoding="utf-8-sig"
    )
    group_metrics.to_csv(output_dir / "tone_group_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(inner_records).to_csv(
        output_dir / "tone_inner_cv_scores.csv", index=False, encoding="utf-8-sig"
    )

    low = assessable["low_confidence"].astype(bool).to_numpy()
    selected_correct = selected_prediction == assessable["expected_tone"].to_numpy(dtype=int)
    baseline_correct = baseline_prediction == assessable["expected_tone"].to_numpy(dtype=int)
    report = {
        "dataset": {
            "data_dir": str(Path(DEFAULT_DATA_DIR) if not counts.get("data_dir") else counts["data_dir"]),
            "n_groups": counts["n_groups"],
            "n_labelled_characters": counts["n_labelled_characters"],
            "n_assessable_characters": counts["n_assessable_characters"],
            "coverage": counts["coverage"],
            "tone_counts_assessable": counts["tone_counts_assessable"],
            "n_low_confidence_assessable": int(low.sum()),
            "n_unassessable_low_confidence": counts["n_unassessable_low_confidence"],
        },
        "baseline_classifier_current_hgb": baseline_raw_metrics,
        "baseline_api_simulated": baseline_api_metrics,
        "nested_selected_candidate": selected_metrics,
        "low_confidence": {
            "baseline_accuracy": float(np.mean(baseline_correct[low])) if low.any() else None,
            "selected_accuracy": float(np.mean(selected_correct[low])) if low.any() else None,
            "baseline_accuracy_other": float(np.mean(baseline_correct[~low])) if (~low).any() else None,
            "selected_accuracy_other": float(np.mean(selected_correct[~low])) if (~low).any() else None,
        },
        "fold_selection": fold_records,
        "final_candidate_selected_on_all_groups": selected_spec,
        "acceptance": acceptance,
        "notes": [
            "Scores exclude no-F0 characters from classifier accuracy; coverage and unassessable characters are reported separately.",
            "API-simulated metrics force the existing neutral-tone character list to tone 5 and should be read as unverified for those chars.",
            "No labels were changed during this benchmark.",
            "The API model is not switched unless the nested grouped-CV acceptance checks pass.",
            "The all_plus_raw_pitch variant appends absolute local mean, local slope, local range, utterance mean, and utterance standard deviation in semitones; the first 19 dimensions remain the deployed features.",
        ],
    }
    (output_dir / "tone_benchmark_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return errors


def main():
    args = parse_args()
    all_characters, feature_frame, X, y, groups = _collect_data(args.data_dir, args.cache)
    print(
        f"Loaded {len(all_characters)} labelled characters; "
        f"{len(y)} have measurable F0 across {len(np.unique(groups))} groups."
    )

    chars = feature_frame["char"].to_numpy(dtype=str)
    (
        baseline_raw_prediction,
        selected_raw_prediction,
        fold_records,
        inner_records,
        specs,
    ) = _cross_validate(
        X, y, groups, chars, args.outer_folds, raw_pitch_focus=args.raw_pitch_focus
    )
    baseline_prediction = _apply_api_neutral_rule(baseline_raw_prediction, chars)
    selected_prediction = _apply_api_neutral_rule(selected_raw_prediction, chars)
    baseline_raw_metrics = _metrics(y, baseline_raw_prediction)
    baseline_api_metrics = _metrics(y, baseline_prediction)
    selected_metrics = _metrics(y, selected_prediction)
    group_metrics = _per_group_metrics(
        feature_frame,
        baseline_raw_prediction,
        baseline_prediction,
        selected_raw_prediction,
        selected_prediction,
    )
    tone_drops = {
        tone: selected_metrics["recall_by_tone"][str(tone)]
        - baseline_api_metrics["recall_by_tone"][str(tone)]
        for tone in TONE_LABELS
    }
    acceptance = {
        "required_non_neutral_accuracy": baseline_api_metrics["accuracy_non_neutral"]
        + MIN_NON_NEUTRAL_GAIN,
        "observed_non_neutral_accuracy": selected_metrics["accuracy_non_neutral"],
        "non_neutral_gain": selected_metrics["accuracy_non_neutral"]
        - baseline_api_metrics["accuracy_non_neutral"],
        "macro_recall_drop": baseline_api_metrics["macro_recall"]
        - selected_metrics["macro_recall"],
        "per_tone_recall_change": {str(k): float(v) for k, v in tone_drops.items()},
        "passed": bool(
            selected_metrics["accuracy_non_neutral"]
            >= baseline_api_metrics["accuracy_non_neutral"] + MIN_NON_NEUTRAL_GAIN
            and selected_metrics["macro_recall"]
            >= baseline_api_metrics["macro_recall"] - MAX_MACRO_RECALL_DROP
            and all(value >= -MAX_PER_TONE_RECALL_DROP for value in tone_drops.values())
        ),
    }

    # Select the deployment candidate using grouped CV over all available groups.
    # Its reported generalization score remains the nested outer-fold score above.
    all_groups_folds = min(INNER_FOLDS, len(np.unique(groups)))
    selected_spec, full_cv_metrics, _ = _select_spec(
        X, y, groups, chars, specs, n_splits=all_groups_folds
    )
    selected_indices = _feature_indices(selected_spec["feature_variant"])
    final_estimator = _build_estimator(selected_spec)
    train_mask = np.ones(len(y), dtype=bool)
    if selected_spec.get("training_filter") == "exclude_api_neutral_examples":
        train_mask &= ~np.isin(chars, list(core.NEUTRAL_TONE_CHARS))
    final_estimator.fit(X[train_mask][:, selected_indices], y[train_mask])

    counts = {
        "data_dir": str(Path(args.data_dir)),
        "n_groups": int(len(np.unique(groups))),
        "n_labelled_characters": int(all_characters["expected_tone"].notna().sum()),
        "n_assessable_characters": int(len(y)),
        "coverage": float(len(y) / max(1, all_characters["expected_tone"].notna().sum())),
        "tone_counts_assessable": {
            str(k): int(v) for k, v in collections.Counter(y.tolist()).items()
        },
        "n_unassessable_low_confidence": int(
            (all_characters["low_confidence"] & ~all_characters["has_features"]).sum()
        ),
    }
    all_candidate_metrics = {
        **selected_metrics,
        "full_data_cv_selected_model": selected_spec,
        "full_data_inner_cv_metrics": full_cv_metrics,
    }
    errors = _save_outputs(
        args.output_dir,
        all_characters,
        feature_frame,
        baseline_raw_prediction,
        baseline_prediction,
        selected_raw_prediction,
        selected_prediction,
        fold_records,
        inner_records,
        baseline_raw_metrics,
        baseline_api_metrics,
        all_candidate_metrics,
        group_metrics,
        counts,
        selected_spec,
        acceptance,
    )

    bundle = {
        "classifier": final_estimator,
        "feature_dim": len(selected_indices),
        "feature_variant": selected_spec["feature_variant"],
        "benchmark_input_feature_dim": core.FEATURE_DIM + 5,
        "feature_indices": selected_indices,
        "n_samples": int(len(y)),
        "cv_accuracy": selected_metrics["accuracy_all_assessable"],
        "cv_accuracy_non_neutral": selected_metrics["accuracy_non_neutral"],
        "cv_macro_recall": selected_metrics["macro_recall"],
        "selected_candidate": selected_spec,
        "validation_passed": acceptance["passed"],
        "nested_cv_metrics": selected_metrics,
    }
    candidate_path = Path(args.output_dir) / "tone_model_candidate.joblib"
    joblib.dump(bundle, candidate_path)

    print("\nBaseline (same 5 grouped folds):")
    print(
        f"  classifier total {baseline_raw_metrics['accuracy_all_assessable'] * 100:.2f}% | "
        f"non-neutral {baseline_raw_metrics['accuracy_non_neutral'] * 100:.2f}% | "
        f"API-simulated total {baseline_api_metrics['accuracy_all_assessable'] * 100:.2f}%"
    )
    print("Nested-CV selected candidates:")
    print(
        f"  total {selected_metrics['accuracy_all_assessable'] * 100:.2f}% | "
        f"non-neutral {selected_metrics['accuracy_non_neutral'] * 100:.2f}% | "
        f"macro recall {selected_metrics['macro_recall'] * 100:.2f}%"
    )
    print(f"  acceptance passed: {acceptance['passed']}")
    print(f"  candidate model: {candidate_path}")
    print(f"  reports: {Path(args.output_dir).resolve()}")
    if len(errors):
        print("\nMost common selected-model error patterns:")
        print(
            errors.groupby(["expected_tone", "selected_prediction", "char", "low_confidence"])
            .size()
            .sort_values(ascending=False)
            .head(12)
            .to_string()
        )


if __name__ == "__main__":
    main()
