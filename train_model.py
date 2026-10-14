"""
Train the tone classifier.

- Timestamps: force-align the script to the audio; do not use the Whisper transcript.
- Labels: expected tones after sandhi (capt_core.expected_tones); exclude erhua.
- Features: capt_core.utterance_features, shared with the API so training and inference stay consistent.
- Characters assigned neutral tone by the API are excluded from classifier training.
- Evaluation: cross-validation grouped by conversation (lesson-story). This estimates performance on unseen data;
  the baseline and candidate use the same folds for a fair comparison.
"""
import argparse
import collections
import json
import os

import joblib
import numpy as np
import opencc
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

import capt_core as core

DATA_DIR = r"D:\hautran\Lab\mandarin-speaking data\Recorded Data\lessons_5_8_FINAL_content_verified_v3_finallllll"
MODEL_PATH = "tone_model_v2.joblib"
CACHE_PATH = "alignment_cache_v2.json"
NEUTRAL_TONE = 5


def make_classifier():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42)


def load_alignments(df, data_dir, rebuild):
    """Align each file's script once and cache the result because alignment is slow."""
    cache = {}
    if os.path.exists(CACHE_PATH) and not rebuild:
        cache = json.load(open(CACHE_PATH, encoding="utf-8"))

    s2twp = opencc.OpenCC("s2twp")
    t2s = opencc.OpenCC("t2s")
    align_model = metadata = None
    changed = False

    for _, row in df.iterrows():
        name, script = str(row["filename"]).strip(), str(row["script"]).strip()
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            continue
        if core.alignment_cache_is_current(cache.get(name), script, path, CACHE_PATH):
            continue
        if align_model is None:
            import whisperx
            print("Loading the alignment model...")
            align_model, metadata = whisperx.load_align_model(language_code="zh", device="cpu")
        import whisperx
        chars = list("".join(core.prepare_target(s2twp.convert(script))))
        items = core.force_align_script(whisperx.load_audio(path), chars, align_model, metadata, "cpu", t2s.convert)
        audio_stat = os.stat(path)
        cache[name] = {
            "script": script,
            "starts": [i["start"] for i in items],
            "ends": [i["end"] for i in items],
            "audio_mtime_ns": audio_stat.st_mtime_ns,
            "audio_size_bytes": audio_stat.st_size,
        }
        changed = True
        print(f"   aligned {name}")

    if changed:
        json.dump(cache, open(CACHE_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    return cache


def collect_samples(df, data_dir, alignments):
    X, y, groups, chars_seen = [], [], [], []
    s2twp = opencc.OpenCC("s2twp")
    for _, row in df.iterrows():
        name = str(row["filename"]).strip()
        path = os.path.join(data_dir, name)
        if name not in alignments or not os.path.exists(path):
            continue
        clauses = core.prepare_target(s2twp.convert(str(row["script"]).strip()))
        chars = list("".join(clauses))
        tones = core.expected_tones(clauses)
        samples, sample_rate = core.load_audio_samples(path)

        items = [{"start": s, "end": e, "interpolated": False}
                 for s, e in zip(alignments[name]["starts"], alignments[name]["ends"])]
        core.fill_missing_timestamps(items, len(samples) / sample_rate)
        feats, _ = core.utterance_features(samples, sample_rate, [i["start"] for i in items], tones,
                                           [len(c) for c in clauses])
        for char, tone, ft in zip(chars, tones, feats):
            if tone is not None and ft is not None:
                X.append(ft)
                y.append(tone)
                groups.append(f"{row['lesson']}-{row['story']}")
                chars_seen.append(char)
    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=int),
        np.asarray(groups, dtype=str),
        np.asarray(chars_seen, dtype=str),
    )


def _apply_api_neutral_rule(prediction, chars):
    prediction = np.asarray(prediction).copy()
    prediction[np.isin(chars, list(core.NEUTRAL_TONE_CHARS))] = NEUTRAL_TONE
    return prediction


def _metrics(y, pred):
    non_neutral = y != NEUTRAL_TONE
    recalls = {
        str(tone): float(np.mean(pred[y == tone] == tone))
        for tone in (1, 2, 3, 4, 5)
        if np.any(y == tone)
    }
    return {
        "accuracy_all_assessable": float(np.mean(pred == y)),
        "accuracy_non_neutral": float(np.mean(pred[non_neutral] == y[non_neutral])),
        "macro_recall": float(np.mean(list(recalls.values()))),
        "recall_by_tone": recalls,
    }


def evaluate(X, y, groups, chars, n_splits=5):
    baseline_pred = np.zeros_like(y)
    filtered_pred = np.zeros_like(y)
    for train_idx, test_idx in GroupKFold(n_splits).split(X, y, groups):
        baseline = make_classifier().fit(X[train_idx], y[train_idx])
        baseline_pred[test_idx] = baseline.predict(X[test_idx])

        train_mask = ~np.isin(chars[train_idx], list(core.NEUTRAL_TONE_CHARS))
        filtered = make_classifier().fit(X[train_idx][train_mask], y[train_idx][train_mask])
        filtered_pred[test_idx] = filtered.predict(X[test_idx])

    baseline_api_pred = _apply_api_neutral_rule(baseline_pred, chars)
    filtered_api_pred = _apply_api_neutral_rule(filtered_pred, chars)
    baseline_metrics = _metrics(y, baseline_api_pred)
    filtered_metrics = _metrics(y, filtered_api_pred)
    tone_changes = {
        tone: filtered_metrics["recall_by_tone"][tone]
        - baseline_metrics["recall_by_tone"][tone]
        for tone in baseline_metrics["recall_by_tone"]
    }
    acceptance = {
        "required_non_neutral_accuracy": baseline_metrics["accuracy_non_neutral"] + 0.02,
        "non_neutral_gain": filtered_metrics["accuracy_non_neutral"]
        - baseline_metrics["accuracy_non_neutral"],
        "macro_recall_drop": baseline_metrics["macro_recall"]
        - filtered_metrics["macro_recall"],
        "per_tone_recall_change": tone_changes,
        "passed": bool(
            filtered_metrics["accuracy_non_neutral"]
            >= baseline_metrics["accuracy_non_neutral"] + 0.02
            and filtered_metrics["macro_recall"] >= baseline_metrics["macro_recall"] - 0.015
            and all(change >= -0.05 for change in tone_changes.values())
        ),
    }

    print(f"\nOut-of-fold results with matching folds ({n_splits}-fold by lesson-story, {len(set(groups))} groups):")
    print(
        f"   - Baseline: {baseline_metrics['accuracy_non_neutral'] * 100:.2f}% excluding neutral tone; "
        f"{baseline_metrics['accuracy_all_assessable'] * 100:.2f}% for the simulated API overall"
    )
    print(
        f"   - Candidate: {filtered_metrics['accuracy_non_neutral'] * 100:.2f}% excluding neutral tone; "
        f"{filtered_metrics['accuracy_all_assessable'] * 100:.2f}% for the simulated API overall"
    )
    print(f"   - Gain excluding neutral tone: {acceptance['non_neutral_gain'] * 100:+.2f} percentage points")
    print(f"   - Meets model replacement threshold: {acceptance['passed']}")
    print("   - Recall by tone (baseline -> candidate):")
    for tone, change in tone_changes.items():
        before = baseline_metrics["recall_by_tone"][tone]
        after = filtered_metrics["recall_by_tone"][tone]
        print(f"       Tone {tone}: {before * 100:.1f}% -> {after * 100:.1f}% ({change * 100:+.1f} percentage points)")
    return baseline_metrics, filtered_metrics, acceptance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-model", default=MODEL_PATH)
    parser.add_argument("--rebuild", action="store_true", help="rebuild all alignments and ignore the cache")
    args = parser.parse_args()

    csv_path = os.path.join(args.data_dir, "FULL_audio_content_audit.csv")
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)

    print("Aligning scripts to audio (forced alignment)...")
    alignments = load_alignments(df, args.data_dir, args.rebuild)
    print("Extracting F0 features...")
    X, y, groups, chars = collect_samples(df, args.data_dir, alignments)
    train_mask = ~np.isin(chars, list(core.NEUTRAL_TONE_CHARS))
    excluded_api_neutrals = int((~train_mask).sum())
    print(f"{len(y)} characters with F0 | label distribution: {dict(sorted(collections.Counter(y.tolist()).items()))}")
    print(f"The API assigned neutral tone to {excluded_api_neutrals} characters; the candidate will exclude them from training.")
    if len(y) == 0:
        print("No data could be extracted for training.")
        return

    baseline_metrics, candidate_metrics, acceptance = evaluate(X, y, groups, chars)

    clf = make_classifier().fit(X[train_mask], y[train_mask])
    joblib.dump({
        "classifier": clf,
        "feature_dim": core.FEATURE_DIM,
        "n_samples": int(train_mask.sum()),
        "n_excluded_api_neutral": int(excluded_api_neutrals),
        "training_filter": "exclude_api_neutral_characters",
        "cv_accuracy": candidate_metrics["accuracy_all_assessable"],
        "cv_accuracy_non_neutral": candidate_metrics["accuracy_non_neutral"],
        "cv_metrics": candidate_metrics,
        "baseline_cv_metrics": baseline_metrics,
        "validation_passed": acceptance["passed"],
    }, args.output_model)
    print(f"\nRetrained on {int(train_mask.sum())} F0-bearing samples used by the API model and saved to '{args.output_model}'")


if __name__ == "__main__":
    main()
