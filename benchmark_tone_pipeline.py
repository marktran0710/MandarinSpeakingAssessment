"""
Benchmark runner for tone_pipeline (v1). Produces syllables.csv, files.csv, report.json and report.md.

    python benchmark_tone_pipeline.py --manifest manifest.csv --out-dir outputs/tone_v1_bench \
        --modes adaptive fixed --cache-dir outputs/tone_v1_bench/alignment_cache

Manifest (CSV with a header row, or a JSON list of objects). One row per recording:

    path             WAV/MP3/... file (relative paths resolve against the manifest's folder)
    script           expected transcript (Han characters; punctuation allowed)
    group            reference | synthetic_wrong | stress
    speaker          speaker / voice id; recordings of one speaker share one F0 profile
    item             (optional) lexical item id
    variant          (optional) e.g. subtle / clear / chaotic / pitch_shift / flat / reversed ...
    wrong_syllables  (optional) 0-based syllable indices whose tone was altered, separated by ; or |
    calibration_path (optional) a calibration recording of the speaker; otherwise the speaker's own
                     recordings are pooled into a profile (benchmark only - no recording-level gates)

Aliases accepted: file/audio -> path, text/transcript -> script, voice -> speaker.

Ground truth is syllable level: a syllable listed in `wrong_syllables` should be flagged `practice`;
every other syllable of every recording should NOT be (and ideally is `likely_correct`).
Synthetic results are a stress test, not real-learner accuracy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import audio_prep
import capt_core
import tone_pipeline as tp

ALIASES = {"file": "path", "audio": "path", "text": "script", "transcript": "script", "voice": "speaker"}
FIXED_RANGE = (75.0, 500.0)
PROFILES = ("low", "mid", "high", "very_high")
SYLLABLE_COLUMNS = [
    "mode", "file", "group", "speaker", "profile", "item", "variant", "index", "character", "expected_surface_tone",
    "truth_wrong", "status", "tone_match", "tone_mismatch", "acoustic_quality", "voiced_ratio", "ac_cc_agreement",
    "detected_alternative", "pitch_floor", "pitch_ceiling", "reasons",
]


# ----------------------------------------------------------------- manifest ----

def load_manifest(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    out = []
    for raw in rows:
        row = {ALIASES.get(k.strip().lower(), k.strip().lower()): v for k, v in raw.items()}
        if not row.get("path") or not row.get("script"):
            raise ValueError(f"Manifest row needs path and script: {raw}")
        audio = Path(row["path"])
        row["path"] = str(audio if audio.is_absolute() else path.parent / audio)
        row.setdefault("group", "reference")
        row["speaker"] = str(row.get("speaker") or "speaker")
        wrong = row.get("wrong_syllables") or ""
        if isinstance(wrong, str):
            wrong = [int(x) for x in wrong.replace("|", ";").split(";") if x.strip() != ""]
        row["wrong_syllables"] = sorted(set(int(x) for x in wrong))
        out.append(row)
    return out


# --------------------------------------------------------------- alignment ----

def cached_syllables(samples, sample_rate, script, config, cache_dir):
    """MFA alignment, cached on disk so the adaptive and fixed runs share one alignment."""
    chars = [c["char"] for c in capt_core.expected_tone_contexts(capt_core.prepare_target(script))]
    key = hashlib.sha1(np.asarray(samples, dtype=np.float32).tobytes() + "".join(chars).encode("utf-8")
                       + tp.dictionary_fingerprint(config).encode("utf-8")).hexdigest()
    cache_file = Path(cache_dir) / f"{key}.json" if cache_dir else None
    if cache_file and cache_file.exists():
        record = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        try:
            record = {"tiers": tp.mfa_align(samples, sample_rate, chars, config)}
        except tp.AlignmentError as exc:
            record = {"error": str(exc)}
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    if "error" in record:
        raise tp.AlignmentError(record["error"])
    return tp.syllables_from_tiers(record["tiers"], chars, config)


# ------------------------------------------------------------------ metrics ----

def _rate(num, den):
    return round(num / den, 4) if den else None


def syllable_metrics(rows):
    """Syllable-level metrics from flat rows (status, truth_wrong, ...). Pure function."""
    total = len(rows)
    sufficient = [r for r in rows if r["status"] != "insufficient_evidence"]
    wrong = [r for r in rows if r["truth_wrong"]]
    unchanged = [r for r in rows if not r["truth_wrong"]]
    tp_ = sum(r["status"] == "practice" for r in wrong)
    fp = sum(r["status"] == "practice" for r in unchanged)
    correct = tp_ + sum(r["status"] == "likely_correct" for r in unchanged)
    coverage = [r["voiced_ratio"] for r in rows if r.get("voiced_ratio") is not None]
    agreement = [r["ac_cc_agreement"] for r in rows if r.get("ac_cc_agreement") is not None]
    return {
        "n_syllables": total,
        "n_sufficient_evidence": len(sufficient),
        "abstention_rate": _rate(total - len(sufficient), total),
        "decision_accuracy_on_decided": _rate(correct, len(sufficient)),
        "n_wrong_syllables": len(wrong),
        "error_recall": _rate(tp_, len(wrong)),
        "wrong_missed_as_likely_correct": _rate(sum(r["status"] == "likely_correct" for r in wrong), len(wrong)),
        "wrong_abstained": _rate(sum(r["status"] == "insufficient_evidence" for r in wrong), len(wrong)),
        "n_unchanged_syllables": len(unchanged),
        "false_positive_rate_unchanged": _rate(fp, len(unchanged)),
        "precision_when_flagging": _rate(tp_, tp_ + fp),
        "mean_voiced_coverage": round(float(np.mean(coverage)), 4) if coverage else None,
        "mean_ac_cc_agreement": round(float(np.mean(agreement)), 4) if agreement else None,
    }


def file_metrics(file_rows):
    """File-level distribution. `file_rows` items: {status, wrong (bool)}."""
    wrong = [f for f in file_rows if f["wrong"]]
    clean = [f for f in file_rows if not f["wrong"]]
    count = lambda rows, s: sum(r["status"] == s for r in rows)  # noqa: E731
    return {
        "n_files": len(file_rows),
        "wrong_files": {
            "n": len(wrong),
            **{s: _rate(count(wrong, s), len(wrong)) for s in tp.STATUSES},
            "false_likely_correct": _rate(count(wrong, "likely_correct"), len(wrong)),   # safety metric
        },
        "unchanged_files": {
            "n": len(clean),
            **{s: _rate(count(clean, s), len(clean)) for s in tp.STATUSES},
            "false_practice": _rate(count(clean, "practice"), len(clean)),
        },
    }


def group_metrics(rows, key):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row[key]].append(row)
    return {str(k): syllable_metrics(v) for k, v in sorted(buckets.items(), key=lambda kv: str(kv[0]))}


def abstention_reasons(rows):
    reasons = Counter()
    for row in rows:
        if row["status"] == "insufficient_evidence":
            for reason in (row["reasons"] or "none").split("|"):
                reasons[reason] += 1
    return dict(reasons.most_common())


def error_cases(rows, limit=25):
    flagged_unchanged = [r for r in rows if not r["truth_wrong"] and r["status"] == "practice"]
    missed_wrong = [r for r in rows if r["truth_wrong"] and r["status"] == "likely_correct"]
    confidence = lambda r: (r["acoustic_quality"] or 0) * (r["tone_mismatch"] or 0)  # noqa: E731
    return {
        "false_positives": sorted(flagged_unchanged, key=confidence, reverse=True)[:limit],
        "false_negatives": sorted(missed_wrong, key=lambda r: r["tone_match"] or 0, reverse=True)[:limit],
    }


def abstention_cases(rows, limit=25):
    abstained = [r for r in rows if r["status"] == "insufficient_evidence"]
    return sorted(abstained, key=lambda r: r["acoustic_quality"] or 0)[:limit]


def build_report(rows, file_rows, profiles, modes):
    """Sections A-I of the evaluation report from flat rows."""
    per_mode = {}
    for mode in modes:
        mode_rows = [r for r in rows if r["mode"] == mode]
        mode_files = [f for f in file_rows if f["mode"] == mode]
        groups = {g: [r for r in mode_rows if r["group"] == g] for g in sorted({r["group"] for r in mode_rows})}
        group_files = {g: [f for f in mode_files if f["group"] == g] for g in groups}
        stress = [r for r in mode_rows if r["group"] == "stress"]
        per_mode[mode] = {
            "B_reference_positive_set": {
                "syllables": syllable_metrics(groups.get("reference", [])),
                "files": file_metrics(group_files.get("reference", [])),
            },
            "C_synthetic_wrong_baseline": {
                "syllables": syllable_metrics(groups.get("synthetic_wrong", [])),
                "files": file_metrics(group_files.get("synthetic_wrong", [])),
                "by_variant": group_metrics(groups.get("synthetic_wrong", []), "variant"),
            },
            "D_stress_test": {
                "syllables": syllable_metrics(stress),
                "files": file_metrics(group_files.get("stress", [])),
                "by_variant": group_metrics(stress, "variant"),
            },
            "F_by_acoustic_profile": {
                p: {"n_files": len({r["file"] for r in mode_rows if r["profile"] == p}),
                    "mean_pitch_floor": _mean([r["pitch_floor"] for r in mode_rows if r["profile"] == p]),
                    "mean_pitch_ceiling": _mean([r["pitch_ceiling"] for r in mode_rows if r["profile"] == p]),
                    **syllable_metrics([r for r in mode_rows if r["profile"] == p])}
                for p in PROFILES if any(r["profile"] == p for r in mode_rows)
            },
            "G_by_expected_tone": group_metrics(mode_rows, "expected_surface_tone"),
            "H_error_cases": error_cases(mode_rows),
            "I_abstention": {"reasons": abstention_reasons(mode_rows), "lowest_quality_cases": abstention_cases(mode_rows)},
            "overall": syllable_metrics(mode_rows),
        }
    report = {"A_calibration_profiles": profiles, "modes": per_mode}
    if {"adaptive", "fixed"} <= set(modes):
        report["E_adaptive_vs_fixed"] = compare_modes(per_mode["adaptive"], per_mode["fixed"])
    return report


def _mean(values):
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 1) if values else None


def compare_modes(adaptive, fixed):
    """Adaptive (calibrated floor/ceiling) vs fixed 75-500 Hz, everything else equal."""
    keys = ["mean_voiced_coverage", "mean_ac_cc_agreement", "abstention_rate", "error_recall",
            "false_positive_rate_unchanged", "wrong_missed_as_likely_correct", "decision_accuracy_on_decided"]
    out = {"fixed_range_hz": list(FIXED_RANGE), "overall": {}, "synthetic_wrong_files": {}}
    for key in keys:
        a, f = adaptive["overall"][key], fixed["overall"][key]
        out["overall"][key] = {"adaptive": a, "fixed": f, "delta": None if a is None or f is None else round(a - f, 4)}
    for mode_name, section in (("adaptive", adaptive), ("fixed", fixed)):
        out["synthetic_wrong_files"][mode_name] = section["C_synthetic_wrong_baseline"]["files"]["wrong_files"]
    return out


def render_markdown(report):
    lines = ["# Tone pipeline v1 benchmark", "",
             "Synthetic results are a stress test, not real-learner accuracy.", "",
             "## A. Calibration / profiles", ""]
    for speaker, profile in report["A_calibration_profiles"].items():
        lines.append(f"- {speaker}: median {profile['f0_median']} Hz, P95 {profile['f0_p95']} Hz, "
                     f"floor {profile['pitch_floor']}, ceiling {profile['pitch_ceiling']}, profile {profile['acoustic_profile']}")
    for mode, sections in report["modes"].items():
        lines += ["", f"## Mode: {mode}", "", "```json", json.dumps(
            {k: v for k, v in sections.items() if k not in ("H_error_cases",)}, indent=1, ensure_ascii=False), "```"]
    if "E_adaptive_vs_fixed" in report:
        lines += ["", "## E. Adaptive vs fixed", "", "```json",
                  json.dumps(report["E_adaptive_vs_fixed"], indent=1), "```"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- runner ----

def build_profiles(rows, config, audio_cache):
    profiles = {}
    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)
    for speaker, speaker_rows in by_speaker.items():
        calibration_path = next((r.get("calibration_path") for r in speaker_rows if r.get("calibration_path")), None)
        if calibration_path:
            samples, rate = audio_prep.decode_audio(calibration_path)
            profiles[speaker] = tp.calibrate_voice(samples, rate, config)
        else:
            prepared = [audio_cache(r["path"]) for r in speaker_rows]
            usable = [a for a in prepared if a.passed] or prepared     # recordings that fail the quality gate do not shape a profile
            profiles[speaker] = tp.pooled_calibration([(a.samples, a.sample_rate) for a in usable], config)
    return profiles


def run(manifest, out_dir, modes, cache_dir=None, limit=None, config=None):
    config = config or tp.load_config()
    rows = load_manifest(manifest)[: limit or None]
    audio = {}

    def audio_cache(path):
        if path not in audio:
            audio[path] = audio_prep.prepare_recording(path, config)
        return audio[path]

    profiles = build_profiles(rows, config, audio_cache)
    syllable_rows, file_rows, failures = [], [], []
    for mode in modes:
        pitch_range = FIXED_RANGE if mode == "fixed" else None
        for row in rows:
            prepared = audio_cache(row["path"])
            samples, rate = prepared.samples, prepared.sample_rate
            profile = profiles[row["speaker"]]
            if not prepared.passed:   # rejected by the audio quality gate: alignment is skipped
                result = tp.verify_utterance(samples, rate, row["script"], profile, config=config,
                                             pitch_range=pitch_range, rejected_reasons=prepared.rejected_reasons)
                failures.append({"file": row["path"], "error": "audio quality: " + ",".join(prepared.reasons)})
            else:
                try:
                    syllables = (cached_syllables(samples, rate, row["script"], config, cache_dir)
                                 if tp.needs_mfa(tp.syllable_count(row["script"]), config) else None)
                    result = tp.verify_utterance(samples, rate, row["script"], profile, alignment=syllables,
                                                 config=config, pitch_range=pitch_range)
                except tp.AlignmentError as exc:   # keep going; record as an all-abstained file
                    # An empty alignment is rejected by verify_utterance, which marks every syllable abstained.
                    result = tp.verify_utterance(samples, rate, row["script"], profile,
                                                 alignment=[], config=config, pitch_range=pitch_range)
                    failures.append({"file": row["path"], "error": str(exc)})
            for syllable in result["syllables"]:
                syllable_rows.append({
                    "mode": mode, "file": row["path"], "group": row["group"], "speaker": row["speaker"],
                    "profile": profile["acoustic_profile"], "item": row.get("item", ""), "variant": row.get("variant", ""),
                    "index": syllable["index"], "character": syllable["character"],
                    "expected_surface_tone": syllable["expected_surface_tone"],
                    "truth_wrong": syllable["index"] in row["wrong_syllables"],
                    "status": syllable["status"], "tone_match": syllable["tone_match"],
                    "tone_mismatch": syllable["tone_mismatch"], "acoustic_quality": syllable["acoustic_quality"],
                    "voiced_ratio": syllable.get("voiced_ratio"), "ac_cc_agreement": syllable.get("ac_cc_agreement"),
                    "detected_alternative": syllable["detected_alternative"],
                    "pitch_floor": syllable["pitch_floor"], "pitch_ceiling": syllable["pitch_ceiling"],
                    "reasons": "|".join(syllable["reasons"]),
                })
            file_rows.append({
                "mode": mode, "file": row["path"], "group": row["group"], "wrong": bool(row["wrong_syllables"]),
                "status": tp.aggregate_status(result["syllables"]),
            })

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "syllables.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SYLLABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(syllable_rows)
    with open(out / "files.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "file", "group", "wrong", "status"])
        writer.writeheader()
        writer.writerows(file_rows)
    report = build_report(syllable_rows, file_rows, profiles, modes)
    report["alignment_and_quality_failures"] = failures
    report["config"] = config
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--modes", nargs="+", default=["adaptive", "fixed"], choices=["adaptive", "fixed"])
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-boundary-search", action="store_true",
                        help="trust the aligner's boundaries as they are (for comparison)")
    parser.add_argument("--max-shift-ms", type=int, default=None, help="boundary search window (default from config)")
    args = parser.parse_args()
    search = {"enabled": not args.no_boundary_search}
    if args.max_shift_ms is not None:
        search["max_shift_ms"] = args.max_shift_ms
    config = tp.load_config(overrides={"boundary_search": search})
    report = run(args.manifest, args.out_dir, args.modes, args.cache_dir, args.limit, config)
    for mode, sections in report["modes"].items():
        print(mode, json.dumps(sections["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
