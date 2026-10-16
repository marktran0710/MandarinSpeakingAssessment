"""Compare the Chapter 5 direct Praat audit before/after WhisperX boundaries."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams["font.family"] = ["Microsoft JhengHei"]
rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "tone_audit_results"
FINAL_DIR = RESULTS / "whisperx_aligned_boundaries_20261016"
PLOTS_DIR = FINAL_DIR / "plots"
sys.path.insert(0, str(ROOT.parent))
import capt_core as core


UNCERTAIN_REPRESENTATIVES = {
    "W014": "No reliable WhisperX alignment (single-character phrase).",
    "W006": "F0 gap exceeds the unchanged 80 ms quality limit.",
    "W003": "T2-to-T3 sandhi context with ambiguous whole-phrase motion.",
    "W021": "Allowed longer transcript variant aligns; phrase motion remains ambiguous.",
    "W024": "F0 gap plus a raw local pitch jump of at least 6 semitones.",
    "W045": "F0 gap plus a raw local pitch jump of at least 6 semitones.",
    "W217": "Catalog Pinyin resolves kòng; internal F0 gap still exceeds the limit.",
}


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def phrase_from(row):
    response = row.get("response") or {}
    return response, (response.get("phrases") or [{}])[0]


def phrase_analysis(phrase):
    return phrase.get("phrase_analysis") or {}


def phrase_status(phrase):
    value = phrase.get("phrase_status", phrase.get("status", "missing"))
    return "good" if value == "likely_correct" else value


def json_cell(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) if value is not None else ""


def window_duration(value):
    try:
        return float(value[1]) - float(value[0])
    except (TypeError, ValueError, IndexError):
        return None


def median_or_none(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(values) if values else None


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plot_diagnostic(item, response, phrase, destination: Path):
    audio_path = ROOT / "audio" / item["audio_file"]
    samples, sample_rate = core.load_audio_samples(audio_path)
    times, raw_midi, _, _ = core.praat_pitch_track(samples, sample_rate)
    smooth_midi = core.smooth_voiced_midi(raw_midi)
    raw_hz = np.full_like(raw_midi, np.nan, dtype=float)
    voiced = np.isfinite(raw_midi)
    raw_hz[voiced] = 440.0 * (2.0 ** ((raw_midi[voiced] - 69.0) / 12.0))

    pa = phrase_analysis(phrase)
    align_quality = (response.get("alignment_quality") or {}).get("learner") or {}
    phrase_window = pa.get("phrase_window_sec") or phrase.get("phrase_window_sec")
    if phrase_window:
        inside = voiced & (times >= phrase_window[0]) & (times < phrase_window[1])
    else:
        inside = voiced
    pitch_center = float(np.median(raw_midi[inside])) if inside.any() else 0.0
    relative_raw = raw_midi - pitch_center
    relative_smooth = smooth_midi - pitch_center

    valid_steps = np.diff(times)
    jumps = np.flatnonzero(
        voiced[:-1] & voiced[1:] & (valid_steps <= 0.0201)
        & (np.abs(np.diff(raw_midi)) >= 6.0)
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.3), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"{item['word_id']}  |  {item['script']}  |  {phrase_status(phrase)}\n"
        f"Expected tones {phrase.get('expected_tone_sequence')}",
        fontsize=13,
    )
    axes[0].plot(times, raw_hz, color="#2463a8", linewidth=1.05, label="Praat raw F0")
    axes[0].set_ylabel("F0 (Hz)")
    axes[0].set_ylim(55, 625)
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="upper right")
    axes[1].plot(times, relative_raw, color="#6e8fb8", linewidth=0.85, alpha=0.75, label="Raw relative pitch")
    axes[1].plot(times, relative_smooth, color="#c26711", linewidth=1.2, label="Smoothed (voiced runs only)")
    axes[1].axhline(0, color="#555", linewidth=0.7, alpha=0.5)
    axes[1].set_ylabel("Relative semitones")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper right")

    if phrase_window:
        for axis in axes:
            axis.axvspan(phrase_window[0], phrase_window[1], color="#55aa66", alpha=0.08)

    alignment = pa.get("syllable_alignment") or []
    expected = phrase.get("expected_tone_sequence") or []
    for index, part in enumerate(alignment):
        analysis_window = part.get("analysis_window_sec")
        if analysis_window:
            axes[1].axvspan(
                analysis_window[0], analysis_window[1],
                color="#4f86c6" if index % 2 == 0 else "#e9a24a", alpha=0.11,
            )
            tone = expected[index] if index < len(expected) else "?"
            label = f"{part.get('character') or '?'} / T{tone}"
            axes[1].text(
                (analysis_window[0] + analysis_window[1]) / 2,
                0.97,
                label,
                transform=axes[1].get_xaxis_transform(),
                ha="center", va="top", fontsize=8,
            )
        elif part.get("start_sec") is not None and part.get("end_sec") is not None:
            axes[1].axvspan(part["start_sec"], part["end_sec"], color="#777", alpha=0.13)
            if part.get("anchor_sec") is not None:
                axes[1].axvline(part["anchor_sec"], color="#555", linewidth=0.7, linestyle=":")

    for jump in jumps:
        axes[0].scatter([times[jump + 1]], [raw_hz[jump + 1]], color="#c5232b", marker="x", s=40, zorder=5)
        axes[1].scatter([times[jump + 1]], [relative_raw[jump + 1]], color="#c5232b", marker="x", s=40, zorder=5)

    reason = phrase.get("assessment_note") or "whole phrase pattern matched"
    coverage = pa.get("trimmed_voiced_coverage")
    gap = pa.get("maximum_internal_unvoiced_gap_sec")
    aligned = f"{align_quality.get('reliable_character_count', 0)}/{align_quality.get('character_count', 0)}"
    fig.text(
        0.01, -0.015,
        f"Reason: {reason} | Alignment: {aligned} | coverage: {coverage} | max internal gap: {gap}s | raw jumps >=6 st/20ms: {len(jumps)}",
        fontsize=8,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(destination.relative_to(FINAL_DIR)).replace("\\", "/"),
        "raw_pitch_jump_count": int(len(jumps)),
        "raw_pitch_jumps": json_cell([
            {"time_sec": round(float(times[index]), 3), "delta_semitones": round(float(raw_midi[index + 1] - raw_midi[index]), 2)}
            for index in jumps
        ]),
        "voiced_frames": int(np.count_nonzero(voiced)),
        "whole_clip_voiced_fraction": round(float(np.mean(voiced)), 3) if len(voiced) else 0.0,
    }


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "items.json").read_text(encoding="utf-8"))
    items = {item["word_id"]: item for item in manifest["items"]}
    audio_hash_mismatches = [
        word_id for word_id, item in items.items()
        if sha256(ROOT / "audio" / item["audio_file"]) != item.get("audio_sha256")
    ]
    before_summary = {row["word_id"]: row for row in read_csv(RESULTS / "item_summary.csv")}
    after_summary = {row["word_id"]: row for row in read_csv(FINAL_DIR / "item_summary.csv")}
    before_raw = {row["item"]["word_id"]: row for row in read_jsonl(RESULTS / "api_responses.jsonl")}
    after_raw = {row["item"]["word_id"]: row for row in read_jsonl(FINAL_DIR / "api_responses.jsonl")}
    if not (len(items) == len(before_summary) == len(after_summary) == len(before_raw) == len(after_raw) == 53):
        raise RuntimeError("Expected exactly 53 matched manifest, CSV, and JSONL rows for both runs.")

    compare_columns = [
        "word_id", "section", "item_type", "script", "pinyin", "audio_file",
        "before_status", "after_status", "status_changed", "before_reason", "after_reason",
        "before_expected_tones", "after_expected_tones", "selected_script_variant",
        "alignment_status", "alignment_method", "aligned_characters",
        "before_phrase_window_sec", "after_phrase_window_sec",
        "before_window_duration_sec", "after_window_duration_sec",
        "before_voiced_coverage", "after_voiced_coverage",
        "before_max_internal_gap_sec", "after_max_internal_gap_sec",
        "syllable_region_voiced_frames", "response_error",
    ]
    comparison_rows = []
    status_transitions = Counter()
    reason_counts = Counter()
    tone_context_rows = []
    tone_context_mismatches = []
    before_durations, after_durations, before_coverages, after_coverages = [], [], [], []
    for word_id, item in items.items():
        old_row, new_row = before_raw[word_id], after_raw[word_id]
        old_response, old_phrase = phrase_from(old_row)
        new_response, new_phrase = phrase_from(new_row)
        old_pa, new_pa = phrase_analysis(old_phrase), phrase_analysis(new_phrase)
        old_status, new_status = phrase_status(old_phrase), phrase_status(new_phrase)
        status_transitions[(old_status, new_status)] += 1
        reason_counts[new_phrase.get("assessment_note") or "whole_phrase_pattern_match"] += 1
        old_window = old_pa.get("phrase_window_sec") or old_phrase.get("phrase_window_sec")
        new_window = new_pa.get("phrase_window_sec") or new_phrase.get("phrase_window_sec")
        old_f0 = old_phrase.get("phrase_f0") or {}
        new_f0 = new_phrase.get("phrase_f0") or {}
        old_cov = old_pa.get("trimmed_voiced_coverage", old_f0.get("trimmed_voiced_coverage"))
        new_cov = new_pa.get("trimmed_voiced_coverage", new_f0.get("trimmed_voiced_coverage"))
        old_gap = old_pa.get("maximum_internal_unvoiced_gap_sec")
        new_gap = new_pa.get("maximum_internal_unvoiced_gap_sec")
        for values, value in ((before_durations, window_duration(old_window)), (after_durations, window_duration(new_window))):
            if value is not None:
                values.append(value)
        for values, value in ((before_coverages, old_cov), (after_coverages, new_cov)):
            if value is not None:
                values.append(float(value))
        quality = (new_response.get("alignment_quality") or {}).get("learner") or {}
        comparison_rows.append({
            "word_id": word_id,
            "section": item["section"],
            "item_type": item["item_type"],
            "script": item["script"],
            "pinyin": item["pinyin"],
            "audio_file": item["audio_file"],
            "before_status": old_status,
            "after_status": new_status,
            "status_changed": old_status != new_status,
            "before_reason": old_phrase.get("assessment_note") or "whole_phrase_pattern_match",
            "after_reason": new_phrase.get("assessment_note") or "whole_phrase_pattern_match",
            "before_expected_tones": json_cell(old_phrase.get("expected_tone_sequence")),
            "after_expected_tones": json_cell(new_phrase.get("expected_tone_sequence")),
            "selected_script_variant": new_response.get("aligned_script_variant", ""),
            "alignment_status": quality.get("status", ""),
            "alignment_method": quality.get("method", ""),
            "aligned_characters": f"{quality.get('reliable_character_count', '')}/{quality.get('character_count', '')}",
            "before_phrase_window_sec": json_cell(old_window),
            "after_phrase_window_sec": json_cell(new_window),
            "before_window_duration_sec": round(window_duration(old_window), 3) if window_duration(old_window) is not None else "",
            "after_window_duration_sec": round(window_duration(new_window), 3) if window_duration(new_window) is not None else "",
            "before_voiced_coverage": old_cov if old_cov is not None else "",
            "after_voiced_coverage": new_cov if new_cov is not None else "",
            "before_max_internal_gap_sec": old_gap if old_gap is not None else "",
            "after_max_internal_gap_sec": new_gap if new_gap is not None else "",
            "syllable_region_voiced_frames": json_cell(new_pa.get("syllable_region_voiced_frames")),
            "response_error": new_row.get("error", ""),
        })

        # Reconstruct the expected lexical/surface-tone context used for this exact
        # selected script and compare it to the API's target sequence.
        selected_chars = str(new_phrase.get("word") or "")
        contexts = core.expected_tone_contexts([selected_chars]) if selected_chars else []
        if word_id == "W217" and selected_chars.endswith("空") and contexts and "kòng" in item["pinyin"]:
            contexts[-1] = {**contexts[-1], "lexical_tone": 4, "target_tone": 4, "tone_rule": "catalog_pinyin_disambiguation"}
        reconstructed = [0 if context["target_tone"] == 5 else context["target_tone"] for context in contexts]
        api_tones = new_phrase.get("expected_tone_sequence") or []
        if reconstructed != api_tones:
            tone_context_mismatches.append({"word_id": word_id, "api": api_tones, "reconstructed": reconstructed})
        for index, context in enumerate(contexts):
            tone_context_rows.append({
                "word_id": word_id,
                "script_variant": new_response.get("aligned_script_variant", ""),
                "character_index": index,
                "character": context["char"],
                "lexical_tone": context["lexical_tone"],
                "target_tone_after_sandhi": 0 if context["target_tone"] == 5 else context["target_tone"],
                "tone_rule": context["tone_rule"],
                "previous_character": context.get("previous_char"),
                "previous_target_tone": context.get("previous_tone"),
                "next_character": context.get("next_char"),
                "next_target_tone": context.get("next_tone"),
                "allow_half_t3": context.get("allow_half_t3"),
                "api_expected_sequence": json_cell(api_tones),
            })

    with (FINAL_DIR / "before_after.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=compare_columns)
        writer.writeheader()
        writer.writerows(comparison_rows)
    tone_columns = list(tone_context_rows[0]) if tone_context_rows else []
    with (FINAL_DIR / "tone_context_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tone_columns)
        writer.writeheader()
        writer.writerows(tone_context_rows)

    plot_index = []
    all_practice = [word_id for word_id in items if phrase_status(phrase_from(after_raw[word_id])[1]) == "practice"]
    selected_uncertain = [
        word_id for word_id in UNCERTAIN_REPRESENTATIVES
        if word_id in items and phrase_status(phrase_from(after_raw[word_id])[1]) == "uncertain"
    ]
    to_plot = [(word_id, "practice", "Every final practice result") for word_id in all_practice]
    to_plot.extend((word_id, "uncertain", UNCERTAIN_REPRESENTATIVES[word_id]) for word_id in selected_uncertain)
    pitch_stats = {}
    for word_id, category, rationale in to_plot:
        item = items[word_id]
        response, phrase = phrase_from(after_raw[word_id])
        image_path = PLOTS_DIR / f"{category}_{word_id}.png"
        stats = plot_diagnostic(item, response, phrase, image_path)
        pitch_stats[word_id] = stats
        plot_index.append({
            "word_id": word_id,
            "category": category,
            "reason_selected": rationale,
            "plot_path": stats["path"],
            "raw_pitch_jump_count": stats["raw_pitch_jump_count"],
            "raw_pitch_jumps": stats["raw_pitch_jumps"],
            "whole_clip_voiced_fraction": stats["whole_clip_voiced_fraction"],
        })
    with (FINAL_DIR / "diagnostic_plot_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plot_index[0]) if plot_index else [])
        writer.writeheader()
        writer.writerows(plot_index)

    before_counts = Counter(row["before_status"] for row in comparison_rows)
    after_counts = Counter(row["after_status"] for row in comparison_rows)
    after_reasons = Counter(row["after_reason"] for row in comparison_rows)
    alignment_counts = Counter(row["alignment_status"] for row in comparison_rows)
    gap_limited = sum(
        row["after_reason"] == "direct_praat_region_evidence_unreliable"
        and row["after_max_internal_gap_sec"] != ""
        and float(row["after_max_internal_gap_sec"]) > 0.080
        for row in comparison_rows
    )
    raw_pitch_jumps = [
        (word_id, pitch_stats[word_id]["raw_pitch_jumps"])
        for word_id in items if word_id in pitch_stats and pitch_stats[word_id]["raw_pitch_jump_count"]
    ]
    # Check every recording's raw Praat pitch once, independently of which plots
    # were selected, so pitch-tracker artifacts are audited across all 53 inputs.
    full_track_stats = {}
    for word_id, item in items.items():
        samples, sample_rate = core.load_audio_samples(ROOT / "audio" / item["audio_file"])
        times, midi, _, _ = core.praat_pitch_track(samples, sample_rate)
        voiced = np.isfinite(midi)
        step = np.diff(times)
        jumps = np.flatnonzero(
            voiced[:-1] & voiced[1:] & (step <= 0.0201)
            & (np.abs(np.diff(midi)) >= 6.0)
        )
        f0 = 440.0 * (2.0 ** ((midi[voiced] - 69.0) / 12.0)) if voiced.any() else np.array([])
        full_track_stats[word_id] = {
            "has_f0": bool(voiced.any()),
            "voiced_frame_count": int(np.count_nonzero(voiced)),
            "whole_clip_voiced_fraction": round(float(np.mean(voiced)), 3) if len(voiced) else 0.0,
            "raw_pitch_jump_count": int(len(jumps)),
            "near_pitch_floor": bool(len(f0) and np.any(f0 < 67.0)),
            "near_pitch_ceiling": bool(len(f0) and np.any(f0 > 590.0)),
        }
    with (FINAL_DIR / "pitch_track_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        columns = ["word_id", *next(iter(full_track_stats.values())).keys()]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for word_id, values in full_track_stats.items():
            writer.writerow({"word_id": word_id, **values})

    jumps_all = [word_id for word_id, values in full_track_stats.items() if values["raw_pitch_jump_count"]]
    floor_hits = [word_id for word_id, values in full_track_stats.items() if values["near_pitch_floor"]]
    ceiling_hits = [word_id for word_id, values in full_track_stats.items() if values["near_pitch_ceiling"]]
    api_errors = [row["word_id"] for row in comparison_rows if row["response_error"]]
    before_window_count = len(before_durations)
    after_window_count = len(after_durations)
    run_meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chapter": 5,
        "item_count": len(items),
        "final_api_successes": 53 - len(api_errors),
        "audio_hashes_verified": len(items) - len(audio_hash_mismatches),
        "audio_hash_mismatches": audio_hash_mismatches,
        "thresholds": {
            "minimum_whisperx_ctc_score_exclusive": core.MIN_ALIGNMENT_SCORE,
            "minimum_phrase_voiced_coverage": 0.50,
            "maximum_internal_unvoiced_gap_sec": 0.080,
            "minimum_phrase_voiced_frames": 8,
            "minimum_voiced_frames_per_non_neutral_syllable": 4,
            "phrase_motion_thresholds_changed": False,
            "gap_comparison_float_epsilon": 1e-9,
        },
        "source_hashes_sha256": {
            "manifest": sha256(ROOT / "items.json"),
            "capt_core": sha256(ROOT.parent / "capt_core.py"),
            "main": sha256(ROOT.parent / "main.py"),
            "audit_runner": sha256(ROOT / "run_api_tone_audit.py"),
        },
        "status_counts_before": dict(before_counts),
        "status_counts_after": dict(after_counts),
        "alignment_status_counts_after": dict(alignment_counts),
        "reason_counts_after": dict(after_reasons),
        "tone_context_mismatches": tone_context_mismatches,
    }
    (FINAL_DIR / "audit_metadata.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    transition_lines = [f"| {before} | {after} | {count} |" for (before, after), count in sorted(status_transitions.items())]
    reason_lines = [f"| `{reason}` | {count} |" for reason, count in sorted(after_reasons.items())]
    mismatch_text = "None" if not tone_context_mismatches else json.dumps(tone_context_mismatches, ensure_ascii=False)
    report = f"""# Chapter 5 Praat phrase-tone audit: forced-alignment rerun

## Scope and decision

- Compared the original 53-row `item_summary.csv` / `api_responses.jsonl` with the final WhisperX-aligned run. The rerun had **{run_meta['final_api_successes']}/53 successful API responses**.
- Verified SHA-256 for the source audio: **{run_meta['audio_hashes_verified']}/53** manifest hashes matched.
- These are textbook teacher recordings. This is a pipeline diagnostic, not a student-accuracy or independent formative assessment benchmark. Teacher clips were not automatically labeled `good`.
- The endpoint stayed `praat_phrase_tone_direct_v1`; Praat used 65–600 Hz, 10 ms. The sample audio remained playback-only and did not affect scoring.
- No tone-motion or F0 decision threshold was tuned. Existing quality cutoffs were retained: CTC score >{core.MIN_ALIGNMENT_SCORE:.2f}, phrase coverage ≥0.50, at least 8 voiced frames, no internal F0 gap >80 ms, and ≥4 voiced frames per non-neutral syllable. The 1e-9 addition is only floating-point tolerance at the unchanged 80 ms boundary.

## Before / after

| Run | Good | Practice | Uncertain | Total |
|---|---:|---:|---:|---:|
| Original equal-duration split | {before_counts.get('good',0)} | {before_counts.get('practice',0)} | {before_counts.get('uncertain',0)} | 53 |
| Forced-alignment phrase edges | {after_counts.get('good',0)} | {after_counts.get('practice',0)} | {after_counts.get('uncertain',0)} | 53 |

Status transitions:

| Before | After | Items |
|---|---|---:|
{chr(10).join(transition_lines)}

Median phrase window duration for rows with a window: **{median_or_none(before_durations):.3f}s** before ({before_window_count}/53) → **{median_or_none(after_durations):.3f}s** after ({after_window_count}/53). Median trimmed voiced coverage: **{median_or_none(before_coverages):.3f}** → **{median_or_none(after_coverages):.3f}**.

## Audit findings

1. **Root cause: CTC character spans are anchors, not complete syllable durations.** In the first forced-alignment pass, the 53 final-character spans had a median duration of about 20 ms and 53 of 96 character spans were ≤30 ms. Treating them as full syllables left only about 2 Praat frames in many final-syllable regions. This produced 19 alignment-gated and 33 region-gated uncertain results in that diagnostic pass. The final implementation uses the CTC span centers as transcript-guided anchors, midpoint boundaries for analysis regions, and the nearby voiced F0 edges for the phrase window. Internal gaps remain unfilled and are still checked.
2. **Final alignment evidence:** {alignment_counts.get('reliable',0)}/53 clips pass the existing alignment reliability rule; {alignment_counts.get('unreliable',0)} remain `phrase_alignment_unreliable` because WhisperX did not provide reliable alignment for every target character. These clips are intentionally not scored as good/practice.
3. **Final voiced-evidence gate:** {gap_limited}/53 clips have usable CTC alignment but still fail the unchanged Praat region gate, all because their largest internal F0 gap exceeds 80 ms. Other insufficient-evidence conditions remain visible in `before_after.csv` and the JSON responses.
4. **Phrase pattern remains conservative:** {after_reasons.get('ambiguous_or_incomplete_whole_phrase_motion_sequence',0)} aligned clips remain ambiguous after quality gates. `observed_tone_sequence` is diagnostic only; it does not decide the phrase result.
5. **Expected tones / sandhi:** reconstructed lexical and surface-tone contexts matched the API sequence for all {53-len(tone_context_mismatches)}/53 clips. The catalog Pinyin for `沒（有）空` disambiguates final `空` as tone 4 (`kòng`); this correction is limited to this Chapter 5 catalog route. Tone details and sandhi reasons are in `tone_context_audit.csv`. Mismatches: {mismatch_text}.
6. **Praat tracking check over all 53 raw recordings:** F0 was extractable in {sum(1 for x in full_track_stats.values() if x['has_f0'])}/53. Adjacent-frame jumps ≥6 semitones within 20 ms occurred in {len(jumps_all)} clips: {', '.join(jumps_all) if jumps_all else 'none'}. Near-floor tracks (<67 Hz) occurred in {', '.join(floor_hits) if floor_hits else 'none'}; near-ceiling tracks (>590 Hz): {', '.join(ceiling_hits) if ceiling_hits else 'none'}. These are tracking flags, not proof that every marked movement is an octave error.

Final reason counts:

| Reason | Items |
|---|---:|
{chr(10).join(reason_lines)}

## Diagnostic plots

- Plots are in `plots/`. They show raw Praat F0 in Hz, phrase-relative semitone contour, selected phrase/syllable regions, CTC anchors, and suspected ≥6-semitone/20-ms jumps.
- All {len(all_practice)} final `practice` clips have plots. {len(selected_uncertain)} representative `uncertain` clips have plots, covering failed alignment, large F0 gaps, third-tone context, ambiguous contour, local pitch jumps, and the `kòng` reading. The plot index is `diagnostic_plot_index.csv`.

## Files

- `before_after.csv`: one-row-per-clip status and alignment comparison.
- `tone_context_audit.csv`: every selected character, lexical tone, tone after sandhi, and rule.
- `pitch_track_audit.csv`: Praat F0 availability, voiced-frame counts, and raw jump/floor/ceiling flags for all 53.
- `api_responses.jsonl`: complete final API responses.
- `audit_metadata.json`: threshold values and SHA-256 provenance for the input manifest and code files.

"""
    (FINAL_DIR / "audit_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "status_before": dict(before_counts),
        "status_after": dict(after_counts),
        "alignment": dict(alignment_counts),
        "reasons_after": dict(after_reasons),
        "practice_plots": len(all_practice),
        "uncertain_plots": len(selected_uncertain),
        "tone_mismatches": tone_context_mismatches,
        "raw_pitch_jump_clips": jumps_all,
        "report": str(FINAL_DIR / "audit_report.md"),
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
