"""
Service layer between the API and tone_pipeline.

Runs the v1 tone-verification pipeline on one recording and returns the payload shape the web UI and
practice_store already read (`phrases`, `summary`, `transcript_check`, `phrase_f0.contour`, ...), plus the full
per-syllable diagnostics under `tone_verification`. No web framework and no model loading happens here.
"""
from __future__ import annotations

import re

import numpy as np

import audio_prep
import capt_core
import tone_pipeline as tp

MODE = "tone_pipeline_v1"
_CONFIG = None


def get_config():
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = tp.load_config()
    return _CONFIG


# ------------------------------------------------------------------ grouping ----

def make_groups(clauses, annotated_words, normalize):
    """Phrase groups as (label, start, end) over the flattened characters, plus the annotation status.

    Word-level annotations are used when they reproduce the script exactly; otherwise punctuation clauses.
    """
    target_chars = list("".join(clauses))
    word_mapping = []
    for word in annotated_words:
        chars = list("".join(capt_core.prepare_target(normalize(word))))
        if chars:
            word_mapping.append((word, chars))
    annotation_hanzi = "".join(character for _, chars in word_mapping for character in chars)
    if not annotated_words:
        status = "missing"
    elif annotation_hanzi != "".join(target_chars):
        status = "does_not_match_script"
    else:
        status = "matched"
    groups, cursor = [], 0
    pieces = [(word, len(chars)) for word, chars in word_mapping] if status == "matched" else [
        (clause, len(clause)) for clause in clauses]
    for label, length in pieces:
        groups.append((label, cursor, cursor + length))
        cursor += length
    return groups, status


# ------------------------------------------------------------------ chapter 5 ----

def chapter5_optional_text_variants(text):
    """Expand one parenthesized optional span into present/omitted transcript forms."""
    variants = [text]
    for opening, closing in ((chr(0xFF08), chr(0xFF09)), ("(", ")")):
        start = text.find(opening)
        end = text.find(closing, start + 1) if start >= 0 else -1
        if start >= 0 and end > start:
            prefix, optional, suffix = text[:start], text[start + 1:end], text[end + 1:]
            variants.extend((prefix + suffix, prefix + optional + suffix))
            break
    return variants


def split_variants(script):
    """'哪裡/哪兒' -> ['哪裡', '哪兒']: every slash alternative is its own pronunciation path."""
    return [part.strip() for part in re.split(r"[/／]", script) if part.strip()]


def expand_optional(alternative):
    """'裡(面)' -> ['裡', '裡面'] (omitted, present); text without optional parts is returned as is."""
    options = chapter5_optional_text_variants(alternative)
    return options[1:] if len(options) > 1 else options


def chapter5_candidates(item, requested_script, normalize):
    """Distinct allowed scripts to align. Each variant (a slash alternative, optional text present or omitted)
    is its own candidate path; paths are never concatenated into one longer script."""
    candidates, seen = [], set()
    for script in [requested_script, item.get("script", ""), *item.get("script_variants", [])]:
        for alternative in split_variants(normalize(str(script or "").strip())):
            for variant in expand_optional(alternative):
                clauses = [part for part in capt_core.prepare_target(variant) if part.strip()]
                chars = tuple("".join(clauses))
                if chars and chars not in seen:
                    seen.add(chars)
                    candidates.append({"script_variant": variant, "clauses": clauses, "chars": list(chars)})
    return candidates


def chapter5_contexts(clauses, item):
    """pypinyin sandhi plus the catalog's disambiguating Pinyin where the workbook gives one."""
    contexts = capt_core.expected_tone_contexts(clauses)
    supplied_pinyin = str(item.get("pinyin") or "")
    # The workbook gives kòng in 沒（有）空; pypinyin picks kōng in the longer context.
    if "".join(clauses) in {"沒空", "沒有空"} and supplied_pinyin.endswith("kòng") and contexts:
        contexts[-1] = {**contexts[-1], "lexical_tone": 4, "target_tone": 4, "tone_rule": "catalog_pinyin_disambiguation"}
    return contexts


# -------------------------------------------------------------------- payload ----

def phrase_f0(contour, start_sec, end_sec, baseline_hz):
    """Chart contour for one phrase window: trims outer unvoiced frames, keeps inner gaps as nulls."""
    if not contour or end_sec <= start_sec or not baseline_hz:
        return None
    times = np.asarray(contour["times"], dtype=float)
    st = np.array([np.nan if v is None else v for v in contour["st"]], dtype=float)
    window = np.flatnonzero((times >= start_sec) & (times < end_sec))
    if not len(window):
        return None
    voiced = window[np.isfinite(st[window])]
    if not len(voiced):
        return None
    selected = np.arange(voiced[0], voiced[-1] + 1)
    origin, step = float(times[selected[0]]), float(contour["step_sec"])
    points = [{
        "offset_sec": round(float(times[i]) - origin, 3),
        "relative_semitone": None if not np.isfinite(st[i]) else round(float(st[i]), 2),
        "hz": None if not np.isfinite(st[i]) else round(float(baseline_hz * 2 ** (st[i] / 12)), 1),
    } for i in selected]
    window_voiced = int(np.isfinite(st[window]).sum())
    return {
        "duration_sec": round(float(times[selected[-1]]) + step - origin, 3),
        "aligned_window_sec": [round(start_sec, 3), round(end_sec, 3)],
        "frame_count": int(len(selected)),
        "voiced_frame_count": int(np.isfinite(st[selected]).sum()),
        "aligned_voiced_coverage": round(window_voiced / len(window), 3),
        "pitch_reference_hz": round(float(baseline_hz), 1),
        "contour": points,
    }


def _tone_or_zero(tone):
    return 0 if tone == 5 else tone


def build_payload(result, *, groups, word_annotation_status, boundaries_from, practice_mode, sample_id,
                  target_script, practice_item=None, audio_quality=None):
    syllables = result["syllables"]
    baseline = (result.get("f0_extraction") or {}).get("speaker_f0_median")
    phrases = []
    for label, start, end in groups:
        group = syllables[start:end]
        status = tp.aggregate_status(group)
        window = (group[0]["start_ms"] / 1000.0, group[-1]["end_ms"] / 1000.0) if group and "start_ms" in group[0] else None
        flagged = next((s for s in group if s["status"] == "practice"), None)
        note = None
        if status == "practice":
            note = flagged["feedback"]
        elif status == "insufficient_evidence":
            note = (group[0]["feedback"] if all(s["status"] == "insufficient_evidence" for s in group)
                    else result["config"]["feedback"]["insufficient_evidence"])
        phrases.append({
            "word": "".join(s["character"] for s in group),
            "annotation": label,
            "expected_tone_sequence": [_tone_or_zero(s["expected_surface_tone"]) for s in group],
            "lexical_tone_sequence": [_tone_or_zero(s["lexical_tone"]) for s in group],
            "observed_tone_sequence": [None] * len(group),   # the verifier checks expected tones; it does not name a tone
            "status": status,
            "phrase_status": status,
            "phrase_window_sec": [round(window[0], 3), round(window[1], 3)] if window else None,
            "phrase_f0": phrase_f0(result.get("contour"), *window, baseline) if window else None,
            "sequence_evidence": [
                {"character": s["character"], "status": s["status"], "tone_match": s["tone_match"],
                 "acoustic_quality": s["acoustic_quality"]} for s in group],
            "syllable_diagnostics": group,
            "assessment_note": note,
        })

    counts = {k: sum(p["status"] == k for p in phrases) for k in tp.STATUSES}
    total = len(syllables)
    return {
        "success": True,
        "mode": MODE,
        "practice_mode": practice_mode,
        "sample_id": sample_id,
        "reference_audio_used": False,
        "reference_audio_role": "optional_playback_only",
        "practice_item": practice_item,
        "target_script": target_script,
        "phrases": phrases,
        "word_segmentation": {
            "status": word_annotation_status,
            "scored_phrases": len(phrases),
            "phrase_boundaries_from": boundaries_from,
        },
        "transcript_checked": False,
        "transcript_check": {
            "status": "script_provided", "recognized_text": None,
            "matched_syllables": None, "total_syllables": total, "match_ratio": None,
        },
        "feedback": result["feedback"],
        "audio_quality": audio_quality,
        "tone_verification": {
            "method": result["method"],
            "calibrated": result["calibrated"],
            "calibration_version": result["calibration_version"],
            "acoustic_profile": result["acoustic_profile"],
            "alignment": {
                "aligner": {"mfa": "MFA (fixed transcript, no ASR)", "voiced_run": "voiced run (one syllable, no aligner)",
                            "injected": "supplied alignment"}.get(result["aligner"], result["aligner"]),
                "error": result["alignment_error"],
                "boundary_search": result["boundary_search"],
            },
            "f0_extraction": result["f0_extraction"],
            "syllable_summary": result["summary"],
            "syllables": syllables,
        },
        "score_semantics": {
            "status": "conservative expected-tone verification from the learner's Praat F0; not a calibrated probability",
            "gating": "practice requires high acoustic quality and a high mismatch; otherwise insufficient_evidence",
            "transcript_check": "ASR is not used; the supplied script is force-aligned",
        },
        "summary": {
            "n_phrases": len(phrases),
            "n_assessable": counts["likely_correct"] + counts["practice"],
            "n_with_audio_evidence": sum(1 for p in phrases if p["phrase_f0"]),
            "likely_correct": counts["likely_correct"],
            "uncertain": counts["insufficient_evidence"],
            "insufficient_evidence": counts["insufficient_evidence"],
            "practice": counts["practice"],
            "text_mismatch": 0,
            "transcript_unchecked": 0,
            "n_syllables": total,
            "syllables": result["summary"],
        },
    }


# ---------------------------------------------------------------- entry points ----

def _verify(prepared, clauses, contexts, calibration, force_mfa=False):
    """`prepared` is audio_prep.PreparedAudio: a recording that failed the quality check is not analysed."""
    return tp.verify_utterance(prepared.samples, prepared.sample_rate, "".join(clauses), calibration,
                               contexts=contexts, config=get_config(), rejected_reasons=prepared.rejected_reasons,
                               force_mfa=force_mfa)


def assess_recording(prepared, *, clauses, contexts, groups, word_annotation_status,
                     calibration, practice_mode, sample_id, target_script):
    """Book / custom practice: one script, grouped into phrases by word annotation or clause."""
    result = _verify(prepared, clauses, contexts, calibration)
    return build_payload(
        result, groups=groups, word_annotation_status=word_annotation_status,
        boundaries_from=("word-level JSON grouped over MFA syllable boundaries"
                         if word_annotation_status == "matched" else "punctuation clauses"),
        practice_mode=practice_mode, sample_id=sample_id, target_script=target_script,
        audio_quality=prepared.report())


def _decided_fraction(result):
    syllables = result["syllables"]
    return sum(s["status"] != "insufficient_evidence" for s in syllables) / max(1, len(syllables))


def _candidate_key(result):
    """Rank (reading, path) pairs; higher is better.

    1. an alignment that worked, 2. the least voiced speech left unexplained (a path that covers what was said),
    3. fewest `practice` syllables, 4. most of the scored syllables accepted, 5. most syllables decided, 6. quality.
    Unexplained speech comes before the verdict, so a learner who says the right word with a wrong tone is not moved
    onto some other variant just because that one happens to look better."""
    syllables = result["syllables"]
    scored = [s for s in syllables if s.get("expected_surface_tone", 0) != 5]
    quality = float(np.mean([s["acoustic_quality"] for s in syllables])) if syllables else 0.0
    outside = (result.get("f0_extraction") or {}).get("voiced_outside_syllables")
    unexplained = 1.0 if outside is None else outside
    accepted = sum(s["status"] == "likely_correct" for s in scored) / max(1, len(scored))
    return (0 if result["alignment_error"] else 1, -round(unexplained / 0.05),
            -sum(s["status"] == "practice" for s in syllables), round(accepted, 2), _decided_fraction(result), quality)


def best_variant(prepared, candidates, verify, config=None):
    """Pick ONE (reading, path) to score: the part of the recording that matches one of the variants.

    A clip may hold more than one reading (裡面 then 裡, 哪裡 then 哪兒) and a path never has to explain all of it.
    With several variants the whole recording and each separate reading are scored against each path, and the best
    pair wins (alignment succeeded, least speech left unexplained, most syllables decided, best quality).
    With one variant, or one reading, this is a single plain run. `verify(audio, candidate)` returns a result."""
    config = config or get_config()
    windows = [None]                                           # None = the whole recording
    if len(candidates) > 1 and prepared.passed:
        chunks = audio_prep.speech_chunks(prepared, config)
        if len(chunks) > 1:
            windows += chunks
    runs = []
    for window in windows:
        audio = prepared if window is None else audio_prep.slice_prepared(prepared, *window)
        for candidate in candidates:
            runs.append((candidate, window, verify(audio, candidate)))
    candidate, window, result = max(runs, key=lambda run: _candidate_key(run[2]))
    tried = [{"script_variant": c["script_variant"], "reading": None if w is None else [round(w[0], 2), round(w[1], 2)],
              "decided_fraction": round(_decided_fraction(r), 3),
              "voiced_outside_syllables": (r.get("f0_extraction") or {}).get("voiced_outside_syllables"),
              "alignment_error": r["alignment_error"]} for c, w, r in runs]
    return candidate, window, result, tried


def assess_chapter5_item(prepared, *, item, sample_id, target_script, calibration, normalize):
    """Chapter 5 word: score the reading in the recording that matches one of the allowed variants."""
    candidates = [c for c in chapter5_candidates(item, target_script, normalize)
                  if len(chapter5_contexts(c["clauses"], item)) == len(c["chars"])]
    if not prepared.passed:
        candidates = candidates[:1]   # nothing is analysed, so there is nothing to choose between
    if not candidates:
        raise ValueError("The script contains no valid Han characters to score pronunciation.")

    def verify(audio, candidate):
        # With several variants every one is force-aligned so they compare fairly.
        return _verify(audio, candidate["clauses"], chapter5_contexts(candidate["clauses"], item), calibration,
                       force_mfa=len(candidates) > 1)

    candidate, window, result, tried = best_variant(prepared, candidates, verify)
    payload = build_payload(
        result, groups=[(item["traditional_chinese"], 0, len(candidate["chars"]))],
        word_annotation_status="selected_chapter5_item",
        boundaries_from="MFA forced alignment of the supplied or allowed text variant",
        practice_mode="chapter5_word", sample_id=sample_id, target_script=target_script,
        practice_item={key: item[key] for key in
                       ("word_key", "item_type", "traditional_chinese", "script", "pinyin", "section")},
        audio_quality=prepared.report())
    payload["aligned_script_variant"] = candidate["script_variant"]
    payload["tone_verification"]["scored_reading"] = None if window is None else {
        "start_sec": round(window[0], 2), "end_sec": round(window[1], 2),
        "note": "the recording holds more than one reading; only this one was scored"}
    payload["tone_verification"]["script_variants_tried"] = tried
    return payload
