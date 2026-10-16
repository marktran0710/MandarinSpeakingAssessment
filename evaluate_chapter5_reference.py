"""
Run the tone pipeline over one-word reference recordings (native-speaker audio) and compare against the script.

    # clips named by Word Key / word id, scripts from the question-bank workbook (--pattern "C*.mp3" = every chapter)
    python evaluate_chapter5_reference.py --audio-dir ".../audio" --xlsx ".../question_bank_overview.xlsx" \
        --out-dir outputs/reference_eval --cache-dir outputs/reference_eval/alignment_cache
    # or an explicit script per clip: a CSV with file,chinese[,lesson,pinyin]
    python evaluate_chapter5_reference.py --audio-dir ".../audio" --scripts-csv scripts.csv --out-dir outputs/reference_eval

What it checks
  1. Script check: the workbook's tone-marked Pinyin vs the lexical tones the pipeline derives (pypinyin + Taiwan
     overrides) for every word, so a wrong *expected* tone is separated from a wrong *detected* tone.
  2. Pipeline run per recording (audio quality gate, MFA alignment, adaptive F0, verifier), adaptive vs fixed range.
  3. Reference audio is correct speech, so every `practice` is a false positive; `insufficient_evidence` is
     abstention. The best-fitting tone (argmax of the verifier's T1-T4 scores) is reported against the
     expected tone as a confusion matrix.

A slash item such as 哪裡/哪兒 gets one candidate path per variant; the reading in the clip that matches one of
them is the one scored (tone_assessment.best_variant).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pypinyin import Style, lazy_pinyin

import audio_prep
import benchmark_tone_pipeline as bench
import capt_core
import tone_assessment as ta
import tone_pipeline as tp
import xlsx_reader

TONE_MARKS = {"ā": "a", "á": "a", "à": "a", "ǎ": "a", "ē": "e", "é": "e", "è": "e", "ě": "e", "ī": "i", "í": "i",
              "ì": "i", "ǐ": "i", "ō": "o", "ó": "o", "ò": "o", "ǒ": "o", "ū": "u", "ú": "u", "ù": "u", "ǔ": "u",
              "ǖ": "ü", "ǘ": "ü", "ǜ": "ü", "ǚ": "ü"}


# ------------------------------------------------------------------- script ----

def load_items_from_dir(audio_dir, xlsx_path, pattern="C5-*.mp3"):
    """Clips from a folder named by word id (C5-5-1-W012.mp3) or full Word Key (C5-5-1-I10-W012.mp3), joined to the
    workbook's Questions sheet. `pattern` picks the chapters (default Chapter 5; "C*.mp3" for all).

    The workbook's Word Key carries an item number (C5-5-1-I10-W012); the folder drops it, so files are matched on
    the key without it. The lesson (5-1 / 5-2 / 5-3) stands in for the source recording when pooling a speaker."""
    questions = xlsx_reader.read_xlsx(xlsx_path)["Questions"]
    header = next(i for i, row in enumerate(questions) if "Word Key" in row)
    columns = {name: questions[header].index(name) for name in ("Word Key", "Traditional Chinese", "Pinyin")}
    short_to_workbook = {}
    for row in questions[header + 1:]:
        if len(row) > max(columns.values()) and row[columns["Word Key"]]:
            key = row[columns["Word Key"]]
            short = __import__("re").sub(r"-I\d+(?=-W)", "", key)
            entry = (key, row[columns["Traditional Chinese"]], row[columns["Pinyin"]])
            short_to_workbook.setdefault(short, entry)
            short_to_workbook.setdefault(key, entry)
    items = []
    for path in sorted(Path(audio_dir).glob(pattern)):
        short = path.stem
        if short not in short_to_workbook:
            continue
        key, chinese, pinyin = short_to_workbook[short]
        lesson = "-".join(key.split("-")[1:3])
        items.append({
            "word_key": key, "lesson": lesson, "audio": str(path), "source": lesson, "source_spans": "",
            "confidence": "", "chinese": chinese, "pinyin": pinyin, "workbook_matches_manifest": True,
            "in_workbook": True, "file": path.name,
        })
    return items


def load_items_from_scripts(audio_dir, scripts_csv):
    """Clips whose script is given explicitly: a CSV with `file` and `chinese` (and optionally `lesson`, `pinyin`).

    Use it when the script of each clip is known from somewhere other than the question-bank workbook."""
    with open(scripts_csv, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    items = []
    for row in rows:
        path = Path(audio_dir) / row["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        lesson = row.get("lesson") or "unknown"
        items.append({
            "word_key": path.stem, "lesson": lesson, "audio": str(path), "source": lesson, "source_spans": "",
            "confidence": "", "chinese": row["chinese"], "pinyin": row.get("pinyin", ""),
            "workbook_matches_manifest": True, "in_workbook": False, "file": path.name,
        })
    return items


def candidate_scripts(chinese):
    """One candidate path per variant: each slash alternative, optional text present or omitted. A recording is
    scored against ONE path; the paths are never joined into a longer script."""
    scripts = []
    for alternative in ta.split_variants(chinese):
        for variant in ta.expand_optional(alternative):
            script = "".join(capt_core.prepare_target(variant))
            if script and script not in scripts:
                scripts.append(script)
    return scripts


PINYIN_SYLLABLE = __import__("re").compile(
    r"(?:[zcs]h|[bpmfdtnlgkhjqxrzcswy])?[aeiouüāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]+(?:ng|n|r)?(?![aeiouüāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ])")
MARK_TONE = {c: t for t, group in enumerate(("āēīōūǖ", "áéíóúǘ", "ǎěǐǒǔǚ", "àèìòùǜ"), start=1) for c in group}


def workbook_tones(pinyin):
    """Tone numbers (5 = unmarked / neutral) for every spoken path of a tone-marked Pinyin string:
    each slash variant, with optional (…) text omitted and present."""
    paths = []
    for variant in pinyin.split("/"):
        text = variant.replace(" ", "").replace("'", "").lower()
        for form in ta.expand_optional(text):
            form = form.replace("（", "").replace("）", "").replace("(", "").replace(")", "")
            paths.append([next((MARK_TONE[ch] for ch in m.group(0) if ch in MARK_TONE), 5)
                          for m in PINYIN_SYLLABLE.finditer(form)])
    return paths


def script_check(item, script_used):
    """Do the tones the pipeline expects for the script it aligned agree with the workbook's tone-marked Pinyin?

    The book writes citation tones (你好 nǐhǎo) but also some surface forms (不錯 búcuò, 哥哥 gēge), so agreeing
    with either the lexical or the surface sequence counts."""
    clauses = capt_core.prepare_target(script_used)
    contexts = capt_core.expected_tone_contexts(clauses)
    if not item["pinyin"]:
        return {"workbook_pinyin": "", "workbook_tones": [], "pipeline_lexical": [], "pipeline_surface": [],
                "agrees": None, "agrees_with": None}
    paths = workbook_tones(item["pinyin"])
    lexical = [c["lexical_tone"] for c in contexts]
    surface = [c["target_tone"] for c in contexts]
    agrees_with = "surface" if surface in paths else ("lexical" if lexical in paths else None)
    return {"workbook_pinyin": item["pinyin"], "workbook_tones": paths, "pipeline_lexical": lexical,
            "pipeline_surface": surface, "agrees": agrees_with is not None, "agrees_with": agrees_with}


# ---------------------------------------------------------------------- run ----

def compatible_tones(syllable, threshold=0.7):
    """Tones whose verifier score reaches `threshold` ('' = none fits; None = no tone scoring happened).

    T2 and T3 (and a half-T3 and T4) legitimately overlap, so one winner would be misleading."""
    scores = syllable.get("tone_scores")
    if not scores:
        return None
    return "|".join(tone for tone in ("1", "2", "3", "4") if scores[tone] >= threshold)


def run_one(prepared, item, profile, config, cache_dir, pitch_range):
    """Score the reading in the recording that matches one of the item's variants (one path, never joined).

    Returns ((script, result), tried, window): `window` is the (start, end) of the reading that was scored when the
    clip holds several readings, else None."""
    scripts = candidate_scripts(item["chinese"])
    candidates = [{"script_variant": s, "clauses": [s], "chars": list(s)} for s in scripts]

    def verify(audio, candidate):
        script = candidate["script_variant"]
        try:
            use_mfa = len(scripts) > 1 or tp.needs_mfa(tp.syllable_count(script), config)
            syllables = (bench.cached_syllables(audio.samples, audio.sample_rate, script, config, cache_dir)
                         if use_mfa else None)
            return tp.verify_utterance(audio.samples, audio.sample_rate, script, profile, alignment=syllables,
                                       config=config, pitch_range=pitch_range)
        except tp.AlignmentError as exc:
            result = tp.verify_utterance(audio.samples, audio.sample_rate, script, profile, alignment=[],
                                         config=config, pitch_range=pitch_range)
            result["alignment_error"] = result["alignment_error"] or str(exc)
            return result

    candidate, window, result, tried = ta.best_variant(prepared, candidates, verify, config)
    return (candidate["script_variant"], result), tried, window


def evaluate(xlsx_path, out_dir, cache_dir, limit=None, boundary_search=True, max_shift_ms=None,
             audio_dir=None, pattern="C5-*.mp3", scripts_csv=None):
    search = {"enabled": boundary_search}
    if max_shift_ms is not None:
        search["max_shift_ms"] = max_shift_ms
    config = tp.load_config(overrides={"boundary_search": search})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, words, profiles = [], [], {}
    items = (load_items_from_scripts(audio_dir, scripts_csv) if scripts_csv
             else load_items_from_dir(audio_dir, xlsx_path, pattern))[: limit or None]
    prepared = {i["word_key"]: audio_prep.prepare_recording(i["audio"], config) for i in items}
    by_source = defaultdict(list)
    for item in items:
        by_source[item["source"]].append(prepared[item["word_key"]])
    for source, recordings in by_source.items():
        usable = [p for p in recordings if p.passed] or recordings
        profiles[source] = tp.pooled_calibration([(p.samples, p.sample_rate) for p in usable], config)

    for mode, pitch_range in (("adaptive", None), ("fixed", bench.FIXED_RANGE)):
        for item in items:
            audio, profile = prepared[item["word_key"]], profiles[item["source"]]
            if not audio.passed:
                result = tp.verify_utterance(audio.samples, audio.sample_rate, candidate_scripts(item["chinese"])[0],
                                             profile, config=config, pitch_range=pitch_range,
                                             rejected_reasons=audio.rejected_reasons)
                script, tried, window = candidate_scripts(item["chinese"])[0], [], None
            else:
                (script, result), tried, window = run_one(audio, item, profile, config, cache_dir, pitch_range)
            statuses = [s["status"] for s in result["syllables"]]
            words.append({
                "mode": mode, "word_key": item["word_key"], "chinese": item["chinese"], "pinyin": item["pinyin"],
                "confidence": item["confidence"], "source": item["source"], "script_used": script,
                "reading_scored": "" if window is None else f"{window[0]:.2f}-{window[1]:.2f}s",
                "audio_passed": audio.passed, "audio_reasons": "|".join(audio.reasons),
                "alignment_error": result["alignment_error"] or "",
                "status": tp.aggregate_status(result["syllables"]), "syllables": len(statuses),
                **{k: statuses.count(k) for k in tp.STATUSES},
            })
            for syllable in result["syllables"]:
                rows.append({
                    "mode": mode, "word_key": item["word_key"], "chinese": item["chinese"],
                    "confidence": item["confidence"], "source": item["source"], "index": syllable["index"],
                    "character": syllable["character"], "pinyin": syllable["pinyin"],
                    "lexical_tone": syllable["lexical_tone"], "expected_tone": syllable["expected_surface_tone"],
                    "tone_rule": syllable["tone_rule"], "status": syllable["status"],
                    "compatible_tones": compatible_tones(syllable), "tone_match": syllable["tone_match"],
                    "acoustic_quality": syllable["acoustic_quality"], "voiced_ratio": syllable.get("voiced_ratio"),
                    "ac_cc_agreement": syllable.get("ac_cc_agreement"), "duration_ms": syllable.get("duration_ms"),
                    "start_st": syllable.get("start_st"), "end_st": syllable.get("end_st"),
                    "min_st": syllable.get("min_st"), "max_st": syllable.get("max_st"),
                    "min_position": syllable.get("min_position"),
                    "fall_before_min_st": syllable.get("fall_before_min_st"),
                    "rise_after_min_st": syllable.get("rise_after_min_st"),
                    "early_to_late_delta_st": syllable.get("early_to_late_delta_st"),
                    "reasons": "|".join(syllable["reasons"]),
                })
    chosen = {w["word_key"]: w["script_used"] for w in words if w["mode"] == "adaptive"}
    checks = {i["word_key"]: {**script_check(i, chosen[i["word_key"]]), "in_workbook": i["in_workbook"],
                              "workbook_matches_manifest": i["workbook_matches_manifest"]} for i in items}

    for name, data in (("syllables.csv", rows), ("words.csv", words)):
        with open(out / name, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    report = summarize(rows, words, checks, profiles)
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return report, rows, words, checks


def summarize(rows, words, checks, profiles):
    report = {"script_check": {
        "words": len(checks), "pinyin_agrees": sum(bool(c["agrees"]) for c in checks.values()),
        "without_pinyin_to_check": sum(c["agrees"] is None for c in checks.values()),
        "agrees_with": dict(Counter(c["agrees_with"] for c in checks.values())),
        "disagreements": {k: {"workbook": c["workbook_pinyin"], "workbook_tones": c["workbook_tones"],
                              "lexical": c["pipeline_lexical"], "surface": c["pipeline_surface"]}
                          for k, c in checks.items() if c["agrees"] is False},
        "not_in_workbook": [k for k, c in checks.items() if not c["in_workbook"]],
    }, "profiles": {s: {k: p[k] for k in ("f0_median", "f0_p95", "pitch_floor", "pitch_ceiling", "acoustic_profile")}
                    for s, p in profiles.items()}, "modes": {}}
    for mode in ("adaptive", "fixed"):
        mode_rows = [r for r in rows if r["mode"] == mode]
        decided = [r for r in mode_rows if r["expected_tone"] != 5 and r["compatible_tones"] is not None]
        kinds = Counter()
        confusion = Counter()
        for r in decided:
            compatible = r["compatible_tones"].split("|") if r["compatible_tones"] else []
            expected = str(r["expected_tone"])
            if not compatible:
                kinds["no_tone_fits"] += 1
            elif compatible == [expected]:
                kinds["unique_and_correct"] += 1
            elif expected in compatible:
                kinds["correct_but_ambiguous"] += 1
            else:
                kinds["unique_or_set_without_expected"] += 1
            confusion[(r["expected_tone"], r["compatible_tones"] or "none")] += 1
        total = Counter(r["status"] for r in mode_rows)
        scored = [r for r in mode_rows if r["expected_tone"] != 5]
        report["modes"][mode] = {
            "syllables": len(mode_rows), "status": dict(total),
            "false_practice_rate": round(total["practice"] / len(mode_rows), 4) if mode_rows else None,
            "abstention_rate": round(total["insufficient_evidence"] / len(mode_rows), 4) if mode_rows else None,
            "scored_syllables_non_neutral": len(scored),
            "syllables_with_tone_scores": len(decided),
            "tone_compatibility": dict(kinds),
            "confusion_expected_to_compatible_set": {f"T{a}->{b}": n for (a, b), n in sorted(confusion.items())},
            "by_expected_tone": {
                f"T{t}": dict(Counter(r["status"] for r in mode_rows if r["expected_tone"] == t))
                for t in (1, 2, 3, 4, 5)},
            "by_cut_confidence": {c: dict(Counter(r["status"] for r in mode_rows if r["confidence"] == c))
                                  for c in sorted({r["confidence"] for r in mode_rows})},
            "words": dict(Counter(w["status"] for w in words if w["mode"] == mode)),
            "audio_rejected": [w["word_key"] for w in words if w["mode"] == mode and not w["audio_passed"]],
            "alignment_failed": [w["word_key"] for w in words if w["mode"] == mode and w["alignment_error"]],
            "mean_voiced_ratio": round(float(np.mean([r["voiced_ratio"] for r in mode_rows if r["voiced_ratio"] is not None])), 4),
            "mean_ac_cc_agreement": round(float(np.mean([r["ac_cc_agreement"] for r in mode_rows if r["ac_cc_agreement"] is not None])), 4),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", required=True, help="folder of C5-5-1-W012.mp3 style files")
    parser.add_argument("--pattern", default="C5-*.mp3", help='which files of --audio-dir ("C*.mp3" = every chapter)')
    parser.add_argument("--xlsx", default=None, help="question-bank workbook (not needed with --scripts-csv)")
    parser.add_argument("--scripts-csv", default=None, help="CSV with file,chinese[,lesson,pinyin]: explicit script per clip")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-boundary-search", action="store_true",
                        help="trust the aligner's boundaries as they are (for comparison)")
    parser.add_argument("--max-shift-ms", type=int, default=None, help="boundary search window (default from config)")
    args = parser.parse_args()
    report, *_ = evaluate(args.xlsx, args.out_dir, args.cache_dir, args.limit,
                          boundary_search=not args.no_boundary_search, max_shift_ms=args.max_shift_ms,
                          audio_dir=args.audio_dir, pattern=args.pattern, scripts_csv=args.scripts_csv)
    print(json.dumps(report, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
