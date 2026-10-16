"""
Sentence-level evaluation on the teacher recordings (lessons 5-8, one dialogue line per file).

    python evaluate_sentences.py --out-dir outputs/sentences_baseline --cache-dir outputs/sentences_cache

Each recording goes through the same service call the UI uses for book practice (tone_assessment.assess_recording,
phrases from the word annotations), so the numbers are what a learner would see. The recordings are native speech with
the script as the target, so every `practice` is a false alarm and `insufficient_evidence` is an abstention.

This is a read-only measurement: it imports the pipeline and changes nothing in it. MFA alignments are cached on disk
(keyed by audio, characters and dictionary) by wrapping tone_pipeline.mfa_align inside this process only.

Reported at three levels (syllable, phrase as the UI shows it, whole sentence), with the causes a fix would have to
address: how long the analysed window of a syllable was, its tone, its neighbours, its speaker.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np

import audio_prep
import capt_core
import tone_assessment as ta
import tone_pipeline as tp

DATA = Path(r"D:\hautran\Lab\mandarin-speaking data\Recorded Data")
DEFAULT_SAMPLE_DIR = DATA / "lessons_5_8_FINAL_content_verified_v3_finallllll"
DEFAULT_WORD_JSON = DATA / "lessons_5_8_word_level.json"
SCRIPT_CSV = "lessons-5-8_with_character_audio_synced.csv"
WINDOW_BUCKETS = ((60, "<=60 ms"), (100, "61-100 ms"), (150, "101-150 ms"), (10 ** 9, ">150 ms"))


# --------------------------------------------------------------------------------------------------- data ----

def load_catalog(sample_dir):
    """The teacher lines exactly as main.py's catalog reads them: id lesson-story-scene, script, speaker, audio file."""
    sample_dir = Path(sample_dir)
    samples = []
    with open(sample_dir / SCRIPT_CSV, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                lesson, story, scene = int(row["lesson"]), int(row["story"]), int(row["scene"])
            except (KeyError, TypeError, ValueError):
                continue
            script = (row.get("script") or "").strip()
            audio = sample_dir / f"{lesson}-{story}-{scene:02d}.mp3"
            if not script or not audio.is_file():
                continue
            samples.append({"id": f"{lesson}-{story}-{scene}", "lesson": f"{lesson}-{story}", "speaker": (row.get("character") or "").strip(),
                            "script": script, "audio": str(audio)})
    return samples


def load_word_annotations(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return {str(k): [str(w) for w in words if str(w).strip()] for k, words in data.items() if isinstance(words, list)}


@contextmanager
def alignment_cache(cache_dir):
    """Cache tone_pipeline.mfa_align on disk for the duration of the block (this process only)."""
    original = tp.mfa_align
    cache = Path(cache_dir) if cache_dir else None

    def cached(samples, sample_rate, chars, config=None):
        if cache is None:
            return original(samples, sample_rate, chars, config)
        key = hashlib.sha1(np.asarray(samples, dtype=np.float32).tobytes() + "".join(chars).encode("utf-8")
                           + tp.dictionary_fingerprint(config or tp.load_config()).encode("utf-8")).hexdigest()
        file = cache / f"{key}.json"
        if file.exists():
            record = json.loads(file.read_text(encoding="utf-8"))
        else:
            try:
                record = {"tiers": original(samples, sample_rate, chars, config)}
            except tp.AlignmentError as exc:
                record = {"error": str(exc)}
            cache.mkdir(parents=True, exist_ok=True)
            file.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        if "error" in record:
            raise tp.AlignmentError(record["error"])
        return record["tiers"]

    with mock.patch.object(tp, "mfa_align", cached):
        yield


# ------------------------------------------------------------------------------------------------ one line ----

def assess_sample(sample, prepared, profile, annotations, normalize):
    """The book-mode path of main._assess_pronunciation, minus the HTTP and storage parts."""
    target_script = normalize(sample["script"])
    clauses = [c for c in capt_core.prepare_target(target_script) if c.strip()]
    groups, annotation_status = ta.make_groups(clauses, annotations.get(sample["id"], []), normalize)
    contexts = ta.sentence_contexts(clauses, groups, annotation_status)
    captured = {}
    build_payload = ta.build_payload

    def observe(result, **kwargs):          # read-only tap: the payload is built exactly as before
        captured["result"] = result
        return build_payload(result, **kwargs)

    with mock.patch.object(ta, "build_payload", observe):
        payload = ta.assess_recording(prepared, clauses=clauses, contexts=contexts, groups=groups,
                                      word_annotation_status=annotation_status, calibration=profile,
                                      practice_mode="book", sample_id=sample["id"], target_script=target_script)
    return payload, groups, annotation_status, captured["result"]


def contour_record(result):
    """The sentence F0 in semitones every 10 ms (None = unvoiced), for the review page."""
    contour = (result or {}).get("contour")
    if not contour:
        return None
    times, st = contour["times"], contour["st"]
    values = [None if v is None or v != v else round(float(v), 1) for v in st[::2]]
    return {"t0": round(float(times[0]), 3), "step": round(float(contour["step_sec"]) * 2, 4), "st": values}


def window_bucket(window_ms):
    if window_ms is None:
        return "no window"
    return next(label for limit, label in WINDOW_BUCKETS if window_ms <= limit)


def flatten(sample, payload, groups, annotation_status):
    """-> (syllable rows, phrase rows, sentence row)."""
    verification = payload["tone_verification"]
    syllables = verification["syllables"]
    group_of = {}
    for label, start, end in groups:
        for index in range(start, end):
            group_of[index] = (label, end - start)
    rows = []
    for i, s in enumerate(syllables):
        segment = s.get("analysis_segment")
        window = (segment["end_ms"] - segment["start_ms"]) if segment else None
        features = s.get("features") or {}
        label, size = group_of.get(i, ("", 0))
        rows.append({
            "sample_id": sample["id"], "lesson": sample["lesson"], "speaker": sample["speaker"], "index": s["index"],
            "character": s["character"], "phrase": label, "phrase_syllables": size,
            "expected_tone": s["expected_surface_tone"], "lexical_tone": s["lexical_tone"],
            "prev_tone": syllables[i - 1]["expected_surface_tone"] if i else "",
            "next_tone": syllables[i + 1]["expected_surface_tone"] if i + 1 < len(syllables) else "",
            "status": s["status"], "tone_match": s["tone_match"], "acoustic_quality": s["acoustic_quality"],
            "syllable_ms": s.get("duration_ms"), "window_ms": window, "window_bucket": window_bucket(window),
            "voiced_frames": features.get("voiced_frames"),
            "start_ms": s.get("start_ms"), "end_ms": s.get("end_ms"),
            "window_start_ms": segment["start_ms"] if segment else None,
            "window_end_ms": segment["end_ms"] if segment else None,
            "also_accepted": "|".join(str(t) for t in s.get("also_accepted_tones", [])),
            "clause_final_half_t3": bool(s.get("clause_final_half_t3")),
            "tone_rule": s.get("tone_rule", ""),
            "reasons": "|".join(s["reasons"]),
        })
    phrase_rows = []
    for label, start, end in groups:
        group = syllables[start:end]
        status = tp.aggregate_status(group) if group else "insufficient_evidence"
        phrase_rows.append({"sample_id": sample["id"], "lesson": sample["lesson"], "speaker": sample["speaker"],
                            "phrase": label, "syllables": end - start, "status": status})
    sentence = {
        "sample_id": sample["id"], "lesson": sample["lesson"], "speaker": sample["speaker"], "script": sample["script"],
        "syllables": len(syllables), "status": tp.aggregate_status(syllables),
        "annotation": annotation_status, "alignment_error": verification["alignment"]["error"] or "",
        "audio_passed": payload["audio_quality"]["passed"] if payload.get("audio_quality") else True,
        **{k: sum(r["status"] == k for r in syllables) for k in tp.STATUSES},
    }
    return rows, phrase_rows, sentence


# ---------------------------------------------------------------------------------------------- summaries ----

def _share(part, whole):
    return round(100.0 * part / whole, 1) if whole else None


def status_counts(rows, key="status"):
    counts = Counter(r[key] for r in rows)
    total = len(rows)
    return {"n": total, **{k: counts.get(k, 0) for k in tp.STATUSES},
            **{f"{k}_pct": _share(counts.get(k, 0), total) for k in tp.STATUSES}}


def breakdown(rows, field):
    groups = defaultdict(list)
    for r in rows:
        groups[str(r[field])].append(r)
    return {k: status_counts(v) for k, v in sorted(groups.items())}


def summarize(syllables, phrases, sentences):
    flagged = [r for r in syllables if r["status"] == "practice"]
    return {
        "syllables": status_counts(syllables),
        "phrases": status_counts(phrases),
        "sentences": {**status_counts(sentences),
                      "with_any_practice": sum(1 for s in sentences if s["practice"]),
                      "with_any_practice_pct": _share(sum(1 for s in sentences if s["practice"]), len(sentences))},
        "by_window": breakdown(syllables, "window_bucket"),
        "by_expected_tone": breakdown(syllables, "expected_tone"),
        "by_lesson": breakdown(syllables, "lesson"),
        "by_speaker": breakdown(syllables, "speaker"),
        "practice_by_window": dict(Counter(r["window_bucket"] for r in flagged)),
        "abstention_reasons": dict(Counter(x for r in syllables if r["status"] == "insufficient_evidence"
                                           for x in (r["reasons"] or "none").split("|")).most_common()),
        "sentences_not_scored": [s["sample_id"] for s in sentences if s["alignment_error"] or not s["audio_passed"]],
    }


def render_markdown(report):
    def table(title, data, cols=("n", "likely_correct_pct", "practice_pct", "insufficient_evidence_pct")):
        lines = [f"### {title}", "", "| group | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for key, row in data.items():
            lines.append(f"| {key} | " + " | ".join(str(row.get(c)) for c in cols) + " |")
        return "\n".join(lines) + "\n"

    s = report["summary"]
    out = [f"# Sentence evaluation: {s['sentences']['n']} teacher recordings, {s['syllables']['n']} syllables", "",
           "Native speech: every `practice` below is a false alarm; `insufficient_evidence` is an abstention.", "",
           table("Levels", {"syllables": s["syllables"], "phrases (as the UI shows them)": s["phrases"],
                            "sentences (all syllables)": s["sentences"]}),
           f"Sentences with at least one `practice` syllable: {s['sentences']['with_any_practice']} "
           f"({s['sentences']['with_any_practice_pct']}%)", "",
           table("By analysed-window length", s["by_window"]), table("By expected tone", s["by_expected_tone"]),
           table("By lesson", s["by_lesson"]), table("By speaker", s["by_speaker"]),
           f"`practice` by window: {s['practice_by_window']}", "", f"Abstention reasons: {s['abstention_reasons']}", ""]
    if s["sentences_not_scored"]:
        out.append(f"Not scored (alignment or audio quality): {s['sentences_not_scored']}")
    out += ["", "## Every `practice` syllable", "", "| sample | char | tone | prev/next | window ms | syllable ms | match | speaker |", "|---|---|---|---|---|---|---|---|"]
    for r in report["practice_syllables"]:
        out.append(f"| {r['sample_id']} | {r['character']} ({r['phrase']}) | T{r['expected_tone']} | {r['prev_tone'] or '-'}/{r['next_tone'] or '-'} "
                   f"| {r['window_ms']} | {r['syllable_ms']} | {r['tone_match']} | {r['speaker']} |")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------------------- run ----

def evaluate(sample_dir, word_json, out_dir, cache_dir=None, group_by="speaker", limit=None, lessons=None, normalize=None):
    if normalize is None:
        import opencc
        normalize = opencc.OpenCC("s2twp").convert           # the same normalisation main.py applies
    config = ta.get_config()
    samples = [s for s in load_catalog(sample_dir) if not lessons or s["lesson"].split("-")[0] in lessons][: limit or None]
    annotations = load_word_annotations(word_json)
    prepared = {s["id"]: audio_prep.prepare_recording(s["audio"], config) for s in samples}
    key_of = {"speaker": lambda s: s["speaker"], "lesson": lambda s: s["lesson"], "all": lambda s: "all",
              "recording": lambda s: s["id"]}[group_by]
    pools = defaultdict(list)
    for s in samples:
        pools[key_of(s)].append(prepared[s["id"]])
    profiles = {}
    for key, members in pools.items():
        usable = [p for p in members if p.passed] or members
        profiles[key] = tp.pooled_calibration([(p.samples, p.sample_rate) for p in usable], config)

    syllable_rows, phrase_rows, sentence_rows, contours = [], [], [], {}
    with alignment_cache(cache_dir):
        for number, sample in enumerate(samples, 1):
            payload, groups, annotation_status, raw = assess_sample(sample, prepared[sample["id"]],
                                                                    profiles[key_of(sample)], annotations, normalize)
            rows, phrases, sentence = flatten(sample, payload, groups, annotation_status)
            contours[sample["id"]] = {"audio": sample["audio"], "script": sample["script"], "speaker": sample["speaker"],
                                      "contour": contour_record(raw)}
            syllable_rows += rows
            phrase_rows += phrases
            sentence_rows.append(sentence)
            print(f"[{number}/{len(samples)}] {sample['id']} {sentence['status']:22s} "
                  f"{sentence['likely_correct']}/{sentence['practice']}/{sentence['insufficient_evidence']}", flush=True)

    report = {
        "settings": {"group_by": group_by, "limit": limit, "lessons": sorted(lessons) if lessons else None,
                     "profiles": {k: {f: v.get(f) for f in ("pitch_floor", "pitch_ceiling", "f0_median")} for k, v in profiles.items()}},
        "summary": summarize(syllable_rows, phrase_rows, sentence_rows),
        "practice_syllables": [r for r in syllable_rows if r["status"] == "practice"],
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("syllables.csv", syllable_rows), ("phrases.csv", phrase_rows), ("sentences.csv", sentence_rows)):
        if data:
            with open(out / name, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "contours.json").write_text(json.dumps(contours, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--word-json", default=str(DEFAULT_WORD_JSON))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--group-by", choices=["speaker", "lesson", "all", "recording"], default="speaker",
                        help="which recordings share one pooled speaker profile (the app uses the learner's own calibration)")
    parser.add_argument("--lessons", nargs="+", default=None, help="e.g. 5 6 (default: all)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    report = evaluate(args.sample_dir, args.word_json, args.out_dir, args.cache_dir, args.group_by, args.limit,
                      set(args.lessons) if args.lessons else None)
    s = report["summary"]
    print(json.dumps({k: s[k] for k in ("syllables", "phrases", "sentences")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
