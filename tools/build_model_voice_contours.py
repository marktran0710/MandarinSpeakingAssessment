"""Build model_voice_contours.json for the "Theo giọng mẫu" mode: the teachers' contours per tone and the four per-tone sigmoids.

    python tools/build_model_voice_contours.py collect [--learners 5]     # slow: runs the pipeline on the teacher sentences, the synthetic wrong tones and the
                                                                          # archived attempts of the most active learners (alignments come from the caches)
    python tools/build_model_voice_contours.py fit                        # fast: picks the contours, fits the sigmoids, writes the JSON and prints the measurements

`collect` keeps the judged pitch windows (and the rules' verdicts) in outputs/model_voice/collected.pkl; `fit` can be repeated on it. The sigmoid of a tone is
chosen so that its score is 0.30 at the 1.5% quantile and 0.70 at the 7% quantile of the syllables known to be right: the teachers' and the learners' that the
rules call Good (a third tone that also accepts tone 2 is left out of tone 3's fit: it is scored against both). Measured: see the printed report.
"""
import argparse
import collections
import contextlib
import io
import json
import math
import os
import pickle
import random
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("TONE_MFA_WARM", "1")
os.environ.setdefault("MANDARIN_USER_DATA_DIR", "")
os.environ.setdefault("MANDARIN_PRACTICE_DATA_DIR", tempfile.mkdtemp(prefix="model_voice_build_"))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model_voice  # noqa: E402

COLLECTED = ROOT / "outputs" / "model_voice" / "collected.pkl"
SEED = 20261019
FIT_FALSE_ALARM, FIT_GOOD_MISS = 0.015, 0.07          # score 0.30 at this quantile of the correct syllables, 0.70 at that one
OPTIONAL_RULE = "third_tone_sandhi_optional"


# ------------------------------------------------------------------------------------------------ collect ----

def collect(learners):
    import compare_configs as cc
    import evaluate_sentences as es
    import audio_prep
    import main
    import practice_store as ps
    import tone_assessment as ta
    import tone_pipeline as tp

    captured, calls, call_file = [], [0], {}
    original, original_utterance = tp.verify_syllable, tp.verify_utterance

    def hooked_utterance(*args, **kwargs):
        calls[0] += 1
        row = sys._getframe(1).f_locals.get("row")             # the synthetic benchmark loops over `row`: remember which file this call is for
        if isinstance(row, dict) and row.get("path"):
            call_file[calls[0]] = Path(str(row["path"])).name
        return original_utterance(*args, **kwargs)

    def hooked(syllable, context, extraction, clean, config, speaker, *args, **kwargs):
        result = original(syllable, context, extraction, clean, config, speaker, *args, **kwargs)
        try:
            start, end, _ = tp._segment_for(syllable, config)
            _, st, _ = tp.segment_pitch(clean, start, end, config)
            values, spread = model_voice.judged_values(st), model_voice.recording_std(clean["st"])
        except Exception:
            values, spread = None, 1.0
        captured.append({"char": syllable.get("char"), "tone": context.get("target_tone"), "values": values, "spread": spread, "call": calls[0],
                         "accepted": [context.get("target_tone"), *context.get("also_accept_tones", ())]})
        return result

    tp.verify_syllable, tp.verify_utterance = hooked, hooked_utterance
    data = {"teacher": [], "synthetic": [], "users": []}

    print("teacher sentences ...", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        sentence = cc.run_sentence(cc.DEFAULT_CACHE_ROOT)
    rows = list(sentence["rows"].values())
    assert len(rows) == len(captured), (len(rows), len(captured))
    for row, cap in zip(rows, captured):
        tone = int(row["expected_tone"]) if str(row["expected_tone"]).isdigit() else None
        data["teacher"].append({**cap, "rec": row["sample_id"], "tone": tone, "status": row["status"]})

    print("synthetic wrong tones ...", flush=True)
    captured.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        synth = cc.run_synthetic(cc.DEFAULT_CACHE_ROOT)
    by_call = collections.defaultdict(list)
    for cap in captured:
        by_call[cap["call"]].append(cap)
    last_call = {call_file[c]: c for c in sorted(by_call) if c in call_file}
    by_file = collections.OrderedDict()
    for row in synth:
        by_file.setdefault(Path(row["file"]).name, []).append(row)
    for file, rows_of_file in by_file.items():
        caps = by_call.get(last_call.get(file), [])
        if len(caps) == len(rows_of_file):
            for row, cap in zip(rows_of_file, caps):
                data["synthetic"].append({**cap, "file": file, "wrong": bool(row["truth_wrong"]), "status": row.get("status")})

    print(f"top {learners} learners ...", flush=True)
    real = ROOT / "practice_data"
    snapshot = Path(tempfile.mkdtemp(prefix="model_voice_db_"))
    shutil.copyfile(real / "practice.sqlite3", snapshot / "practice.sqlite3")
    names = [r["name"] for r in ps.PracticeStore(snapshot / "practice.sqlite3", snapshot / "rec").leaderboard(limit=learners)["top"][:learners]]
    print("  ", names, flush=True)
    connection = sqlite3.connect(f"file:{(real / 'practice.sqlite3').as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    for name in names:
        student = connection.execute("select id from students where name=?", (name,)).fetchone()
        calibration = json.loads(connection.execute("select calibration_json from student_voice_calibrations where student_id=?", (student["id"],)).fetchone()[0])
        attempts = connection.execute("select id, sample_id, target_script, audio_path from practice_attempts where student_id=? and sample_id <> 'custom' order by created_at",
                                      (student["id"],)).fetchall()
        done = skipped = 0
        with es.alignment_cache(cc.DEFAULT_CACHE_ROOT / "learner"):
            for r in attempts:
                mode = "chapter5_word" if r["sample_id"].startswith("C5-") else "book"
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        prepared = audio_prep.prepare_recording(r["audio_path"], ta.get_config())
                        captured.clear()
                        target = main.converter.convert(r["target_script"].strip())
                        if mode == "chapter5_word":
                            payload = ta.assess_chapter5_item(prepared, item=main.CHAPTER5_WORD_ITEMS[r["sample_id"]], sample_id=r["sample_id"], target_script=target,
                                                              calibration=calibration, normalize=main.converter.convert, level="easy")
                        else:
                            clauses = [c for c in main.core.prepare_target(target) if c.strip()]
                            groups, status = ta.make_groups(clauses, main.WORD_ANNOTATIONS.get(r["sample_id"], []), main.converter.convert)
                            payload = ta.assess_recording(prepared, clauses=clauses, contexts=ta.sentence_contexts(clauses, groups, status), groups=groups,
                                                          word_annotation_status=status, calibration=calibration, practice_mode=mode, sample_id=r["sample_id"],
                                                          target_script=target, level="easy")
                except Exception:
                    skipped += 1
                    continue
                syllables = payload["tone_verification"]["syllables"]
                if len(syllables) != len(captured):
                    skipped += 1
                    continue
                done += 1
                for index, (s, cap) in enumerate(zip(syllables, captured)):
                    data["users"].append({**cap, "user": name, "attempt_id": r["id"], "sample_id": r["sample_id"], "index": index, "tone": s.get("expected_surface_tone"),
                                          "status": s["status"], "rule": s.get("tone_rule"), "start_ms": s.get("start_ms")})
        print(f"   {name}: {done} attempts, {skipped} skipped", flush=True)
    COLLECTED.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(data, open(COLLECTED, "wb"))
    print("saved", COLLECTED, {k: len(v) for k, v in data.items()})


# ---------------------------------------------------------------------------------------------------- fit ----

def load_settings():
    import yaml
    return yaml.safe_load(open(ROOT / "tone_v1_config.yaml", encoding="utf-8"))["model_voice"]


def usable(rows):
    return [r for r in rows if r["tone"] in model_voice.TONES and r["values"] is not None]


def auc(right, wrong):
    if not len(right) or not len(wrong):
        return float("nan")
    right = np.sort(np.asarray(right))
    lo, hi = np.searchsorted(right, wrong, side="left"), np.searchsorted(right, wrong, side="right")
    return float(np.mean((len(right) - hi + 0.5 * (hi - lo)) / len(right)))


def fit(output):
    settings = load_settings()
    data = pickle.load(open(COLLECTED, "rb"))
    teacher = usable(data["teacher"])
    pools = {t: [(r["rec"], model_voice.normalise(r["values"], r["spread"], settings["std_floor"])) for r in teacher if r["tone"] == t] for t in model_voice.TONES}
    shuffler = random.Random(SEED)
    for tone in model_voice.TONES:
        shuffler.shuffle(pools[tone])
    count = int(settings["references_per_tone"])
    dummy = {t: {"midpoint": 0.0, "steepness": 1.0} for t in model_voice.TONES}

    def contours_without(rec):
        return model_voice.Contours({t: [s for r, s in pools[t] if r != rec][:count] for t in model_voice.TONES}, dummy)

    everything = contours_without(None)
    leave_one_out = {}

    def similarity(row, tone, rec=None):
        sequence = model_voice.normalise(row["values"], row["spread"], settings["std_floor"])
        if rec is None:
            return everything.similarity(tone, sequence, settings)
        if rec not in leave_one_out:
            leave_one_out[rec] = contours_without(rec)
        return leave_one_out[rec].similarity(tone, sequence, settings)

    print("scoring the teachers ...", flush=True)
    teacher_raw = [similarity(r, r["tone"], r["rec"]) for r in teacher]
    users = usable(data["users"])
    print("scoring the learners ...", flush=True)
    user_raw = [similarity(r, r["tone"]) for r in users]
    correct = {t: [] for t in model_voice.TONES}
    for r, value in zip(teacher, teacher_raw):
        correct[r["tone"]].append(value)
    for r, value in zip(users, user_raw):
        if r["status"] == "likely_correct" and r.get("rule") != OPTIONAL_RULE:
            correct[r["tone"]].append(value)
    sigmoid = {}
    for tone in model_voice.TONES:
        low, high = float(np.quantile(correct[tone], FIT_FALSE_ALARM)), float(np.quantile(correct[tone], FIT_GOOD_MISS))
        sigmoid[tone] = {"midpoint": round((low + high) / 2, 4), "steepness": round(2 * math.log(7 / 3) / (high - low), 3)}
        print(f"  tone {tone}: {len(correct[tone])} correct syllables, cut points {low:.3f} / {high:.3f} -> midpoint {sigmoid[tone]['midpoint']}, steepness {sigmoid[tone]['steepness']}")
    document = {"version": 1, "seed": SEED, "references_per_tone": count, "std_floor": settings["std_floor"], "step_ms": 10,
                "fitted_on": {"teacher_syllables": len(teacher), "learner_syllables": len(users), "learners": sorted({r["user"] for r in users})},
                "sigmoid": {str(t): v for t, v in sigmoid.items()},
                "references": {str(t): [[round(float(v), 4) for v in s] for _, s in pools[t][:count]] for t in model_voice.TONES}}
    Path(output).write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    print("written", output, f"({Path(output).stat().st_size // 1024} KB)")

    # ---- the measurements (no guards here: they are the rules' own and are applied in the pipeline)
    contours = model_voice.Contours({t: [s for _, s in pools[t][:count]] for t in model_voice.TONES}, sigmoid)

    def score(row, rec=None):
        accepted = [t for t in row["accepted"] if t in model_voice.TONES] or [row["tone"]]
        use = contours if rec is None else model_voice.Contours({t: [s for r, s in pools[t] if r != rec][:count] for t in model_voice.TONES}, sigmoid)
        judged = model_voice.judge(row["values"], row["spread"], accepted, settings, use)
        return None if judged is None else judged["score"]

    good_at, practice_at = settings["good_at"], settings["practice_at"]
    print("\nmeasurements (model-voice score alone, before the rules' guards):")
    t_scores = [score(r, r["rec"]) for r in teacher]
    t_scores = [s for s in t_scores if s is not None]
    print(f"  teachers ({len(t_scores)}): Good {100*np.mean(np.array(t_scores) >= good_at):.1f}%, Needs practice {100*np.mean(np.array(t_scores) <= practice_at):.1f}%")
    u_scores = [(r, score(r)) for r in users]
    u_scores = [(r, s) for r, s in u_scores if s is not None]
    for status in ("likely_correct", "insufficient_evidence", "practice"):
        s = np.array([v for r, v in u_scores if r["status"] == status])
        if len(s):
            print(f"  learners the rules call {status:22s} ({len(s):4d}): Good {100*np.mean(s >= good_at):5.1f}%  Needs practice {100*np.mean(s <= practice_at):5.1f}%")
    synth = usable(data["synthetic"])
    sy = [(r, score(r)) for r in synth]
    sy = [(r, s) for r, s in sy if s is not None]
    wrong = np.array([s for r, s in sy if r["wrong"]])
    right = np.array([s for r, s in sy if not r["wrong"]])
    if len(wrong) and len(right):
        print(f"  synthetic wrong tones ({len(wrong)}): flagged {100*np.mean(wrong <= practice_at):.1f}%, shown Good {100*np.mean(wrong >= good_at):.1f}% | unchanged ({len(right)}): Good {100*np.mean(right >= good_at):.1f}%")
        rules_wrong = [r["status"] for r, _ in sy if r["wrong"]]
        print(f"  (the rules on the same wrong syllables: flagged {100*np.mean([s == 'practice' for s in rules_wrong]):.1f}%, shown Good {100*np.mean([s == 'likely_correct' for s in rules_wrong]):.1f}%; AUC right vs wrong: {auc(right, wrong):.3f})")


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--learners", type=int, default=5)
    f = sub.add_parser("fit")
    f.add_argument("--output", default=str(model_voice.CONTOURS_PATH))
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.learners)
    else:
        fit(args.output)


if __name__ == "__main__":
    main_cli()
