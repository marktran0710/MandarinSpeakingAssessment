"""
Compare tone-decision settings on the same recordings, without touching tone_v1_config.yaml.

    python compare_configs.py trials/user_config_a.yaml
    python compare_configs.py trials/user_config_a.yaml --split          # also each top-level section on its own
    python compare_configs.py trials/a.yaml trials/b.yaml --suite word sentence

A trial file holds the sections/keys of tone_v1_config.yaml it wants to change (tone:, confidence:, ...); the baseline
(the current file) is always run first. Three suites, each with its own question:

  word       teacher recordings read as single words (53 Chapter 5 words + 86 verified-correct clips).
             Everything is correct, so every `practice` is a FALSE ALARM; the Good share shows how often it says so.
  sentence   100 teacher sentences through the sentence profile (partial credit, half-third, short-window floor).
             Same reading: `practice` = false alarm.
  synthetic  520 synthetic recordings with known wrong syllables (a stress test, not real learner accuracy):
             error_recall = wrong syllables flagged `practice`; false_accept = wrong syllables shown as Good.

Alignment (MFA) does not depend on the tone thresholds, so it is cached on disk and shared by every config:
the first run pays for MFA (about an hour for everything), later runs only for pitch analysis.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import json
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml

import tone_assessment as ta
import tone_pipeline as tp

ROOT = Path(__file__).resolve().parent
SUITES = ("word", "sentence", "synthetic")
DEFAULT_CACHE_ROOT = ROOT / "outputs" / "config_trials" / "cache"
SHORT = {"likely_correct": "Good", "insufficient_evidence": "Okay", "uncertain": "Okay", "practice": "Practice", "not_heard": "NotHeard"}


# --------------------------------------------------------------------------------------------- config switch ----

def deep_update(target, changes):
    """Merge nested dicts: a trial may change `sentence: tone: x` without having to repeat the sentence section."""
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


@contextlib.contextmanager
def trial_config(trial):
    """Every tp.load_config() call (and ta.get_config(), which uses it) sees the trial values on top of the file."""
    original = tp.load_config

    def patched(path=None, overrides=None, **_):             # same signature as tp.load_config (callers use overrides=)
        # Layers, as in production: the file, then the trial's values, then a profile's own settings (the `sentence:`
        # section, already merged with the trial's `sentence:` values). Applying the trial last would let a word-mode
        # value overwrite the phrase-mode value for the same key.
        config = original(path, None)
        deep_update(config, trial)
        for section, values in (overrides or {}).items():
            config.setdefault(section, {}).update(values)
        return config

    tp.load_config = patched
    ta._CONFIG = ta._SENTENCE_CONFIG = None
    ta._LEVEL_CONFIGS.clear()
    try:
        yield
    finally:
        tp.load_config = original
        ta._CONFIG = ta._SENTENCE_CONFIG = None
        ta._LEVEL_CONFIGS.clear()


# ---------------------------------------------------------------------------------------------------- suites ----

def run_word(cache_root, limit=None):
    """{ "service_53:key#i" | "true_86:key#i": status } for the single-word teacher recordings."""
    import evaluate_sentences as es
    import word_regression as wr

    statuses = {}
    with contextlib.redirect_stdout(io.StringIO()):
        with es.alignment_cache(cache_root / "service_53"):
            for key, entry in wr.run_service_53().items():
                statuses["service_53:" + key] = entry[0]
        for key, entry in wr.run_true_86(str(cache_root / "true_86")).items():
            statuses["true_86:" + key] = entry[0]
    return statuses


def run_sentence(cache_root, limit=None):
    """{"syllables": {"id#i": status}, "phrases": [status...], "sentences": n_with_practice, "n_sentences": n}"""
    import evaluate_sentences as es

    with tempfile.TemporaryDirectory(prefix="trial_sentences_") as out, contextlib.redirect_stdout(io.StringIO()):
        es.evaluate(es.DEFAULT_SAMPLE_DIR, es.DEFAULT_WORD_JSON, out, cache_dir=str(cache_root / "sentence"), limit=limit)
        read = lambda name: list(csv.DictReader(open(Path(out) / name, newline="", encoding="utf-8-sig")))  # noqa: E731
        syllables, phrases, sentences = read("syllables.csv"), read("phrases.csv"), read("sentences.csv")
    detail = ("sample_id", "character", "expected_tone", "prev_tone", "next_tone", "status", "tone_match", "acoustic_quality",
              "syllable_ms", "window_ms", "reasons", "speaker")
    return {"syllables": {f"{r['sample_id']}#{r['index']}": r["status"] for r in syllables},
            "rows": {f"{r['sample_id']}#{r['index']}": {k: r[k] for k in detail} for r in syllables},
            "phrases": [r["status"] for r in phrases],
            "sentences": sum(int(r["practice"]) > 0 for r in sentences), "n_sentences": len(sentences)}


def run_synthetic(cache_root, limit=None):
    """Rows of the synthetic benchmark (truth_wrong as bool)."""
    import benchmark_tone_pipeline as bench
    import word_regression as wr

    with tempfile.TemporaryDirectory(prefix="trial_synthetic_") as work, contextlib.redirect_stdout(io.StringIO()):
        bench.run(str(wr.SYNTHETIC_MANIFEST), work, ["adaptive"], cache_dir=str(cache_root / "synthetic"), limit=limit)
        rows = list(csv.DictReader(open(Path(work) / "syllables.csv", newline="", encoding="utf-8-sig")))
    for row in rows:
        row["truth_wrong"] = row["truth_wrong"] == "True"
        row["key"] = f"{Path(row['file']).name}#{row['index']}"
    return rows


# ------------------------------------------------------------------------------------------------ summaries ----

def pct(part, whole):
    return round(100.0 * part / whole, 1) if whole else None


def mix(statuses):
    counts = Counter(SHORT.get(s, s) for s in statuses)
    n = len(statuses)
    return {"n": n, "Good": pct(counts["Good"], n), "Okay": pct(counts["Okay"], n), "Practice": pct(counts["Practice"], n),
            "practice_n": counts["Practice"], "NotHeard": counts["NotHeard"]}


def synthetic_summary(rows):
    def block(subset):
        wrong = [r for r in subset if r["truth_wrong"]]
        same = [r for r in subset if not r["truth_wrong"]]
        count = lambda rs, s: sum(r["status"] == s for r in rs)  # noqa: E731
        return {"n_wrong": len(wrong), "error_recall": pct(count(wrong, "practice"), len(wrong)),
                "false_accept": pct(count(wrong, "likely_correct"), len(wrong)), "wrong_okay": pct(count(wrong, "insufficient_evidence"), len(wrong)),
                "n_unchanged": len(same), "false_alarm": pct(count(same, "practice"), len(same)),
                "unchanged_good": pct(count(same, "likely_correct"), len(same))}
    by_variant = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
    return {"overall": block(rows), "by_variant": {k: block(v) for k, v in sorted(by_variant.items())}}


def summarize(results):
    summary = {}
    if "word" in results:
        words = results["word"]
        summary["word"] = {name: mix([s for k, s in words.items() if k.startswith(name + ":")]) for name in ("service_53", "true_86")}
        summary["word"]["all"] = mix(list(words.values()))
    if "sentence" in results:
        sentence = results["sentence"]
        summary["sentence"] = {"syllables": mix(list(sentence["syllables"].values())), "phrases": mix(sentence["phrases"]),
                               "sentences_with_practice": sentence["sentences"], "n_sentences": sentence["n_sentences"],
                               "sentences_with_practice_pct": pct(sentence["sentences"], sentence["n_sentences"])}
    if "synthetic" in results:
        summary["synthetic"] = synthetic_summary(results["synthetic"])
    return summary


def transitions(base, trial):
    """How the same syllables moved between baseline and trial: {"Okay->Good": n, ...} (unchanged ones left out)."""
    moves = Counter()
    for key in base.keys() & trial.keys():
        a, b = SHORT.get(base[key], base[key]), SHORT.get(trial[key], trial[key])
        if a != b:
            moves[f"{a}->{b}"] += 1
    return dict(sorted(moves.items(), key=lambda kv: -kv[1]))


def newly_flagged(base_results, trial_results):
    """Sentence syllables that are `practice` under the trial but were not under the baseline (teacher audio: new false alarms)."""
    base, trial = base_results.get("sentence"), trial_results.get("sentence")
    if not base or not trial:
        return []
    return [dict(trial["rows"][k], was=base["syllables"][k]) for k, status in trial["syllables"].items()
            if status == "practice" and base["syllables"].get(k) != "practice"]


def newly_flagged_words(base_results, trial_results):
    """Single-word syllables (teacher audio, all correct) that are `practice` under the trial but were not under the baseline."""
    base, trial = base_results.get("word"), trial_results.get("word")
    if not base or not trial:
        return []
    return [(key, SHORT.get(base.get(key), base.get(key))) for key, status in sorted(trial.items())
            if status == "practice" and base.get(key) != "practice"]


def flat_statuses(results):
    flat = {}
    if "word" in results:
        flat.update({"word|" + k: v for k, v in results["word"].items()})
    if "sentence" in results:
        flat.update({"sentence|" + k: v for k, v in results["sentence"]["syllables"].items()})
    if "synthetic" in results:
        flat.update({"synthetic|" + r["key"]: r["status"] for r in results["synthetic"]})
    return flat


# -------------------------------------------------------------------------------------------------- report ----

def render(names, summaries, moves, flagged=None, flagged_words=None):
    lines = []
    cols = " | ".join(names)
    sep = "|---|" + "---|" * len(names)

    def row(label, getter):
        cells = []
        for summary in summaries:
            try:
                value = getter(summary)
            except (KeyError, TypeError):
                value = None
            cells.append("-" if value is None else str(value))
        return f"| {label} | " + " | ".join(cells) + " |"

    if "word" in summaries[0]:
        lines += ["### Single words, teacher audio (all correct: Practice = false alarm)", "", f"| | {cols} |", sep]
        for suite in ("service_53", "true_86", "all"):
            n = summaries[0]["word"][suite]["n"]
            for key in ("Good", "Okay", "Practice"):
                lines.append(row(f"{suite} ({n} syll.) {key} %", lambda s, suite=suite, key=key: s["word"][suite][key]))
        lines.append(row("false alarms (syllables)", lambda s: s["word"]["all"]["practice_n"]))
        lines.append("")
    if "sentence" in summaries[0]:
        lines += ["### Sentences, teacher audio (all correct: Practice = false alarm)", "", f"| | {cols} |", sep]
        for key in ("Good", "Okay", "Practice"):
            lines.append(row(f"syllables {key} % (n={summaries[0]['sentence']['syllables']['n']})", lambda s, key=key: s["sentence"]["syllables"][key]))
        lines.append(row("false alarms (syllables)", lambda s: s["sentence"]["syllables"]["practice_n"]))
        for key in ("Good", "Okay", "Practice"):
            lines.append(row(f"phrases {key} %", lambda s, key=key: s["sentence"]["phrases"][key]))
        lines.append(row("sentences with a Practice phrase %", lambda s: s["sentence"]["sentences_with_practice_pct"]))
        lines.append("")
    if "synthetic" in summaries[0]:
        lines += ["### Synthetic wrong tones (stress test; truth = known wrong syllables)", "", f"| | {cols} |", sep]
        for label, key in (("error recall % (wrong flagged Practice)", "error_recall"), ("false accept % (wrong shown Good)", "false_accept"),
                           ("wrong shown Okay %", "wrong_okay"), ("false alarm % (unchanged flagged Practice)", "false_alarm"),
                           ("unchanged shown Good %", "unchanged_good")):
            lines.append(row(f"overall {label}", lambda s, key=key: s["synthetic"]["overall"][key]))
        lines += ["", "By variant (recall % / false accept %):", "", f"| variant (n wrong) | {cols} |", sep]
        for variant, info in summaries[0]["synthetic"]["by_variant"].items():
            lines.append(row(f"{variant} ({info['n_wrong']})", lambda s, v=variant: f"{s['synthetic']['by_variant'][v]['error_recall']} / {s['synthetic']['by_variant'][v]['false_accept']}"))
        lines.append("")
    if any(moves):
        lines += ["### What moved versus the baseline (same syllables)", ""]
        for name, move in zip(names, moves):
            if move:
                lines.append(f"- **{name}**: " + ", ".join(f"{k} {v}" for k, v in move.items()))
        lines.append("")
    for name, keys in (flagged_words or {}).items():
        if keys:
            lines += [f"### New false alarms under {name} (single words that became Practice): {len(keys)}", ""]
            lines += [f"- {key} (was {was})" for key, was in keys[:40]]
            lines.append("")
    for name, rows in (flagged or {}).items():
        if rows:
            lines += [f"### New false alarms under {name} (teacher sentences that became Practice)", "",
                      "| sample | char | tone | prev/next | syllable ms | window ms | match | quality | was | reasons (before) |", "|---|---|---|---|---|---|---|---|---|---|"]
            lines += [f"| {r['sample_id']} | {r['character']} | T{r['expected_tone']} | {r['prev_tone'] or '-'}/{r['next_tone'] or '-'} | {r['syllable_ms']} | "
                      f"{r['window_ms']} | {r['tone_match']} | {r['acoustic_quality']} | {SHORT.get(r['was'], r['was'])} | {r['reasons'] or '-'} |" for r in rows]
            lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------ main ----

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trials", nargs="+", help="YAML files with the settings to try")
    parser.add_argument("--suite", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--split", action="store_true", help="also run each top-level section of each trial on its own")
    parser.add_argument("--limit", type=int, default=None, help="first N recordings of the sentence and synthetic suites (smoke test)")
    parser.add_argument("--base", default=None, help="YAML merged into EVERY config, the baseline included (e.g. trials/_legacy_dict_base.yaml: "
                                                      "the older dictionary, so alignment caches made before a dictionary change stay valid)")
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--out", default=None, help="folder for report.json / report.md (default outputs/config_trials/<time>)")
    args = parser.parse_args()

    base = (yaml.safe_load(Path(args.base).read_text(encoding="utf-8")) or {}) if args.base else {}
    with_base = lambda trial: deep_update(copy.deepcopy(base), copy.deepcopy(trial))      # noqa: E731
    configs = [("baseline", with_base({}))]
    for file in args.trials:
        trial = yaml.safe_load(Path(file).read_text(encoding="utf-8")) or {}
        configs.append((Path(file).stem, with_base(trial)))
        if args.split:
            configs += [(f"{Path(file).stem}:{section}", with_base({section: values})) for section, values in trial.items()]
    out = Path(args.out or ROOT / "outputs" / "config_trials" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    # If a step ever stalls (a hung subprocess, a lock), the stack of every thread is written here every 4 minutes.
    import faulthandler
    stall_log = open(out / "stalls.log", "w")
    faulthandler.dump_traceback_later(240, repeat=True, file=stall_log)
    cache_root = Path(args.cache_root)
    runners = {"word": run_word, "sentence": run_sentence, "synthetic": run_synthetic}

    names, summaries, flats, flagged, flagged_words, base_results = [], [], [], {}, {}, None
    for name, overrides in configs:
        started = time.time()
        print(f"[{name}] running {', '.join(args.suite)} ...", flush=True)
        with trial_config(overrides):
            results = {suite: runners[suite](cache_root, args.limit) for suite in args.suite}
        summary = summarize(results)
        if base_results is None:
            base_results = results
        else:
            flagged[name] = newly_flagged(base_results, results)
            flagged_words[name] = newly_flagged_words(base_results, results)
        names.append(name)
        summaries.append(summary)
        flats.append(flat_statuses(results))
        print(f"[{name}] done in {time.time() - started:.0f}s", flush=True)
        moves = [{}] + [transitions(flats[0], f) for f in flats[1:]]
        report = {"configs": dict(configs), "names": names, "summaries": summaries, "moves": moves, "new_false_alarms": flagged,
                  "new_false_alarm_words": flagged_words}
        (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        (out / "report.md").write_text(render(names, summaries, moves, flagged, flagged_words), encoding="utf-8")   # written after every config: nothing is lost
    print("\n" + render(names, summaries, moves, flagged, flagged_words))
    print(f"\nreport: {out / 'report.md'}")


if __name__ == "__main__":
    sys.exit(main())
