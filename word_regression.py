"""
Word-level regression gate: the word-mode results must not move while the sentence work changes shared code.

    python word_regression.py --freeze      # write golden/word_level_snapshot.json from the current code
    python word_regression.py --check       # re-run and compare; exit 1 on any difference
    python word_regression.py --check --fresh    # ignore the alignment caches (re-run MFA everywhere)

Three suites, all native or synthetic audio whose per-syllable verdicts are recorded as they are today:

  service_53   the 53 Chapter 5 words through the real chapter5_word service path (tone_assessment.assess_chapter5_item,
               items and audio exactly as main.py loads them), one pooled speaker profile
  true_86      the 86 verified-correct recordings of lessons 5-1..8-1 (pipeline level, adaptive range)
  synthetic_520  the synthetic wrong-tone benchmark (pipeline level, adaptive range)

A syllable is compared on (status, tone_match, acoustic_quality, reasons). A change in any of them, a syllable that
appears or disappears, fails the check, so "the word level is unchanged" is something that can be shown, not claimed.
The sentence profile may only add behaviour that is off by default; this script is the proof that it is.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GOLDEN_DIR = ROOT / "golden"
SNAPSHOT = GOLDEN_DIR / "word_level_snapshot.json"
TRUE_DIR = Path(r"D:\hautran\Lab\mandarin-speaking data\Recorded Data\audio test\true")
TRUE_SCRIPTS = GOLDEN_DIR / "true_scripts_verified.csv"
SYNTHETIC_MANIFEST = GOLDEN_DIR / "synthetic_manifest.csv"
SUITES = ("service_53", "true_86", "synthetic_520")
FIELDS = ("status", "tone_match", "acoustic_quality", "reasons")


def entry(status, tone_match, acoustic_quality, reasons):
    return [status, None if tone_match in (None, "") else round(float(tone_match), 3),
            None if acoustic_quality in (None, "") else round(float(acoustic_quality), 3),
            "|".join(reasons) if isinstance(reasons, (list, tuple)) else (reasons or "")]


def compare(golden, current):
    """Differences between two snapshots: {suite: {"missing": [...], "extra": [...], "changed": {key: (old, new)}}}."""
    report = {}
    for suite in sorted(set(golden) | set(current)):
        old, new = golden.get(suite), current.get(suite)
        if old is None or new is None:
            report[suite] = {"missing": sorted(old or []) if new is None else [], "extra": sorted(new or []) if old is None else [],
                             "changed": {}, "suite": "absent from " + ("current run" if new is None else "golden")}
            continue
        changed = {k: (old[k], new[k]) for k in old.keys() & new.keys() if old[k] != new[k]}
        missing, extra = sorted(old.keys() - new.keys()), sorted(new.keys() - old.keys())
        if changed or missing or extra:
            report[suite] = {"missing": missing, "extra": extra, "changed": changed}
    return report


def describe(report, limit=15):
    lines = []
    for suite, diff in report.items():
        lines.append(f"[{suite}] {len(diff['changed'])} changed, {len(diff['missing'])} missing, {len(diff['extra'])} extra"
                     + (f" ({diff['suite']})" if "suite" in diff else ""))
        for key, (old, new) in list(diff["changed"].items())[:limit]:
            lines.append(f"    {key}: {old} -> {new}")
        for key in diff["missing"][:limit]:
            lines.append(f"    missing {key}")
        for key in diff["extra"][:limit]:
            lines.append(f"    extra   {key}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ suites ----

def run_service_53(cache_dir=None):
    """The real word-mode service path. main.py is imported for its item loader only, with a throw-away data dir."""
    os.environ.setdefault("MANDARIN_PRACTICE_DATA_DIR", tempfile.mkdtemp(prefix="word_regression_"))
    sys.path.insert(0, str(ROOT))
    import audio_prep
    import main
    import tone_assessment as ta
    import tone_pipeline as tp

    config = ta.get_config()
    items = main.CHAPTER5_WORD_ITEMS
    prepared = {key: audio_prep.prepare_recording(str(item["audio_path"]), config) for key, item in items.items()}
    profile = tp.pooled_calibration([(p.samples, p.sample_rate) for p in prepared.values() if p.passed], config)
    out = {}
    for key, item in items.items():
        payload = ta.assess_chapter5_item(prepared[key], item=item, sample_id=key,
                                          target_script=main.converter.convert(item["script"]),
                                          calibration=profile, normalize=main.converter.convert)
        for s in payload["tone_verification"]["syllables"]:
            out[f"{key}#{s['index']}"] = entry(s["status"], s["tone_match"], s["acoustic_quality"], s["reasons"])
    return out


def run_true_86(cache_dir=None):
    sys.path.insert(0, str(ROOT))
    import evaluate_chapter5_reference as ev

    with tempfile.TemporaryDirectory(prefix="word_regression_") as work:
        _, rows, _, _ = ev.evaluate(None, work, cache_dir, audio_dir=str(TRUE_DIR), scripts_csv=str(TRUE_SCRIPTS))
    return {f"{r['word_key']}#{r['index']}": entry(r["status"], r["tone_match"], r["acoustic_quality"], r["reasons"])
            for r in rows if r["mode"] == "adaptive"}


def run_synthetic_520(cache_dir=None):
    sys.path.insert(0, str(ROOT))
    import benchmark_tone_pipeline as bench

    with tempfile.TemporaryDirectory(prefix="word_regression_") as work:
        bench.run(str(SYNTHETIC_MANIFEST), work, ["adaptive"], cache_dir=cache_dir)
        with open(Path(work) / "syllables.csv", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    return {f"{Path(r['file']).name}#{r['index']}": entry(r["status"], r["tone_match"], r["acoustic_quality"], r["reasons"])
            for r in rows if r["mode"] == "adaptive"}


RUNNERS = {"service_53": run_service_53, "true_86": run_true_86, "synthetic_520": run_synthetic_520}


def collect(suites, cache_dirs, fresh=False):
    snapshot = {}
    for name in suites:
        cache = None
        if not fresh:
            cache = cache_dirs.get(name)
        print(f"running {name} ...", flush=True)
        snapshot[name] = RUNNERS[name](cache)
        print(f"  {len(snapshot[name])} syllables", flush=True)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true", help="record the current results as the golden snapshot")
    mode.add_argument("--check", action="store_true", help="compare the current results with the golden snapshot")
    parser.add_argument("--suite", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--fresh", action="store_true", help="do not read or write alignment caches")
    parser.add_argument("--cache-root", default=str(ROOT / "outputs" / "word_regression_cache"),
                        help="alignment caches (one folder per suite); safe to share between runs")
    args = parser.parse_args()
    cache_dirs = {name: str(Path(args.cache_root) / name) for name in SUITES}
    current = collect(args.suite, cache_dirs, fresh=args.fresh)

    if args.freeze:
        golden = json.loads(SNAPSHOT.read_text(encoding="utf-8")) if SNAPSHOT.is_file() else {}
        golden.update(current)
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(golden, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
        print(f"froze {sum(len(v) for v in current.values())} syllables into {SNAPSHOT}")
        return 0

    golden = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    report = compare({k: golden[k] for k in args.suite if k in golden}, current)
    if report:
        print("WORD LEVEL CHANGED:\n" + describe(report))
        return 1
    print(f"word level unchanged: {sum(len(v) for v in current.values())} syllables identical in {', '.join(args.suite)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
