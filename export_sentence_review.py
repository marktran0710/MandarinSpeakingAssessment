"""
A page to double-check the sentences and words that still get `practice` (and what the sentence rules removed).

    python export_sentence_review.py --run outputs/sentences_after_ab --before outputs/sentences_baseline

Reads the files evaluate_sentences.py wrote. For every sentence with a `practice` syllable it shows the script, the
flagged syllable(s), an audio player for the teacher recording (it plays the flagged syllable by itself, with a little
context either side, or the whole sentence), the sentence F0 in semitones with the syllable boundaries, and the numbers.
A second section lists the syllables that were `practice` before and are not any more, with the rule that changed it.

The audio is not copied: the page links the recordings in place, so open it from this machine.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

CONTEXT_SEC = 0.25
W, H, PAD = 980, 150, 28


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def why_changed(before, after):
    """Short reason a syllable stopped being `practice` (compare the rows of the two runs)."""
    if after["status"] == "practice":
        return ""
    if after.get("also_accepted"):
        return f"T3 before T3 across a word end: T{after['also_accepted']} also accepted"
    if after["tone_rule"] != before.get("tone_rule", after["tone_rule"]) or after["expected_tone"] != before["expected_tone"]:
        return f"expected T{before['expected_tone']} -> T{after['expected_tone']} ({after['tone_rule']})"
    if "short_analysis_window" in after["reasons"]:
        return f"analysed window {after['window_ms']} ms is under the 100 ms minimum for an accusation"
    if after.get("clause_final_half_t3") in (True, "True"):
        return "clause-final T3 may be a low fall (half-third); the final rise is not required"
    return after["reasons"] or after["status"]


def contour_svg(record, syllables, total_sec):
    """Semitone contour with syllable bands; flagged syllables in red, abstained in grey."""
    if not record or not record["st"]:
        return "<p class=muted>no F0 contour</p>"
    values = record["st"]
    voiced = [v for v in values if v is not None]
    if not voiced:
        return "<p class=muted>no voiced frames</p>"
    lo, hi = min(voiced) - 1, max(voiced) + 1
    t0, step = record["t0"], record["step"]
    x = lambda t: PAD + (t - 0.0) / total_sec * (W - 2 * PAD)       # noqa: E731
    y = lambda v: H - PAD / 2 - (v - lo) / (hi - lo) * (H - PAD)    # noqa: E731
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img">']
    for s in syllables:
        a, b = int(s["start_ms"]) / 1000, int(s["end_ms"]) / 1000
        fill = {"practice": "#f8d7d7", "insufficient_evidence": "#ececec"}.get(s["status"], "none")
        parts.append(f'<rect x="{x(a):.1f}" y="4" width="{max(x(b) - x(a), 1):.1f}" height="{H - 8}" fill="{fill}" '
                     f'stroke="#bbb" stroke-width="0.5"/>')
        label = f'{html.escape(s["character"])} T{s["expected_tone"]}'
        parts.append(f'<text x="{(x(a) + x(b)) / 2:.1f}" y="{H - 2}" font-size="11" text-anchor="middle" fill="#555">{label}</text>')
        if s["window_start_ms"] not in (None, "") and s["status"] == "practice":
            wa, wb = int(s["window_start_ms"]) / 1000, int(s["window_end_ms"]) / 1000
            parts.append(f'<rect x="{x(wa):.1f}" y="4" width="{max(x(wb) - x(wa), 1):.1f}" height="6" fill="#c0392b"/>')
    segment = []

    def flush():
        if len(segment) > 1:
            parts.append('<polyline fill="none" stroke="#1f5fbf" stroke-width="1.6" points="' + " ".join(segment) + '"/>')
        segment.clear()

    for i, v in enumerate(values):
        if v is None:
            flush()
        else:
            segment.append(f"{x(t0 + i * step):.1f},{y(v):.1f}")
    flush()
    parts.append("</svg>")
    return "".join(parts)


def audio_tag(path, start, end, label):
    src = Path(path).resolve().as_uri() + f"#t={max(start, 0):.2f},{end:.2f}"
    return f'<div class=aud><span>{html.escape(label)}</span><audio controls preload="none" src="{src}"></audio></div>'


def sentence_card(sample_id, info, rows, contours):
    flagged = [r for r in rows if r["status"] == "practice"]
    total_sec = max(int(r["end_ms"]) for r in rows) / 1000 + 0.2
    players = [audio_tag(info["audio"], 0, total_sec, "whole sentence")]
    for r in flagged:
        a, b = int(r["start_ms"]) / 1000, int(r["end_ms"]) / 1000
        players.append(audio_tag(info["audio"], a - CONTEXT_SEC, b + CONTEXT_SEC, f"{r['character']} (±{CONTEXT_SEC:.2f}s)"))
    table = ["<table><tr><th>syllable<th>phrase<th>expected<th>prev/next<th>window<th>syllable<th>match<th>quality<th>notes"]
    for r in flagged:
        table.append(
            f"<tr class=flag><td><b>{html.escape(r['character'])}</b><td>{html.escape(r['phrase'])}<td>T{r['expected_tone']}"
            f"<td>{r['prev_tone'] or '-'} / {r['next_tone'] or '-'}<td>{r['window_ms']} ms<td>{r['syllable_ms']} ms"
            f"<td>{r['tone_match']}<td>{r['acoustic_quality']}<td>{html.escape(r['tone_rule'])}")
    table.append("</table>")
    marked = "".join(
        f'<span class="ch {r["status"]}">{html.escape(r["character"])}</span>' for r in rows)
    return (f'<section><h3>{sample_id} <small>{html.escape(info["speaker"])}</small></h3>'
            f'<p class=script>{marked}</p>{"".join(players)}{contour_svg(info["contour"], rows, total_sec)}{"".join(table)}</section>')


def changed_table(before_rows, after_rows):
    before = {(r["sample_id"], r["index"]): r for r in before_rows}
    out = ["<table><tr><th>sample<th>syllable<th>phrase<th>was<th>now<th>why"]
    count = 0
    for r in after_rows:
        b = before.get((r["sample_id"], r["index"]))
        if b and b["status"] == "practice" and r["status"] != "practice":
            count += 1
            out.append(f"<tr><td>{r['sample_id']}<td><b>{html.escape(r['character'])}</b><td>{html.escape(r['phrase'])}"
                       f"<td>practice (expected T{b['expected_tone']})<td>{r['status']}<td>{html.escape(why_changed(b, r))}")
    out.append("</table>")
    return count, "".join(out)


CSS = """
body{font:14px/1.5 system-ui,sans-serif;margin:0;padding:0 16px 40px;background:#fff;color:#1a1a1a}
@media (prefers-color-scheme:dark){body{background:#16181c;color:#e6e6e6}table td,table th{border-color:#444}
section{border-color:#444}.ch.likely_correct{color:#9ad39a}.muted{color:#999}}
h1{font-size:20px}h3{margin:0 0 4px}small{color:#888;font-weight:400}
section{border:1px solid #ddd;border-radius:8px;padding:12px 14px;margin:14px 0;max-width:1010px}
.script{font-size:24px;margin:2px 0 8px}.ch{padding:0 1px;border-bottom:3px solid transparent}
.ch.practice{background:#f8d7d7;color:#a21;border-bottom-color:#c0392b}.ch.insufficient_evidence{color:#888}
.aud{display:inline-block;margin:0 14px 6px 0}.aud span{display:block;font-size:12px;color:#777}
audio{height:30px;width:260px}table{border-collapse:collapse;margin-top:8px;font-size:13px}
td,th{border:1px solid #ddd;padding:3px 8px;text-align:left}.flag td{background:rgba(192,57,43,.07)}
.muted{color:#777}
"""


def render(run, before):
    run, before = Path(run), (Path(before) if before else None)
    rows = read_rows(run / "syllables.csv")
    contours = json.loads((run / "contours.json").read_text(encoding="utf-8"))
    by_sample = defaultdict(list)
    for r in rows:
        by_sample[r["sample_id"]].append(r)
    flagged_ids = [sid for sid, rs in by_sample.items() if any(r["status"] == "practice" for r in rs)]
    total_flags = sum(r["status"] == "practice" for r in rows)
    parts = [f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
             f"<title>Sentence review</title><style>{CSS}</style>",
             f"<h1>{total_flags} syllables still marked <i>practice</i> in {len(flagged_ids)} of {len(by_sample)} teacher sentences</h1>",
             "<p class=muted>The teacher recordings are native speech, so each flag below is something to check by ear: "
             "play the flagged syllable (±0.25 s of context) and the whole sentence. Red = practice, grey = not enough evidence.</p>"]
    for sid in sorted(flagged_ids, key=lambda k: [int(x) for x in k.split("-")]):
        parts.append(sentence_card(sid, contours[sid], by_sample[sid], contours))
    if before:
        count, table = changed_table(read_rows(before / "syllables.csv"), rows)
        parts.append(f"<h2>{count} syllables that were <i>practice</i> before the sentence rules and are not any more</h2>"
                     "<p class=muted>Listen to a few of these as well: the rules should only have removed false alarms.</p>" + table)
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--before", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = Path(args.out or Path(args.run) / "review.html")
    out.write_text(render(args.run, args.before), encoding="utf-8")
    print(out.resolve())


if __name__ == "__main__":
    main()
