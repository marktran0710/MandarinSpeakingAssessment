import csv
import json
import tempfile
import unittest
from pathlib import Path

import export_sentence_review as er

COLUMNS = ["sample_id", "index", "character", "phrase", "expected_tone", "prev_tone", "next_tone", "status", "tone_match",
           "acoustic_quality", "syllable_ms", "window_ms", "start_ms", "end_ms", "window_start_ms", "window_end_ms",
           "also_accepted", "tone_rule", "reasons"]


def row(sample, index, char, status, start, end, **extra):
    base = {"sample_id": sample, "index": index, "character": char, "phrase": char, "expected_tone": "3", "prev_tone": "",
            "next_tone": "3", "status": status, "tone_match": "0.0", "acoustic_quality": "0.95", "syllable_ms": end - start,
            "window_ms": end - start, "start_ms": start, "end_ms": end, "window_start_ms": start, "window_end_ms": end,
            "also_accepted": "", "tone_rule": "lexical_tone", "reasons": ""}
    base.update(extra)
    return base


def write_run(folder, rows, contours):
    folder = Path(folder)
    with open(folder / "syllables.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (folder / "contours.json").write_text(json.dumps(contours, ensure_ascii=False), encoding="utf-8")


CONTOUR = {"t0": 0.0, "step": 0.01, "st": [None, 0.5, 1.0, 1.5, None, -1.0, -2.0, -3.0]}


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def test_only_sentences_with_a_practice_syllable_get_a_card_with_audio_and_a_contour(self):
        rows = [row("5-1-1", 0, "我", "practice", 0, 150), row("5-1-1", 1, "要", "likely_correct", 150, 300),
                row("5-1-2", 0, "好", "likely_correct", 0, 200)]
        contours = {sid: {"audio": str(self.root / f"{sid}.mp3"), "script": "x", "speaker": "A", "contour": CONTOUR}
                    for sid in ("5-1-1", "5-1-2")}
        write_run(self.root, rows, contours)
        page = er.render(self.root, None)
        self.assertIn("5-1-1", page)
        self.assertNotIn("<h3>5-1-2", page)
        self.assertIn("<audio", page)
        self.assertIn("#t=0.00,0.40", page)                         # the flagged syllable with context
        self.assertIn("<polyline", page)
        self.assertIn("1 syllables still marked", page)

    def test_the_before_section_lists_what_the_rules_removed_and_why(self):
        before_dir, after_dir = self.root / "before", self.root / "after"
        before_dir.mkdir(), after_dir.mkdir()
        contours = {"5-1-1": {"audio": str(self.root / "a.mp3"), "script": "x", "speaker": "A", "contour": CONTOUR}}
        write_run(before_dir, [row("5-1-1", 0, "我", "practice", 0, 150), row("5-1-1", 1, "要", "practice", 150, 400)], contours)
        write_run(after_dir, [row("5-1-1", 0, "我", "likely_correct", 0, 150, also_accepted="2"),
                              row("5-1-1", 1, "要", "practice", 150, 400)], contours)
        page = er.render(after_dir, before_dir)
        self.assertIn("1 syllables that were", page)
        self.assertIn("also accepted", page)

    def test_a_short_window_is_explained(self):
        before = row("a", 0, "我", "practice", 0, 60)
        after = row("a", 0, "我", "insufficient_evidence", 0, 60, reasons="short_analysis_window")
        self.assertIn("100 ms minimum", er.why_changed(before, after))

    def test_a_missing_contour_does_not_break_the_page(self):
        self.assertIn("no F0", er.contour_svg(None, [], 1.0))


if __name__ == "__main__":
    unittest.main()
