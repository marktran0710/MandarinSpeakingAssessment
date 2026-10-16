import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import evaluate_sentences as es
import tone_pipeline as tp


def row(status, window=120, **extra):
    return {"status": status, "window_ms": window, "window_bucket": es.window_bucket(window), "expected_tone": 2,
            "lesson": "5-1", "speaker": "A", "reasons": extra.get("reasons", ""), **extra}


class CatalogTests(unittest.TestCase):
    def test_lines_without_audio_or_script_are_skipped_and_ids_follow_the_app(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / es.SCRIPT_CSV).write_text(
                "lesson,story,scene,character,script\n5,1,1,中明,你好嗎？\n5,1,2,友美,\n5,1,3,友美,我很好。\nx,1,1,a,b\n",
                encoding="utf-8-sig")
            (folder / "5-1-01.mp3").write_bytes(b"x")
            (folder / "5-1-03.mp3").write_bytes(b"x")
            catalog = es.load_catalog(folder)
        self.assertEqual([s["id"] for s in catalog], ["5-1-1", "5-1-3"])
        self.assertEqual(catalog[0]["speaker"], "中明")
        self.assertEqual(catalog[0]["lesson"], "5-1")

    def test_word_annotations_keep_lists_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "w.json"
            path.write_text(json.dumps({"5-1-1": ["你", "好", " "], "bad": "x"}, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(es.load_word_annotations(path), {"5-1-1": ["你", "好"]})


class WindowTests(unittest.TestCase):
    def test_buckets(self):
        self.assertEqual([es.window_bucket(w) for w in (30, 60, 61, 100, 101, 150, 151, None)],
                         ["<=60 ms", "<=60 ms", "61-100 ms", "61-100 ms", "101-150 ms", "101-150 ms", ">150 ms", "no window"])


class SummaryTests(unittest.TestCase):
    def test_counts_percentages_and_practice_by_window(self):
        syllables = [row("likely_correct"), row("likely_correct"), row("practice", window=50),
                     row("insufficient_evidence", reasons="ambiguous_contour")]
        sentences = [{"sample_id": "a", "practice": 1, "alignment_error": "", "audio_passed": True, "status": "practice"},
                     {"sample_id": "b", "practice": 0, "alignment_error": "", "audio_passed": True, "status": "likely_correct"}]
        report = es.summarize(syllables, [], sentences)
        self.assertEqual(report["syllables"]["n"], 4)
        self.assertEqual(report["syllables"]["practice_pct"], 25.0)
        self.assertEqual(report["sentences"]["with_any_practice"], 1)
        self.assertEqual(report["practice_by_window"], {"<=60 ms": 1})
        self.assertEqual(report["abstention_reasons"], {"ambiguous_contour": 1})
        self.assertIn("Native speech", es.render_markdown({"summary": report, "practice_syllables": []}))

    def test_an_empty_group_does_not_divide_by_zero(self):
        self.assertIsNone(es.status_counts([])["practice_pct"])


class AlignmentCacheTests(unittest.TestCase):
    def test_second_call_is_served_from_disk_and_errors_are_replayed(self):
        calls = []

        def fake(samples, sample_rate, chars, config=None):
            calls.append(chars)
            if chars == ["x"]:
                raise tp.AlignmentError("no alignment")
            return [{"tier": 1}]

        samples = [0.1, 0.2, 0.3]
        with tempfile.TemporaryDirectory() as cache, mock.patch.object(tp, "mfa_align", fake):
            with es.alignment_cache(cache):
                self.assertEqual(tp.mfa_align(samples, 16000, ["a", "b"]), [{"tier": 1}])
                self.assertEqual(tp.mfa_align(samples, 16000, ["a", "b"]), [{"tier": 1}])
                for _ in range(2):
                    with self.assertRaises(tp.AlignmentError):
                        tp.mfa_align(samples, 16000, ["x"])
            self.assertEqual(len(calls), 2)                 # one real call per distinct input
            self.assertIs(tp.mfa_align, fake)               # restored after the block


if __name__ == "__main__":
    unittest.main()
