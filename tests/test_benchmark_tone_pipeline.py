import csv
import tempfile
import unittest
from pathlib import Path

import benchmark_tone_pipeline as bench


def row(status, wrong=False, **extra):
    base = {"mode": "adaptive", "file": "a.wav", "group": "synthetic_wrong", "speaker": "s", "profile": "mid",
            "item": "x", "variant": "clear", "index": 0, "character": "媽", "expected_surface_tone": 1,
            "truth_wrong": wrong, "status": status, "tone_match": 0.5, "tone_mismatch": 0.5,
            "acoustic_quality": 0.9, "voiced_ratio": 0.8, "ac_cc_agreement": 0.9, "detected_alternative": None,
            "pitch_floor": 90.0, "pitch_ceiling": 450.0, "reasons": ""}
    base.update(extra)
    return base


class MetricTests(unittest.TestCase):
    def test_syllable_metrics(self):
        rows = [
            row("practice", wrong=True), row("practice", wrong=True), row("likely_correct", wrong=True),
            row("insufficient_evidence", wrong=True),
            row("likely_correct"), row("likely_correct"), row("practice"), row("insufficient_evidence"),
        ]
        m = bench.syllable_metrics(rows)
        self.assertEqual(m["n_syllables"], 8)
        self.assertEqual(m["n_sufficient_evidence"], 6)
        self.assertEqual(m["abstention_rate"], 0.25)
        self.assertEqual(m["error_recall"], 0.5)                       # 2 of 4 wrong flagged
        self.assertEqual(m["wrong_missed_as_likely_correct"], 0.25)
        self.assertEqual(m["wrong_abstained"], 0.25)
        self.assertAlmostEqual(m["false_positive_rate_unchanged"], 0.25)   # 1 of 4 unchanged flagged
        self.assertAlmostEqual(m["precision_when_flagging"], 2 / 3, places=3)
        self.assertAlmostEqual(m["decision_accuracy_on_decided"], 4 / 6, places=3)  # 2 TP + 2 TN of 6 decided

    def test_empty_input_does_not_divide_by_zero(self):
        m = bench.syllable_metrics([])
        self.assertEqual(m["n_syllables"], 0)
        self.assertIsNone(m["error_recall"])

    def test_file_metrics_report_the_false_accept_rate(self):
        files = [{"wrong": True, "status": "practice"}] * 9 + [{"wrong": True, "status": "likely_correct"}] + [
            {"wrong": False, "status": "likely_correct"}] * 2
        m = bench.file_metrics(files)
        self.assertEqual(m["wrong_files"]["false_likely_correct"], 0.1)
        self.assertEqual(m["wrong_files"]["practice"], 0.9)
        self.assertEqual(m["unchanged_files"]["false_practice"], 0.0)

    def test_abstention_reasons_and_error_cases(self):
        rows = [row("insufficient_evidence", reasons="too_few_voiced_frames|low_acoustic_quality"),
                row("insufficient_evidence", reasons="too_few_voiced_frames"),
                row("practice", tone_mismatch=0.95), row("likely_correct", wrong=True, tone_match=0.9)]
        self.assertEqual(bench.abstention_reasons(rows), {"too_few_voiced_frames": 2, "low_acoustic_quality": 1})
        cases = bench.error_cases(rows)
        self.assertEqual(len(cases["false_positives"]), 1)
        self.assertEqual(len(cases["false_negatives"]), 1)

    def test_report_has_every_section_and_the_adaptive_vs_fixed_comparison(self):
        rows = [row("practice", wrong=True), row("likely_correct"),
                row("practice", wrong=True, mode="fixed"), row("insufficient_evidence", mode="fixed")]
        files = [{"mode": "adaptive", "group": "synthetic_wrong", "wrong": True, "status": "practice"},
                 {"mode": "fixed", "group": "synthetic_wrong", "wrong": True, "status": "practice"}]
        profiles = {"s": {"f0_median": 200, "f0_p95": 300, "pitch_floor": 80, "pitch_ceiling": 400,
                          "acoustic_profile": "mid"}}
        report = bench.build_report(rows, files, profiles, ["adaptive", "fixed"])
        self.assertIn("A_calibration_profiles", report)
        self.assertIn("E_adaptive_vs_fixed", report)
        adaptive = report["modes"]["adaptive"]
        for section in ("B_reference_positive_set", "C_synthetic_wrong_baseline", "D_stress_test",
                        "F_by_acoustic_profile", "G_by_expected_tone", "H_error_cases", "I_abstention"):
            self.assertIn(section, adaptive)
        self.assertEqual(report["E_adaptive_vs_fixed"]["fixed_range_hz"], [75.0, 500.0])
        self.assertIn("mid", adaptive["F_by_acoustic_profile"])
        self.assertIn("1", adaptive["G_by_expected_tone"])
        self.assertIn("# Tone pipeline v1 benchmark", bench.render_markdown(report))


class ManifestTests(unittest.TestCase):
    def test_csv_manifest_with_aliases_and_wrong_syllables(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.csv"
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["file", "text", "group", "voice", "wrong_syllables"])
                writer.writerow(["a.wav", "你好", "synthetic_wrong", "v1", "0;1"])
                writer.writerow(["b.wav", "謝謝", "reference", "v2", ""])
            rows = bench.load_manifest(path)
        self.assertEqual(rows[0]["script"], "你好")
        self.assertEqual(rows[0]["speaker"], "v1")
        self.assertEqual(rows[0]["wrong_syllables"], [0, 1])
        self.assertEqual(rows[1]["wrong_syllables"], [])
        self.assertTrue(rows[0]["path"].endswith("a.wav"))

    def test_manifest_row_without_script_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            path.write_text('[{"path": "a.wav"}]', encoding="utf-8")
            with self.assertRaises(ValueError):
                bench.load_manifest(path)


if __name__ == "__main__":
    unittest.main()
