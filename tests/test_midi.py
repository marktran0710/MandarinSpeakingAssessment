import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import tone_assessment as ta  # noqa: E402
import tone_pipeline as tp  # noqa: E402
from test_tone_pipeline import CONFIG, SR, tone_audio  # noqa: E402


class MidiFormulaTests(unittest.TestCase):
    def test_the_formula_is_69_plus_12_log2_of_f0_over_440(self):
        self.assertEqual(float(tp.hz_to_midi(440.0)), 69.0)
        self.assertAlmostEqual(float(tp.hz_to_midi(220.0)), 57.0)
        self.assertAlmostEqual(float(tp.hz_to_midi(880.0)), 81.0)
        self.assertAlmostEqual(float(tp.hz_to_midi(261.6256)), 60.0, places=3)                      # middle C
        np.testing.assert_allclose(tp.hz_to_midi([220.0, 440.0]), [57.0, 69.0])

    def test_semitones_above_the_speaker_median_are_the_difference_of_two_midi_numbers(self):
        rng = np.random.default_rng(7)
        f0 = rng.uniform(70, 500, 500)
        reference = float(rng.uniform(90, 300))
        np.testing.assert_allclose(tp.hz_to_st(f0, reference), tp.hz_to_midi(f0) - float(tp.hz_to_midi(reference)), atol=1e-12)
        np.testing.assert_allclose(tp.hz_to_st(f0, reference), 12.0 * np.log2(f0 / reference), atol=1e-9)  # the old expression

    def test_a_rise_or_fall_is_the_same_whatever_the_reference_so_every_threshold_is_unchanged(self):
        for reference in (90.0, 130.0, 440.0):
            rise = tp.hz_to_st(220.0, reference) - tp.hz_to_st(130.0, reference)
            self.assertAlmostEqual(float(rise), float(tp.hz_to_midi(220.0) - tp.hz_to_midi(130.0)), places=9)


class MidiOutputTests(unittest.TestCase):
    def run_utterance(self):
        audio, spans = tone_audio([[200, 200, 200, 200], [170, 165, 200, 235]])
        syllables = [{"char": c, "start": s, "end": e, "phones": [], "nucleus": (s + 0.04, e)} for c, (s, e) in zip("媽麻", spans)]
        calibration = {"status": "calibrated", "pitch_floor": 90.0, "pitch_ceiling": 450.0, "f0_median": 200.0}
        return tp.verify_utterance(audio, SR, "媽麻", calibration, alignment=syllables, config=CONFIG)

    def test_each_syllable_reports_its_four_points_in_midi_and_in_semitones_from_the_median(self):
        result = self.run_utterance()
        median_midi = float(tp.hz_to_midi(200.0))
        self.assertAlmostEqual(result["f0_extraction"]["speaker_f0_median_midi"], median_midi, places=1)
        for syllable in result["syllables"]:
            for point in ("start", "end", "min", "max"):
                self.assertAlmostEqual(syllable[f"{point}_midi"] - median_midi, syllable[f"{point}_st"], delta=0.02, msg=point)
        self.assertAlmostEqual(result["contour"]["baseline_midi"], median_midi, places=1)

    def test_the_chart_points_carry_midi_next_to_hz_and_relative_semitones(self):
        result = self.run_utterance()
        contour = result["contour"]
        chart = ta.phrase_f0({"times": contour["times"], "st": contour["st"], "step_sec": contour["step_sec"]}, 0.2, 0.6, 200.0)
        points = [p for p in chart["contour"] if p["midi"] is not None]
        self.assertTrue(points)
        for p in points:
            self.assertAlmostEqual(p["midi"], chart["pitch_reference_midi"] + p["relative_semitone"], delta=0.02)
            self.assertAlmostEqual(p["midi"], float(tp.hz_to_midi(p["hz"])), delta=0.1)           # the same pitch, three ways

    def test_the_page_offers_the_midi_view_in_every_language_and_the_chart_title_has_no_stray_question_mark(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn('<option value="midi" data-i18n="midiNote">MIDI note</option>', source)
        self.assertIn('const valueKey = useHz ? "hz" : useMidi ? "midi" : "relative_semitone";', source)
        for key in ("midiNote", "midiTitle", "midiLabel"):
            self.assertEqual(len(re.findall(r"\b%s:" % key, source)), 5, key)
        self.assertNotIn('} ? ${t("phrase")}', source)


if __name__ == "__main__":
    unittest.main()
