import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tone_assessment as ta  # noqa: E402

SOURCE = (ROOT / "main.py").read_text(encoding="utf-8")


class PhraseContourTests(unittest.TestCase):
    def test_the_contour_says_where_its_time_axis_starts(self):
        times = np.round(np.arange(0.0, 2.0, 0.005), 3)
        st = [None] * len(times)
        for i, t in enumerate(times):
            if 0.62 <= t < 1.40:                                   # voiced only in the middle of the phrase window
                st[i] = 1.0
        contour = {"times": times.tolist(), "st": st, "step_sec": 0.005}
        result = ta.phrase_f0(contour, 0.5, 1.6, 200.0)
        self.assertEqual(result["origin_sec"], 0.62)                  # offset 0 of the chart is the first voiced frame, in recording time
        self.assertEqual(result["contour"][0]["offset_sec"], 0.0)


class ExpectedToneLineTests(unittest.TestCase):
    def test_every_language_has_the_new_texts(self):
        for key in ("showExpectedTone", "expectedLine", "expectedNote"):
            self.assertEqual(len(re.findall(r"\b%s: " % key, SOURCE)), 5, key)

    def test_the_pitch_controls_have_a_switch_and_a_legend_entry(self):
        controls = SOURCE[SOURCE.index('<div class="pitch-controls">'): SOURCE.index('<div id="clausesContainer">')]
        self.assertIn('id="showExpectedTone"', controls)
        self.assertIn('onchange="toggleExpectedTone()"', controls)
        self.assertIn('class="pitch-swatch expected"', controls)

    def test_the_guide_is_off_until_the_learner_switches_it_on_and_comes_with_a_caveat(self):
        controls = SOURCE[SOURCE.index('<div class="pitch-controls">'): SOURCE.index('<div id="clausesContainer">')]
        self.assertNotRegex(controls, r'id="showExpectedTone"[^>]*checked')
        self.assertIn('localStorage.getItem("showExpectedTone") === "1"', SOURCE)
        self.assertIn('data-i18n="expectedNote"', controls)

    def test_the_four_tones_have_a_shape_and_the_neutral_tone_has_none(self):
        shapes = SOURCE[SOURCE.index("const EXPECTED_TONE_SHAPES = {"): SOURCE.index("function expectedToneSpans(")]
        for tone in ("1:", "2:", "3:", "4:"):
            self.assertIn(tone, shapes)
        self.assertNotIn("5:", shapes)

    def test_the_line_is_drawn_before_the_learners_line_so_it_stays_behind_it(self):
        chart = SOURCE[SOURCE.index("function renderPhrasePitchChart("): SOURCE.index("function refreshPitchPlots()")]
        self.assertLess(chart.index('class="expected-line"'), chart.index('class="learner-line"'))

    def test_the_choice_is_remembered_in_this_browser(self):
        self.assertIn('localStorage.setItem("showExpectedTone", on ? "1" : "0")', SOURCE)
        self.assertIn('document.getElementById("showExpectedTone").checked = showExpectedToneChoice();', SOURCE)


if __name__ == "__main__":
    unittest.main()
