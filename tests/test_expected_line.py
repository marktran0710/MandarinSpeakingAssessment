import json
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tone_assessment as ta  # noqa: E402

SOURCE = (ROOT / "main.py").read_text(encoding="utf-8")
NODE = shutil.which("node")


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
        for key in ("showExpectedTone", "expectedLine"):
            self.assertEqual(len(re.findall(r"\b%s: " % key, SOURCE)), 5, key)

    def test_the_pitch_controls_have_a_switch_and_a_legend_entry(self):
        controls = SOURCE[SOURCE.index('<div class="pitch-controls">'): SOURCE.index('<div id="resultExtras">')]
        self.assertIn('id="showExpectedTone"', controls)
        self.assertIn('onchange="toggleExpectedTone()"', controls)
        self.assertIn('class="pitch-swatch expected"', controls)

    def test_the_guide_is_on_until_the_learner_switches_it_off_and_has_no_explanation_under_the_chart(self):
        controls = SOURCE[SOURCE.index('<div class="pitch-controls">'): SOURCE.index('<div id="resultExtras">')]
        self.assertRegex(controls, r'id="showExpectedTone"[^>]*checked')
        self.assertIn('localStorage.getItem("showExpectedTone") !== "0"', SOURCE)
        for removed in ("expectedNote", "pitchNote", "pitch-note"):        # the two long notes were taken out on request (2026-10-19)
            self.assertNotIn(removed, SOURCE)

    def test_the_guide_is_built_from_the_learners_own_start_end_lowest_and_highest_points(self):
        guide = SOURCE[SOURCE.index("function shapeFromFourPoints("): SOURCE.index("function showExpectedToneChoice()")]
        for field in ("start_st", "end_st", "min_st", "max_st"):
            self.assertIn(field, guide)
        self.assertIn('syllable.status === "likely_correct"', guide)                  # right already: the learner's own size; otherwise at least a basic movement
        self.assertIn("function fallbackShape(", guide)                              # a syllable that could not be measured

    def test_the_four_tones_have_a_shape_and_the_neutral_tone_has_none(self):
        shapes = SOURCE[SOURCE.index("const EXPECTED_TONE_SHAPES = {"): SOURCE.index("const clampNumber")]
        for tone in ("1:", "2:", "3:", "4:"):
            self.assertIn(tone, shapes)
        self.assertNotIn("5:", shapes)

    def test_the_line_is_drawn_before_the_learners_line_so_it_stays_behind_it(self):
        chart = SOURCE[SOURCE.index("function renderPhrasePitchChart("): SOURCE.index("function refreshPitchPlots()")]
        self.assertLess(chart.index('class="expected-line"'), chart.index('class="learner-line"'))

    def test_the_choice_is_remembered_in_this_browser(self):
        self.assertIn('localStorage.setItem("showExpectedTone", on ? "1" : "0")', SOURCE)
        self.assertIn('document.getElementById("showExpectedTone").checked = showExpectedToneChoice();', SOURCE)


def shapes_for(cases):
    start = SOURCE.index("const clampNumber")
    helpers = SOURCE[start: SOURCE.index("\n", start) + 1]
    block_start = SOURCE.index("const MIN_GUIDE_MOVEMENT")
    block = SOURCE[block_start: SOURCE.index("// Without those four numbers", block_start)]
    script = f"{helpers}\n{block}\nconsole.log(JSON.stringify({json.dumps(cases)}.map(c => shapeFromFourPoints(c.tone, c.syllable))));"
    return json.loads(subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30, check=True).stdout)


def syllable(status, start, end, lowest, highest, **more):
    return {"status": status, "start_st": start, "end_st": end, "min_st": lowest, "max_st": highest, **more}


@unittest.skipUnless(NODE, "node is needed to run the page's JavaScript")
class FourPointShapeTests(unittest.TestCase):
    def test_a_good_tone_four_keeps_the_learners_own_range(self):
        shape, = shapes_for([{"tone": 4, "syllable": syllable("likely_correct", 6.0, 1.0, 0.8, 6.2)}])
        self.assertAlmostEqual(shape[0][1], 6.2, places=6)                         # from their highest to their lowest point
        self.assertAlmostEqual(shape[1][1], 0.8, places=6)

    def test_a_good_shallow_fall_is_not_blown_up(self):
        shape, = shapes_for([{"tone": 4, "syllable": syllable("likely_correct", 3.0, 2.5, 2.4, 3.1)}])
        self.assertAlmostEqual(shape[0][1] - shape[1][1], 0.7, places=6)           # 0.7 st: the guide agrees with the verdict

    def test_a_flat_tone_four_that_is_not_right_yet_gets_a_basic_fall_around_its_own_height(self):
        shape, = shapes_for([{"tone": 4, "syllable": syllable("insufficient_evidence", 2.0, 2.1, 1.9, 2.2)}])
        self.assertAlmostEqual(shape[0][1] - shape[1][1], 4.0, places=6)
        self.assertAlmostEqual((shape[0][1] + shape[1][1]) / 2, 2.05, places=6)

    def test_a_rising_contour_asked_to_fall_gets_a_fall_inside_its_own_range(self):
        shape, = shapes_for([{"tone": 4, "syllable": syllable("practice", 0.0, 6.0, 0.0, 6.0)}])
        self.assertEqual(shape, [[0, 6.0], [1, 0.0]])

    def test_tone_two_rises_tone_one_stays_level_tone_three_dips(self):
        two, one, three = shapes_for([
            {"tone": 2, "syllable": syllable("practice", 3.0, 3.1, 3.0, 3.1)},
            {"tone": 1, "syllable": syllable("practice", 8.0, 4.0, 4.0, 8.0)},
            {"tone": 3, "syllable": syllable("insufficient_evidence", 1.0, 1.0, 0.0, 2.0, min_position=0.4)}])
        self.assertLess(two[0][1], two[1][1])
        self.assertEqual(one, [[0, 6.0], [1, 6.0]])                                # level at the average of where it starts and ends
        self.assertEqual([point[0] for point in three], [0, 0.4, 1])
        self.assertLess(three[1][1], three[0][1])
        self.assertLess(three[1][1], three[2][1])

    def test_a_syllable_without_the_four_numbers_has_no_such_shape(self):
        shape, = shapes_for([{"tone": 4, "syllable": {"status": "insufficient_evidence", "start_st": None}}])
        self.assertIsNone(shape)


if __name__ == "__main__":
    unittest.main()
