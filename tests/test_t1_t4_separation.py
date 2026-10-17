import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import tone_assessment as ta  # noqa: E402
import tone_pipeline as tp  # noqa: E402


def features(**changes):
    base = {"early_to_late_delta": 0.0, "range_core_ST": 1.0, "rise_after_min": 0.0, "min_position": 0.5, "fall_before_min": 0.0,
            "start_ST": 0.0, "end_ST": 0.0, "min_ST": -0.5, "max_ST": 0.5, "syllable_ms": 250.0}
    base.update(changes)
    return base


class AsymmetricToneOneTests(unittest.TestCase):
    def setUp(self):
        self.sentence = ta.get_config("sentence")
        self.words = ta.get_config()

    def t1(self, delta, config):
        return tp.tone_scores(features(early_to_late_delta=delta), False, config)[1]

    def test_in_sentences_a_fall_is_good_up_to_about_two_semitones_and_a_rise_up_to_two_point_six(self):
        self.assertGreaterEqual(self.t1(1.9, self.sentence), 0.7)
        self.assertLess(self.t1(2.4, self.sentence), 0.7)
        self.assertEqual(self.t1(2.8, self.sentence), 0.0)
        self.assertGreaterEqual(self.t1(-2.5, self.sentence), 0.7)                   # a rise keeps the old allowance
        self.assertLess(self.t1(-3.0, self.sentence), 0.7)

    def test_the_boundary_is_where_the_setting_says(self):
        cfg = self.sentence["tone"]
        good_until = cfg["t1_max_fall_st"] + 0.3 * cfg["t1_fall_width_st"]
        self.assertAlmostEqual(good_until, 2.0)
        self.assertAlmostEqual(self.t1(good_until, self.sentence), 0.7, places=6)

    def test_single_words_keep_the_symmetrical_allowance(self):
        self.assertNotIn("t1_max_fall_st", self.words["tone"])
        self.assertGreaterEqual(self.t1(2.5, self.words), 0.7)
        self.assertGreaterEqual(self.t1(-2.5, self.words), 0.7)

    def test_the_easy_level_keeps_the_rule(self):
        easy = ta.get_config("sentence", "easy")
        self.assertEqual((easy["tone"]["t1_max_fall_st"], easy["tone"]["t1_fall_width_st"]), (1.7, 1.0))
        self.assertLess(self.t1(2.4, easy), 0.7)

    def test_it_never_touches_a_real_tone_1_that_is_level_or_slightly_rising(self):
        for delta in (-1.8, -0.9, 0.0, 0.5, 1.2):
            self.assertEqual(self.t1(delta, self.sentence), 1.0, delta)


class SteadyShallowToneFourTests(unittest.TestCase):
    def setUp(self):
        self.sentence = ta.get_config("sentence")

    def t4(self, delta, steps, config=None, **more):
        f = features(early_to_late_delta=delta, start_ST=delta, end_ST=0.0, **more)
        if steps is not None:
            f["fall_steps_down"] = steps
        return tp.tone_scores(f, False, config or self.sentence)[4]

    def test_a_shallow_fall_must_be_steady(self):
        self.assertEqual(self.t4(1.9, 1.0), 1.0)                                    # steady: Good (1.9 st clears the 1.7 st bar of a 250 ms syllable)
        self.assertLess(self.t4(1.9, 0.4), 0.5)                                     # wobbly or flat: not a tone 4
        self.assertGreaterEqual(self.t4(1.9, 0.7), 0.7)                             # exactly the bar
        self.assertEqual(self.t4(1.9, 0.0), 0.0)

    def test_a_clear_fall_needs_no_steadiness(self):
        self.assertEqual(self.t4(3.0, 0.3), 1.0)
        self.assertEqual(self.t4(2.0, 0.3), 1.0)                                    # at below_st the rule stops

    def test_a_syllable_with_nothing_to_judge_the_steadiness_by_is_not_penalised(self):
        self.assertEqual(self.t4(1.9, None), self.t4(1.9, 1.0))

    def test_single_words_have_no_such_rule(self):
        words = ta.get_config()
        self.assertNotIn("t4_steady", words["tone"])
        self.assertGreater(self.t4(2.6, 0.2, words), 0.0)

    def test_the_feature_tells_a_steady_fall_from_a_wobbly_line(self):
        times = np.arange(0, 0.2, 0.005)
        steady = tp.syllable_features(times, np.linspace(1.5, 0.0, len(times)), 0.0, 0.2)
        rng = np.random.default_rng(3)
        wobbly = tp.syllable_features(times, 0.7 + rng.normal(0, 0.5, len(times)), 0.0, 0.2)
        self.assertEqual(steady["fall_steps_down"], 1.0)
        self.assertLess(wobbly["fall_steps_down"], 0.7)
        self.assertNotIn("fall_steps_down", tp.syllable_features(times[:6], np.linspace(1, 0, 6), 0.0, 0.03))      # too few frames

    def test_the_rules_are_verdict_only(self):
        for key in ("t1_max_fall_st", "t1_fall_width_st", "t4_steady"):
            self.assertIn(key, tp.VERDICT_ONLY_TONE_RULES)
            self.assertIn(key, self.sentence["tone"])


if __name__ == "__main__":
    unittest.main()
