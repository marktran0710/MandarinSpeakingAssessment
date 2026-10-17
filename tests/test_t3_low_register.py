import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import tone_assessment as ta  # noqa: E402
import tone_pipeline as tp  # noqa: E402


def features(**changes):
    base = {"early_to_late_delta": 0.0, "range_core_ST": 3.0, "rise_after_min": 0.0, "min_position": 0.2, "fall_before_min": 0.0,
            "start_ST": -4.0, "end_ST": -4.0, "min_ST": -4.5, "max_ST": -3.5, "syllable_ms": 400.0}
    base.update(changes)
    return base


class LowRegisterThirdToneTests(unittest.TestCase):
    def setUp(self):
        self.sentence = ta.get_config("sentence")
        self.words = ta.get_config()

    def test_a_low_rising_third_tone_with_no_fall_into_the_low_point_is_a_tone_3_in_sentences(self):
        f = features(start_ST=-4.5, min_ST=-4.6, max_ST=-0.8, end_ST=-1.6, rise_after_min=3.8)        # a learner's 我: starts low, rises 4 st
        self.assertGreaterEqual(tp.tone_scores(f, True, self.sentence)[3], 0.7)
        without = copy.deepcopy(self.sentence)
        del without["tone"]["t3_low_register"]
        self.assertEqual(tp.tone_scores(f, True, without)[3], 0.0)                                  # the old rules scored it 0

    def test_a_low_flat_syllable_is_a_tone_3_too(self):
        f = features(start_ST=-3.0, min_ST=-3.2, max_ST=-2.6, end_ST=-2.6)
        self.assertGreaterEqual(tp.tone_scores(f, True, self.sentence)[3], 0.7)

    def test_it_needs_a_deep_low_not_just_a_flat_one_near_the_median(self):
        f = features(start_ST=-0.5, min_ST=-0.6, max_ST=0.2, end_ST=0.0)
        self.assertLess(tp.tone_scores(f, True, self.sentence)[3], 0.2)

    def test_a_contour_that_gets_high_is_not_a_tone_3(self):
        f = features(start_ST=-3.0, min_ST=-3.2, max_ST=4.5, end_ST=4.0, rise_after_min=7.0)           # a rising tone 2 from a low start
        self.assertLess(tp.tone_scores(f, True, self.sentence)[3], 0.2)

    def test_a_high_fall_is_not_a_tone_3(self):
        f = features(start_ST=6.0, min_ST=-2.0, max_ST=6.0, end_ST=-2.0, early_to_late_delta=7.0)       # a tone 4
        self.assertLess(tp.tone_scores(f, True, self.sentence)[3], 0.2)

    def test_it_is_a_sentence_rule_and_follows_the_half_third_permission(self):
        f = features(start_ST=-4.5, min_ST=-4.6, max_ST=-0.8, end_ST=-1.6, rise_after_min=3.8)
        self.assertNotIn("t3_low_register", self.words["tone"])
        self.assertEqual(tp.tone_scores(f, True, self.words)[3], 0.0)                               # single words: not applied
        self.assertEqual(tp.tone_scores(f, False, self.sentence)[3], 0.0)                           # where a half third is not allowed: not applied

    def test_the_easy_level_keeps_it(self):
        self.assertIn("t3_low_register", ta.get_config("sentence", "easy")["tone"])

    def test_it_is_verdict_only_so_the_boundary_search_never_sees_it(self):
        self.assertIn("t3_low_register", tp.VERDICT_ONLY_TONE_RULES)


if __name__ == "__main__":
    unittest.main()
