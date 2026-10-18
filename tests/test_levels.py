import copy
import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import tone_assessment as ta  # noqa: E402
import tone_pipeline as tp  # noqa: E402


def falling(delta):
    return {"early_to_late_delta": delta, "range_core_ST": 3.0, "rise_after_min": 0.0, "min_position": 0.5, "fall_before_min": 0.0,
            "start_ST": delta, "end_ST": 0.0, "syllable_ms": 250.0}


class LevelConfigTests(unittest.TestCase):
    def test_the_standard_config_is_what_it_was(self):
        for level in (None, "standard"):
            cfg = ta.get_config(None, level)
            self.assertEqual((cfg["tone"]["t2_min_rise_st"], cfg["tone"]["t3_min_fall_st"], cfg["tone"]["t4_min_fall_st"]), (2.5, 2.0, 3.0))
            self.assertIs(ta.get_config(None, level), ta.get_config())
            self.assertIs(ta.get_config("sentence", level), ta.get_config("sentence"))

    def test_easy_asks_for_a_smaller_movement_in_tones_2_3_and_4_only(self):
        std, easy = ta.get_config(), ta.get_config(None, "easy")
        self.assertEqual((easy["tone"]["t2_min_rise_st"], easy["tone"]["t3_min_fall_st"], easy["tone"]["t4_min_fall_st"]), (2.0, 1.6, 1.8))
        for key in ("t1_max_delta_st", "t1_max_range_st", "t2_max_min_position", "t3_min_position", "t3_max_position"):
            self.assertEqual(easy["tone"][key], std["tone"][key], key)
        self.assertEqual(easy["confidence"], std["confidence"])               # the decision bars are not part of a level

    def test_easy_in_sentence_practice_keeps_the_sentence_profile_and_relaxes_its_rules(self):
        std, easy = ta.get_config("sentence"), ta.get_config("sentence", "easy")
        self.assertTrue(easy["tone"]["clause_final_half_t3"])
        self.assertEqual(easy["confidence"]["practice_min_window_ms"], 130)
        self.assertEqual(easy["confidence"]["unstable_scope"], "syllable")
        self.assertEqual(std["tone"]["t4_by_duration"]["min_fall_short_st"], 1.2)
        self.assertEqual(easy["tone"]["t4_by_duration"]["min_fall_short_st"], 0.5)
        self.assertEqual((std["tone"]["t4_by_duration"]["tiny_ms"], std["tone"]["t4_by_duration"]["min_fall_tiny_st"]), (140, 0.9))
        self.assertEqual((easy["tone"]["t4_by_duration"]["tiny_ms"], easy["tone"]["t4_by_duration"]["min_fall_tiny_st"]), (140, 0.5))
        self.assertEqual({easy["tone"]["t4_by_duration"][k] for k in ("min_fall_tiny_st", "min_fall_short_st", "min_fall_medium_st", "min_fall_long_st")}, {0.5})
        self.assertTrue(easy["tone"]["t4_by_duration"]["require_end_below_start"])
        self.assertEqual(easy["tone"]["t4_before_t4"], {"relaxation_st": 0.5, "floor_st": 0.8, "next_tones": [1, 4]})
        self.assertEqual([rule["set"]["t2_min_rise_st"] for rule in easy["tone"]["left_context"]], [1.6, 1.6])
        self.assertEqual([rule["set"]["t2_min_rise_st"] for rule in std["tone"]["left_context"]], [2.0, 2.0])

    def test_a_level_never_changes_the_configs_of_the_other_levels(self):
        before = ta.get_config("sentence")["tone"]["t4_by_duration"]["min_fall_short_st"]
        ta.get_config("sentence", "easy")
        ta.get_config(None, "easy")
        self.assertEqual(ta.get_config("sentence")["tone"]["t4_by_duration"]["min_fall_short_st"], before)
        self.assertEqual(ta.get_config()["tone"]["t4_min_fall_st"], 3.0)

    def test_an_unknown_level_is_an_error(self):
        with self.assertRaises(ValueError):
            ta.get_config(None, "impossible")
        with self.assertRaises(ValueError):
            ta.resolve_level("impossible", "foreigner")


class LevelChoiceTests(unittest.TestCase):
    def test_automatic_means_easy_for_everybody(self):
        for group in ("foreigner", "taiwanese", None):
            self.assertEqual(ta.resolve_level("", group), "easy", group)
            self.assertEqual(ta.resolve_level(None, group), "easy", group)

    def test_a_group_can_still_get_its_own_default_from_the_config(self):
        config = copy.deepcopy(ta.get_config())
        config["levels"]["default_by_group"] = {"taiwanese": "standard"}
        self.assertEqual(ta.resolve_level("", "taiwanese", config), "standard")
        self.assertEqual(ta.resolve_level("", "foreigner", config), "easy")
        config["levels"]["default_by_group"] = {"taiwanese": "no_such_level"}
        self.assertEqual(ta.resolve_level("", "taiwanese", config), "standard")             # an unknown name falls back to the safe level

    def test_a_request_can_name_a_level(self):
        self.assertEqual(ta.resolve_level("standard", "foreigner"), "standard")
        self.assertEqual(ta.resolve_level(" Easy ", "taiwanese"), "easy")

    def test_the_assess_endpoint_takes_a_level(self):
        import main
        self.assertIn("level", inspect.signature(main.assess_pronunciation).parameters)
        self.assertIn("level", inspect.signature(ta.assess_recording).parameters)
        self.assertIn("level", inspect.signature(ta.assess_chapter5_item).parameters)


class LevelEffectTests(unittest.TestCase):
    def test_a_modest_tone_four_fall_is_partial_in_standard_and_good_in_easy(self):
        standard, easy = ta.get_config(), ta.get_config(None, "easy")
        f = falling(2.2)
        self.assertLess(tp.tone_scores(f, False, standard)[4], 0.70)
        self.assertGreaterEqual(tp.tone_scores(f, False, easy)[4], 0.70)

    def test_a_flat_syllable_is_still_not_a_tone_four_in_easy(self):
        easy = ta.get_config(None, "easy")
        self.assertLessEqual(tp.tone_scores(falling(0.5), False, easy)[4], 0.20)


if __name__ == "__main__":
    unittest.main()
