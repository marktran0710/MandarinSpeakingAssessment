import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import compare_configs as cc  # noqa: E402
import tone_assessment as ta  # noqa: E402
import tone_pipeline as tp  # noqa: E402


class TrialConfigTests(unittest.TestCase):
    def test_a_nested_change_keeps_the_rest_of_the_section(self):
        merged = cc.deep_update({"sentence": {"tone": {"clause_final_half_t3": True}}}, {"sentence": {"tone": {"x": 1}}})
        self.assertEqual(merged, {"sentence": {"tone": {"clause_final_half_t3": True, "x": 1}}})

    def test_a_sentence_only_trial_reaches_the_sentence_profile_and_not_the_base(self):
        before = tp.load_config
        with cc.trial_config({"sentence": {"tone": {"t4_by_duration": {"short_ms": 123}}}}):
            rule = ta.get_config("sentence")["tone"]["t4_by_duration"]
            self.assertEqual(rule["short_ms"], 123)                                         # the trial's value ...
            self.assertEqual(rule["medium_ms"], 300)                                        # ... merged into the production rule
            self.assertTrue(ta.get_config("sentence")["tone"]["clause_final_half_t3"])      # the profile's own setting survives
            self.assertNotIn("t4_by_duration", ta.get_config()["tone"])
        self.assertIs(tp.load_config, before)                                                # restored
        self.assertEqual(ta.get_config("sentence")["tone"]["t4_by_duration"]["short_ms"], 180)

    def test_word_and_phrase_values_for_the_same_key_stay_separate(self):
        trial = {"tone": {"t1_max_delta_st": 2.5}, "confidence": {"practice_min_window_ms": 100},
                 "sentence": {"tone": {"t1_max_delta_st": 3.0}, "confidence": {"practice_min_window_ms": 130}}}
        with cc.trial_config(trial):
            self.assertEqual(ta.get_config()["tone"]["t1_max_delta_st"], 2.5)                  # single words
            self.assertEqual(ta.get_config("sentence")["tone"]["t1_max_delta_st"], 3.0)        # phrases keep their own value
            self.assertEqual(ta.get_config()["confidence"]["practice_min_window_ms"], 100)
            self.assertEqual(ta.get_config("sentence")["confidence"]["practice_min_window_ms"], 130)
        self.assertEqual(ta.get_config()["tone"]["t1_max_delta_st"], 2.0)

    def test_a_base_trial_changes_both_profiles(self):
        with cc.trial_config({"tone": {"t4_min_fall_st": 2.5}, "confidence": {"likely_correct_match": 0.6}}):
            self.assertEqual(ta.get_config()["tone"]["t4_min_fall_st"], 2.5)
            self.assertEqual(ta.get_config("sentence")["tone"]["t4_min_fall_st"], 2.5)
            self.assertEqual(ta.get_config("sentence")["confidence"]["likely_correct_match"], 0.6)
            self.assertEqual(ta.get_config("sentence")["confidence"]["practice_min_window_ms"], 130)   # sentence value kept
        self.assertEqual(ta.get_config()["tone"]["t4_min_fall_st"], 3.0)

    def test_the_shipped_trial_files_load(self):
        import yaml
        root = Path(__file__).resolve().parents[1] / "trials"
        for name in ("user_config_a.yaml", "t4_by_duration_sentence.yaml", "t4_by_duration_proxy.yaml", "half_t3_only.yaml",
                     "t4_before_t4_sentence.yaml", "t4_before_t4_1_5_sentence.yaml", "t4_both_rules_sentence.yaml",
                     "rule7_final_tail_sentence.yaml", "rule7_final_tail_30_sentence.yaml", "rule9_carryover_sentence.yaml",
                     "rule9_carryover_strong_sentence.yaml", "rule8_short_130_sentence.yaml", "rule8_short_180_sentence.yaml",
                     "rules_combined_sentence.yaml", "rule10_reduplication_sentence.yaml",
                     "rules_recommended_sentence.yaml", "final_config_as_pasted.yaml", "final_config_keep_decisions.yaml", "final_config_thresholds_only.yaml", "final_config_geomean_only.yaml", "user_ruleset_b.yaml", "half_t3_end_as_okay.yaml", "user_config_c.yaml", "user_config_c_floor130.yaml",
                     "left_context_t2_prev12_20_sentence.yaml", "left_context_t2_prev12_15_sentence.yaml", "left_context_t2_end3_20_sentence.yaml", "left_context_t2_either_20_sentence.yaml"):
            self.assertIsInstance(yaml.safe_load((root / name).read_text(encoding="utf-8")), dict, name)


if __name__ == "__main__":
    unittest.main()
