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
        with cc.trial_config({"sentence": {"tone": {"t4_by_duration": {"short_ms": 180}}}}):
            self.assertEqual(ta.get_config("sentence")["tone"]["t4_by_duration"], {"short_ms": 180})
            self.assertTrue(ta.get_config("sentence")["tone"]["clause_final_half_t3"])      # the profile's own setting survives
            self.assertNotIn("t4_by_duration", ta.get_config()["tone"])
        self.assertIs(tp.load_config, before)                                                # restored
        self.assertNotIn("t4_by_duration", ta.get_config("sentence")["tone"])

    def test_a_base_trial_changes_both_profiles(self):
        with cc.trial_config({"tone": {"t4_min_fall_st": 2.5}, "confidence": {"likely_correct_match": 0.6}}):
            self.assertEqual(ta.get_config()["tone"]["t4_min_fall_st"], 2.5)
            self.assertEqual(ta.get_config("sentence")["tone"]["t4_min_fall_st"], 2.5)
            self.assertEqual(ta.get_config("sentence")["confidence"]["likely_correct_match"], 0.6)
            self.assertEqual(ta.get_config("sentence")["confidence"]["practice_min_window_ms"], 100)   # sentence value kept
        self.assertEqual(ta.get_config()["tone"]["t4_min_fall_st"], 3.0)

    def test_the_shipped_trial_files_load(self):
        import yaml
        root = Path(__file__).resolve().parents[1] / "trials"
        for name in ("user_config_a.yaml", "t4_by_duration_sentence.yaml", "t4_by_duration_proxy.yaml", "half_t3_only.yaml",
                     "t4_before_t4_sentence.yaml", "t4_before_t4_1_5_sentence.yaml", "t4_both_rules_sentence.yaml",
                     "rule7_final_tail_sentence.yaml", "rule7_final_tail_30_sentence.yaml", "rule9_carryover_sentence.yaml",
                     "rule9_carryover_strong_sentence.yaml", "rule8_short_130_sentence.yaml", "rule8_short_180_sentence.yaml",
                     "rules_combined_sentence.yaml", "rule10_reduplication_sentence.yaml",
                     "rules_recommended_sentence.yaml"):
            self.assertIsInstance(yaml.safe_load((root / name).read_text(encoding="utf-8")), dict, name)


if __name__ == "__main__":
    unittest.main()
