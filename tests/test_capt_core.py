import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import capt_core as core


class TestStripErhua(unittest.TestCase):
    def test_drops_erhua_suffix(self):
        self.assertEqual(core.strip_erhua("我有一點兒累"), "我有一點累")
        self.assertEqual(core.strip_erhua("一會兒"), "一會")
        self.assertEqual(core.strip_erhua("玩兒"), "玩")

    def test_keeps_er_when_it_is_a_real_syllable(self):
        for word in ["兒子", "女兒", "兒童", "嬰兒"]:
            self.assertEqual(core.strip_erhua(word), word)


class TestPrepareTarget(unittest.TestCase):
    def test_splits_into_clauses_on_punctuation(self):
        self.assertEqual(
            core.prepare_target("你不要常常在家上網、看電腦，應該去運動。"),
            ["你不要常常在家上網", "看電腦", "應該去運動"],
        )

    def test_drops_erhua_and_non_chinese(self):
        self.assertEqual(core.prepare_target("我覺得有一點兒累，OK！"), ["我覺得有一點累"])


class TestExpectedTones(unittest.TestCase):
    def tones(self, *clauses):
        return core.expected_tones(list(clauses))

    def test_third_tone_sandhi_pair(self):
        self.assertEqual(self.tones("友美"), [2, 3])
        self.assertEqual(self.tones("你想去哪裡"), [2, 3, 4, 2, 3])

    def test_third_tone_run_of_three_becomes_2_2_3(self):
        self.assertEqual(self.tones("我很好"), [2, 2, 3])

    def test_sandhi_is_blocked_by_clause_break(self):
        self.assertEqual(self.tones("好", "我們"), [3, 3, 5])

    def test_no_sandhi_before_neutral_tone(self):
        self.assertEqual(self.tones("我們"), [3, 5])

    def test_bu_before_fourth_tone_is_second(self):
        self.assertEqual(self.tones("不錯"), [2, 4])
        self.assertEqual(self.tones("不是"), [2, 4])

    def test_bu_otherwise_stays_fourth(self):
        self.assertEqual(self.tones("不好"), [4, 3])

    def test_bu_in_verb_not_verb_is_neutral(self):
        self.assertEqual(self.tones("好不好"), [3, 5, 3])

    def test_yi_rules(self):
        self.assertEqual(self.tones("一家"), [4, 1])    # before 1st tone -> 4
        self.assertEqual(self.tones("一起"), [4, 3])    # before 3rd tone -> 4
        self.assertEqual(self.tones("一個"), [2, 4])    # before 4th tone -> 2
        self.assertEqual(self.tones("第一"), [4, 1])    # ordinal / clause-final -> 1

    def test_taiwan_readings_override_pypinyin(self):
        self.assertEqual(self.tones("星期六"), [1, 2, 4])

    def test_neutral_particles_come_from_pypinyin(self):
        self.assertEqual(self.tones("什麼"), [2, 5])

    def test_taiwan_neutral_second_syllables(self):
        self.assertEqual(self.tones("覺得"), [2, 5])
        self.assertEqual(self.tones("衣服"), [1, 5])

    def test_reduplicated_kinship_terms_have_a_neutral_second_syllable(self):
        # Keep neutral second syllables stable across pypinyin versions. Without
        # these readings, elder sister can get 3-3 sandhi and younger brother 4-4.
        self.assertEqual(self.tones("哥哥"), [1, 5])
        self.assertEqual(self.tones("姐姐"), [3, 5])
        self.assertEqual(self.tones("妹妹"), [4, 5])
        self.assertEqual(self.tones("弟弟"), [4, 5])
        self.assertTrue(core.expected_tone_contexts(["姐姐"])[0]["allow_half_t3"])

    def test_compound_function_words_keep_third_tone_before_a_following_third_tone(self):
        # 所以 / 哪裡 are followed by a prosodic break, so the last syllable does not sandhi into the next word
        self.assertEqual(self.tones("所以我"), [2, 3, 3])
        self.assertEqual(self.tones("哪裡有"), [2, 3, 3])
        self.assertEqual(self.tones("所以"), [2, 3])

    def test_context_records_lexical_and_surface_tones_and_third_tone_sandhi(self):
        contexts = core.expected_tone_contexts(["友美", "你好嗎"])
        self.assertEqual([item["target_tone"] for item in contexts], [2, 3, 2, 3, 5])
        self.assertEqual(contexts[0]["lexical_tone"], 3)
        self.assertEqual(contexts[0]["tone_rule"], "third_tone_sandhi")
        self.assertFalse(contexts[1]["allow_half_t3"])
        self.assertTrue(contexts[3]["allow_half_t3"])
        self.assertEqual(contexts[3]["next_tone"], 5)

    def test_context_records_bu_and_yi_sandhi_rules(self):
        bu = core.expected_tone_contexts(["不錯"])[0]
        yi = core.expected_tone_contexts(["一個"])[0]
        self.assertEqual((bu["lexical_tone"], bu["target_tone"], bu["tone_rule"]), (4, 2, "bu_sandhi"))
        self.assertEqual((yi["lexical_tone"], yi["target_tone"], yi["tone_rule"]), (1, 2, "yi_sandhi"))

    def test_explicit_phrase_boundaries_block_cross_phrase_third_tone_sandhi(self):
        contexts = core.expected_tone_contexts(["我很好"], phrase_end_indices=[1, 2, 3])
        self.assertEqual([item["target_tone"] for item in contexts], [3, 3, 3])
        self.assertIsNone(contexts[0]["next_tone"])
        self.assertTrue(contexts[0]["is_phrase_final"])

    def test_5_1_1_phrase_tone_sequences(self):
        contexts = core.expected_tone_contexts(
            ["友美", "妳這個週末要做什麼"],
            phrase_end_indices=[2, 3, 5, 7, 11],
        )
        phrase_ranges = [(0, 2), (2, 3), (3, 5), (5, 7), (7, 11)]
        sequences = [
            [0 if contexts[index]["target_tone"] == 5 else contexts[index]["target_tone"]
             for index in range(start, end)]
            for start, end in phrase_ranges
        ]
        self.assertEqual(sequences, [[2, 3], [3], [4, 4], [1, 4], [4, 4, 2, 0]])


class TestNormalizeAudio(unittest.TestCase):
    def test_peak_is_scaled_to_one_whatever_the_loudness(self):
        for gain in (0.05, 0.5, 3.0):
            out = core.normalize_audio(gain * np.array([0.0, 0.5, -1.0, 0.25]))
            self.assertAlmostEqual(float(np.max(np.abs(out))), 1.0, places=6)

    def test_silence_stays_silent(self):
        self.assertEqual(float(np.max(np.abs(core.normalize_audio(np.zeros(8))))), 0.0)


if __name__ == "__main__":
    unittest.main()
