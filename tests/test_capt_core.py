import os
import sys
import unittest

import numpy as np
from pypinyin.contrib.tone_convert import to_tone3

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


class TestCrossWordSandhi(unittest.TestCase):
    """Sentence practice: word ends from the annotation stop blocking sandhi. Everything else is as before."""

    def contexts(self, text, word_ends, cross=True):
        return core.expected_tone_contexts([text], phrase_end_indices=word_ends, cross_word_sandhi=cross)

    def test_the_default_is_unchanged_and_adds_no_new_fields(self):
        default = core.expected_tone_contexts(["你想去哪裡"], phrase_end_indices=[1, 2, 3, 5])
        self.assertEqual(default, self.contexts("你想去哪裡", [1, 2, 3, 5], cross=False))
        self.assertTrue(all("also_accept_tones" not in c for c in default))
        self.assertEqual(default[0]["target_tone"], 3)                 # blocked at the word end, as before

    def test_t3_before_t3_across_a_word_end_keeps_t3_and_also_accepts_t2(self):
        first = self.contexts("我也好", [1, 2, 3])[0]
        self.assertEqual((first["target_tone"], first["also_accept_tones"], first["tone_rule"]),
                         (3, (2,), "third_tone_sandhi_optional"))
        self.assertNotIn("also_accept_tones", self.contexts("我也好", [1, 2, 3])[2])   # the last syllable has no successor

    def test_bu_and_yi_take_their_sandhi_tone_across_a_word_end(self):
        self.assertEqual(self.contexts("我不要", [1, 2, 3], cross=False)[1]["target_tone"], 4)
        bu = self.contexts("我不要", [1, 2, 3])[1]
        self.assertEqual((bu["target_tone"], bu["tone_rule"]), (2, "bu_sandhi"))
        yi = self.contexts("一個", [1, 2])[0]
        self.assertEqual((yi["target_tone"], yi["tone_rule"]), (2, "yi_sandhi"))

    def test_the_previous_and_next_tones_follow_the_sandhi_tone(self):
        contexts = self.contexts("我不要", [1, 2, 3])
        self.assertEqual(contexts[2]["previous_tone"], None)           # 要 starts its own word: no neighbour carried over
        self.assertEqual(contexts[0]["next_tone"], None)

    def test_punctuation_and_the_fixed_phrase_breaks_still_block_sandhi(self):
        split = core.expected_tone_contexts(["我", "也好"], phrase_end_indices=[1, 2, 3], cross_word_sandhi=True)
        self.assertNotIn("also_accept_tones", split[0])               # a comma between the words
        fixed = self.contexts("哪裡很好", [2, 3, 4])[1]               # 哪裡 is a fixed phrase end
        self.assertNotIn("also_accept_tones", fixed)

    def test_without_word_ends_the_flag_changes_nothing(self):
        self.assertEqual(core.expected_tone_contexts(["我也好"], cross_word_sandhi=True),
                         core.expected_tone_contexts(["我也好"]))


class TestDisplayPinyin(unittest.TestCase):
    def test_music_is_yinyue_not_yinle(self):
        self.assertEqual(core.display_pinyin("音樂"), ["yīn", "yuè"])
        self.assertEqual(core.display_pinyin("我要在家看書、聽音樂。")[-3:], ["yīn", "yuè", "。"])
        self.assertEqual(core.display_pinyin("樂器"), ["yuè", "qì"])
        self.assertEqual(core.display_pinyin("快樂"), ["kuài", "lè"])             # the other reading is untouched

    def test_the_tones_shown_are_the_tones_the_app_scores(self):
        for text in ("音樂", "我覺得有一點累", "這個", "隻", "妳這個週末要做什麼", "女兒", "你好嗎"):
            numbered = [to_tone3(p) for p in core.display_pinyin(text)]                 # yuè -> yue4, ma -> ma
            digits = [int(p[-1]) if p[-1].isdigit() else 5 for p in numbered]
            self.assertEqual(digits, core.display_tones(text), text)
            self.assertEqual([d for d, ch in zip(digits, text) if ch not in "不一"],
                             [t for t, ch in zip(core._base_tones(text), text) if ch not in "不一"], text)    # every other syllable: the dictionary tone

    def test_the_taiwan_and_catalog_readings_are_kept(self):
        self.assertEqual(core.display_pinyin("這個"), ["zhè", "gè"])
        self.assertEqual(core.display_pinyin("隻"), ["zhī"])
        self.assertEqual(core.display_pinyin("你好嗎"), ["nǐ", "hǎo", "ma"])        # neutral tone has no mark
        self.assertEqual(core.display_pinyin("女"), ["nǚ"])

    def test_same_shape_as_lazy_pinyin_so_the_page_can_line_it_up_with_the_script(self):
        from pypinyin import lazy_pinyin, Style
        for text in ("友美，妳這個週末要做什麼？", "我要在家看書、聽音樂。", "媽，我跟友美要去喝下午茶。", "OK！好"):
            self.assertEqual(len(core.display_pinyin(text)), len(lazy_pinyin(text, style=Style.TONE)), text)

    def test_the_pages_use_it(self):
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py"), encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("_sample_pinyin = lru_cache(maxsize=1024)(core.display_pinyin)", source)               # worked out once per script
        self.assertIn('_sample_pinyin(sample["script"])', source)
        self.assertIn('tone_assessment.chapter5_character_pinyin(item)', source)                       # once per word (chapter5_display)
        self.assertNotIn("lazy_pinyin(sample", source)


class TestDisplayPinyinWritesBuAndYiAsSpoken(unittest.TestCase):
    """不錯 is read bú cuò and 一個 yí gè: the pinyin over the script must say so, not print the dictionary tone (bù cuò, yī gè)."""

    def test_bu_before_a_fourth_tone_is_second_tone(self):
        self.assertEqual(core.display_pinyin("不錯"), ["bú", "cuò"])
        self.assertEqual(core.display_pinyin("不要"), ["bú", "yào"])
        self.assertEqual(core.display_pinyin("我知道一家很不錯的咖啡廳")[6:8], ["bú", "cuò"])

    def test_bu_elsewhere_stays_fourth_tone_and_between_two_copies_of_a_verb_is_light(self):
        self.assertEqual(core.display_pinyin("不好"), ["bù", "hǎo"])
        self.assertEqual(core.display_pinyin("不"), ["bù"])
        self.assertEqual(core.display_pinyin("好不好"), ["hǎo", "bu", "hǎo"])

    def test_yi_takes_the_tone_it_is_spoken_with(self):
        self.assertEqual(core.display_pinyin("一個"), ["yí", "gè"])             # before a fourth tone: second
        self.assertEqual(core.display_pinyin("一起"), ["yì", "qǐ"])             # before the other tones: fourth
        self.assertEqual(core.display_pinyin("一杯"), ["yì", "bēi"])
        self.assertEqual(core.display_pinyin("一"), ["yī"])                     # alone, in a number or at the end of a word: first
        self.assertEqual(core.display_pinyin("第一"), ["dì", "yī"])
        self.assertEqual(core.display_pinyin("統一"), ["tǒng", "yī"])

    def test_a_third_tone_before_a_third_tone_is_not_rewritten(self):
        self.assertEqual(core.display_pinyin("你好"), ["nǐ", "hǎo"])            # pinyin does not mark that change

    def test_the_text_a_learner_types_is_written_the_same_way(self):
        self.assertEqual(core.pinyin_per_character("不錯，一個"), ["bú", "cuò", "", "yí", "gè"])

    def test_the_tone_printed_is_the_tone_scored_for_bu_and_yi(self):
        for text in ("不錯", "不要", "一個", "一起", "不好", "好不好", "第一"):
            printed = [int(to_tone3(p)[-1]) if to_tone3(p)[-1].isdigit() else 5 for p in core.display_pinyin(text)]
            scored = [item["target_tone"] for item in core.expected_tone_contexts([text])]
            self.assertEqual([p for p, ch in zip(printed, text) if ch in "不一"], [s for s, ch in zip(scored, text) if ch in "不一"], text)


class TestDisplayPinyinOverridesAndFixes(unittest.TestCase):
    def test_a_reading_can_be_supplied(self):
        self.assertEqual(core.display_pinyin("沒有空", tones=[2, 3, 4]), ["méi", "yǒu", "kòng"])
        self.assertEqual(core.display_pinyin("沒有空", tones=[2, 3]), core.display_pinyin("沒有空"))   # wrong length: ignored
        self.assertEqual(core.display_pinyin("好，空", tones=[3, 4]), ["hǎo", "，", "kòng"])           # non-Han passes through

    def test_sleep_is_shuijiao(self):
        self.assertEqual(core._base_tones("睡覺"), [4, 4])
        self.assertEqual(core.display_pinyin("睡覺"), ["shuì", "jiào"])
        self.assertEqual(core.display_pinyin("我覺得"), ["wǒ", "jué", "de"])                            # 覺得 is unchanged


class TestPinyinPerCharacter(unittest.TestCase):
    def test_one_entry_per_character_so_the_page_can_line_it_up(self):
        text = "我要，聽音樂。ok 好"
        entries = core.pinyin_per_character(text)
        self.assertEqual(len(entries), len(text))
        self.assertEqual(entries[:2], ["wǒ", "yào"])
        self.assertEqual(entries[3:6], ["tīng", "yīn", "yuè"])
        self.assertEqual([entries[2], entries[6], entries[7], entries[8], entries[9]], ["", "", "", "", ""])
        self.assertEqual(entries[-1], "hǎo")

    def test_simplified_keeps_the_length(self):
        self.assertEqual(core.simplified("音樂"), "音乐")
        self.assertEqual(len(core.simplified("我們這個")), 4)


if __name__ == "__main__":
    unittest.main()
