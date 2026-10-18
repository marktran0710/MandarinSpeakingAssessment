import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import capt_core as core  # noqa: E402
import tone_assessment as ta  # noqa: E402
from pypinyin.contrib.tone_convert import to_tone3  # noqa: E402


class LightWordsTests(unittest.TestCase):
    """認識 is read rèn‧shi, not rèn shí: everyday words with a light last syllable are printed and treated as such."""

    def test_the_page_prints_the_light_syllable_without_a_tone_mark(self):
        for text, expected in (("認識", ["rèn", "shi"]), ("謝謝", ["xiè", "xie"]), ("意思", ["yì", "si"]), ("看看", ["kàn", "kan"]), ("爸爸", ["bà", "ba"]),
                               ("媽媽", ["mā", "ma"]), ("麻煩", ["má", "fan"]), ("明白", ["míng", "bai"]), ("石頭", ["shí", "tou"]), ("記得", ["jì", "de"])):
            self.assertEqual(core.display_pinyin(text), expected, text)

    def test_it_works_inside_a_sentence_and_for_what_the_learner_types(self):
        self.assertEqual(core.display_pinyin("很開心認識你們"), ["hěn", "kāi", "xīn", "rèn", "shi", "nǐ", "men"])
        self.assertEqual(core.pinyin_per_character("我認識你，謝謝"), ["wǒ", "rèn", "shi", "nǐ", "", "xiè", "xie"])

    def test_the_light_syllable_is_expected_as_a_light_tone_and_not_scored(self):
        contexts = core.expected_tone_contexts(["我很高興認識你"])
        by_char = {c["char"]: c for c in contexts}
        self.assertEqual((by_char["識"]["lexical_tone"], by_char["識"]["target_tone"]), (5, 5))
        self.assertEqual(by_char["認"]["target_tone"], 4)                           # the first syllable keeps its full tone

    def test_a_third_tone_before_a_light_syllable_is_not_a_sandhi_trigger(self):
        contexts = core.expected_tone_contexts(["我的爸爸"])
        self.assertEqual([c["target_tone"] for c in contexts], [3, 5, 4, 5])
        self.assertEqual([c["target_tone"] for c in core.expected_tone_contexts(["我的爸爸們"])], [3, 5, 4, 5, 5])

    def test_words_that_are_full_tone_stay_full(self):
        # the textbook teachers read these in full (measured), and the Ministry of Education dictionary gives a full last syllable for the others
        shown = {"喜歡": ["xǐ", "huān"], "知道": ["zhī", "dào"], "漂亮": ["piào", "liàng"], "朋友": ["péng", "yǒu"], "東西": ["dōng", "xī"], "晚上": ["wǎn", "shàng"],
                 "時候": ["shí", "hòu"], "地方": ["dì", "fāng"], "舒服": ["shū", "fú"], "學生": ["xué", "shēng"], "先生": ["xiān", "shēng"], "休息": ["xiū", "xí"],
                 "清楚": ["qīng", "chǔ"], "寶寶": ["bǎo", "bǎo"], "師傅": ["shī", "fù"], "親戚": ["qīn", "qī"], "眼睛": ["yǎn", "jīng"], "頭髮": ["tóu", "fǎ"],
                 "打算": ["dǎ", "suàn"], "行李": ["xíng", "lǐ"], "風箏": ["fēng", "zhēng"], "小姐": ["xiǎo", "jiě"]}
        self.assertEqual(set(shown), set(core.FULL_TONE_WORDS))
        for word, pinyin in shown.items():
            self.assertEqual(core.display_pinyin(word), pinyin, word)
            self.assertNotIn(word, core.LIGHT_FINAL_WORDS, word)

    def test_the_two_light_states_are_listed_apart_and_both_are_light_and_not_scored(self):
        self.assertFalse(set(core.REQUIRED_LIGHT_WORDS) & set(core.OPTIONAL_LIGHT_WORDS))
        self.assertEqual(core.LIGHT_FINAL_WORDS, core.REQUIRED_LIGHT_WORDS + core.OPTIONAL_LIGHT_WORDS)
        for word in ("爸爸", "謝謝", "意思", "看看", "葡萄", "兄弟", "生意"):
            self.assertIn(word, core.LIGHT_FINAL_WORDS, word)
            contexts = core.expected_tone_contexts([word])
            self.assertEqual(contexts[-1]["target_tone"], 5, word)                    # not scored: a learner who keeps the full tone is never marked wrong
        for word in ("看看", "葡萄", "兄弟", "姑娘", "便宜", "生意", "客氣"):
            self.assertIn(word, core.OPTIONAL_LIGHT_WORDS, word)
        for word in ("爸爸", "謝謝", "意思", "麻煩", "明白", "耳朵", "脾氣", "暖和", "玻璃", "豆腐", "腦袋", "屁股", "早上", "蘑菇"):
            self.assertIn(word, core.REQUIRED_LIGHT_WORDS, word)

    def test_what_the_displayed_tones_say_is_what_is_scored(self):
        for text in ("我很高興認識你們", "爸爸們都認得", "看看這個"):
            printed = [int(to_tone3(p)[-1]) if to_tone3(p)[-1].isdigit() else 5 for p in core.display_pinyin(text)]
            self.assertEqual(printed, core.display_tones(text), text)

    def test_words_the_textbook_prints_with_a_full_tone_are_not_in_the_list(self):
        for word in ("窗戶", "家具", "客廳", "咖啡廳", "房間", "旁邊", "上面", "下面", "裡面", "外面", "前面", "後面", "太陽", "可是", "沙發", "廚房", "電視"):
            self.assertNotIn(word, core.LIGHT_FINAL_WORDS, word)
        self.assertTrue(all(len(word) >= 2 for word in core.LIGHT_FINAL_WORDS))

    def test_every_chapter_5_word_still_agrees_with_the_textbooks_own_pinyin(self):
        sys.argv = sys.argv[:1]
        import evaluate_chapter5_reference as ev
        items = json.loads((ROOT / "chapter5_word_practice" / "items.json").read_text(encoding="utf-8"))["items"]
        checked = 0
        for item in items:
            if not item.get("pinyin"):
                continue
            shown = [p for p in ta.chapter5_character_pinyin(item) if p and p[0].isalpha()]
            digits = [int(to_tone3(p)[-1]) if to_tone3(p)[-1].isdigit() else 5 for p in shown]
            self.assertIn(digits, ev.workbook_tones(item["pinyin"]), (item["script"], item["pinyin"], shown))
            checked += 1
        self.assertGreater(checked, 40)


if __name__ == "__main__":
    unittest.main()
