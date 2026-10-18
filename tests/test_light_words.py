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
        for text, expected in (("認識", ["rèn", "shi"]), ("朋友", ["péng", "you"]), ("謝謝", ["xiè", "xie"]), ("知道", ["zhī", "dao"]),
                               ("喜歡", ["xǐ", "huan"]), ("東西", ["dōng", "xi"]), ("學生", ["xué", "sheng"]), ("時候", ["shí", "hou"]),
                               ("看看", ["kàn", "kan"]), ("爸爸", ["bà", "ba"]), ("媽媽", ["mā", "ma"]), ("石頭", ["shí", "tou"]), ("記得", ["jì", "de"])):
            self.assertEqual(core.display_pinyin(text), expected, text)

    def test_it_works_inside_a_sentence_and_for_what_the_learner_types(self):
        self.assertEqual(core.display_pinyin("很開心認識你們"), ["hěn", "kāi", "xīn", "rèn", "shi", "nǐ", "men"])
        self.assertEqual(core.pinyin_per_character("我知道，謝謝"), ["wǒ", "zhī", "dao", "", "xiè", "xie"])

    def test_the_light_syllable_is_expected_as_a_light_tone_and_not_scored(self):
        contexts = core.expected_tone_contexts(["我很高興認識你"])
        by_char = {c["char"]: c for c in contexts}
        self.assertEqual((by_char["識"]["lexical_tone"], by_char["識"]["target_tone"]), (5, 5))
        self.assertEqual(by_char["認"]["target_tone"], 4)                           # the first syllable keeps its full tone

    def test_a_third_tone_before_a_light_syllable_is_not_a_sandhi_trigger(self):
        contexts = core.expected_tone_contexts(["我的朋友"])
        self.assertEqual([c["target_tone"] for c in contexts], [3, 5, 2, 5])
        self.assertEqual([c["target_tone"] for c in core.expected_tone_contexts(["你喜歡"])], [2, 3, 5])   # 你喜 is still a 3-3; the light 歡 is not scored

    def test_the_compass_directions_are_not_light(self):
        self.assertEqual(core.display_pinyin("東西南北"), ["dōng", "xī", "nán", "běi"])
        self.assertEqual(core.display_pinyin("買東西"), ["mǎi", "dōng", "xi"])

    def test_what_the_displayed_tones_say_is_what_is_scored(self):
        for text in ("我很高興認識你們", "朋友們都知道", "看看這個東西"):
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
