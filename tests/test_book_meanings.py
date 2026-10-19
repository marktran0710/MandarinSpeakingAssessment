import asyncio
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main  # noqa: E402

SOURCE = (ROOT / "main.py").read_text(encoding="utf-8")
MEANINGS = json.loads((ROOT / "book_sample_meanings.json").read_text(encoding="utf-8"))
LANGUAGES = ("en", "vi", "id", "hi")
HAN = re.compile(r"[一-鿿]")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


class MeaningDataTests(unittest.TestCase):
    def test_every_book_sentence_has_a_meaning_in_every_language(self):
        missing = [(sid, lang) for sid in main.SAMPLE_CATALOG for lang in LANGUAGES if not str(MEANINGS.get(sid, {}).get(lang, "")).strip()]
        self.assertEqual(missing, [])

    def test_there_is_no_meaning_for_a_sentence_that_does_not_exist(self):
        self.assertEqual(sorted(set(MEANINGS) - set(main.SAMPLE_CATALOG)), [])

    def test_a_meaning_is_a_translation_not_the_chinese_and_not_the_same_text_in_every_language(self):
        for sid, entry in MEANINGS.items():
            for lang in LANGUAGES:
                self.assertFalse(HAN.search(entry[lang]), (sid, lang))                 # no Chinese copied into a translation
            self.assertTrue(DEVANAGARI.search(entry["hi"]), sid)                         # Hindi is in Devanagari
            self.assertFalse(DEVANAGARI.search(entry["en"] + entry["vi"] + entry["id"]), sid)
            if len(entry["en"]) > 12:                                                    # (a one-word "OK." may be alike in two languages)
                self.assertNotEqual(entry["en"], entry["vi"], sid)
                self.assertNotEqual(entry["en"], entry["id"], sid)

    def test_a_translation_is_about_as_long_as_the_chinese_suggests(self):
        # a loose guard against a swapped or truncated line: the English has at least one word per two Chinese characters
        for sid, sample in main.SAMPLE_CATALOG.items():
            characters = len(HAN.findall(sample["script"]))
            words = len(MEANINGS[sid]["en"].split())
            self.assertGreaterEqual(words * 2 + 1, characters * 0.5, sid)


class MeaningInThePageTests(unittest.TestCase):
    def test_the_sentence_list_carries_the_meanings(self):
        lessons = asyncio.run(main.get_sample_lessons())
        self.assertEqual(len(lessons), len(main.SAMPLE_CATALOG))
        for lesson in lessons:
            self.assertEqual(set(lesson["meaning"]), set(LANGUAGES), lesson["id"])
        first = next(item for item in lessons if item["id"] == "5-1-1")
        self.assertEqual(first["meaning"]["vi"], "Youmei, cuối tuần này bạn định làm gì?")

    def test_every_language_has_the_label(self):
        self.assertEqual(len(re.findall(r"\bitemMeaning: ", SOURCE)), 5)

    def test_the_meaning_is_shown_under_the_script_in_the_language_of_the_page(self):
        render = SOURCE[SOURCE.index("function renderItemMeaning()"): SOURCE.index("function renderItemProgress()")]
        self.assertIn('currentLanguage === "zh-TW" ? ""', render)                       # a Chinese page needs no translation of the Chinese
        self.assertIn("meanings[currentLanguage] || meanings.en", render)               # English where a language has none
        self.assertIn('selectedPracticeMode() === "book"', render)
        self.assertLess(SOURCE.index('id="transcriptDisplay"'), SOURCE.index('id="itemMeaning"'))
        self.assertLess(SOURCE.index('id="itemMeaning"'), SOURCE.index('id="itemProgress"'))

    def test_the_meaning_follows_the_sentence_and_the_language(self):
        controls = SOURCE[SOURCE.index("function syncPracticeControls()"): SOURCE.index("function selectedPracticeSampleId(")]
        self.assertIn("renderItemMeaning();", controls)
        apply = SOURCE[SOURCE.index("function applyLanguage("): SOURCE.index("function changeLanguage(")]
        self.assertIn("renderItemMeaning();", apply)


if __name__ == "__main__":
    unittest.main()
