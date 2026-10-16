import tempfile
import unittest
from pathlib import Path

import tone_pipeline as tp

import build_mfa_dictionary as bd

DICT = Path(__file__).resolve().parents[1] / "mfa" / "mandarin_taiwan_catalog.dict"
CATALOG_CHARS = "巧克淇淋裡末午咖啡桌哥沙廚臺漂游泳趣累休息踏朋方郵舒服線裙顏鞋梯汽雜椅弟俱姊"


class ToneLetterTests(unittest.TestCase):
    def test_swap_tone_only_touches_the_tone_bearing_phone(self):
        self.assertEqual(bd.swap_tone("aw˨˩˦", "˥˩"), "aw˥˩")
        self.assertEqual(bd.swap_tone("ɕ e˥˩ n", "˧˥"), "ɕ e˧˥ n")
        self.assertEqual(bd.tone_letters("tɕʰ aw˨˩˦"), "˨˩˦")
        self.assertEqual(bd.tone_letters("m"), "")

    def test_tone_letter_map_is_learned_from_the_entries(self):
        entries = {"媽": [(0.99, "m a˥")], "麻": [(0.99, "m a˧˥")], "馬": [(0.99, "m a˨˩˦")], "罵": [(0.99, "m a˥˩")]}
        self.assertEqual(bd.tone_letter_map(entries), {1: "˥", 2: "˧˥", 3: "˨˩˦", 4: "˥˩"})

    def test_a_missing_character_borrows_phones_from_a_same_syllable_entry(self):
        entries = {"媽": [(0.99, "m a˥")], "麻": [(0.99, "m a˧˥")], "馬": [(0.99, "m a˨˩˦")], "罵": [(0.99, "m a˥˩")]}
        by_syllable = {"ma": [(c, bd.syllable_and_tone(c)[1], p[0][1]) for c, p in entries.items()]}
        letters = bd.tone_letter_map(entries)
        self.assertEqual(bd.pronunciation_for("嗎", entries, by_syllable, letters), None)  # neutral tone has no letters
        self.assertEqual(bd.pronunciation_for("螞", entries, by_syllable, letters), "m a˨˩˦")   # ma3, same tone: borrowed


class DictionaryCacheNameTests(unittest.TestCase):
    """MFA caches compiled dictionaries by file name, so the name has to change when the content does."""

    def test_the_copy_handed_to_mfa_is_named_by_content(self):
        with tempfile.TemporaryDirectory() as work:
            first = Path(tp._dictionary_argument("mfa/mandarin_taiwan_catalog.dict", work))
            self.assertTrue(first.is_file())
            self.assertRegex(first.name, r"^mandarin_taiwan_catalog_[0-9a-f]{10}\.dict$")
            self.assertEqual(first.read_bytes(), (tp.CONFIG_PATH.parent / "mfa" / "mandarin_taiwan_catalog.dict").read_bytes())
            self.assertEqual(Path(tp._dictionary_argument("mfa/mandarin_taiwan_catalog.dict", work)).name, first.name)

    def test_the_fingerprint_follows_the_dictionary_content(self):
        import hashlib
        config = tp.load_config()
        expected = hashlib.sha1((tp.CONFIG_PATH.parent / "mfa" / "mandarin_taiwan_catalog.dict").read_bytes()).hexdigest()[:10]
        self.assertEqual(tp.dictionary_fingerprint(config), expected)
        named = {**config, "alignment": {**config["alignment"], "mfa_dictionary": "mandarin_taiwan_mfa"}}
        self.assertNotEqual(tp.dictionary_fingerprint(named), expected)      # an installed name hashes its name

    def test_an_installed_dictionary_name_is_passed_through(self):
        with tempfile.TemporaryDirectory() as work:
            self.assertEqual(tp._dictionary_argument("mandarin_taiwan_mfa", work), "mandarin_taiwan_mfa")


class GeneratedDictionaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries = bd.read_dictionary(DICT)

    def test_every_catalog_character_has_an_entry(self):
        self.assertEqual([c for c in CATALOG_CHARS if c not in self.entries], [])

    def test_added_entries_have_the_right_tone_letters(self):
        for char, expected in (("末", "˥˩"), ("裡", "˨˩˦"), ("淇", "˧˥"), ("咖", "˥"), ("桌", "˥")):
            self.assertEqual(bd.tone_letters(self.entries[char][0][1]), expected, char)

    def test_stock_entries_are_kept(self):
        self.assertIn("你", self.entries)
        self.assertGreater(len(self.entries), 15000)


if __name__ == "__main__":
    unittest.main()
