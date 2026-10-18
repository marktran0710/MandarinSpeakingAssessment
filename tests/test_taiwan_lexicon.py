import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import capt_core as core  # noqa: E402
import tone_pipeline as tp  # noqa: E402
from test_tone_pipeline import CONFIG, SR, tone_audio  # noqa: E402

CAL = {"status": "calibrated", "pitch_floor": 90, "pitch_ceiling": 450, "f0_median": 200}


def lexicon_file(text):
    folder = tempfile.TemporaryDirectory()
    path = Path(folder.name) / "lexicon.yaml"
    path.write_text(text, encoding="utf-8")
    return folder, path


class LoadTests(unittest.TestCase):
    def test_the_shipped_lexicon_is_for_taiwan_and_loads(self):
        self.assertEqual(core.PRONUNCIATION_LOCALE, "zh-TW")
        self.assertGreater(len(core.LIGHT_FINAL_WORDS), 60)
        self.assertEqual(core.TAIWAN_WORD_TONES["休息"], [1, 2])
        self.assertEqual(core.TEXTBOOK_TONES["哥哥"], [1, 5])
        self.assertEqual(core.TAIWAN_WORD_TONES["丈夫"], [4, 1])
        self.assertEqual(set(core.LEXICON_VARIANTS), {"知道", "東西"})

    def test_a_tone_list_must_have_one_tone_1_to_5_per_character(self):
        for bad in ("tones:\n  休息: [1]\n", "tones:\n  休息: [1, 6]\n", "tones:\n  休息: 12\n"):
            folder, path = lexicon_file("locale: zh-TW\n" + bad)
            self.addCleanup(folder.cleanup)
            with self.assertRaises(ValueError, msg=bad):
                core.load_lexicon(path)

    def test_the_states_cannot_contradict_each_other(self):
        for bad in ("light_required: [看看]\nlight_optional: [看看]\n",
                    "light_required: [學生]\nfull_tone: [學生]\n",
                    "light_optional: [好]\n",
                    "light_required: 爸爸 媽媽\n",
                    "light_required: [爸爸, 媽 媽]\n",
                    "variants:\n  知道:\n    - {tones: [1, 4], before: 南}\n",
                    "variants:\n  知道:\n    - {tones: [1]}\n",
                    "variants:\n  知道:\n    - {tones: [1, 5]}\nfull_tone: [知道]\n"):
            folder, path = lexicon_file("locale: zh-TW\n" + bad)
            self.addCleanup(folder.cleanup)
            with self.assertRaises(ValueError, msg=bad):
                core.load_lexicon(path)


class ResolutionTests(unittest.TestCase):
    def test_each_syllable_knows_which_layer_gave_its_tone(self):
        tones, sources, states = core._resolve_tones("學生謝謝看看房子我們休息")
        self.assertEqual(tones, [2, 1, 4, 5, 4, 5, 2, 5, 3, 5, 1, 2])
        self.assertEqual(sources, ["generic_g2p", "generic_g2p", "generic_g2p", "taiwan_lexicon", "generic_g2p", "taiwan_lexicon",
                                   "textbook", "textbook", "generic_g2p", "neutral_rule", "taiwan_lexicon", "taiwan_lexicon"])      # a whole-word reading owns both syllables
        self.assertEqual(states, [None, None, None, "required", None, "optional", None, "required", None, None, None, None])

    def test_the_textbook_beats_the_lexicon_beats_the_light_lists_beats_generic(self):
        with mock.patch.object(core, "REQUIRED_LIGHT_WORDS", ("休息",)), mock.patch.object(core, "LEXICON_TONES", {"休息": [1, 2]}), \
                mock.patch.object(core, "TEXTBOOK_TONES", {}):
            tones, sources, _ = core._resolve_tones("休息")
            self.assertEqual((tones, sources[1]), ([1, 2], "taiwan_lexicon"))                 # the lexicon reading beats the light list
        with mock.patch.object(core, "LEXICON_TONES", {"休息": [1, 2]}), mock.patch.object(core, "TEXTBOOK_TONES", {"休息": [1, 1]}):
            tones, sources, _ = core._resolve_tones("休息")
            self.assertEqual((tones, sources), ([1, 1], ["textbook", "textbook"]))            # and the textbook beats the lexicon
        with mock.patch.object(core, "REQUIRED_LIGHT_WORDS", ()), mock.patch.object(core, "OPTIONAL_LIGHT_WORDS", ()), \
                mock.patch.object(core, "LEXICON_TONES", {}), mock.patch.object(core, "TEXTBOOK_TONES", {}):
            self.assertEqual(core._resolve_tones("休息")[1], ["generic_g2p", "generic_g2p"])

    def test_confidence_follows_the_source(self):
        contexts = core.expected_tone_contexts(["學生謝謝房子我們"])
        got = [(c["char"], c["tone_source"], c["tone_confidence"], c["neutral_state"]) for c in contexts]
        self.assertEqual(got, [("學", "generic_g2p", 0.7, None), ("生", "generic_g2p", 0.7, None),           # pypinyin alone: not trusted like the lexicon
                               ("謝", "generic_g2p", 0.7, None), ("謝", "taiwan_lexicon", 1.0, "required"),
                               ("房", "textbook", 1.0, None), ("子", "textbook", 1.0, "required"),
                               ("我", "generic_g2p", 0.7, None), ("們", "neutral_rule", 0.9, None)])

    def test_the_scored_tones_did_not_change(self):
        # the data moved into a file: the expected surface tones of everyday words are what they were
        for text, expected in (("認識", [4, 5]), ("謝謝", [4, 5]), ("學生", [2, 1]), ("喜歡", [3, 1]), ("休息", [1, 2]), ("頭髮", [2, 3]), ("哥哥", [1, 5]),
                               ("房子", [2, 5]), ("我們", [3, 5]), ("不錯", [2, 4]), ("你好", [2, 3]), ("漂亮", [4, 5]), ("晚上", [3, 5]), ("丈夫", [4, 1]), ("朋友", [2, 3])):
            self.assertEqual([c["target_tone"] for c in core.expected_tone_contexts([text])], expected, text)


class PayloadTests(unittest.TestCase):
    def test_the_result_says_where_each_expected_tone_came_from_and_for_which_locale(self):
        audio, spans = tone_audio([[200] * 4, [200] * 4])
        syllables = [{"char": c, "start": a, "end": b, "phones": [], "nucleus": None} for c, (a, b) in zip("爸爸", spans)]
        result = tp.verify_utterance(audio, SR, "爸爸", CAL, alignment=syllables, config=CONFIG)
        self.assertEqual(result["pronunciation_locale"], "zh-TW")
        first, second = result["syllables"]
        self.assertEqual((first["tone_source"], first["tone_confidence"], first["neutral_state"]), ("generic_g2p", 0.7, None))
        self.assertEqual((second["tone_source"], second["tone_confidence"], second["neutral_state"]), ("taiwan_lexicon", 1.0, "required"))


if __name__ == "__main__":
    unittest.main()
