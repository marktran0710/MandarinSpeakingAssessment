import copy
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import tone_assessment as ta  # noqa: E402
import tone_pipeline as tp  # noqa: E402
from test_tone_pipeline import CONFIG, SR, tone_audio  # noqa: E402

CAL = {"status": "calibrated", "pitch_floor": 90, "pitch_ceiling": 450, "f0_median": 200}


def spans_of(audio_spans, count, chars="媽"):
    return [{"char": chars[0], "start": a, "end": b, "phones": [], "nucleus": None} for a, b in audio_spans[:count]]


def verify(syllables, audio, chars, config=CONFIG):
    return tp.verify_utterance(audio, SR, chars, CAL, alignment=syllables, config=config)


class LengthCheckTests(unittest.TestCase):
    def test_a_script_that_matches_what_was_said_is_not_flagged(self):
        audio, spans = tone_audio([[200] * 4] * 6, gap_sec=0.12)
        result = verify(spans_of(spans, 6), audio, "媽" * 6)
        self.assertIsNone(result["f0_extraction"]["length_check"])

    def test_extra_speech_after_the_last_syllable_means_the_speaker_said_more(self):
        audio, spans = tone_audio([[200] * 4] * 7, gap_sec=0.12)                 # seven spoken, the script has five
        result = verify(spans_of(spans, 5), audio, "媽" * 5)
        check = result["f0_extraction"]["length_check"]
        self.assertEqual(check["direction"], "longer")
        self.assertFalse(check["gate"])
        self.assertGreaterEqual(check["extra_syllables"], 1.5)

    def test_extra_speech_before_the_first_syllable_counts_too(self):
        audio, spans = tone_audio([[200] * 4] * 7, gap_sec=0.12)
        syllables = [{"char": "媽", "start": a, "end": b, "phones": [], "nucleus": None} for a, b in spans[2:]]
        check = verify(syllables, audio, "媽" * 5)["f0_extraction"]["length_check"]
        self.assertEqual(check["direction"], "longer")

    def test_script_words_that_had_to_be_squeezed_mean_the_speaker_said_less(self):
        audio, spans = tone_audio([[200] * 4] * 5, gap_sec=0.12)                  # five spoken, the script has seven
        end = spans[-1][1]
        squeezed = [(end + 0.02, end + 0.05), (end + 0.05, end + 0.08)]           # the two extra words fitted into 30 ms each
        syllables = [{"char": "媽", "start": a, "end": b, "phones": [], "nucleus": None} for a, b in spans + squeezed]
        check = verify(syllables, audio, "媽" * 7)["f0_extraction"]["length_check"]
        self.assertEqual(check["direction"], "shorter")
        self.assertFalse(check["gate"])
        self.assertEqual(check["squeezed_syllables"], 2)

    def test_three_squeezed_syllables_mean_the_alignment_is_not_trusted_at_all(self):
        audio, spans = tone_audio([[200] * 4] * 5, gap_sec=0.12)
        end = spans[-1][1]
        squeezed = [(end + 0.02 + 0.03 * i, end + 0.05 + 0.03 * i) for i in range(3)]
        syllables = [{"char": "媽", "start": a, "end": b, "phones": [], "nucleus": None} for a, b in spans + squeezed]
        result = verify(syllables, audio, "媽" * 8)
        self.assertTrue(result["f0_extraction"]["length_check"]["gate"])
        self.assertTrue(result["f0_extraction"]["speech_outside_alignment"])
        self.assertEqual({s["status"] for s in result["syllables"]}, {"insufficient_evidence"})
        self.assertTrue(all("bad_alignment" in s["reasons"] for s in result["syllables"]))

    def test_a_flagged_recording_is_judged_but_cannot_accuse_anyone(self):
        # five flat syllables read where tone 4 is expected (a clear mistake), with two more words spoken after them
        audio, spans = tone_audio([[200] * 4] * 7, gap_sec=0.12)
        flagged = verify(spans_of(spans, 5, "罵"), audio, "罵" * 5)
        self.assertEqual(flagged["f0_extraction"]["length_check"]["direction"], "longer")
        self.assertNotIn("practice", {s["status"] for s in flagged["syllables"]})
        self.assertTrue(all(s["tone_match"] == 0.0 for s in flagged["syllables"]))       # the shape was still measured
        audio, spans = tone_audio([[200] * 4] * 5, gap_sec=0.12)
        clean = verify(spans_of(spans, 5, "罵"), audio, "罵" * 5)
        self.assertIsNone(clean["f0_extraction"]["length_check"])
        self.assertEqual({s["status"] for s in clean["syllables"]}, {"practice"})        # the same mistake, said exactly as scripted, is flagged

    def test_short_texts_are_left_alone(self):
        audio, spans = tone_audio([[200] * 4] * 5, gap_sec=0.12)
        self.assertIsNone(verify(spans_of(spans, 2), audio, "媽媽")["f0_extraction"]["length_check"])

    def test_the_check_can_be_switched_off_in_the_config(self):
        config = copy.deepcopy(CONFIG)
        config["length_check"]["min_syllables"] = 999
        audio, spans = tone_audio([[200] * 4] * 7, gap_sec=0.12)
        self.assertIsNone(verify(spans_of(spans, 5), audio, "媽" * 5, config)["f0_extraction"]["length_check"])

    def test_production_settings_and_the_payload_field(self):
        cfg = ta.get_config()["length_check"]
        self.assertEqual((cfg["min_syllables"], cfg["longer_extra_syllables"], cfg["shorter_squeezed"], cfg["shorter_gate_squeezed"]), (4, 1.5, 2, 3))
        source = (ROOT / "tone_assessment.py").read_text(encoding="utf-8")
        self.assertEqual(source.count('payload["length_check"]'), 2)                      # sentence practice and the chapter-5 words


class StrictModePageTests(unittest.TestCase):
    CODES = ("en", "zh-TW", "vi", "id", "hi")

    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "main.py").read_text(encoding="utf-8")

    def test_the_difficulty_select_became_a_strict_mode_switch_with_a_visible_explanation(self):
        self.assertIn('<input type="checkbox" id="strictMode" onchange="saveLevelChoice()">', self.source)
        self.assertIn('id="levelHelp"', self.source)
        for gone in ("levelSelect", "levelAuto", "levelStandard", "levelEasy"):
            self.assertNotIn(gone, self.source, gone)

    def test_off_is_the_default_easy_and_on_asks_for_the_standard_level(self):
        self.assertIn('document.getElementById("strictMode").checked ? "standard" : ""', self.source)
        self.assertEqual(ta.resolve_level("", "foreigner"), "easy")                       # switch off: the request carries no level

    def test_every_language_names_the_switch_and_explains_both_positions(self):
        names = {"en": "Strict mode", "zh-TW": "嚴格模式", "vi": "Chế độ nghiêm ngặt", "id": "Mode ketat", "hi": "सख़्त मोड"}
        for code in self.CODES:
            chunk = re.search(r'\n {16}(?:%s|"%s"): \{(.*?)\n {16}(?:"?[\w-]+"?: \{|\};)' % (code, code), self.source, re.S).group(1)
            self.assertIn('levelLabel: "%s"' % names[code], chunk, code)
            hint = re.search(r'levelHint: "([^"]*)"', chunk).group(1)
            self.assertGreater(len(hint), 40, code)

    def test_the_length_messages_exist_in_every_language_and_the_results_show_them(self):
        for key in ("lengthLonger", "lengthShorter", "lengthDifferent"):
            self.assertEqual(len(re.findall(r"\b%s: " % key, self.source)), 5, key)
        self.assertIn('{ longer: "lengthLonger", shorter: "lengthShorter", different: "lengthDifferent" }[data.length_check?.direction]', self.source)


if __name__ == "__main__":
    unittest.main()
