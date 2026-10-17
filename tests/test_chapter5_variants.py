"""Variants are separate candidate paths (never one long concatenated script); neutral tone is never `practice`."""
import unittest
from unittest import mock

import numpy as np

import audio_prep
import capt_core
import evaluate_chapter5_reference as ev
import tone_assessment as ta
import tone_pipeline as tp
from tests.test_tone_pipeline import CONFIG, SR, tone_audio

IDENTITY = lambda text: text  # noqa: E731
ITEM = {"word_key": "K", "item_type": "word", "traditional_chinese": "x", "pinyin": "", "section": "5-1"}


def scripts(item, requested):
    return ["".join(c["chars"]) for c in ta.chapter5_candidates(item, requested, IDENTITY)]


class VariantPathTests(unittest.TestCase):
    def test_slash_variants_are_separate_paths(self):
        item = {**ITEM, "script": "哪裡", "script_variants": ["哪裡", "哪兒"]}
        self.assertEqual(scripts(item, "哪裡/哪兒"), ["哪裡", "哪"])        # 兒 is erhua, not a syllable
        self.assertEqual(scripts({**ITEM, "script": "姐姐", "script_variants": ["姐姐", "姊姊"]}, "姐姐/姊姊"),
                         ["姐姐", "姊姊"])
        self.assertEqual(scripts({**ITEM, "script": "曬太陽", "script_variants": ["曬太陽", "晒太陽"]}, "曬太陽/晒太陽"),
                         ["曬太陽", "晒太陽"])

    def test_no_candidate_is_the_concatenation_of_two_variants(self):
        for requested, text in (("哪裡/哪兒", "哪裡哪兒"), ("姐姐/姊姊", "姐姐姊姊"), ("曬太陽/晒太陽", "曬太陽晒太陽")):
            item = {**ITEM, "script": requested.split("/")[0], "script_variants": requested.split("/")}
            self.assertNotIn(text, scripts(item, requested))
            self.assertTrue(all(len(s) <= max(len(p) for p in requested.split("/")) for s in scripts(item, requested)))

    def test_optional_text_gives_one_path_with_and_one_without(self):
        for requested, expected in (("裡(面)", ["裡", "裡面"]), ("外（面）", ["外", "外面"]), ("電視(機)", ["電視", "電視機"])):
            self.assertEqual(scripts({**ITEM, "script": requested}, requested), expected)

    def test_each_optional_path_is_one_clause_so_sandhi_is_not_cut(self):
        candidates = ta.chapter5_candidates({**ITEM, "script": "沒（有）空"}, "沒（有）空", IDENTITY)
        self.assertEqual({tuple(c["clauses"]) for c in candidates}, {("沒空",), ("沒有空",)})

    def test_evaluation_script_uses_the_same_paths(self):
        self.assertEqual(ev.candidate_scripts("哪裡/哪兒"), ["哪裡", "哪"])
        self.assertEqual(ev.candidate_scripts("姐姐/姊姊"), ["姐姐", "姊姊"])
        self.assertEqual(ev.candidate_scripts("裡(面)"), ["裡", "裡面"])
        self.assertEqual(ev.candidate_scripts("錢包"), ["錢包"])
        self.assertEqual(ev.candidate_scripts("曬太陽/晒太陽"), ["曬太陽", "晒太陽"])

    def test_workbook_tones_are_per_path(self):
        self.assertEqual(ev.workbook_tones("nǎlǐ/nǎr"), [[3, 3], [3]])
        self.assertEqual(ev.workbook_tones("méi(yǒu) kòng"), [[2, 4], [2, 3, 4]])
        self.assertEqual(ev.workbook_tones("jiějie"), [[3, 5]])

    def test_each_path_is_checked_against_the_workbook_separately(self):
        item = {"pinyin": "nǎlǐ/nǎr"}
        self.assertTrue(ev.script_check(item, "哪裡")["agrees"])
        self.assertTrue(ev.script_check(item, "哪兒")["agrees"])
        self.assertTrue(ev.script_check({"pinyin": "jiějie"}, "姐姐")["agrees"])
        self.assertTrue(ev.script_check({"pinyin": "jiějie"}, "姊姊")["agrees"])      # 姊姊 is read jiějie too
        self.assertFalse(ev.script_check({"pinyin": "nǎlǐ/nǎr"}, "哪裡哪兒")["agrees"])  # a joined path matches nothing

    def test_sister_variant_has_the_same_neutral_second_syllable(self):
        for text in ("姐姐", "姊姊"):
            self.assertEqual([c["target_tone"] for c in capt_core.expected_tone_contexts([text])], [3, 5])


class OnePathIsChosenTests(unittest.TestCase):
    """assess_chapter5_item must align and score each variant on its own and keep exactly one."""

    def run_item(self, item, requested):
        seen = []
        audio, _ = tone_audio([[200] * 4])
        prepared = audio_prep.prepare_samples(audio, SR, CONFIG)

        def fake_verify(prepared_audio, clauses, contexts, calibration, force_mfa=False, config=None):
            seen.append(["".join(clauses), force_mfa])
            return tp.verify_utterance(prepared_audio.samples, prepared_audio.sample_rate, "".join(clauses),
                                       calibration, contexts=contexts, alignment=[], config=CONFIG)

        calibration = {"status": "calibrated", "pitch_floor": 90.0, "pitch_ceiling": 450.0, "f0_median": 200.0}
        with mock.patch.object(ta, "_verify", side_effect=fake_verify):
            payload = ta.assess_chapter5_item(prepared, item=item, sample_id="x", target_script=requested,
                                              calibration=calibration, normalize=IDENTITY)
        return seen, payload

    def test_every_variant_is_scored_as_its_own_script(self):
        seen, payload = self.run_item({**ITEM, "script": "哪裡", "script_variants": ["哪裡", "哪兒"], "pinyin": "nǎlǐ/nǎr"},
                                      "哪裡/哪兒")
        self.assertEqual([s[0] for s in seen], ["哪裡", "哪"])
        self.assertTrue(all(s[1] for s in seen))                      # several variants -> all force-aligned
        self.assertIn(payload["aligned_script_variant"], ("哪裡", "哪兒"))
        self.assertEqual(len(payload["tone_verification"]["script_variants_tried"]), 2)
        self.assertEqual(len(payload["phrases"]), 1)                  # one answer, not one per variant

    def test_the_variant_that_explains_the_speech_wins(self):
        explains = {"alignment_error": None, "f0_extraction": {"voiced_outside_syllables": 0.0},
                    "syllables": [{"status": "likely_correct", "acoustic_quality": 0.9}]}
        leaves_speech = {"alignment_error": None, "f0_extraction": {"voiced_outside_syllables": 0.5},
                         "syllables": [{"status": "likely_correct", "acoustic_quality": 0.9}] * 2}
        self.assertGreater(ta._candidate_key(explains), ta._candidate_key(leaves_speech))


class ScoreTheReadingThatMatchesTests(unittest.TestCase):
    """A clip may hold several readings; only the one matching a variant is scored."""

    CALIBRATION = {"status": "calibrated", "pitch_floor": 90.0, "pitch_ceiling": 450.0, "f0_median": 200.0}
    CANDIDATES = [{"script_variant": "裡面", "clauses": ["裡面"], "chars": list("裡面")},
                  {"script_variant": "裡", "clauses": ["裡"], "chars": ["裡"]}]

    @staticmethod
    def result(outside, decided=1.0, quality=0.9, error=None):
        n = 4
        statuses = ["likely_correct"] * round(n * decided) + ["insufficient_evidence"] * (n - round(n * decided))
        return {"alignment_error": error, "f0_extraction": {"voiced_outside_syllables": outside},
                "syllables": [{"status": s, "acoustic_quality": quality} for s in statuses]}

    def prepared(self, gap):
        audio, _ = tone_audio([[200] * 4, [220] * 4], gap_sec=gap)
        return audio_prep.prepare_samples(audio, SR, CONFIG)

    def test_two_readings_are_scored_separately_and_one_is_kept(self):
        prepared, calls = self.prepared(0.25), []

        def verify(audio, candidate):
            calls.append((len(audio.samples), candidate["script_variant"]))
            whole = len(audio.samples) == len(prepared.samples)
            # The whole clip leaves a reading unexplained; each single reading is fully explained by a path.
            return self.result(0.5 if whole else 0.0)

        candidate, window, result, tried = ta.best_variant(prepared, self.CANDIDATES, verify, CONFIG)
        self.assertIsNotNone(window)                                        # a single reading was chosen
        self.assertEqual(result["f0_extraction"]["voiced_outside_syllables"], 0.0)
        self.assertEqual(len(calls), 6)                                     # whole + 2 readings, each x 2 variants
        self.assertEqual(len(tried), 6)
        self.assertEqual(len({c[0] for c in calls}), 2)

    def test_the_path_that_matches_the_reading_wins(self):
        prepared = self.prepared(0.25)

        def verify(audio, candidate):
            whole = len(audio.samples) == len(prepared.samples)
            if whole:
                return self.result(0.5)
            return self.result(0.0 if candidate["script_variant"] == "裡" else 0.3)

        candidate, window, _, _ = ta.best_variant(prepared, self.CANDIDATES, verify, CONFIG)
        self.assertEqual(candidate["script_variant"], "裡")
        self.assertIsNotNone(window)

    def test_one_reading_is_scored_whole_with_no_extra_work(self):
        prepared, calls = self.prepared(0.05), []
        ta.best_variant(prepared, self.CANDIDATES, lambda a, c: calls.append(c["script_variant"]) or self.result(0.0), CONFIG)
        self.assertEqual(calls, ["裡面", "裡"])                              # whole clip x each variant only

    def test_a_single_variant_is_never_split(self):
        prepared, calls = self.prepared(0.3), []
        candidate, window, _, _ = ta.best_variant(
            prepared, self.CANDIDATES[:1], lambda a, c: calls.append(len(a.samples)) or self.result(0.0), CONFIG)
        self.assertEqual(calls, [len(prepared.samples)])
        self.assertIsNone(window)

    def test_the_whole_clip_wins_when_it_is_one_reading_with_an_inner_pause(self):
        prepared = self.prepared(0.25)

        def verify(audio, candidate):
            whole = len(audio.samples) == len(prepared.samples)
            return self.result(0.0 if whole else 0.4)            # splitting would cut the word in two

        _, window, _, _ = ta.best_variant(prepared, self.CANDIDATES, verify, CONFIG)
        self.assertIsNone(window)

    def test_the_payload_says_which_reading_was_scored(self):
        prepared = self.prepared(0.25)
        calls = []

        def fake_verify(audio, clauses, contexts, calibration, force_mfa=False, config=None):
            calls.append(force_mfa)
            return tp.verify_utterance(audio.samples, audio.sample_rate, "".join(clauses), calibration,
                                       contexts=contexts, alignment=[], config=CONFIG)

        item = {**ITEM, "script": "裡(面)", "script_variants": ["裡", "裡面"], "pinyin": "lǐ(miàn)"}
        with mock.patch.object(ta, "_verify", side_effect=fake_verify):
            payload = ta.assess_chapter5_item(prepared, item=item, sample_id="x", target_script="裡(面)",
                                              calibration=self.CALIBRATION, normalize=IDENTITY)
        self.assertEqual(len(payload["phrases"]), 1)                       # one answer for the item
        self.assertTrue(all(calls))                                         # several variants -> all force-aligned
        readings = [t["reading"] for t in payload["tone_verification"]["script_variants_tried"]]
        self.assertIn(None, readings)
        self.assertTrue(any(r is not None for r in readings))


class NeutralToneNeverPracticeTests(unittest.TestCase):
    CAL = {"status": "calibrated", "pitch_floor": 90.0, "pitch_ceiling": 450.0, "f0_median": 200.0}
    # Shapes that look like T1, T2, T3, T4 and a wide excursion: none of them may produce `practice`.
    SHAPES = {
        "high level": [260, 260, 260, 260], "low level": [150, 150, 150, 150],
        "rising": [150, 190, 240, 290], "falling": [300, 250, 190, 140],
        "dipping": [220, 150, 140, 230], "wide": [320, 140, 330, 130],
    }
    WORDS = {"房子": [2, 5], "哥哥": [1, 5], "姐姐": [3, 5], "姊姊": [3, 5], "弟弟": [4, 5], "妹妹": [4, 5], "吧": [5]}

    def test_the_word_list_really_ends_in_a_neutral_syllable(self):
        for word, tones in self.WORDS.items():
            self.assertEqual([c["target_tone"] for c in capt_core.expected_tone_contexts([word])], tones, word)

    def neutral_results(self, word, shape):
        tones = self.WORDS[word]
        first = {1: [260] * 4, 2: [170, 165, 200, 235], 3: [190, 150, 140, 175], 4: [260, 230, 190, 150]}
        if len(word) == 1:
            audio, _ = tone_audio([self.SHAPES[shape]])
            with mock.patch.object(tp, "mfa_align", side_effect=AssertionError("MFA must not run")):
                return tp.verify_utterance(audio, SR, word, self.CAL, config=CONFIG)["syllables"]
        audio, spans = tone_audio([first[tones[0]], (self.SHAPES[shape], 0.25)])
        alignment = [{"char": c, "start": s, "end": e, "phones": [], "nucleus": (s + 0.03, e)}
                     for c, (s, e) in zip(word, spans)]
        return tp.verify_utterance(audio, SR, word, self.CAL, alignment=alignment, config=CONFIG)["syllables"]

    def test_neutral_syllables_are_only_likely_correct_or_insufficient(self):
        for word in self.WORDS:
            for shape in self.SHAPES:
                neutral = [s for s in self.neutral_results(word, shape) if s["expected_surface_tone"] == 5]
                self.assertEqual(len(neutral), 1, (word, shape))
                self.assertIn(neutral[0]["status"], ("likely_correct", "insufficient_evidence"), (word, shape))
                self.assertNotIn("tone_scores", neutral[0])             # never scored with the T1-T4 rules

    def test_a_neutral_syllable_does_not_turn_the_word_into_practice(self):
        for word in ("房子", "哥哥", "姐姐", "弟弟", "妹妹"):
            results = self.neutral_results(word, "wide")
            self.assertNotEqual(tp.aggregate_status(results), "practice", word)

    def test_the_policy_downgrades_any_neutral_practice_that_slips_through(self):
        results = [{"expected_surface_tone": 5, "status": "practice", "reasons": [], "feedback": "x"},
                   {"expected_surface_tone": 4, "status": "practice", "reasons": [], "feedback": "y"}]
        tp.enforce_neutral_policy(results, CONFIG)
        self.assertEqual(results[0]["status"], "insufficient_evidence")
        self.assertEqual(results[0]["reasons"], [])                      # silent: no reason shown to the user
        self.assertNotIn("neutral_never_practice", str(tp._public(results[0])))
        self.assertTrue(results[0]["_neutral_policy_downgrade"])         # private flag, removed from the output
        self.assertNotIn("_neutral_policy_downgrade", tp._public(results[0]))
        self.assertEqual(results[1]["status"], "practice")               # other tones are untouched

    def test_the_policy_can_be_switched_off_explicitly(self):
        results = [{"expected_surface_tone": 5, "status": "practice", "reasons": [], "feedback": "x"}]
        tp.enforce_neutral_policy(results, {**CONFIG, "neutral": {**CONFIG["neutral"], "allow_practice": True}})
        self.assertEqual(results[0]["status"], "practice")

    def test_the_shipped_config_forbids_neutral_practice(self):
        self.assertFalse(CONFIG["neutral"]["allow_practice"])


if __name__ == "__main__":
    unittest.main()
