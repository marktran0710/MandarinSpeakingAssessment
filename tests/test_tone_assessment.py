import json
import unittest
from unittest import mock

import numpy as np

import capt_core
import tone_assessment as ta
import tone_pipeline as tp
from tests.test_tone_pipeline import CONFIG, SR, tone_audio

IDENTITY = lambda text: text  # noqa: E731


def syllable(status, tone=1):
    return {"status": status, "expected_surface_tone": tone}


class AggregateStatusTests(unittest.TestCase):
    def test_any_practice_wins(self):
        self.assertEqual(tp.aggregate_status([syllable("likely_correct"), syllable("practice")]), "practice")

    def test_all_scored_syllables_must_be_accepted(self):
        self.assertEqual(tp.aggregate_status([syllable("likely_correct")] * 2), "likely_correct")
        self.assertEqual(tp.aggregate_status(
            [syllable("likely_correct"), syllable("insufficient_evidence")]), "insufficient_evidence")

    def test_an_unaccepted_neutral_syllable_does_not_block_the_group(self):
        self.assertEqual(tp.aggregate_status(
            [syllable("likely_correct"), syllable("insufficient_evidence", tone=5)]), "likely_correct")
        self.assertEqual(tp.aggregate_status([syllable("insufficient_evidence", tone=5)]), "insufficient_evidence")


class GroupingTests(unittest.TestCase):
    def test_word_annotations_group_the_script_when_they_match(self):
        groups, status = ta.make_groups(["我愛你"], ["我", "愛你"], IDENTITY)
        self.assertEqual((status, groups), ("matched", [("我", 0, 1), ("愛你", 1, 3)]))

    def test_clauses_are_used_without_matching_annotations(self):
        self.assertEqual(ta.make_groups(["你好", "再見"], [], IDENTITY),
                         ([("你好", 0, 2), ("再見", 2, 4)], "missing"))
        groups, status = ta.make_groups(["你好"], ["他好"], IDENTITY)
        self.assertEqual((status, groups), ("does_not_match_script", [("你好", 0, 2)]))


class Chapter5Tests(unittest.TestCase):
    ITEM = {"script": "沒（有）空", "script_variants": ["沒空", "沒有空"], "pinyin": "méi (yǒu) kòng"}

    def test_optional_text_expands_to_present_and_omitted_forms(self):
        self.assertEqual(ta.chapter5_optional_text_variants("沒（有）空"), ["沒（有）空", "沒空", "沒有空"])

    def test_candidates_are_distinct_clean_scripts(self):
        candidates = ta.chapter5_candidates(self.ITEM, "沒（有）空", IDENTITY)
        self.assertEqual(sorted("".join(c["chars"]) for c in candidates), ["沒有空", "沒空"])

    def test_catalog_pinyin_overrides_the_kong_reading(self):
        contexts = ta.chapter5_contexts(["沒空"], self.ITEM)
        self.assertEqual(contexts[-1]["target_tone"], 4)
        self.assertEqual(contexts[-1]["tone_rule"], "catalog_pinyin_disambiguation")


class CandidateChoiceTests(unittest.TestCase):
    @staticmethod
    def result(outside, statuses, error=None):
        return {"alignment_error": error, "f0_extraction": {"voiced_outside_syllables": outside},
                "syllables": [{"status": s, "acoustic_quality": 0.9} for s in statuses]}

    def test_the_variant_that_leaves_less_speech_unexplained_wins(self):
        # Speaker said 裡面; the 'omit 面' variant leaves the second syllable's voicing unexplained.
        omitted = self.result(0.45, ["likely_correct"])
        full = self.result(0.02, ["likely_correct", "insufficient_evidence"])
        self.assertGreater(ta._candidate_key(full), ta._candidate_key(omitted))

    def test_a_pair_with_no_practice_beats_one_with_practice_when_the_speech_is_equally_explained(self):
        clean = self.result(0.0, ["likely_correct"])
        accused = self.result(0.0, ["likely_correct", "practice"])
        self.assertGreater(ta._candidate_key(clean), ta._candidate_key(accused))

    def test_unexplained_speech_still_comes_before_the_verdict(self):
        # A learner who says the right word with a wrong tone must not be moved to a variant that leaves a
        # syllable of speech unexplained just because that one gives no `practice`.
        right_word_wrong_tone = self.result(0.0, ["likely_correct", "practice"])
        other_variant_misses_speech = self.result(0.4, ["likely_correct"])
        self.assertGreater(ta._candidate_key(right_word_wrong_tone), ta._candidate_key(other_variant_misses_speech))

    def test_neutral_syllables_do_not_count_as_scored(self):
        with_neutral = {"alignment_error": None, "f0_extraction": {"voiced_outside_syllables": 0.0},
                        "syllables": [{"status": "likely_correct", "acoustic_quality": 0.9, "expected_surface_tone": 1},
                                      {"status": "insufficient_evidence", "acoustic_quality": 0.9, "expected_surface_tone": 5}]}
        self.assertEqual(ta._candidate_key(with_neutral)[3], 1.0)                  # 1 of 1 scored syllables accepted

    def test_a_failed_alignment_ranks_last(self):
        self.assertGreater(ta._candidate_key(self.result(0.1, ["insufficient_evidence"])),
                           ta._candidate_key({**self.result(None, ["insufficient_evidence"]), "alignment_error": "x",
                                              "f0_extraction": None}))


class Chapter5AlignmentChoiceTests(unittest.TestCase):
    """A voiced run cannot tell 裡 from 裡面, so variants of different length must all go through MFA."""

    def run_item(self, item):
        flags = []
        audio, spans = tone_audio([[200] * 4])
        syllable = [{"char": "媽", "start": spans[0][0], "end": spans[0][1], "phones": [], "nucleus": None}]

        def fake_verify(prepared, clauses, contexts, calibration, force_mfa=False):
            flags.append(force_mfa)
            chars = "".join(clauses)
            return tp.verify_utterance(prepared.samples, prepared.sample_rate, chars, calibration, contexts=contexts,
                                       alignment=[dict(syllable[0], char=c) for c in chars][:len(contexts)] if len(chars) == 1
                                       else [], config=CONFIG)

        import audio_prep
        prepared = audio_prep.prepare_samples(audio, SR, CONFIG)
        calibration = {"status": "calibrated", "pitch_floor": 90.0, "pitch_ceiling": 450.0, "f0_median": 200.0}
        with mock.patch.object(ta, "_verify", side_effect=fake_verify):
            ta.assess_chapter5_item(prepared, item=item, sample_id="x", target_script=item["script"],
                                    calibration=calibration, normalize=IDENTITY)
        return flags

    ITEM = {"word_key": "K", "item_type": "word", "traditional_chinese": "x", "pinyin": "", "section": "5-1"}

    def test_optional_text_forces_mfa_for_every_variant(self):
        self.assertEqual(self.run_item({**self.ITEM, "script": "裡（面）"}), [True, True])

    def test_a_single_fixed_one_syllable_script_may_skip_mfa(self):
        self.assertEqual(self.run_item({**self.ITEM, "script": "媽"}), [False])


class PhraseF0Tests(unittest.TestCase):
    CONTOUR = {"step_sec": 0.005, "times": [round(i * 0.005, 3) for i in range(100)],
               "st": [None] * 10 + [0.0] * 30 + [None] * 8 + [2.0] * 40 + [None] * 12}

    def test_outer_unvoiced_frames_are_trimmed_and_inner_gaps_stay_null(self):
        f0 = ta.phrase_f0(self.CONTOUR, 0.0, 0.5, 200.0)
        self.assertEqual(f0["contour"][0]["offset_sec"], 0.0)
        self.assertEqual(f0["contour"][0]["hz"], 200.0)
        self.assertTrue(any(p["relative_semitone"] is None for p in f0["contour"]))
        self.assertIsNotNone(f0["contour"][-1]["relative_semitone"])
        self.assertAlmostEqual(f0["contour"][-1]["hz"], round(200 * 2 ** (2 / 12), 1), places=1)

    def test_no_pitch_or_no_baseline_gives_none(self):
        self.assertIsNone(ta.phrase_f0(self.CONTOUR, 0.0, 0.04, 200.0))
        self.assertIsNone(ta.phrase_f0(self.CONTOUR, 0.0, 0.5, None))
        self.assertIsNone(ta.phrase_f0(None, 0.0, 0.5, 200.0))


class PayloadTests(unittest.TestCase):
    def payload(self, contours, script, groups=None, alignment=True):
        audio, spans = tone_audio(contours)
        syllables = [{"char": c, "start": s, "end": e, "phones": [], "nucleus": (s + 0.04, e)}
                     for c, (s, e) in zip(script, spans)] if alignment else []
        calibration = {"status": "calibrated", "pitch_floor": 90.0, "pitch_ceiling": 450.0, "f0_median": 200.0,
                       "calibration_version": "speaker_f0_v1", "acoustic_profile": "mid"}
        result = tp.verify_utterance(audio, SR, script, calibration, alignment=syllables, config=CONFIG)
        return ta.build_payload(
            result, groups=groups or [(script, 0, len(script))], word_annotation_status="missing",
            boundaries_from="punctuation clauses", practice_mode="custom", sample_id="custom", target_script=script)

    def test_payload_has_the_fields_the_ui_and_store_read(self):
        payload = self.payload([[200] * 4, [170, 165, 200, 235], [260, 230, 190, 150]], "媽麻罵",
                               groups=[("媽麻", 0, 2), ("罵", 2, 3)])
        json.dumps(payload)                                    # must be storable
        self.assertTrue(payload["success"])
        self.assertEqual(payload["transcript_check"]["status"], "script_provided")
        self.assertEqual(payload["summary"]["n_phrases"], 2)
        first, second = payload["phrases"]
        self.assertEqual(first["annotation"], "媽麻")
        self.assertEqual(first["expected_tone_sequence"], [1, 2])
        self.assertEqual(first["status"], "likely_correct")
        self.assertEqual(second["expected_tone_sequence"], [4])
        self.assertTrue(first["phrase_f0"]["contour"])
        self.assertTrue(first["phrase_f0"]["duration_sec"] > 0)
        for key in ("word", "annotation", "expected_tone_sequence", "observed_tone_sequence", "sequence_evidence",
                    "assessment_note", "status"):                # read by practice_store.list_admin_dashboard
            self.assertIn(key, first)
        self.assertEqual(payload["summary"]["likely_correct"], 2)
        self.assertEqual(payload["tone_verification"]["syllables"][0]["character"], "媽")
        self.assertNotIn("config", payload)

    def test_practice_phrase_carries_conservative_feedback(self):
        payload = self.payload([[200] * 4], "罵")
        phrase = payload["phrases"][0]
        self.assertEqual(phrase["status"], "practice")
        self.assertEqual(phrase["assessment_note"], CONFIG["feedback"]["practice"][4])
        self.assertEqual(payload["summary"]["practice"], 1)
        self.assertEqual(payload["summary"]["n_assessable"], 1)

    def test_failed_alignment_is_an_insufficient_evidence_payload_not_an_error(self):
        payload = self.payload([[200] * 4, [200] * 4], "媽罵", alignment=False)
        self.assertTrue(payload["tone_verification"]["alignment"]["error"])
        self.assertEqual({p["status"] for p in payload["phrases"]}, {"insufficient_evidence"})
        self.assertIsNone(payload["phrases"][0]["phrase_f0"])
        self.assertEqual(payload["feedback"], CONFIG["feedback"]["insufficient_evidence"])
        json.dumps(payload)

    def test_neutral_tone_is_reported_as_zero_in_the_tone_sequence(self):
        payload = self.payload([[200] * 4, ([190, 190, 190], 0.18)], "媽吧")
        self.assertEqual(payload["phrases"][0]["expected_tone_sequence"], [1, 0])


class SentenceProfileTests(unittest.TestCase):
    def test_words_keep_the_plain_config_and_sentences_get_the_window_rule(self):
        self.assertEqual(ta.get_config()["confidence"]["practice_min_window_ms"], 0)
        self.assertEqual(ta.get_config("sentence")["confidence"]["practice_min_window_ms"], 100)

    def test_the_sentence_profile_changes_nothing_else_and_leaves_the_base_config_alone(self):
        base, sentence = ta.get_config(), ta.get_config("sentence")
        changed = {"confidence": {"practice_min_window_ms"}, "tone": {"clause_final_half_t3"}}
        for section in sorted(set(base) - {"sentence"}):
            if not isinstance(base[section], dict):
                self.assertEqual(base[section], sentence[section], section)
                continue
            keep = lambda config: {k: v for k, v in config[section].items() if k not in changed.get(section, ())}  # noqa: E731
            self.assertEqual(keep(base), keep(sentence), section)
        for section, keys in changed.items():                          # the only two differences, and only in the profile
            for key in keys:
                self.assertNotEqual(base[section][key], sentence[section][key])
        self.assertIsNot(base, sentence)
        self.assertEqual(base["confidence"]["practice_min_window_ms"], 0)
        self.assertFalse(base["tone"]["clause_final_half_t3"])
        self.assertTrue(sentence["tone"]["clause_final_half_t3"])

    def test_an_unknown_profile_is_an_error(self):
        with self.assertRaises(ValueError):
            ta.get_config("paragraph")

    def test_sentence_contexts_cross_word_sandhi_only_with_matched_annotations(self):
        clauses = ["我也好"]
        groups = [("我", 0, 1), ("也", 1, 2), ("好", 2, 3)]
        matched = ta.sentence_contexts(clauses, groups, "matched")
        self.assertEqual(matched[0]["also_accept_tones"], (2,))
        missing = ta.sentence_contexts(clauses, [("我也好", 0, 3)], "missing")
        self.assertEqual(missing[0]["target_tone"], 2)                 # plain sandhi inside one clause, as always
        self.assertNotIn("also_accept_tones", missing[0])

    def test_word_practice_contexts_are_the_plain_ones(self):
        item = {"pinyin": ""}
        self.assertEqual(ta.chapter5_contexts(["我也"], item), capt_core.expected_tone_contexts(["我也"]))


if __name__ == "__main__":
    unittest.main()
