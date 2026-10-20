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

        def fake_verify(prepared, clauses, contexts, calibration, force_mfa=False, config=None):
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
        self.assertEqual(ta.get_config("sentence")["confidence"]["practice_min_window_ms"], 130)

    def test_the_sentence_profile_changes_nothing_else_and_leaves_the_base_config_alone(self):
        base, sentence = ta.get_config(), ta.get_config("sentence")
        changed = {"confidence": {"practice_min_window_ms", "practice_min_window_ms_by_tone"}, "tone": {"clause_final_half_t3", "t4_by_duration", "t4_before_t4", "left_context", "t3_low_register", "t1_max_fall_st", "t1_fall_width_st", "t4_steady"}}
        for section in sorted(set(base) - {"sentence"}):
            if not isinstance(base[section], dict):
                self.assertEqual(base[section], sentence[section], section)
                continue
            keep = lambda config: {k: v for k, v in config[section].items() if k not in changed.get(section, ())}  # noqa: E731
            self.assertEqual(keep(base), keep(sentence), section)
        for section, keys in changed.items():                          # the only differences, and only in the profile
            for key in keys:
                self.assertNotEqual(base[section].get(key), sentence[section].get(key))
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

    def test_book_mode_uses_printed_sentence_tones_without_rules_sandhi(self):
        entry = ta._BOOK_SENTENCE_TONES["5-1-1"]
        clauses = capt_core.prepare_target(entry["script"])
        groups = [("sentence", 0, sum(map(len, clauses)))]
        rules = ta.sentence_contexts(clauses, groups, "matched")
        book = ta.book_sentence_contexts("5-1-1", clauses, rules)

        self.assertEqual(rules[0]["target_tone"], 2)                  # 友美: spoken third-tone sandhi
        self.assertEqual(book[0]["target_tone"], 3)                   # the book prints yǒu měi
        self.assertEqual(book[0]["tone_source"], "book_pinyin")
        self.assertEqual(book[0]["tone_rule"], "book_pinyin")

    def test_book_pinyin_tones_cover_neutral_and_variant_paths(self):
        self.assertEqual(ta.book_pinyin_tone_paths("nǎlǐ/nǎr"), [[3, 3], [3]])
        self.assertEqual(ta.book_pinyin_tone_paths("méi(yǒu) kòng"), [[2, 4], [2, 3, 4]])
        self.assertEqual(ta.book_pinyin_tone_paths("jiějie"), [[3, 5]])

        clauses = capt_core.prepare_target("你好")
        rules = capt_core.expected_tone_contexts(clauses)
        book = ta._contexts_with_book_tones(rules, [3, 3])
        self.assertEqual(rules[1]["adjacent_previous_tone"], 2)
        self.assertEqual(book[1]["adjacent_previous_tone"], 3)

        item = {"traditional_chinese": "哪裡/哪兒", "pinyin": "nǎlǐ/nǎr"}
        contexts = ta.chapter5_book_contexts(["哪"], item, "哪兒", IDENTITY)
        self.assertEqual([context["target_tone"] for context in contexts], [3])

    def test_word_practice_contexts_are_the_plain_ones(self):
        item = {"pinyin": ""}
        self.assertEqual(ta.chapter5_contexts(["我也"], item), capt_core.expected_tone_contexts(["我也"]))


class DisplayStatusTests(unittest.TestCase):
    """Sentence practice: more than half of a phrase's scored syllables correct -> yellow (uncertain), not practice."""

    @staticmethod
    def group(*statuses, tones=None):
        tones = tones or [2] * len(statuses)
        return [syllable(status, tone) for status, tone in zip(statuses, tones)]

    def test_off_by_default_so_words_keep_the_strict_status(self):
        group = self.group("likely_correct", "likely_correct", "practice")
        self.assertEqual(ta.display_status(group), "practice")
        self.assertEqual(ta.display_status(group, partial_credit=False), "practice")

    def test_more_than_half_correct_shows_uncertain(self):
        self.assertEqual(ta.display_status(self.group("likely_correct", "likely_correct", "practice"), True), "uncertain")
        self.assertEqual(ta.display_status(self.group(*["likely_correct"] * 6, "practice", "practice"), True), "uncertain")

    def test_exactly_half_or_fewer_stays_practice(self):
        self.assertEqual(ta.display_status(self.group("likely_correct", "practice"), True), "practice")
        self.assertEqual(ta.display_status(self.group("likely_correct", "likely_correct", "practice", "insufficient_evidence"), True), "practice")
        self.assertEqual(ta.display_status(self.group("practice", "practice", "likely_correct"), True), "practice")

    def test_doubtful_syllables_count_against_the_phrase(self):
        group = self.group("likely_correct", "likely_correct", "likely_correct", "practice", "insufficient_evidence")
        self.assertEqual(ta.display_status(group, True), "uncertain")            # 3 of 5
        group = self.group("likely_correct", "likely_correct", "practice", "insufficient_evidence", "insufficient_evidence")
        self.assertEqual(ta.display_status(group, True), "practice")             # 2 of 5

    def test_neutral_syllables_are_not_scored_so_they_do_not_pad_the_share(self):
        group = self.group("practice", "likely_correct", tones=[3, 5])
        self.assertEqual(ta.display_status(group, True), "practice")             # 0 of 1 scored syllable
        group = self.group("likely_correct", "likely_correct", "practice", "likely_correct", tones=[3, 2, 4, 5])
        self.assertEqual(ta.display_status(group, True), "uncertain")            # 2 of 3 scored

    def test_other_statuses_never_change(self):
        self.assertEqual(ta.display_status(self.group("likely_correct", "likely_correct"), True), "likely_correct")
        self.assertEqual(ta.display_status(self.group("insufficient_evidence", "likely_correct"), True), "insufficient_evidence")


class PartialCreditPayloadTests(unittest.TestCase):
    CAL = {"status": "calibrated", "pitch_floor": 90.0, "pitch_ceiling": 450.0, "f0_median": 200.0}

    def result(self):
        audio, spans = tone_audio([([200, 200, 200, 200], 0.35)] * 3)          # three level syllables
        syllables = [{"char": c, "start": s, "end": e, "phones": [], "nucleus": (s, e), "alignment_ok": True}
                     for c, (s, e) in zip("天天罵", spans)]                      # the last one should fall (T4)
        result = tp.verify_utterance(audio, SR, "天天罵", self.CAL, alignment=syllables, config=CONFIG)
        self.assertEqual([s["status"] for s in result["syllables"]], ["likely_correct", "likely_correct", "practice"])
        return result

    def payload(self, **kwargs):
        return ta.build_payload(self.result(), groups=[("天天罵", 0, 3)], word_annotation_status="matched",
                                boundaries_from="test", practice_mode="custom", sample_id="custom",
                                target_script="天天罵", audio_quality={"passed": True, "reasons": []}, **kwargs)

    def test_a_mostly_correct_phrase_is_yellow_in_sentence_mode_but_the_syllable_still_says_practice(self):
        payload = self.payload(partial_credit=True)
        phrase = payload["phrases"][0]
        self.assertEqual((phrase["status"], phrase["strict_status"]), ("uncertain", "practice"))
        self.assertEqual([s["status"] for s in phrase["syllable_diagnostics"]], ["likely_correct", "likely_correct", "practice"])
        self.assertEqual(phrase["feedback_items"], [{"type": "tone", "char": "罵", "tone": 4}])    # the advice is kept
        summary = payload["summary"]
        self.assertEqual((summary["practice"], summary["uncertain"], summary["insufficient_evidence"], summary["partly_correct"]),
                         (0, 1, 1, 1))
        self.assertEqual(summary["n_assessable"], 1)

    def test_words_and_the_default_stay_strict(self):
        payload = self.payload()
        phrase = payload["phrases"][0]
        self.assertEqual((phrase["status"], phrase["strict_status"]), ("practice", "practice"))
        self.assertEqual((payload["summary"]["practice"], payload["summary"]["uncertain"]), (1, 0))

    def test_the_word_service_path_does_not_ask_for_partial_credit(self):
        import inspect
        self.assertIn("partial_credit=True", inspect.getsource(ta.assess_recording))
        self.assertNotIn("partial_credit", inspect.getsource(ta.assess_chapter5_item))


class Chapter5PinyinTests(unittest.TestCase):
    def test_the_catalog_readings_show_over_the_word(self):
        item = {"script": "沒（有）空", "pinyin": "méi(yǒu) kòng"}
        self.assertEqual(ta.chapter5_character_pinyin(item), ["méi", "（", "yǒu", "）", "kòng"])
        self.assertEqual(ta.chapter5_character_pinyin({"script": "音樂", "pinyin": "yīnyuè"}), ["yīn", "yuè"])
        self.assertEqual(ta.chapter5_character_pinyin({"script": "哥哥", "pinyin": "gēge"}), ["gē", "ge"])


class SegmentWordsTests(unittest.TestCase):
    LEXICON = ta.build_lexicon(["下午茶", "咖啡廳", "我", "要", "哪裡/哪兒", "沒（有）空"], IDENTITY)

    def test_the_lexicon_keeps_multi_character_han_words_and_every_variant(self):
        self.assertEqual(self.LEXICON, {"下午茶", "咖啡廳", "哪裡", "沒有空"})        # 哪兒: the erhua 兒 is dropped, as in every script

    def test_known_words_stay_whole(self):
        words = ta.segment_words(["我要喝下午茶"], self.LEXICON)
        self.assertIn("下午茶", words)
        self.assertEqual("".join(words), "我要喝下午茶")

    def test_every_character_is_covered_in_order_and_words_never_cross_a_clause(self):
        clauses = ["我今天想學中文", "你呢", "我們一起去喝下午茶"]
        words = ta.segment_words(clauses, self.LEXICON)
        self.assertEqual("".join(words), "".join(clauses))
        position = 0
        edges = set()
        for clause in clauses:
            position += len(clause)
            edges.add(position)
        boundaries, total = set(), 0
        for word in words:
            total += len(word)
            boundaries.add(total)
        self.assertTrue(edges <= boundaries)                        # each clause end is a word end
        self.assertTrue(all(len(word) <= 4 for word in words))

    def test_unknown_text_is_split_into_short_words(self):
        words = ta.segment_words(["我今天想學中文"], self.LEXICON)
        self.assertGreater(len(words), 2)
        self.assertTrue(all(len(word) <= 3 for word in words))

    def test_typed_text_is_grouped_by_word_and_counts_as_annotated(self):
        clauses = ["我今天想學中文", "我們一起去喝下午茶"]
        groups, status = ta.make_groups(clauses, ta.segment_words(clauses, self.LEXICON), IDENTITY)
        self.assertEqual(status, "matched")
        self.assertGreater(len(groups), len(clauses))
        self.assertEqual(groups[-1][2], sum(len(c) for c in clauses))

    def test_the_book_mode_still_uses_the_teacher_annotation(self):
        import inspect
        source = open(ta.__file__, encoding="utf-8").read()
        self.assertIn("def segment_words", source)
        main_source = open(str(__import__("pathlib").Path(ta.__file__).with_name("main.py")), encoding="utf-8").read()
        self.assertIn('practice_mode == "custom"', main_source)
        self.assertIn("else WORD_ANNOTATIONS.get(sample_id, [])", main_source)


class DisplayStatusNotHeardTests(unittest.TestCase):
    def test_silence_shows_not_heard_in_both_modes_and_never_okay_or_practice(self):
        silent = [{"status": "insufficient_evidence", "expected_surface_tone": 2, "reasons": ["too_few_voiced_frames"]}]
        self.assertEqual(ta.display_status(silent), "not_heard")
        self.assertEqual(ta.display_status(silent, partial_credit=True), "not_heard")

    def test_a_phrase_with_a_measured_voice_keeps_its_strict_status(self):
        undecided = [{"status": "insufficient_evidence", "expected_surface_tone": 2, "reasons": ["ambiguous_contour"]}]
        self.assertEqual(ta.display_status(undecided), "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
