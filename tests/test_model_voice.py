import copy
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import capt_core  # noqa: E402
import main  # noqa: E402
import model_voice  # noqa: E402
import tone_assessment as ta  # noqa: E402
import tone_pipeline as tp  # noqa: E402
from practice_store import PracticeStore  # noqa: E402
from test_tone_pipeline import CONFIG, SPEAKER, STEP, SR, make_track, tone_audio  # noqa: E402

SOURCE = (ROOT / "main.py").read_text(encoding="utf-8")
SETTINGS = dict(CONFIG["model_voice"])
SPREAD = 3.0


def shape(kind, n=24):
    x = np.linspace(0.0, 1.0, n)
    return {"fall": 6.0 - 12.0 * x, "rise": -4.0 + 8.0 * x, "flat": 0.2 * np.sin(2 * np.pi * x), "dip": 8.0 * (x - 0.45) ** 2 - 2.0}[kind]


def fixture_contours():
    """Four tones, five slightly different contours each (a fall, rise, level line, or dip)."""
    kinds = {1: "flat", 2: "rise", 3: "dip", 4: "fall"}
    references = {tone: [model_voice.normalise(shape(kind, n) * scale, SPREAD, SETTINGS["std_floor"]) for n, scale in ((20, 1.0), (24, 0.9), (28, 1.1), (22, 1.0), (26, 0.95))]
                  for tone, kind in kinds.items()}
    sigmoid = {str(tone): {"midpoint": 0.35, "steepness": 14.0} for tone in kinds}
    return model_voice.Contours(references, sigmoid)


FIXTURE = fixture_contours()


def sequence(kind, n=30):
    return model_voice.normalise(shape(kind, n), SPREAD, SETTINGS["std_floor"])


class NormaliseTests(unittest.TestCase):
    def test_the_level_of_a_syllable_plays_no_part(self):
        x = shape("fall")
        np.testing.assert_allclose(model_voice.normalise(x, SPREAD, 0.2), model_voice.normalise(x + 5.0, SPREAD, 0.2))

    def test_a_level_tone_keeps_a_small_amplitude_instead_of_having_its_tracking_noise_blown_up(self):
        flat = 0.05 * np.sin(np.linspace(0, 9, 30))                       # std ~0.035 st, far below 0.2 x 3 st
        assert float(flat.std()) < 0.2 * SPREAD
        self.assertLess(float(model_voice.normalise(flat, SPREAD, 0.2).std()), 0.1)
        self.assertAlmostEqual(float(model_voice.normalise(shape("fall"), SPREAD, 0.2).std()), shape("fall").std() / max(shape("fall").std(), 0.2 * SPREAD), places=6)

    def test_judged_values_are_the_finite_frames_every_second_one(self):
        st = np.array([np.nan, 1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_allclose(model_voice.judged_values(st), [1.0, 3.0, 5.0])
        self.assertIsNone(model_voice.judged_values(np.array([np.nan, 2.0, np.nan])))


class SimilarityTests(unittest.TestCase):
    def test_a_contour_is_identical_to_itself_and_a_fall_is_closer_to_the_falling_tone(self):
        reference = FIXTURE.references[4][0]
        solo = model_voice.Contours({4: [reference]}, {"4": {"midpoint": 0.35, "steepness": 14.0}})
        self.assertAlmostEqual(solo.similarity(4, reference, SETTINGS), 1.0, places=9)
        falling = sequence("fall")
        self.assertGreater(FIXTURE.similarity(4, falling, SETTINGS), FIXTURE.similarity(2, falling, SETTINGS) + 0.3)
        self.assertGreater(FIXTURE.score(4, falling, SETTINGS), 0.9)
        self.assertLess(FIXTURE.score(2, falling, SETTINGS), 0.1)

    def test_the_plain_python_path_matches_the_numba_path(self):
        if model_voice.numba is None:
            self.skipTest("numba is not installed")
        rng = np.random.default_rng(3)
        for n in (5, 13, 30, 41):
            seq = model_voice.normalise(np.cumsum(rng.normal(size=n)), SPREAD, 0.2)
            for tone in (1, 2, 3, 4):
                with mock.patch.object(model_voice, "USE_NUMBA", True):
                    fast = FIXTURE.similarity(tone, seq, SETTINGS)
                with mock.patch.object(model_voice, "USE_NUMBA", False):
                    slow = FIXTURE.similarity(tone, seq, SETTINGS)
                self.assertAlmostEqual(fast, slow, places=9, msg=(n, tone))

    def test_dtw_path_starts_and_ends_at_both_contour_endpoints(self):
        short, long_ = sequence("fall", 8), sequence("fall", 40)
        a, b = model_voice._dtw_python(short.tolist(), long_.tolist(), 1)
        self.assertEqual((int(a[0]), int(b[0]), int(a[-1]), int(b[-1])), (0, 0, 7, 39))

    def test_five_point_moving_average_smooths_without_changing_length(self):
        smoothed = model_voice._smooth([0.0, 0.0, 5.0, 0.0, 0.0], 5)
        np.testing.assert_allclose(smoothed, [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_per_tone_sigmoid_is_centered_on_its_fitted_midpoint(self):
        solo = model_voice.Contours({4: [sequence("fall")]}, {"4": {"midpoint": 0.35, "steepness": 14.0}})
        with mock.patch.object(solo, "similarity", return_value=0.35):
            self.assertAlmostEqual(solo.score(4, sequence("fall"), SETTINGS), 0.5)

    def test_the_number_of_contours_used_can_be_limited(self):
        full = FIXTURE.similarity(4, sequence("rise"), SETTINGS)
        self.assertNotEqual(full, FIXTURE.similarity(4, sequence("rise"), SETTINGS, count=1))

    def test_same_sample_cosine_is_high_for_the_same_contour_and_low_for_a_different_one(self):
        rising = shape("rise")
        same = model_voice.same_sample_similarity(rising, SPREAD, rising, SPREAD, SETTINGS)
        different = model_voice.same_sample_similarity(rising, SPREAD, shape("fall"), SPREAD, SETTINGS)
        self.assertGreaterEqual(same, 0.99)
        self.assertLess(different, 0.65)


class JudgeTests(unittest.TestCase):
    def values_of(self, kind, n=30):
        return shape(kind, n)

    def test_cosine_compares_all_tones_and_expected_tone_is_still_reported(self):
        judged = model_voice.judge(self.values_of("fall"), SPREAD, [4], SETTINGS, FIXTURE)
        self.assertGreater(judged["score"], 0.9)
        self.assertEqual(set(judged["scores"]), {1, 2, 3, 4})
        self.assertEqual(set(judged["similarities"]), {1, 2, 3, 4})
        self.assertEqual(judged["best_tone"], 4)

    def test_a_clearer_competing_cosine_prevents_a_high_expected_sigmoid_from_accepting_the_tone(self):
        similarities = {1: 0.548, 2: 0.273, 3: 0.202, 4: 0.337}
        with mock.patch.dict(FIXTURE.sigmoid, {3: (-0.5213, 3.014)}):
            with mock.patch.object(FIXTURE, "similarity", side_effect=lambda tone, *_args, **_kwargs: similarities[tone]):
                judged = model_voice.judge(self.values_of("flat"), SPREAD, [3], SETTINGS, FIXTURE)
        self.assertEqual(judged["best_tone"], 1)
        self.assertGreater(judged["score"], SETTINGS["good_at"])
        self.assertFalse(judged["accepted_match"])

    def test_good_cutoff_uses_the_sigmoid_score_for_every_tone(self):
        config = {**CONFIG, "model_voice": SETTINGS}
        self.assertEqual(model_voice.verdict(0.60, 1.0, True, config), "likely_correct")
        self.assertEqual(model_voice.verdict(0.5999, 1.0, True, config), "insufficient_evidence")

    def test_a_second_accepted_tone_counts_only_when_it_clearly_matches(self):
        rising = self.values_of("rise")
        alone = model_voice.judge(rising, SPREAD, [3], SETTINGS, FIXTURE)
        self.assertEqual(alone["best_tone"], 2)
        self.assertFalse(alone["accepted_match"])
        both = model_voice.judge(rising, SPREAD, [3, 2], SETTINGS, FIXTURE)
        self.assertLess(alone["score"], 1.0)
        self.assertGreater(both["score"], 0.8)
        self.assertGreater(both["scores"][2], both["scores"][3])
        self.assertTrue(both["accepted_match"])
        strict = model_voice.judge(rising, SPREAD, [3, 2], {**SETTINGS, "alternative_tone_min": 0.9999}, FIXTURE)
        self.assertFalse(strict["accepted_match"])
        self.assertAlmostEqual(strict["score"], alone["score"], places=9)

    def test_a_weak_optional_tone_does_not_hide_a_good_expected_tone(self):
        similarities = {1: 0.55, 2: 0.66, 3: 0.60, 4: -0.2}
        sigmoids = {1: (0.0, 1.0), 2: (0.30, 1.0), 3: (-1.0, 10.0), 4: (0.0, 1.0)}
        with mock.patch.dict(FIXTURE.sigmoid, sigmoids):
            with mock.patch.object(FIXTURE, "similarity", side_effect=lambda tone, *_args, **_kwargs: similarities[tone]):
                judged = model_voice.judge(self.values_of("flat"), SPREAD, [3, 2], SETTINGS, FIXTURE)
        self.assertEqual(judged["selected_accepted_tone"], 3)
        self.assertAlmostEqual(judged["score"], 1 / (1 + np.exp(-10.0 * (0.60 + 1.0))))
        self.assertTrue(judged["accepted_match"])

    def test_too_little_to_compare_is_not_judged(self):
        self.assertIsNone(model_voice.judge(None, SPREAD, [4], SETTINGS, FIXTURE))
        self.assertIsNone(model_voice.judge(np.array([1.0]), SPREAD, [4], SETTINGS, FIXTURE))
        only_two_tones = model_voice.Contours({1: FIXTURE.references[1], 2: FIXTURE.references[2]},
                                              {"1": {"midpoint": 0.35, "steepness": 14.0}, "2": {"midpoint": 0.35, "steepness": 14.0}})
        self.assertIsNone(model_voice.judge(self.values_of("fall"), SPREAD, [4], SETTINGS, only_two_tones))


class ModelVoiceSandhiContextTests(unittest.TestCase):
    BOOK_T3 = {"target_tone": 3, "tone_source": "book_pinyin"}

    def test_strict_rules_sandhi_replaces_the_book_tone_used_for_scoring(self):
        for rule in ("third_tone_sandhi", "bu_sandhi", "yi_sandhi"):
            with self.subTest(rule=rule):
                self.assertEqual(tp._model_voice_scoring_tones(
                    {"tone_rule": rule, "target_tone": 2, "also_accept_tones": ()}, self.BOOK_T3), [2])

    def test_optional_third_tone_sandhi_keeps_both_rules_accepted_readings(self):
        result = tp._model_voice_scoring_tones(
            {"tone_rule": "third_tone_sandhi_optional", "target_tone": 3, "also_accept_tones": (2,)},
            self.BOOK_T3,
        )
        self.assertEqual(result, [3, 2])

    def test_non_sandhi_rules_do_not_override_book_pinyin(self):
        for rule in ("lexical_tone", "taiwan_lexical_reading", "phrase_final_third_tone"):
            with self.subTest(rule=rule):
                self.assertIsNone(tp._model_voice_scoring_tones(
                    {"tone_rule": rule, "target_tone": 2}, self.BOOK_T3))


class VerdictTests(unittest.TestCase):
    config = {**CONFIG, "model_voice": SETTINGS}

    def test_good_needs_the_score_and_decent_quality(self):
        confidence = CONFIG["confidence"]
        self.assertEqual(model_voice.verdict(SETTINGS["good_at"], 1.0, True, self.config), "likely_correct")
        self.assertEqual(model_voice.verdict(SETTINGS["good_at"] - 0.0001, 1.0, True, self.config), "insufficient_evidence")
        self.assertEqual(model_voice.verdict(0.95, confidence["likely_correct_quality"] - 0.01, True, self.config), "insufficient_evidence")

    def test_needs_practice_needs_a_low_score_high_quality_and_permission_to_accuse(self):
        confidence = CONFIG["confidence"]
        self.assertEqual(model_voice.verdict(0.30, 1.0, True, self.config), "practice")
        self.assertEqual(model_voice.verdict(0.31, 1.0, True, self.config), "insufficient_evidence")
        self.assertEqual(model_voice.verdict(0.05, 1.0, False, self.config), "insufficient_evidence")
        self.assertEqual(model_voice.verdict(0.05, confidence["practice_quality"] - 0.01, True, self.config), "insufficient_evidence")


def verify_model(char, st_values, *, accuse_ok=True, doubtful_alignment=False, alignment_ok=True, context=None,
                 settings=None, confidence=None, enabled=True, reference=None):
    config = copy.deepcopy(CONFIG)
    config["model_voice"].update({"enabled": enabled, **(settings or {})})
    config["confidence"].update(confidence or {})
    extraction, clean = make_track(st_values)
    end = len(st_values) * STEP
    syllable = {"char": char, "start": 0.0, "end": end, "phones": [], "alignment_ok": alignment_ok, "nucleus": (0.0, end)}
    context = context or capt_core.expected_tone_contexts([char])[0]
    with mock.patch.object(model_voice, "load_contours", return_value=FIXTURE):
        return tp.verify_syllable(syllable, context, extraction, clean, config, SPEAKER, True,
                                  accuse_ok=accuse_ok, doubtful_alignment=doubtful_alignment,
                                  model_voice_reference=reference)


FALL, RISE = np.linspace(6.0, -6.0, 60), np.linspace(-4.0, 4.0, 60)


class PipelineTests(unittest.TestCase):
    def test_mode_voice_uses_book_expected_tones_and_retains_the_rules_path(self):
        script = "你好"
        rule_contexts = capt_core.expected_tone_contexts([script])
        book_contexts = ta._contexts_with_book_tones(rule_contexts, [3, 3])
        audio, spans = tone_audio([[170, 175, 190, 210], [190, 150, 140, 175]])
        syllables = [{"char": char, "start": start, "end": end, "phones": [], "nucleus": (start + 0.04, end)}
                     for char, (start, end) in zip(script, spans)]
        calibration = {"status": "calibrated", "pitch_floor": 90.0, "pitch_ceiling": 450.0, "f0_median": 200.0}
        config = ta.get_config("sentence", "easy", "model_voice")

        with mock.patch.object(model_voice, "load_contours", return_value=FIXTURE):
            result = tp.verify_utterance(audio, SR, script, calibration, alignment=syllables, config=config,
                                         contexts=rule_contexts, model_voice_contexts=book_contexts)

        self.assertEqual([item["expected_surface_tone"] for item in result["syllables"]], [3, 3])
        self.assertEqual([item["rule_expected_surface_tone"] for item in result["syllables"]], [2, 3])
        self.assertTrue(all(item["tone_source"] == "book_pinyin" for item in result["syllables"]))
        self.assertTrue(all("rule_tone_match" in item for item in result["syllables"]))

    def test_same_sample_reference_is_only_used_to_rescue_a_non_good_model_voice_result(self):
        context = {**capt_core.expected_tone_contexts(["媽"])[0], "target_tone": 1}
        reference = {"values": model_voice.judged_values(RISE), "spread": SPREAD}
        rescued = verify_model("媽", RISE, context=context, reference=reference)
        unchanged = verify_model("媽", RISE, context=context,
                                 reference={"values": model_voice.judged_values(FALL), "spread": SPREAD})

        self.assertEqual(rescued["status"], "likely_correct")
        self.assertTrue(rescued["model_voice_reference_used"])
        self.assertIn("matched_same_sample_reference", rescued["reasons"])
        self.assertLess(unchanged["model_voice_reference_cosine"], 0.65)
        self.assertFalse(unchanged["model_voice_reference_used"])

    def test_same_sample_reference_does_not_rescue_low_quality_or_doubtful_alignment(self):
        context = {**capt_core.expected_tone_contexts(["媽"])[0], "target_tone": 1}
        reference = {"values": model_voice.judged_values(RISE), "spread": SPREAD}
        low_quality = verify_model("媽", RISE, context=context, reference=reference,
                                   settings={"reference_sample_min_quality": 1.1})
        doubtful = verify_model("媽", RISE, context=context, reference=reference, doubtful_alignment=True)
        self.assertNotEqual(low_quality["status"], "likely_correct")
        self.assertFalse(low_quality.get("model_voice_reference_used", False))
        self.assertNotEqual(doubtful["status"], "likely_correct")
        self.assertFalse(doubtful.get("model_voice_reference_used", False))

    def test_a_clean_fall_for_tone_4_is_good_and_keeps_the_rules_verdict_beside_it(self):
        result = verify_model("去", FALL)
        self.assertEqual(result["status"], "likely_correct")
        self.assertEqual(result["rule_status"], "likely_correct")
        self.assertGreater(result["model_score"], 0.9)
        self.assertNotIn("model_closest_tone", result)

    def test_a_rise_for_tone_4_is_accused_and_says_which_tone_it_sounds_like(self):
        result = verify_model("去", RISE)
        self.assertEqual(result["status"], "practice")
        self.assertEqual(result["model_closest_tone"], 2)
        self.assertEqual(result["model_scores"]["2"], max(result["model_scores"].values()))
        self.assertLess(result["model_score"], 0.3)
        self.assertTrue(result["feedback"])

    def test_the_rules_guards_still_hold_back_an_accusation(self):
        for label, kwargs, settings_, reason in (("unstable recording", {"accuse_ok": False}, None, "unstable_pitch_track"),
                                                 ("doubtful alignment", {"doubtful_alignment": True}, None, "bad_alignment")):
            with self.subTest(label):
                result = verify_model("去", RISE, **kwargs)
                self.assertEqual(result["status"], "insufficient_evidence")
                self.assertIn(reason, result["reasons"])
        short = verify_model("去", RISE, confidence={"practice_min_window_ms": 400, "practice_min_window_ms_by_tone": {}})
        self.assertEqual(short["status"], "insufficient_evidence")
        self.assertIn("short_analysis_window", short["reasons"])

    def test_a_syllable_the_rules_cannot_judge_keeps_the_rules_answer(self):
        gated = verify_model("去", FALL, alignment_ok=False)
        self.assertEqual(gated["status"], "insufficient_evidence")
        self.assertNotIn("model_score", gated)
        neutral = verify_model("的", FALL)
        self.assertNotIn("model_score", neutral)

    def test_a_third_tone_that_also_accepts_tone_2_is_good_for_a_rise_but_not_when_the_bar_is_out_of_reach(self):
        context = {**capt_core.expected_tone_contexts(["我"])[0], "also_accept_tones": (2,)}
        self.assertEqual(context["target_tone"], 3)
        result = verify_model("我", RISE, context=context)
        self.assertEqual(result["status"], "likely_correct")
        self.assertEqual(result["also_accepted_tones"], [2])
        strict = verify_model("我", RISE, context=context, settings={"alternative_tone_min": 0.9999})
        self.assertNotEqual(strict["status"], "likely_correct")

    def test_without_the_mode_the_syllable_is_what_it_always_was(self):
        off = verify_model("去", RISE, enabled=False)
        for key in ("rule_status", "model_score", "model_scores", "model_closest_tone"):
            self.assertNotIn(key, off)
        self.assertEqual(off["status"], "practice")                       # the rules' own verdict for a rise on tone 4


class ConfigAndPayloadTests(unittest.TestCase):
    def test_the_mode_is_a_copy_of_the_configuration_with_the_switch_on(self):
        rules = ta.get_config("sentence", "easy")
        self.assertIs(ta.get_config("sentence", "easy", "rules"), rules)
        self.assertIs(ta.get_config("sentence", "easy", None), rules)
        mode = ta.get_config("sentence", "easy", "model_voice")
        self.assertTrue(mode["model_voice"]["enabled"])
        fusion = ta.get_config("sentence", "easy", "fusion")
        self.assertTrue(fusion["model_voice"]["enabled"])
        self.assertFalse(rules["model_voice"]["enabled"])
        self.assertIs(ta.get_config("sentence", "easy", "model_voice"), mode)
        self.assertEqual(mode["tone"]["t4_by_duration"], rules["tone"]["t4_by_duration"])           # the level is kept
        with self.assertRaises(ValueError):
            ta.get_config(scoring_mode="bogus")

    def test_fusion_accepts_either_good_verdict_and_needs_both_to_flag_practice(self):
        statuses = [
            ("likely_correct", "practice", 2, 3),       # Rules-only good: display the Rules target.
            ("practice", "likely_correct", 3, 4),       # Model Voice-only good: keep the book target.
            ("likely_correct", "likely_correct", 1, 1),
            ("practice", "practice", 2, 3),
            ("practice", "insufficient_evidence", 2, 3),
        ]
        syllables = []
        for rules, voice, rule_tone, book_tone in statuses:
            syllables.append({
                "character": "字", "status": voice, "rule_status": rules,
                "expected_surface_tone": book_tone, "tone_rule": "book_pinyin", "lexical_tone": book_tone,
                "rule_expected_surface_tone": rule_tone, "rule_lexical_tone": rule_tone,
                "rule_tone_rule": "rules", "rule_tone_match": 0.9, "tone_match": 0.1,
                "model_score": 0.2 if voice == "practice" else 0.9, "reasons": [], "feedback": "feedback",
            })
        result = ta._fuse_verification_result({"syllables": syllables, "config": copy.deepcopy(CONFIG),
                                                "summary": {}, "feedback": "old"})
        fused = result["syllables"]
        self.assertEqual([item["status"] for item in fused], [
            "likely_correct", "likely_correct", "likely_correct", "practice", "insufficient_evidence",
        ])
        self.assertEqual([item["fusion_basis"] for item in fused], [
            "rules_only", "model_voice_only", "both", "both_practice", "unresolved",
        ])
        self.assertEqual(fused[0]["expected_surface_tone"], 2)
        self.assertEqual(fused[0]["model_voice_expected_surface_tone"], 3)
        self.assertEqual(fused[1]["expected_surface_tone"], 4)
        self.assertEqual(result["summary"]["likely_correct"], 3)
        self.assertEqual(result["summary"]["practice"], 1)

        no_voice_verdict = ta._fuse_verification_result({
            "syllables": [{"character": "字", "status": "practice", "rule_status": "practice",
                           "expected_surface_tone": 2, "rule_expected_surface_tone": 2, "reasons": []}],
            "config": copy.deepcopy(CONFIG), "summary": {}, "feedback": "old",
        })
        self.assertEqual(no_voice_verdict["syllables"][0]["status"], "insufficient_evidence")
        self.assertEqual(no_voice_verdict["syllables"][0]["fusion_basis"], "unresolved")

    def test_the_payload_and_the_saved_summary_say_which_mode_judged_it(self):
        payload = {"summary": {}}
        ta._mark_scoring_mode(payload, "model_voice")
        self.assertEqual((payload["scoring_mode"], payload["summary"]["scoring_mode"]), ("model_voice", "model_voice"))
        other = {"summary": {}}
        ta._mark_scoring_mode(other, None)
        self.assertEqual(other["summary"]["scoring_mode"], "rules")

    def test_a_syllable_not_good_names_the_tone_it_sounds_like_in_the_advice(self):
        group = [{"character": "去", "status": "practice", "expected_surface_tone": 4, "reasons": [], "model_closest_tone": 2, "model_score": 0.1}]
        self.assertIn({"type": "heard_as", "char": "去", "tone": 2}, ta.phrase_feedback(group))
        good = [{"character": "去", "status": "likely_correct", "expected_surface_tone": 4, "reasons": [], "model_closest_tone": 2, "model_score": 0.9}]
        self.assertEqual(ta.phrase_feedback(good), [])


class TheDataFileTests(unittest.TestCase):
    def test_the_shipped_contours_load_and_cover_every_tone(self):
        self.assertTrue(model_voice.available())
        contours = model_voice.load_contours()
        self.assertEqual(contours.tones(), (1, 2, 3, 4))
        for tone in model_voice.TONES:
            self.assertEqual(len(contours.references[tone]), SETTINGS["references_per_tone"], tone)
        document = json.loads(model_voice.CONTOURS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(document["std_floor"], SETTINGS["std_floor"])
        self.assertEqual(document["references_per_tone"], SETTINGS["references_per_tone"])
        self.assertEqual(document["algorithm"], "dtw_alignment_5point_smoothing_cosine_tone_sigmoid")
        self.assertEqual(set(contours.sigmoid), set(model_voice.TONES))

    def test_the_teachers_fall_is_a_fall_for_tone_4_and_not_a_rise_with_the_real_contours(self):
        contours = model_voice.load_contours()
        self.assertGreater(contours.score(4, sequence("fall"), SETTINGS), contours.score(2, sequence("fall"), SETTINGS))
        self.assertGreater(contours.score(2, sequence("rise"), SETTINGS), contours.score(4, sequence("rise"), SETTINGS))


class EndpointTests(unittest.TestCase):
    class Upload:
        filename = "a.wav"

        def __init__(self):
            self.file = io.BytesIO(b"x")

    def call(self, scoring_mode, practice_mode="book"):
        student = {"id": "s1", "group_type": "foreigner"}
        with mock.patch.object(main.practice_store, "get_student", return_value=student), mock.patch.object(main.practice_store, "get_voice_calibration", return_value=None):
            return main._assess_pronunciation(self.Upload(), "你好", "custom", "s1", practice_mode, "", scoring_mode)

    def test_an_unknown_mode_is_a_400(self):
        with self.assertRaises(HTTPException) as caught:
            self.call("bogus")
        self.assertEqual(caught.exception.status_code, 400)

    def test_the_mode_without_its_data_is_a_503_not_a_silent_fallback_to_the_rules(self):
        with mock.patch.object(model_voice, "available", return_value=False), self.assertRaises(HTTPException) as caught:
            self.call("model_voice")
        self.assertEqual(caught.exception.status_code, 503)

    def test_fusion_requires_the_model_voice_data(self):
        with mock.patch.object(model_voice, "available", return_value=False), self.assertRaises(HTTPException) as caught:
            self.call("fusion")
        self.assertEqual(caught.exception.status_code, 503)

    def test_own_text_is_judged_by_the_rules_whatever_is_asked(self):
        seen = {}

        def fake_assess(prepared, **kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop here")

        with mock.patch.object(main.audio_prep, "prepare_recording", return_value=mock.Mock(samples=np.zeros(10))), \
                mock.patch.object(main.tone_assessment, "assess_recording", side_effect=fake_assess), \
                mock.patch.object(model_voice, "available", return_value=False), self.assertRaises(HTTPException) as caught:
            self.call("model_voice", practice_mode="custom")
        self.assertEqual(caught.exception.status_code, 500)               # got past the 503: the mode was not honoured for own text
        self.assertEqual(seen.get("scoring_mode"), "rules")


class BoardsIgnoreTheModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.store = PracticeStore(root / "practice.sqlite3", root / "recordings")
        self.audio = root / "a.webm"
        self.audio.write_bytes(b"x")
        self.student = self.store.create_or_get_student("Mei", "foreigner")["id"]

    def attempt(self, sample_id, good, okay=0, mode=None):
        summary = {"syllables": {"likely_correct": good, "insufficient_evidence": okay, "practice": 0}}
        if mode:
            summary["scoring_mode"] = mode
        self.store.save_attempt(self.student, sample_id, "文本", self.audio, "a.webm", {"summary": summary})

    def test_a_model_voice_attempt_is_on_neither_board(self):
        for number in range(5):
            self.attempt(f"5-1-{number + 1}", good=5, okay=5, mode="rules")                 # 50 each
            self.attempt(f"5-1-{number + 1}", good=10, mode="model_voice")                   # would be 100 each
        board = self.store.leaderboard()
        self.assertEqual(board["top"][0]["points"], 250)
        self.assertEqual(board["accuracy"]["top"][0]["accuracy"], 50.0)

    def test_fusion_attempts_count_on_both_boards(self):
        for number in range(5):
            self.attempt(f"5-1-{number + 1}", good=5, okay=5, mode="rules")
            self.attempt(f"5-1-{number + 1}", good=9, okay=1, mode="fusion")
        board = self.store.leaderboard()
        self.assertEqual(board["top"][0]["points"], 450)
        self.assertEqual(board["accuracy"]["top"][0]["accuracy"], 70.0)

    def test_the_latest_result_of_a_lesson_is_the_newest_attempt_whichever_way_it_was_judged(self):
        self.attempt("5-1-1", good=5, okay=5, mode="rules")                                  # 50
        self.attempt("5-1-1", good=9, okay=1, mode="model_voice")                            # 90: the newest
        self.attempt("5-1-2", good=10, mode="model_voice")                                   # a lesson only the model voice has been tried on: marked as practised
        lessons = self.store.progress(self.student)["lessons"]
        self.assertEqual((lessons["5-1-1"]["last"], lessons["5-1-1"]["attempts"], lessons["5-1-1"]["best"]), (90, 2, 90))
        self.assertEqual((lessons["5-1-2"]["last"], lessons["5-1-2"]["attempts"]), (100, 1))
        self.attempt("5-1-1", good=6, okay=4, mode="rules")                                  # 60, newest again
        self.assertEqual(self.store.progress(self.student)["lessons"]["5-1-1"]["last"], 60)
        self.assertEqual(self.store.leaderboard()["top"][0]["points"], 60)                  # the boards still count only the rules attempts: the best of 5-1-1 is 60

    def test_attempts_from_before_the_mode_existed_still_count(self):
        self.attempt("5-1-1", good=4, okay=0)
        self.assertEqual(self.store.leaderboard()["top"][0]["points"], 100)


class AdminDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.store = PracticeStore(root / "practice.sqlite3", root / "recordings")
        self.audio = root / "a.webm"
        self.audio.write_bytes(b"x")
        self.student = self.store.create_or_get_student("Mei", "foreigner")["id"]

    def save(self, mode, syllables):
        result = {"scoring_mode": mode, "practice_mode": "book", "phrases": [], "tone_verification": {"syllables": syllables},
                  "summary": {"scoring_mode": mode, "syllables": {"likely_correct": 1, "insufficient_evidence": 0, "practice": 0}}}
        self.store.save_attempt(self.student, "5-1-1", "文本", self.audio, "a.webm", result)

    def test_a_model_voice_attempt_carries_both_results_and_the_syllables_they_disagree_on(self):
        self.save("model_voice", [
            {"character": "我", "expected_surface_tone": 3, "status": "likely_correct", "rule_status": "practice", "model_score": 0.88},
            {"character": "去", "expected_surface_tone": 4, "status": "practice", "rule_status": "likely_correct", "model_score": 0.12, "model_closest_tone": 2},
            {"character": "好", "expected_surface_tone": 3, "status": "likely_correct", "rule_status": "likely_correct", "model_score": 0.9},
            {"character": "的", "expected_surface_tone": 5, "status": "insufficient_evidence"}])
        attempt = self.store.list_admin_dashboard()[0]["attempts"][0]
        self.assertEqual(attempt["scoring_mode"], "model_voice")
        view = attempt["model_voice"]
        self.assertEqual((view["total"], view["rules_good"], view["model_good"]), (4, 2, 2))
        self.assertEqual([(d["character"], d["rules"], d["model"], d["score"], d["closest_tone"]) for d in view["differences"]],
                         [("我", "practice", "likely_correct", 0.88, None), ("去", "likely_correct", "practice", 0.12, 2)])

    def test_a_fusion_attempt_keeps_its_final_result_and_both_original_verdicts(self):
        self.save("fusion", [
            {"character": "你", "expected_surface_tone": 3, "model_voice_expected_surface_tone": 3,
             "status": "likely_correct", "model_voice_status": "practice", "rule_status": "likely_correct", "model_score": 0.12},
        ])
        attempt = self.store.list_admin_dashboard()[0]["attempts"][0]
        self.assertEqual(attempt["scoring_mode"], "fusion")
        self.assertEqual(attempt["model_voice"]["model_good"], 0)
        self.assertEqual(attempt["model_voice"]["rules_good"], 1)

    def test_a_rules_attempt_and_an_old_attempt_have_no_model_voice_view(self):
        self.save("rules", [{"character": "我", "expected_surface_tone": 3, "status": "likely_correct"}])
        self.store.save_attempt(self.student, "5-1-2", "文本", self.audio, "a.webm", {"summary": {"syllables": {"likely_correct": 1}}})
        attempts = self.store.list_admin_dashboard()[0]["attempts"]
        self.assertEqual({a["scoring_mode"] for a in attempts}, {"rules"})
        self.assertEqual({a["model_voice"] for a in attempts}, {None})


class PageTests(unittest.TestCase):
    def test_the_page_defaults_every_visit_to_fusion_even_if_a_browser_has_an_old_choice(self):
        select = SOURCE[SOURCE.index('<select id="scoringModeSelect"'): SOURCE.index('<p id="scoringModeHint"')]
        self.assertLess(select.index('value="fusion"'), select.index('value="model_voice"'))
        self.assertLess(select.index('value="model_voice"'), select.index('value="rules"'))
        page_state = SOURCE[SOURCE.index("// Each page visit starts in Fusion"): SOURCE.index("let isRecording", SOURCE.index("// Each page visit starts in Fusion"))]
        self.assertIn('let scoringMode = "fusion";', page_state)
        self.assertNotIn('localStorage.getItem("scoringMode")', SOURCE)
        self.assertIn('onchange="changeScoringMode()"', select)
        self.assertLess(SOURCE.index('id="practiceMode"'), SOURCE.index('id="scoringModeSelect"'))

    def test_the_choice_is_visit_only_sent_with_every_assessment_and_forced_to_rules_for_own_text(self):
        self.assertNotIn('localStorage.setItem("scoringMode", scoringMode)', SOURCE)
        self.assertIn('formData.append("scoring_mode", effectiveScoringMode(practiceMode));', SOURCE)
        effective = SOURCE[SOURCE.index("function effectiveScoringMode("): SOURCE.index("function changeScoringMode(")]
        self.assertIn('mode === "custom" ? "rules"', effective)
        sync = SOURCE[SOURCE.index("function syncScoringMode("): SOURCE.index("function hasMandarinCharacters(")]
        self.assertIn("select.disabled = custom ||", sync)
        self.assertIn('scoringModeCustomHint', sync)
        self.assertIn('scoringModeNotCounted', sync)
        self.assertIn('scoringModeFusionHint', sync)
        self.assertIn("let scoringMode", SOURCE[: SOURCE.index("function selectedPracticeMode(")])

    def test_the_mode_follows_the_page_state_and_the_language(self):
        controls = SOURCE[SOURCE.index("function syncPracticeControls()"): SOURCE.index("function selectedPracticeSampleId(")]
        self.assertIn("syncScoringMode();", controls)
        apply = SOURCE[SOURCE.index("function applyLanguage("): SOURCE.index("function changeLanguage(")]
        self.assertIn("syncScoringMode();", apply)

    def test_the_learners_result_names_the_closest_tone_but_shows_no_badge_and_no_comparison_of_the_modes(self):
        render = SOURCE[SOURCE.index("function renderVisualResults("): SOURCE.index('@app.post("/api/assess-pronunciation")')]
        self.assertIn('item.type === "heard_as"', render)
        for word in ("scoring-badge", "scoring-compare", "scoringBadgeModelVoice", "scoringCompare", "rule_status"):
            self.assertNotIn(word, SOURCE)                                 # not in the markup, the styles, the strings, or the result card

    def test_the_admin_dashboard_shows_the_mode_the_two_results_of_the_same_recording_and_where_they_differ(self):
        admin = SOURCE[SOURCE.index("function renderAttempt("): SOURCE.index("function renderUser(")]
        self.assertIn('attempt.scoring_mode === "model_voice"', admin)
        self.assertIn(" | model voice", admin)
        self.assertIn("Same recording: rules", admin)
        self.assertIn("view.differences", admin)
        self.assertIn("not counted on the leaderboards", admin)

    def test_any_okay_or_needs_practice_word_sends_the_learner_back_to_the_model_voice_once_under_the_cards_own_text_included(self):
        render = SOURCE[SOURCE.index("function renderVisualResults("): SOURCE.index('@app.post("/api/assess-pronunciation")')]
        flag = render[render.index("const listenAgain = "): render.index("const groups = ")]
        for status in ('"practice"', '"uncertain"', '"insufficient_evidence"'):
            self.assertIn(status, flag)
        self.assertNotIn("likely_correct", flag)
        self.assertNotIn("custom", flag)                                   # own text has generated audio to listen to
        self.assertEqual(render.count('t("adviceListenAgain")'), 1)        # one note for the whole result, not one per word card
        note = render[render.index('${listenAgain ?'):]
        self.assertTrue(note.startswith('${listenAgain ? `<p class="results-advice">'))
        self.assertLess(render.index('class="result-groups"'), render.index("${listenAgain ?"))        # under the cards
        self.assertNotIn("advice.push(", render)
        self.assertIn('adviceListenAgain: "Listen to the model voice again, then try once more."', SOURCE)

    def test_every_language_has_every_string_and_the_vietnamese_name_is_the_owners(self):
        for key in ("scoringModeLabel", "scoringModeRules", "scoringModeModelVoice", "scoringModeNotCounted", "scoringModeCustomHint", "adviceHeardAs", "adviceListenAgain"):
            self.assertEqual(len(re.findall(rf"\b{key}: ", SOURCE)), 5, key)
        self.assertIn('scoringModeModelVoice: "Theo giọng mẫu",', SOURCE)
        self.assertIn('scoringModeModelVoice: "對照範例發音",', SOURCE)

    def test_the_endpoint_takes_the_mode_and_hands_it_to_both_assessments(self):
        self.assertIn('scoring_mode: str = Form(default="fusion")', SOURCE)
        self.assertIn('scoring_mode: str = "fusion",', SOURCE)
        self.assertIn('scoring_mode or model_voice.MODE_FUSION', SOURCE)
        self.assertEqual(SOURCE.count("level=assessment_level, scoring_mode=scoring_mode,"), 2)


if __name__ == "__main__":
    unittest.main()
