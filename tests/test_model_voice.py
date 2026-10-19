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
from test_tone_pipeline import CONFIG, SPEAKER, STEP, make_track  # noqa: E402

SOURCE = (ROOT / "main.py").read_text(encoding="utf-8")
SETTINGS = dict(CONFIG["model_voice"])
SPREAD = 3.0


def shape(kind, n=24):
    x = np.linspace(0.0, 1.0, n)
    return {"fall": 6.0 - 12.0 * x, "rise": -4.0 + 8.0 * x, "flat": 0.2 * np.sin(2 * np.pi * x), "dip": 8.0 * (x - 0.45) ** 2 - 2.0}[kind]


def fixture_contours():
    """Four tones, five slightly different contours each (a fall, a rise, a level line, a dip), one sigmoid for all."""
    kinds = {1: "flat", 2: "rise", 3: "dip", 4: "fall"}
    references = {tone: [model_voice.normalise(shape(kind, n) * scale, SPREAD, SETTINGS["std_floor"]) for n, scale in ((20, 1.0), (24, 0.9), (28, 1.1), (22, 1.0), (26, 0.95))]
                  for tone, kind in kinds.items()}
    return model_voice.Contours(references, {str(t): {"midpoint": 0.35, "steepness": 14.0} for t in kinds})


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
        solo = model_voice.Contours({4: [reference]}, {"4": {"midpoint": 0.0, "steepness": 1.0}})
        self.assertAlmostEqual(solo.similarity(4, reference, SETTINGS), 1.0, places=9)
        falling = sequence("fall")
        self.assertGreater(FIXTURE.similarity(4, falling, SETTINGS), FIXTURE.similarity(2, falling, SETTINGS) + 0.3)
        self.assertGreater(FIXTURE.score(4, falling, SETTINGS), 0.9)
        self.assertLess(FIXTURE.score(2, falling, SETTINGS), 0.1)

    def test_the_plain_python_path_gives_the_numbers_of_the_numba_path(self):
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

    def test_a_band_narrower_than_the_length_difference_is_widened_to_it(self):
        short, long_ = sequence("fall", 8), sequence("fall", 40)
        a, b = model_voice._dtw_python(short.tolist(), long_.tolist(), 1)
        self.assertEqual((int(a[0]), int(b[0]), int(a[-1]), int(b[-1])), (0, 0, 7, 39))      # a path from corner to corner exists

    def test_the_number_of_contours_used_can_be_limited(self):
        full = FIXTURE.similarity(4, sequence("rise"), SETTINGS)
        self.assertNotEqual(full, FIXTURE.similarity(4, sequence("rise"), SETTINGS, count=1))


class JudgeTests(unittest.TestCase):
    def values_of(self, kind, n=30):
        return shape(kind, n)

    def test_the_expected_tone_alone_decides_without_alternatives(self):
        judged = model_voice.judge(self.values_of("fall"), SPREAD, [4], SETTINGS, FIXTURE)
        self.assertGreater(judged["score"], 0.9)
        self.assertEqual(list(judged["scores"]), [4])

    def test_a_second_accepted_tone_counts_only_when_it_clearly_matches(self):
        rising = self.values_of("rise")
        alone = model_voice.judge(rising, SPREAD, [3], SETTINGS, FIXTURE)["score"]
        both = model_voice.judge(rising, SPREAD, [3, 2], SETTINGS, FIXTURE)
        self.assertLess(alone, 0.6)
        self.assertGreater(both["score"], 0.8)
        self.assertGreater(both["scores"][2], both["scores"][3])
        strict = model_voice.judge(rising, SPREAD, [3, 2], {**SETTINGS, "alternative_tone_min": 0.9999}, FIXTURE)
        self.assertAlmostEqual(strict["score"], alone, places=9)

    def test_too_little_to_compare_is_not_judged(self):
        self.assertIsNone(model_voice.judge(None, SPREAD, [4], SETTINGS, FIXTURE))
        self.assertIsNone(model_voice.judge(np.array([1.0]), SPREAD, [4], SETTINGS, FIXTURE))
        only_two_tones = model_voice.Contours({1: FIXTURE.references[1], 2: FIXTURE.references[2]}, {"1": {"midpoint": 0.3, "steepness": 10}, "2": {"midpoint": 0.3, "steepness": 10}})
        self.assertIsNone(model_voice.judge(self.values_of("fall"), SPREAD, [4], SETTINGS, only_two_tones))


class VerdictTests(unittest.TestCase):
    config = {**CONFIG, "model_voice": SETTINGS}

    def test_good_needs_the_score_and_decent_quality(self):
        confidence = CONFIG["confidence"]
        self.assertEqual(model_voice.verdict(0.60, 1.0, True, self.config), "likely_correct")
        self.assertEqual(model_voice.verdict(0.59, 1.0, True, self.config), "insufficient_evidence")
        self.assertEqual(model_voice.verdict(0.95, confidence["likely_correct_quality"] - 0.01, True, self.config), "insufficient_evidence")

    def test_needs_practice_needs_a_low_score_high_quality_and_permission_to_accuse(self):
        confidence = CONFIG["confidence"]
        self.assertEqual(model_voice.verdict(0.30, 1.0, True, self.config), "practice")
        self.assertEqual(model_voice.verdict(0.31, 1.0, True, self.config), "insufficient_evidence")
        self.assertEqual(model_voice.verdict(0.05, 1.0, False, self.config), "insufficient_evidence")
        self.assertEqual(model_voice.verdict(0.05, confidence["practice_quality"] - 0.01, True, self.config), "insufficient_evidence")


def verify_model(char, st_values, *, accuse_ok=True, doubtful_alignment=False, alignment_ok=True, context=None, settings=None, confidence=None, enabled=True):
    config = copy.deepcopy(CONFIG)
    config["model_voice"].update({"enabled": enabled, **(settings or {})})
    config["confidence"].update(confidence or {})
    extraction, clean = make_track(st_values)
    end = len(st_values) * STEP
    syllable = {"char": char, "start": 0.0, "end": end, "phones": [], "alignment_ok": alignment_ok, "nucleus": (0.0, end)}
    context = context or capt_core.expected_tone_contexts([char])[0]
    with mock.patch.object(model_voice, "load_contours", return_value=FIXTURE):
        return tp.verify_syllable(syllable, context, extraction, clean, config, SPEAKER, True, accuse_ok=accuse_ok, doubtful_alignment=doubtful_alignment)


FALL, RISE = np.linspace(6.0, -6.0, 60), np.linspace(-4.0, 4.0, 60)


class PipelineTests(unittest.TestCase):
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
        self.assertFalse(rules["model_voice"]["enabled"])
        self.assertIs(ta.get_config("sentence", "easy", "model_voice"), mode)
        self.assertEqual(mode["tone"]["t4_by_duration"], rules["tone"]["t4_by_duration"])           # the level is kept
        with self.assertRaises(ValueError):
            ta.get_config(scoring_mode="bogus")

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
            midpoint, steepness = contours.sigmoid[tone]
            self.assertTrue(0.0 < midpoint < 1.0 and steepness > 0.0, tone)
        document = json.loads(model_voice.CONTOURS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(document["std_floor"], SETTINGS["std_floor"])
        self.assertEqual(document["references_per_tone"], SETTINGS["references_per_tone"])

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

    def test_a_model_voice_attempt_is_on_neither_board_and_marks_no_lesson(self):
        for number in range(5):
            self.attempt(f"5-1-{number + 1}", good=5, okay=5, mode="rules")                 # 50 each
            self.attempt(f"5-1-{number + 1}", good=10, mode="model_voice")                   # would be 100 each
        board = self.store.leaderboard()
        self.assertEqual(board["top"][0]["points"], 250)
        self.assertEqual(board["accuracy"]["top"][0]["accuracy"], 50.0)
        self.assertEqual({lesson["best"] for lesson in self.store.progress(self.student)["lessons"].values()}, {50})
        self.assertEqual({lesson["attempts"] for lesson in self.store.progress(self.student)["lessons"].values()}, {1})

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

    def test_a_rules_attempt_and_an_old_attempt_have_no_model_voice_view(self):
        self.save("rules", [{"character": "我", "expected_surface_tone": 3, "status": "likely_correct"}])
        self.store.save_attempt(self.student, "5-1-2", "文本", self.audio, "a.webm", {"summary": {"syllables": {"likely_correct": 1}}})
        attempts = self.store.list_admin_dashboard()[0]["attempts"]
        self.assertEqual({a["scoring_mode"] for a in attempts}, {"rules"})
        self.assertEqual({a["model_voice"] for a in attempts}, {None})


class PageTests(unittest.TestCase):
    def test_the_page_has_the_selector_with_both_modes_and_the_model_voice_first_as_the_default(self):
        select = SOURCE[SOURCE.index('<select id="scoringModeSelect"'): SOURCE.index('<p id="scoringModeHint"')]
        self.assertLess(select.index('value="model_voice"'), select.index('value="rules"'))
        # nothing stored (a new browser, a private window, a blocked storage) means the model voice; only an explicit "rules" keeps the rules
        self.assertIn('localStorage.getItem("scoringMode") === "rules" ? "rules" : "model_voice"', SOURCE)
        self.assertIn('catch (error) { return "model_voice"; }', SOURCE)
        self.assertIn('onchange="changeScoringMode()"', select)
        self.assertLess(SOURCE.index('id="practiceMode"'), SOURCE.index('id="scoringModeSelect"'))

    def test_the_choice_is_remembered_sent_with_every_assessment_and_forced_to_rules_for_own_text(self):
        self.assertIn('localStorage.getItem("scoringMode")', SOURCE)
        self.assertIn('localStorage.setItem("scoringMode", scoringMode)', SOURCE)
        self.assertIn('formData.append("scoring_mode", effectiveScoringMode(practiceMode));', SOURCE)
        effective = SOURCE[SOURCE.index("function effectiveScoringMode("): SOURCE.index("function changeScoringMode(")]
        self.assertIn('mode === "custom" ? "rules"', effective)
        sync = SOURCE[SOURCE.index("function syncScoringMode("): SOURCE.index("function hasMandarinCharacters(")]
        self.assertIn("select.disabled = custom ||", sync)
        self.assertIn('scoringModeCustomHint', sync)
        self.assertIn('scoringModeNotCounted', sync)
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
        self.assertIn('scoring_mode: str = Form(default="rules")', SOURCE)
        self.assertEqual(SOURCE.count("level=assessment_level, scoring_mode=scoring_mode,"), 2)


if __name__ == "__main__":
    unittest.main()
