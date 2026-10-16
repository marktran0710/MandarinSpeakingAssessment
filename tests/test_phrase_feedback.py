import json
import re
import unittest
from pathlib import Path

import tone_assessment as ta

ROOT = Path(__file__).resolve().parent.parent


def syllable(char, status, tone=2, reasons=()):
    return {"character": char, "status": status, "expected_surface_tone": tone, "reasons": list(reasons)}


class PhraseFeedbackTests(unittest.TestCase):
    def test_a_good_word_gets_no_advice(self):
        self.assertEqual(ta.phrase_feedback([syllable("媽", "likely_correct", 1), syllable("麻", "likely_correct")]), [])

    def test_practice_names_the_syllable_and_its_tone_and_only_the_flagged_ones(self):
        group = [syllable("友", "likely_correct", 2), syllable("美", "practice", 3), syllable("好", "insufficient_evidence", 3)]
        self.assertEqual(ta.phrase_feedback(group), [{"type": "tone", "char": "美", "tone": 3}])

    def test_advice_for_several_flagged_syllables_is_capped(self):
        group = [syllable(c, "practice", 4) for c in "一二三四五"]
        self.assertEqual(len(ta.phrase_feedback(group)), ta.MAX_TONE_ADVICE)

    def test_not_enough_evidence_says_how_to_try_again_by_cause(self):
        cases = {"slower": ["syllable_too_short"], "slower2": ["short_analysis_window"],
                 "steady": ["unstable_pitch_track"], "steady2": ["octave_jump_in_segment"], "steady3": ["low_acoustic_quality"],
                 "clearer": ["ambiguous_contour"], "read_exactly": ["bad_alignment"], "again": ["something_new"]}
        for name, reasons in cases.items():
            expected = re.sub(r"\d$", "", name)
            items = ta.phrase_feedback([syllable("你", "insufficient_evidence", 3, reasons)])
            self.assertEqual(items, [{"type": "retry", "code": expected}], name)

    def test_an_alignment_problem_outranks_the_other_causes(self):
        reasons = ["unstable_pitch_track", "bad_alignment", "syllable_too_short"]
        self.assertEqual(ta.phrase_feedback([syllable("你", "insufficient_evidence", 3, reasons)])[0]["code"], "read_exactly")

    def test_a_failed_recording_is_explained_once_not_under_every_word(self):
        group = [syllable("你", "insufficient_evidence", 3, ["audio_quality_failed", "too_quiet"])]
        self.assertEqual(ta.phrase_feedback(group), [])

    def test_a_neutral_syllable_never_gets_tone_advice(self):
        # neutral tone can be likely_correct or insufficient_evidence, never practice
        self.assertEqual(ta.phrase_feedback([syllable("嗎", "likely_correct", 5)]), [])
        items = ta.phrase_feedback([syllable("嗎", "insufficient_evidence", 5, ["low_acoustic_quality"])])
        self.assertEqual(items, [{"type": "retry", "code": "steady"}])

    def test_the_advice_carries_no_measurements(self):
        group = [syllable("美", "practice", 3), syllable("你", "insufficient_evidence", 3, ["ambiguous_contour"])]
        for item in ta.phrase_feedback(group[:1]) + ta.phrase_feedback(group[1:]):
            self.assertLessEqual(set(item), {"type", "char", "tone", "code"})
        self.assertTrue(json.dumps(ta.phrase_feedback(group[:1])))        # plain, storable types


class UiStringsTests(unittest.TestCase):
    """Every language of the UI needs the advice texts (a missing key would show the raw key name)."""
    KEYS = (["adviceTone%d" % n for n in (1, 2, 3, 4)]
            + ["retrySlower", "retrySteady", "retryClearer", "retryReadExactly", "retryAgain"]
            + ["qualityClipped", "qualityTooQuiet", "qualityNoisy", "qualityTooShort", "qualityTooLong",
               "qualityNoSpeech", "qualityNoVoicing"]
            + ["busyWait", "notHeard", "retryNotHeard"])

    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "main.py").read_text(encoding="utf-8")

    def test_each_key_is_defined_once_per_language(self):
        for key in self.KEYS:
            self.assertEqual(len(re.findall(r"\b%s:" % key, self.source)), 5, key)         # en, zh-TW, vi, id, hi

    def test_every_retry_code_the_backend_can_send_has_a_ui_key(self):
        codes = {code for code, _ in ta._RETRY_BY_REASON} | {"again", "not_heard"}
        mapping = dict(re.findall(r"(\w+): \"(retry\w+)\"", self.source.split("const retryKeys = {")[1].split("}")[0]))
        self.assertEqual(set(mapping), codes)
        for ui_key in mapping.values():
            self.assertIn(ui_key, self.KEYS)

    def test_advice_texts_have_no_technical_terms(self):
        block = "\n".join(re.findall(r"^\s+(?:advice|retry|quality)\w+: \"[^\"]*\",$", self.source, re.M))
        self.assertGreater(len(block), 500)
        for term in ("semitone", "F0", "Hz", "Praat", "pitch track", "半音", "基頻", "st)", "octave"):
            self.assertNotIn(term, block, term)

    def test_the_card_template_no_longer_shows_a_detected_line(self):
        self.assertNotIn('t("detectedSequence")', self.source)
        self.assertNotIn("diagnosticNotNeeded\")", self.source)


class PayloadTests(unittest.TestCase):
    def test_every_phrase_in_the_payload_carries_feedback_items(self):
        import numpy as np
        from tests.test_tone_pipeline import CONFIG, SR, tone_audio
        import tone_pipeline as tp

        audio, spans = tone_audio([([200, 200, 200, 200], 0.35), ([280, 200, 120, 90], 0.35)])
        syllables = [{"char": c, "start": s, "end": e, "phones": [], "nucleus": (s, e), "alignment_ok": True}
                     for c, (s, e) in zip("媽罵", spans)]
        result = tp.verify_utterance(audio, SR, "媽罵", {"status": "calibrated", "pitch_floor": 90.0, "pitch_ceiling": 450.0,
                                                          "f0_median": 200.0}, alignment=syllables, config=CONFIG)
        payload = ta.build_payload(result, groups=[("媽", 0, 1), ("罵", 1, 2)], word_annotation_status="matched",
                                   boundaries_from="test", practice_mode="custom", sample_id="custom", target_script="媽罵",
                                   audio_quality={"passed": True, "reasons": []})
        for phrase in payload["phrases"]:
            self.assertIsInstance(phrase["feedback_items"], list)
        json.dumps(payload)                                                 # still storable as JSON


class NoCorruptedTextTests(unittest.TestCase):
    """A past save turned some Vietnamese and Chinese UI text into question marks; keep that from coming back."""

    def test_no_word_in_the_pages_is_broken_by_question_marks(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        source = source.replace("/api/pinyin?text=", "")                      # a URL query is not a broken word
        broken = re.findall(r'[^\W\d_]\?[^\W\d_]|"\?{2,}', source)           # a letter glued to ?, or a text that is all ??
        self.assertEqual(broken, [])

    def test_the_script_note_is_correct_in_every_language(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        notes = re.findall(r'scriptProvidedOnly: "([^"]*)"', source)
        self.assertEqual(len(notes), 5)
        for note in notes:
            self.assertNotIn("?", note)
            self.assertNotIn("ASR", note)                     # the old English text claimed alignment is skipped


class StatusLabelTests(unittest.TestCase):
    """Learners see three levels (green / yellow / red), never 'not enough evidence'."""

    def test_the_three_levels_have_plain_names_in_every_language(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        lines = re.findall(r'likelyCorrect: "([^"]*)", practice: "([^"]*)", uncertain: "([^"]*)", insufficientEvidence: "([^"]*)"', source)
        self.assertEqual(lines, [("Good", "Needs practice", "Okay", "Okay"),
                                 ("良好", "需要練習", "還可以", "還可以"),
                                 ("Tốt", "Cần luyện", "Tạm ổn", "Tạm ổn")])


class TopicsModeNameTests(unittest.TestCase):
    def test_the_one_item_at_a_time_mode_is_called_topics_in_every_language(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertEqual(re.findall(r'chapter5WordMode: "([^"]*)"', source), ["Topics", "主題", "Chủ đề", "Topik", "विषय"])
        self.assertIn('<span data-i18n="chapter5WordMode">Topics</span>', source)
        self.assertNotIn("one item at a time", source)


class BusyOverlayTests(unittest.TestCase):
    def test_the_slow_steps_show_a_full_screen_layer_that_cannot_be_dismissed(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn('<dialog id="busyOverlay"', source)
        self.assertIn("overlay.showModal()", source)
        self.assertIn('addEventListener("cancel", event => event.preventDefault())', source)
        self.assertIn("setBusy(isLoading, message);", source)                       # the assessment status
        self.assertIn('setBusy(key === "calibrationProcessing"', source)            # the voice calibration
        self.assertGreaterEqual(source.count("setBusy(false);"), 2)                 # always cleared in a finally block


class TopicsCardTests(unittest.TestCase):
    def test_the_card_texts_are_short_in_every_language(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertEqual(re.findall(r'chapter5Meaning: "([^"]*)"', source), ["{meaning}"] * 5)
        self.assertEqual(re.findall(r'chapter5Progress: "([^"]*)"', source), ["{current} / {total}"] * 5)
        for text in re.findall(r'chapter5ReferenceAudio: "([^"]*)"', source):
            self.assertLess(len(text), 60)

    def test_my_own_text_shows_the_script_with_pinyin_like_topics_and_uses_a_normal_font(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertNotIn("#scriptInput:not([readonly]) ~ #transcriptDisplay", source)       # the preview is not hidden
        self.assertIn("#transcriptDisplay:empty { display: none; }", source)               # only when there is nothing to show
        self.assertIn("/api/pinyin?text=", source)
        self.assertIn('document.getElementById("pinyinToggle").hidden = false;', source)
        self.assertIn("#scriptInput { font-family: inherit;", source)


class NotHeardTests(unittest.TestCase):
    SILENT = [syllable("你", "insufficient_evidence", 3, ["too_few_voiced_frames"]),
              syllable("好", "insufficient_evidence", 3, ["too_few_voiced_frames"])]

    def test_a_phrase_with_no_measurable_voice_is_not_heard_and_asks_for_a_retry(self):
        self.assertTrue(ta.heard_nothing(self.SILENT))
        self.assertEqual(ta.phrase_feedback(self.SILENT), [{"type": "retry", "code": "not_heard"}])

    def test_a_rejected_recording_is_not_heard_and_is_explained_once_for_the_whole_recording(self):
        rejected = [syllable("你", "insufficient_evidence", 3, ["audio_quality_failed", "too_quiet"])]
        self.assertTrue(ta.heard_nothing(rejected))
        self.assertEqual(ta.phrase_feedback(rejected), [])

    def test_some_voice_somewhere_in_the_phrase_means_it_was_heard(self):
        mixed = [syllable("你", "likely_correct", 3), syllable("好", "insufficient_evidence", 3, ["too_few_voiced_frames"])]
        self.assertFalse(ta.heard_nothing(mixed))
        self.assertEqual(ta.phrase_feedback(mixed), [{"type": "retry", "code": "slower"}])

    def test_other_reasons_are_not_silence(self):
        for reason in ("ambiguous_contour", "unstable_pitch_track", "bad_alignment", "syllable_too_short"):
            self.assertFalse(ta.heard_nothing([syllable("你", "insufficient_evidence", 3, [reason])]), reason)
        self.assertFalse(ta.heard_nothing([]))

    def test_it_never_hides_a_practice_or_a_good_syllable(self):
        self.assertFalse(ta.heard_nothing([syllable("你", "practice", 3, ["too_few_voiced_frames"])]))
        self.assertFalse(ta.heard_nothing([syllable("你", "likely_correct", 3)]))


class NotHeardLabelTests(unittest.TestCase):
    def test_it_has_a_name_and_a_style_in_every_language(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertEqual(re.findall(r'notHeard: "([^"]*)"', source), ["Not heard", "沒聽到", "Chưa nghe rõ", "Tidak terdengar", "सुनाई नहीं दिया"])
        self.assertIn('not_heard: t("notHeard"),', source)
        self.assertIn('.word-card.not-heard {', source)
        self.assertIn('status === "not_heard" ? "not-heard"', source)


class OptionalPartTests(unittest.TestCase):
    """Topics items written like 裡(面): the recording says 裡面 and both forms are accepted, so show the full form."""

    @staticmethod
    def cells(workbook_text, script, pinyin=""):
        import tone_assessment as ta
        return ta.chapter5_display_cells({"traditional_chinese": workbook_text, "script": script, "pinyin": pinyin})

    def test_the_optional_part_is_shown_with_its_own_pinyin_and_marked(self):
        cells = self.cells("裡(面)", "裡")
        self.assertEqual([(c["char"], c["pinyin"], c["optional"]) for c in cells], [("裡", "lǐ", False), ("面", "miàn", True)])
        cells = self.cells("電視(機)", "電視")
        self.assertEqual([(c["char"], c["optional"]) for c in cells], [("電", False), ("視", False), ("機", True)])

    def test_brackets_never_become_cells_of_their_own(self):
        cells = self.cells("沒（有）空", "沒（有）空", "méi(yǒu) kòng")
        self.assertEqual([(c["char"], c["pinyin"], c["optional"]) for c in cells],
                         [("沒", "méi", False), ("有", "yǒu", True), ("空", "kòng", False)])      # kòng, not kōng

    def test_items_without_an_optional_part_are_unchanged(self):
        self.assertIsNone(self.cells("錢包", "錢包"))
        self.assertIsNone(self.cells("哪裡/哪兒", "哪裡"))                                       # slash alternatives are not "optional"

    def test_every_catalog_item_with_an_optional_part_gets_cells_and_every_other_item_none(self):
        import json
        import tone_assessment as ta
        items = json.loads((ROOT / "chapter5_word_practice" / "items.json").read_text(encoding="utf-8"))["items"]
        optional = [i for i in items if ta.split_optional(i["traditional_chinese"])]
        self.assertGreaterEqual(len(optional), 8)
        for item in optional:
            cells = ta.chapter5_display_cells(item)
            self.assertTrue(cells and any(c["optional"] for c in cells), item["word_key"])
            self.assertTrue(all(c["pinyin"] and c["pinyin"] not in "()（）" for c in cells), item["word_key"])
            variants = {ta.chapter5_candidates(item, item["script"], lambda t: t)[i]["script_variant"] for i in range(len(ta.chapter5_candidates(item, item["script"], lambda t: t)))}
            self.assertIn("".join(c["char"] for c in cells), variants, item["word_key"])           # the shown form is one that is scored

    def test_the_page_renders_cells_and_explains_the_optional_part_in_every_language(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn('"display_cells": tone_assessment.chapter5_display_cells(item)', source)
        self.assertIn('renderTranscript(cells ? { cells } : { script: item.script, pinyin: item.character_pinyin || [] });', source)
        self.assertIn('id="chapter5OptionalHint"', source)
        self.assertIn(".transcript-char.optional {", source)
        hints = re.findall(r'chapter5OptionalHint: "([^"]*)"', source)
        self.assertEqual(len(hints), 5)
        for hint in hints:
            self.assertIn("{short}", hint)
            self.assertIn("{full}", hint)


class StatusIconTests(unittest.TestCase):
    def test_okay_is_a_half_filled_circle_not_a_question_mark(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertNotIn('"not_heard" ? "&#8212;" : "?"', source)                    # the old "?" icon
        self.assertIn('const iconClass = icon ? "status-icon" : "status-icon half";', source)
        self.assertIn(".status-icon.half { background: linear-gradient(90deg, currentColor 50%, transparent 50%); }", source)
        for icon in ("&#10003;", "&#8635;", "&#8212;"):                              # the other three keep their symbols
            self.assertIn(icon, source)


class CalibrationDialogTests(unittest.TestCase):
    def test_skipping_is_not_offered_up_front_but_nobody_can_get_stuck(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("laterButton.hidden = !(calibrated || calibrationHadError);", source)
        self.assertIn("calibrationHadError = true;", source)
        # state used by the first render must be declared before that render (a late `let` breaks the whole page)
        self.assertLess(source.index("let calibrationHadError = false;"), source.index("applyLanguage(localStorage.getItem"))

    def test_all_dialog_and_recorder_state_is_declared_before_the_first_render(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        first_render = source.index("applyLanguage(localStorage.getItem")
        for declaration in ("let calibrationOutcome", "let calibrationCanStop", "let calibrationClock", "let stopCalibrationMeter",
                            "let practiceIndicatorTimer", "let stopPracticeMeter", "const CALIBRATION_MIN_MS", "const CALIBRATION_MAX_MS"):
            self.assertIn(declaration, source)
            self.assertLess(source.index(declaration), first_render, declaration)

    def test_every_reason_the_server_can_give_has_its_own_translated_tip(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        reasons = set(re.findall(r'CalibrationError\(\s*"(\w+)"', (ROOT / "tone_pipeline.py").read_text(encoding="utf-8")))
        self.assertGreaterEqual(len(reasons), 6)
        mapping = dict(re.findall(r'(\w+): "(\w+)",', source[source.index("function calibrationReasonKey"):][:700]))
        # too_long cannot happen from this page (it stops at 10 s), so it may use the generic message
        for reason in reasons - {"too_long"}:
            self.assertIn(reason, mapping, reason)
        keys = set(mapping.values()) | {"calibrationFailed"}
        UiLanguagesTests.setUpClass()
        for code in UiLanguagesTests.CODES:
            for key in keys:
                self.assertIn(key, UiLanguagesTests.tables[code], (code, key))
        self.assertIn('"reason": exc.reason', source)                         # the server tells the page why it said no

    def test_the_page_lets_the_learner_stop_only_inside_the_servers_duration_limits(self):
        import yaml
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        calibration = yaml.safe_load((ROOT / "tone_v1_config.yaml").read_text(encoding="utf-8"))["calibration"]
        minimum = int(re.search(r"const CALIBRATION_MIN_MS = (\d+);", source).group(1)) / 1000
        maximum = int(re.search(r"const CALIBRATION_MAX_MS = (\d+);", source).group(1)) / 1000
        self.assertGreaterEqual(minimum, calibration["min_duration_sec"])     # "I'm done" never produces a too-short sample
        self.assertLessEqual(maximum, calibration["max_duration_sec"])        # the automatic stop never produces a too-long one

    def test_the_voice_setup_that_opens_by_itself_always_has_a_visible_way_out(self):
        # it covers the whole page for a new speaker; with only a hidden "skip" and Esc, Topics etc. could not be clicked
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn('id="voiceCalibrationCloseBtn"', source)
        self.assertIn('onclick="deferVoiceCalibration()" aria-label="Close" data-i18n-aria-label="calClose"', source)
        self.assertIn("if (outside) deferVoiceCalibration();", source)                       # a click on the dimmed area
        self.assertIn('getElementById("voiceCalibrationCloseBtn").disabled = isVoiceCalibrationRecording', source)
        self.assertIn("laterButton.hidden = !(calibrated || calibrationHadError);", source)  # "skip" itself is still not offered up front

    def test_the_practice_recorder_shows_a_clock_and_level_and_always_clears_them(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn('id="recordingIndicator"', source)
        self.assertIn("showRecordingIndicator(stream);", source)
        # hidden again on every way a recording can end: stop button, recorder stop event, microphone failure
        self.assertGreaterEqual(source.count("hideRecordingIndicator();"), 4)


class UiLanguagesTests(unittest.TestCase):
    CODES = ("en", "zh-TW", "vi", "id", "hi")

    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "main.py").read_text(encoding="utf-8")
        head = cls.source.index("const LANGUAGE_TEXT = {")
        end = cls.source.index("let currentLanguage")
        body = cls.source[head:end]
        starts = {code: re.search(r'\n {16}(?:%s|"%s"): \{' % (code, code), body).start() for code in cls.CODES}
        order = sorted(starts, key=starts.get)
        cls.tables = {}
        for code, nxt in zip(order, order[1:] + [None]):
            chunk = body[starts[code]: starts[nxt] if nxt else len(body)]
            cls.tables[code] = dict(re.findall(r'(\w+): "((?:[^"\\]|\\.)*)"', chunk))

    def test_every_language_has_every_text(self):
        keys = set(self.tables["en"])
        self.assertGreater(len(keys), 200)
        for code in self.CODES:
            self.assertEqual(set(self.tables[code]), keys, code)

    def test_placeholders_survive_translation(self):
        for code in self.CODES:
            for key, value in self.tables[code].items():
                self.assertEqual(sorted(re.findall(r"\{(\w+)\}", value)), sorted(re.findall(r"\{(\w+)\}", self.tables["en"][key])), (code, key))

    def test_the_new_languages_are_really_translated_and_in_their_own_script(self):
        neutral = {"chapter5Progress"}                                       # numbers only: the same in every language
        for code in ("id", "hi"):
            untranslated = [k for k, v in self.tables[code].items()
                            if v == self.tables["en"][k] and len(v) > 12 and k not in neutral]
            self.assertEqual(untranslated, [], code)
        devanagari = sum(bool(re.search(r"[\u0900-\u097F]", v)) for v in self.tables["hi"].values())
        self.assertGreater(devanagari, 190)
        self.assertEqual(self.tables["hi"]["likelyCorrect"], "बढ़िया")
        self.assertEqual(self.tables["id"]["practice"], "Perlu latihan")

    def test_the_selector_lists_the_new_languages_by_their_own_names(self):
        self.assertIn('<option value="id">Bahasa Indonesia</option>', self.source)
        self.assertIn('<option value="hi">हिन्दी</option>', self.source)

    def test_one_font_stack_covers_every_script_the_page_shows(self):
        stack = re.search(r"body \{ font-family: ([^;]*);", self.source).group(1)
        for family in ("Segoe UI", "Nirmala UI", "Noto Sans Devanagari", "Microsoft JhengHei", "PingFang TC", "Noto Sans TC"):
            self.assertIn(family, stack)
        self.assertIn("button, input, select, textarea { font-family: inherit; }", self.source)

    def test_topic_labels_exist_in_every_language_and_chinese_uses_the_zh_key(self):
        import json
        data = json.loads((ROOT / "chapter5_word_practice" / "categories.json").read_text(encoding="utf-8"))
        for category in data["categories"]:
            for entry in (category, *category.get("subcategories", [])):
                self.assertTrue({"en", "zh", "vi", "id", "hi"} <= set(entry["labels"]), entry["id"])
        self.assertIn('currentLanguage === "zh-TW" ? "zh" : currentLanguage', self.source)


if __name__ == "__main__":
    unittest.main()
