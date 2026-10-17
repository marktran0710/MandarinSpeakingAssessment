import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import capt_core  # noqa: E402
import tone_assessment as ta  # noqa: E402
import tone_pipeline as tp  # noqa: E402
from test_tone_pipeline import CONFIG, SR, STEP, make_track, tone_audio  # noqa: E402

CAL = {"status": "calibrated", "pitch_floor": 90, "pitch_ceiling": 450, "f0_median": 200}


def stretch_track(stretches, total=1.8):
    """A cleaned contour (flat voice) that is voiced exactly inside `stretches` [(start_sec, end_sec)]."""
    st = np.full(int(total / STEP), np.nan)
    for a, b in stretches:
        st[int(round(a / STEP)):int(round(b / STEP))] = 0.0
    return make_track(st)[1]


def syllables_at(*spans, chars="媽罵罵"):
    return [{"char": c, "start": a, "end": b, "phones": [], "alignment_ok": True, "nucleus": None} for c, (a, b) in zip(chars, spans)]


def contexts_for(chars):
    return capt_core.expected_tone_contexts([chars])


# 媽 voiced 0.20-0.60 and fading until 0.66 (inside the stub); the real 罵 at 1.00-1.25 and 1.35-1.50; the aligner gave 罵 a ~90 ms stub and the next 罵 all of it
STUB_CASE = dict(stretches=[(0.20, 0.66), (1.00, 1.25), (1.35, 1.50)], spans=[(0.20, 0.57), (0.57, 0.66), (0.95, 1.60)])


class RepairTests(unittest.TestCase):
    def repair(self, stretches=None, spans=None, chars="媽罵罵", config=CONFIG):
        case = STUB_CASE
        clean = stretch_track(stretches if stretches is not None else case["stretches"])
        return tp.repair_collapsed_syllables(syllables_at(*(spans or case["spans"]), chars=chars), contexts_for(chars), clean, config)

    def test_the_stub_gets_the_first_stretch_and_the_next_syllable_the_second(self):
        out, repaired = self.repair()
        self.assertEqual(repaired, [1, 2])
        stub, nxt = out[1], out[2]
        self.assertEqual(stub["nucleus"], (1.00, 1.25))
        self.assertAlmostEqual(nxt["nucleus"][0], 1.35)
        self.assertAlmostEqual(nxt["nucleus"][1], 1.50)
        self.assertAlmostEqual(stub["end"], 1.30)                         # half-way through the 100 ms gap between the stretches
        self.assertAlmostEqual(nxt["start"], 1.30)
        self.assertAlmostEqual(stub["start"], 0.90)                       # 100 ms for the consonant before the voice
        self.assertEqual(nxt["end"], 1.60)                                # the end the aligner chose stays
        self.assertEqual(out[0], syllables_at(*STUB_CASE["spans"])[0])    # the syllable before is untouched
        for syllable in (stub, nxt):
            self.assertTrue(syllable["alignment_repaired"])
            self.assertEqual(syllable["segment_source"], "repaired_voiced_run")

    def test_the_input_is_not_modified(self):
        syllables = syllables_at(*STUB_CASE["spans"])
        before = copy.deepcopy(syllables)
        tp.repair_collapsed_syllables(syllables, contexts_for("媽罵罵"), stretch_track(STUB_CASE["stretches"]), CONFIG)
        self.assertEqual(syllables, before)

    def test_one_stretch_in_the_span_is_left_alone(self):
        out, repaired = self.repair(stretches=[(0.20, 0.66), (1.00, 1.50)])
        self.assertEqual(repaired, [])
        self.assertEqual(out, syllables_at(*STUB_CASE["spans"]))

    def test_three_stretches_in_the_span_are_left_alone(self):
        self.assertEqual(self.repair(stretches=[(0.20, 0.66), (0.95, 1.10), (1.17, 1.30), (1.38, 1.55)])[1], [])

    def test_a_stretch_shorter_than_a_syllable_does_not_count(self):
        out, repaired = self.repair(stretches=[(0.20, 0.66), (1.00, 1.25), (1.30, 1.34), (1.40, 1.55)])     # 40 ms blip
        self.assertEqual(repaired, [1, 2])
        self.assertEqual(out[1]["nucleus"], (1.00, 1.25))
        self.assertAlmostEqual(out[2]["nucleus"][0], 1.40)
        self.assertAlmostEqual(out[2]["nucleus"][1], 1.55)

    def test_a_syllable_with_voice_of_its_own_is_not_a_stub(self):
        # the voice starts inside the short syllable (0.60 -> 0.66 is a separate stretch from 0.20-0.55)
        self.assertEqual(self.repair(stretches=[(0.20, 0.55), (0.60, 0.66), (1.00, 1.25), (1.35, 1.50)])[1], [])

    def test_a_long_syllable_is_not_a_stub(self):
        self.assertEqual(self.repair(spans=[(0.20, 0.50), (0.50, 0.66), (0.95, 1.60)])[1], [])             # 160 ms

    def test_a_neutral_syllable_is_not_repaired(self):
        self.assertEqual(self.repair(chars="媽的罵")[1], [])

    def test_the_last_syllable_has_nothing_to_borrow_from(self):
        self.assertEqual(self.repair(spans=[(0.20, 0.57), (0.95, 1.60), (1.60, 1.69)])[1], [])

    def test_it_can_be_switched_off(self):
        config = copy.deepcopy(CONFIG)
        config["alignment"]["repair_collapsed"]["enabled"] = False
        out, repaired = self.repair(config=config)
        self.assertEqual(repaired, [])
        self.assertEqual(out, syllables_at(*STUB_CASE["spans"]))

    def test_an_alignment_that_is_not_trusted_is_not_repaired(self):
        syllables = syllables_at(*STUB_CASE["spans"])
        syllables[0]["alignment_ok"] = False
        self.assertEqual(tp.repair_collapsed_syllables(syllables, contexts_for("媽罵罵"), stretch_track(STUB_CASE["stretches"]), CONFIG)[1], [])

    def test_production_has_it_on_for_words_and_sentences(self):
        for config in (ta.get_config(), ta.get_config("sentence"), ta.get_config("sentence", "easy"), ta.get_config(None, "easy")):
            self.assertEqual(config["alignment"]["repair_collapsed"],
                             {"enabled": True, "stub_max_ms": 130, "bridge_ms": 40, "min_run_ms": 70, "onset_ms": 100})


class RepairInTheWholePipelineTests(unittest.TestCase):
    """Three tone-4-ish syllables were spoken with pauses; the aligner squeezed the middle one into a stub behind the first."""

    @staticmethod
    def spoken(second, third):
        audio, spans = tone_audio([([200, 200, 200], 0.40), (second, 0.30), (third, 0.20)], gap_sec=0.35)
        (a0, b0), (a1, b1), (a2, b2) = spans
        stub = (b0 - 0.08, b0 + 0.02)                                    # the fading end of the first syllable, called the second
        return audio, syllables_at((a0, b0 - 0.08), stub, (a1 - 0.10, b2))

    def assess(self, second, third, config):
        audio, syllables = self.spoken(second, third)
        return tp.verify_utterance(audio, SR, "媽罵罵", CAL, alignment=syllables, config=config)

    def test_both_falls_are_found_and_judged(self):
        falls = ([300, 150], [300, 150])
        repaired = self.assess(*falls, CONFIG)
        self.assertEqual(repaired["f0_extraction"]["alignment_repaired"], [1, 2])
        self.assertEqual([s["status"] for s in repaired["syllables"]], ["likely_correct"] * 3)
        off = copy.deepcopy(CONFIG)
        off["alignment"]["repair_collapsed"]["enabled"] = False
        before = self.assess(*falls, off)
        self.assertEqual(before["f0_extraction"]["alignment_repaired"], [])
        self.assertNotEqual(before["syllables"][1]["status"], "likely_correct")             # the stub, judged on a tail that is not its own (sentences: Okay "too short")
        self.assertLess(before["syllables"][1]["duration_ms"], 130)
        self.assertGreater(repaired["syllables"][1]["start_ms"], before["syllables"][1]["start_ms"])      # it moved onto its own voice

    def test_a_repair_never_accuses(self):
        flat = ([200, 200], [200, 200])                                   # a clear mistake in both
        repaired = self.assess(*flat, CONFIG)
        self.assertEqual(repaired["f0_extraction"]["alignment_repaired"], [1, 2])
        self.assertNotIn("practice", [s["status"] for s in repaired["syllables"]])
        self.assertTrue(all("bad_alignment" in s["reasons"] for s in repaired["syllables"][1:]))
        self.assertTrue(all(s["tone_match"] == 0.0 for s in repaired["syllables"][1:]))               # the shape is still measured

    def test_the_chart_window_of_the_phrase_starts_at_its_own_voice(self):
        falls = ([300, 150], [300, 150])
        repaired = self.assess(*falls, CONFIG)
        stub, nxt = repaired["syllables"][1], repaired["syllables"][2]
        self.assertGreater(stub["start_ms"], 800)                         # not the 90 ms behind the first syllable
        self.assertEqual(stub["analysis_segment"]["source"], "repaired_voiced_run")


if __name__ == "__main__":
    unittest.main()
