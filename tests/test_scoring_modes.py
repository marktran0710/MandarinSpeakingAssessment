import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import capt_core  # noqa: E402
import tone_assessment as ta  # noqa: E402
import tone_pipeline as tp  # noqa: E402
from test_tone_pipeline import CONFIG, STEP, make_track  # noqa: E402


def features(**changes):
    base = {"early_to_late_delta": 0.0, "range_core_ST": 1.0, "rise_after_min": 0.0, "min_position": 0.5, "fall_before_min": 0.0,
            "start_ST": 0.0, "end_ST": 0.0, "syllable_ms": 250.0}
    base.update(changes)
    return base


def with_scoring(mode):
    config = copy.deepcopy(CONFIG)
    config["scoring"] = {"combine_features": mode}
    return config


class CombineFactorsTests(unittest.TestCase):
    def test_product_is_the_default_and_the_geometric_mean_is_the_nth_root(self):
        self.assertAlmostEqual(tp.combine_factors([0.85] * 3, CONFIG), 0.85 ** 3)
        self.assertAlmostEqual(tp.combine_factors([0.85] * 3, with_scoring("geometric_mean")), 0.85)
        self.assertAlmostEqual(tp.combine_factors([0.5, 1.0], with_scoring("geometric_mean")), 0.5 ** 0.5)
        self.assertAlmostEqual(tp.combine_factors([0.5, 1.0], with_scoring("product")), 0.5)

    def test_one_factor_at_zero_still_gives_zero_in_both_modes(self):
        for mode in ("product", "geometric_mean"):
            self.assertEqual(tp.combine_factors([1.0, 0.0, 1.0], with_scoring(mode)), 0.0)

    def test_three_good_factors_reach_good_only_with_the_geometric_mean(self):
        # each of the three factors of a full third tone is 0.85: fall 1.82 and rise 1.82 (thr 2.0), low point 15% of the margin outside
        f = features(fall_before_min=1.82, rise_after_min=1.82, min_position=0.235)
        product = tp.tone_scores(f, False, with_scoring("product"))[3]
        geometric = tp.tone_scores(f, False, with_scoring("geometric_mean"))[3]
        self.assertAlmostEqual(product, 0.85 ** 3, places=2)
        self.assertLess(product, 0.70)                                   # not Good today
        self.assertAlmostEqual(geometric, 0.85, places=2)
        self.assertGreaterEqual(geometric, 0.70)

    def test_a_clearly_wrong_contour_is_still_a_zero(self):
        flat = features()                                                # no fall, no rise: not a tone 3 or a tone 2 or a tone 4
        for mode in ("product", "geometric_mean"):
            scores = tp.tone_scores(flat, False, with_scoring(mode))
            for tone in (2, 3, 4):
                self.assertEqual(scores[tone], 0.0, (mode, tone))

    def test_the_two_factor_tones_use_the_square_root(self):
        f = features(early_to_late_delta=3.0, range_core_ST=1.0)         # T1: |delta| 3.0 vs 2.0 -> 0.5; range fine -> 1.0
        self.assertAlmostEqual(tp.tone_scores(f, False, with_scoring("product"))[1], 0.5)
        self.assertAlmostEqual(tp.tone_scores(f, False, with_scoring("geometric_mean"))[1], 0.5 ** 0.5)

    def test_the_boundary_search_always_uses_the_plain_product(self):
        seen = []
        original = tp.tone_scores

        def spy(feats, allow_half, config):
            seen.append((config.get("scoring") or {}).get("combine_features"))
            return original(feats, allow_half, config)

        tones = capt_core.expected_tone_contexts(["媽罵"])
        extraction, clean = make_track(np.r_[np.zeros(30), np.linspace(2, -2, 30), np.full(30, -2.0)])
        late = 42 * STEP
        syllables = [{"char": "媽", "start": 0.0, "end": late, "phones": [], "nucleus": (0.0, late)},
                     {"char": "罵", "start": late, "end": 90 * STEP, "phones": [], "nucleus": (late, 90 * STEP)}]
        with mock.patch.object(tp, "tone_scores", spy):
            tp.search_boundaries(syllables, tones, clean, with_scoring("geometric_mean"))
        self.assertTrue(seen)
        self.assertEqual(set(seen), {"product"})

    def test_production_still_uses_the_product(self):
        self.assertEqual(ta.get_config()["scoring"]["combine_features"], "product")
        self.assertEqual(ta.get_config("sentence")["scoring"]["combine_features"], "product")


class BeforeToneFourRelaxationTests(unittest.TestCase):
    RULE = {"short_ms": 180, "medium_ms": 300, "min_fall_short_st": 1.5, "min_fall_medium_st": 2.0, "min_fall_long_st": 2.5}

    def tone_config(self, **before):
        config = copy.deepcopy(CONFIG["tone"])
        config["t4_by_duration"] = self.RULE
        config["t4_before_t4"] = before
        return config

    def test_the_first_tone_four_needs_half_a_semitone_less_than_its_length_rule(self):
        tone = self.tone_config(relaxation_st=0.5)
        self.assertEqual(tp.t4_min_fall(features(syllable_ms=220.0, followed_by_t4=True), tone), 1.5)     # 220 ms: 2.0 -> 1.5
        self.assertEqual(tp.t4_min_fall(features(syllable_ms=220.0), tone), 2.0)                          # the second one: normal
        self.assertEqual(tp.t4_min_fall(features(syllable_ms=350.0, followed_by_t4=True), tone), 2.0)     # 2.5 -> 2.0
        self.assertEqual(tp.t4_min_fall(features(syllable_ms=120.0, followed_by_t4=True), tone), 1.0)     # 1.5 -> 1.0

    def test_the_requirement_never_drops_below_half_a_semitone(self):
        tone = self.tone_config(relaxation_st=5.0)
        self.assertEqual(tp.t4_min_fall(features(syllable_ms=120.0, followed_by_t4=True), tone), 0.5)

    def test_the_older_absolute_form_still_works(self):
        tone = self.tone_config(min_fall_st=2.0)
        self.assertEqual(tp.t4_min_fall(features(syllable_ms=350.0, followed_by_t4=True), tone), 2.0)


if __name__ == "__main__":
    unittest.main()
