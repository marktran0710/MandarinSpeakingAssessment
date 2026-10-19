import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import tone_pipeline as tp  # noqa: E402
from test_tone_pipeline import CONFIG  # noqa: E402


def tiers(words, phones):
    return {"words": {"entries": [list(w) for w in words]}, "phones": {"entries": [list(p) for p in phones]}}


# 末 as the aligner gave it in a slow reading (Hau tran, 2026-10-18): m 70 ms, "w" 200 ms (it took the voiced start of the vowel), o˥˩ 139 ms
MO = tiers([(4.02, 4.43, "末")], [(4.02, 4.09, "m"), (4.09, 4.29, "w"), (4.29, 4.43, "o˥˩")])
YAO = tiers([(4.95, 5.27, "要")], [(4.95, 5.10, "j"), (5.10, 5.27, "aw˥˩")])


class MedialGlideTests(unittest.TestCase):
    def nucleus(self, tier, char, **align):
        config = copy.deepcopy(CONFIG)
        config["alignment"].update(align)
        return tp.syllables_from_tiers(tier, [char], config)[0]["nucleus"]

    def test_the_window_that_is_judged_starts_at_a_medial_glide(self):
        self.assertEqual(self.nucleus(MO, "末", nucleus_includes_medial_glide=True), (4.09, 4.43))

    def test_it_can_be_switched_off(self):
        self.assertEqual(self.nucleus(MO, "末", nucleus_includes_medial_glide=False), (4.29, 4.43))

    def test_a_glide_that_is_the_whole_onset_stays_an_onset(self):
        self.assertEqual(self.nucleus(YAO, "要", nucleus_includes_medial_glide=True), (5.10, 5.27))

    def test_the_shipped_config_includes_the_glide(self):
        self.assertTrue(tp.load_config()["alignment"]["nucleus_includes_medial_glide"])


if __name__ == "__main__":
    unittest.main()
