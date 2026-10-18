import json
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "main.py").read_text(encoding="utf-8")
NODE = shutil.which("node")


def function_source(name):
    start = SOURCE.index(f"function {name}(")
    return SOURCE[start:SOURCE.index("\n            }\n", start) + len("\n            }\n")]


def samples():
    """The shape of /api/sample-lessons: lessons 5-8, a few stories and scenes each, listed in a scrambled order."""
    rows = [{"id": f"{lesson}-{story}-{scene}", "lesson": lesson, "story": story, "scene": scene}
            for lesson in (5, 6, 7, 8) for story in (1, 2, 3) for scene in (1, 2, 3, 4)]
    return rows[::-1]


def run(expression):
    script = f"{function_source('chapterStartSamples')}\n{function_source('startSampleFor')}\nconst samples = {json.dumps(samples())};\nconsole.log(JSON.stringify({expression}));"
    return json.loads(subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30, check=True).stdout)


@unittest.skipUnless(NODE, "node is needed to run the page's JavaScript")
class StartSampleTests(unittest.TestCase):
    def test_the_starts_are_the_first_scene_of_the_first_story_of_each_chapter(self):
        self.assertEqual([s["id"] for s in run("chapterStartSamples(samples)")], ["5-1-1", "6-1-1", "7-1-1", "8-1-1"])

    def test_the_same_person_always_starts_at_the_same_place(self):
        for student in ("481dd328468d4612ac8974cfc3533d91", "b836173d54df497db9d9900a7b7c359e"):
            first = run(f"startSampleFor(samples, {json.dumps(student)})")
            self.assertEqual(first, run(f"startSampleFor(samples, {json.dumps(student)})"))
            self.assertEqual(first["scene"], 1)
            self.assertEqual(first["story"], 1)

    def test_different_people_are_spread_over_all_the_chapters(self):
        ids = [f"{n:032x}" for n in range(0, 4000, 37)]
        starts = run("[" + ",".join(f"startSampleFor(samples, {json.dumps(i)}).id" for i in ids) + "]")
        counts = {start: starts.count(start) for start in set(starts)}
        self.assertEqual(set(counts), {"5-1-1", "6-1-1", "7-1-1", "8-1-1"})
        self.assertGreater(min(counts.values()), len(ids) * 0.12)                 # not lopsided (about a quarter each)

    def test_a_visitor_without_a_profile_gets_one_of_the_starts_and_an_empty_list_none(self):
        self.assertIn(run("startSampleFor(samples, null).id"), ["5-1-1", "6-1-1", "7-1-1", "8-1-1"])
        self.assertIsNone(run("startSampleFor([], 'abc')"))

    def test_one_chapter_only_always_starts_there(self):
        self.assertEqual(run("startSampleFor(samples.filter(s => s.lesson === 6), 'anyone').id"), "6-1-1")


class WiringTests(unittest.TestCase):
    def test_the_first_choice_uses_the_start_and_picking_a_sample_stops_the_page_choosing(self):
        self.assertIn("const start = startSampleFor(sampleLessons, activeStudent && activeStudent.id);\n                    if (start) select.value = start.id;", SOURCE)
        self.assertIn('onchange="sampleTouched = true; selectSample()"', SOURCE)
        self.assertIn('if (profile && !sampleTouched && sampleLessons.length && selectedPracticeMode() === "book") applyStartSample();', SOURCE)

    def test_the_board_counts_in_sentences(self):
        for needle in ('leaderboardLessons: "{n} sentences"', 'leaderboardLessons: "{n} câu"', 'leaderboardLessons: "{n} 句"',
                       'leaderboardLessons: "{n} kalimat"', 'leaderboardLessons: "{n} वाक्य"'):
            self.assertIn(needle, SOURCE)


if __name__ == "__main__":
    unittest.main()
