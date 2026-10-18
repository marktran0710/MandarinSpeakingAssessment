import asyncio
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main  # noqa: E402
from practice_store import PracticeStore  # noqa: E402

SOURCE = (ROOT / "main.py").read_text(encoding="utf-8")


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.store = PracticeStore(root / "practice.sqlite3", root / "recordings")
        self.audio = root / "a.webm"
        self.audio.write_bytes(b"x")

    def attempt(self, name, sample_id, good, okay=0, practice=0, script="文本"):
        student = self.store.create_or_get_student(name, "foreigner")
        self.store.save_attempt(student["id"], sample_id, script, self.audio, "a.webm",
                                {"summary": {"syllables": {"likely_correct": good, "insufficient_evidence": okay, "practice": practice}}})
        return student["id"]


class AccuracyBoardPoolsEveryAttemptTests(StoreCase):
    def lessons_twice(self, name, first, second):
        """Five lessons, each read twice: `first` and `second` are (good, okay, practice)."""
        student = None
        for number in range(5):
            self.attempt(name, f"5-1-{number + 1}", *first)
            student = self.attempt(name, f"5-1-{number + 1}", *second)
        return student

    def test_accuracy_counts_every_attempt_in_syllables_and_an_okay_is_half(self):
        self.lessons_twice("Mei", (5, 0, 5), (10, 0, 0))                    # 5 + 10 Good out of 20 syllables: 75, although the best tries are perfect
        board = self.store.leaderboard()
        self.assertEqual(board["accuracy"]["top"], [{"rank": 1, "name": "Mei", "accuracy": 75.0, "lessons": 5, "is_me": False}])
        self.assertEqual(board["top"][0]["points"], 500)                     # the points board still counts the best attempt of each lesson

    def test_a_longer_sentence_weighs_more_than_a_short_one(self):
        for number in range(4):
            self.attempt("Lin", f"5-1-{number + 1}", 1, 0, 0)               # four one-syllable lessons, all Good
        self.attempt("Lin", "5-1-5", 5, 0, 15)                              # one 20-syllable lesson, a quarter Good
        accuracy = self.store.leaderboard()["accuracy"]["top"][0]["accuracy"]
        self.assertEqual(accuracy, round(100.0 * 9 / 24, 1))                 # 37.5, not the 85 a plain average of the lessons would give

    def test_own_text_and_attempts_without_a_good_syllable_do_not_count(self):
        for number in range(5):
            self.attempt("Mei", f"5-1-{number + 1}", 8, 0, 2)               # 80
        self.attempt("Mei", "custom", 1, 0, 9)
        self.attempt("Mei", "5-1-1", 0, 10, 0)                               # a failed alignment: all Okay
        self.assertEqual(self.store.leaderboard()["accuracy"]["top"][0]["accuracy"], 80.0)


class ProgressTests(StoreCase):
    def test_a_lesson_knows_its_attempts_best_and_latest_points(self):
        learner = self.attempt("Mei", "5-1-1", 5, 0, 5)                     # 50
        self.attempt("Mei", "5-1-1", 9, 0, 1)                               # 90
        self.attempt("Mei", "5-1-1", 6, 2, 2)                               # 70: the latest, not the best
        entry = self.store.progress(learner)["lessons"]["5-1-1"]
        self.assertEqual((entry["attempts"], entry["best"], entry["last"]), (3, 90, 70))
        self.assertTrue(entry["last_at"])

    def test_different_lessons_have_their_own_latest_result(self):
        learner = self.attempt("Mei", "5-1-1", 9, 0, 1)                     # 90
        self.attempt("Mei", "5-1-2", 5, 0, 5)                               # 50
        self.attempt("Mei", "C5-5-1-I1-W001", 2, 0, 0, script="你好")        # 100: a Chapter 5 word is a lesson too
        lessons = self.store.progress(learner)["lessons"]
        self.assertEqual({key: value["last"] for key, value in lessons.items()}, {"5-1-1": 90, "5-1-2": 50, "C5-5-1-I1-W001": 100})

    def test_own_text_is_not_a_lesson(self):
        learner = self.attempt("Mei", "custom", 4, 0, 0, script="我愛你")
        self.assertEqual(self.store.progress(learner), {"lessons": {}})

    def test_an_attempt_without_a_good_syllable_is_not_a_result_and_leaves_the_earlier_one_standing(self):
        learner = self.attempt("Mei", "5-1-1", 8, 0, 2)                     # 80
        self.attempt("Mei", "5-1-1", 0, 8, 0)                               # silence / a failed alignment: all Okay
        entry = self.store.progress(learner)["lessons"]["5-1-1"]
        self.assertEqual((entry["attempts"], entry["last"]), (1, 80))
        self.assertEqual(self.store.progress(self.attempt("Lin", "5-1-2", 0, 8, 0))["lessons"], {})

    def test_a_learner_sees_only_their_own_attempts(self):
        mine = self.attempt("Mei", "5-1-1", 5, 0, 5)
        self.attempt("Lin", "5-1-2", 10)
        self.assertEqual(list(self.store.progress(mine)["lessons"]), ["5-1-1"])
        self.assertEqual(self.store.progress("nobody"), {"lessons": {}})

    def test_the_endpoint_answers_for_a_learner_and_for_nobody(self):
        mine = self.attempt("Mei", "5-1-1", 5, 0, 5)
        with mock.patch.object(main, "practice_store", self.store):
            known = asyncio.run(main.get_progress(student_id=mine))
            anonymous = asyncio.run(main.get_progress())
        self.assertEqual(known["lessons"]["5-1-1"]["last"], 50)
        self.assertEqual(anonymous, {"lessons": {}})


class PageTests(unittest.TestCase):
    KEYS = ("resultPrevious", "resultNext", "itemProgress", "itemNotPracticed")

    def test_every_language_has_the_new_texts(self):
        for key in self.KEYS:
            self.assertEqual(len(re.findall(r"\b%s: " % key, SOURCE)), 5, key)

    def test_there_is_no_list_of_recent_results_any_more(self):
        for gone in ("recentCard", "recentList", "recentTitle", "renderRecent"):
            self.assertNotIn(gone, SOURCE, gone)

    def test_the_result_card_has_a_previous_and_a_next_button_and_no_script_line(self):
        card = SOURCE[SOURCE.index('<div id="resultCard"'): SOURCE.index('<div id="pronunciationResults">')]
        self.assertIn('id="prevItemBtn"', card)
        self.assertIn('onclick="goToPreviousItem()"', card)
        self.assertIn('id="nextItemBtn"', card)
        self.assertIn('onclick="goToNextItem()"', card)
        for gone in ("retryBtn", "Read again", "targetScriptDisplay"):
            self.assertNotIn(gone, SOURCE, gone)

    def test_the_lists_are_marked_with_the_latest_result_and_the_progress_is_loaded_after_each_assessment(self):
        self.assertIn("loadProgress();                                      // the attempt is now the latest result of that sentence", SOURCE)
        self.assertIn("function renderItemProgress()", SOURCE)
        self.assertIn("fetch(`/api/progress?student_id=${encodeURIComponent(activeStudent.id)}`)", SOURCE)
        self.assertIn("`✓ ${done.last}% · ${label}`", SOURCE)
        self.assertIn('t("itemProgress", { last: done.last, when: relativeTime(done.last_at) })', SOURCE)


if __name__ == "__main__":
    unittest.main()
