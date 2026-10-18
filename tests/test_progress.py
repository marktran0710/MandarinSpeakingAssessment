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
        progress = self.store.progress(learner)
        entry = progress["lessons"]["5-1-1"]
        self.assertEqual((entry["attempts"], entry["best"], entry["last"]), (3, 90, 70))
        self.assertTrue(entry["last_at"])

    def test_own_text_marks_no_lesson_but_is_among_the_recent_results(self):
        learner = self.attempt("Mei", "custom", 4, 0, 0, script="我愛你")
        progress = self.store.progress(learner)
        self.assertEqual(progress["lessons"], {})
        self.assertEqual([(r["sample_id"], r["mode"], r["script"], r["points"]) for r in progress["recent"]], [("custom", "custom", "我愛你", 100)])

    def test_chapter_5_words_are_told_apart_from_book_sentences(self):
        learner = self.attempt("Mei", "C5-5-1-I1-W001", 2, 0, 0, script="你好")
        self.attempt("Mei", "5-1-1", 3, 0, 1, script="友美")
        modes = {r["sample_id"]: r["mode"] for r in self.store.progress(learner)["recent"]}
        self.assertEqual(modes, {"C5-5-1-I1-W001": "chapter5_word", "5-1-1": "book"})

    def test_an_attempt_without_a_good_syllable_is_not_a_result(self):
        learner = self.attempt("Mei", "5-1-1", 0, 8, 0)                     # silence / a failed alignment: all Okay
        progress = self.store.progress(learner)
        self.assertEqual(progress["lessons"], {})                            # it marks nothing ...
        self.assertEqual(progress["recent"][0]["points"], None)              # ... and shows no points

    def test_the_recent_results_are_newest_first_and_limited(self):
        learner = None
        for number in range(12):
            learner = self.attempt("Mei", f"5-1-{number + 1}", 10 - number // 2, 0, number // 2, script=f"句{number}")
        recent = self.store.progress(learner)["recent"]
        self.assertEqual(len(recent), 10)
        self.assertEqual([r["script"] for r in recent][:3], ["句11", "句10", "句9"])
        self.assertEqual(len(self.store.progress(learner, recent_limit=3)["recent"]), 3)

    def test_a_learner_sees_only_their_own_attempts(self):
        mine = self.attempt("Mei", "5-1-1", 5, 0, 5)
        self.attempt("Lin", "5-1-2", 10)
        progress = self.store.progress(mine)
        self.assertEqual(list(progress["lessons"]), ["5-1-1"])
        self.assertEqual(self.store.progress("nobody"), {"lessons": {}, "recent": []})

    def test_the_endpoint_answers_for_a_learner_and_for_nobody(self):
        mine = self.attempt("Mei", "5-1-1", 5, 0, 5)
        with mock.patch.object(main, "practice_store", self.store):
            known = asyncio.run(main.get_progress(student_id=mine))
            anonymous = asyncio.run(main.get_progress())
        self.assertEqual(known["lessons"]["5-1-1"]["best"], 50)
        self.assertEqual(anonymous, {"lessons": {}, "recent": []})


class PageTests(unittest.TestCase):
    KEYS = ("resultRetry", "resultNext", "itemProgress", "itemNotPracticed", "recentTitle", "recentEmpty", "recentPoints")

    def test_every_language_has_the_new_texts(self):
        for key in self.KEYS:
            self.assertEqual(len(re.findall(r"\b%s: " % key, SOURCE)), 5, key)

    def test_the_result_card_has_a_read_again_and_a_next_button(self):
        card = SOURCE[SOURCE.index('<div id="resultCard"'): SOURCE.index('<div id="pronunciationResults">')]
        self.assertIn('id="retryBtn"', card)
        self.assertIn('onclick="toggleRecording()"', card)                   # the same record / stop as the button at the top
        self.assertIn('id="nextItemBtn"', card)
        self.assertIn('onclick="goToNextItem()"', card)

    def test_the_lists_are_marked_and_the_progress_is_loaded_after_each_assessment(self):
        self.assertIn("loadProgress();                                      // the attempt is now a mark and a recent result", SOURCE)
        self.assertIn("function renderItemProgress()", SOURCE)
        self.assertIn("fetch(`/api/progress?student_id=${encodeURIComponent(activeStudent.id)}`)", SOURCE)
        self.assertIn('id="recentList"', SOURCE)


if __name__ == "__main__":
    unittest.main()
