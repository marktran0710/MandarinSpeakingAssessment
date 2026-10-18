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


class LeaderboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.store = PracticeStore(root / "practice.sqlite3", root / "recordings")
        self.audio = root / "a.webm"
        self.audio.write_bytes(b"x")

    def attempt(self, name, sample_id, good, okay=0, practice=0):
        student = self.store.create_or_get_student(name, "foreigner")
        self.store.save_attempt(student["id"], sample_id, "文本", self.audio, "a.webm",
                                {"summary": {"syllables": {"likely_correct": good, "insufficient_evidence": okay, "practice": practice}}})
        return student["id"]

    def test_a_lesson_is_worth_its_best_attempt_and_an_okay_counts_half(self):
        self.attempt("Mei", "5-1-1", good=2, okay=0, practice=2)            # 50
        self.attempt("Mei", "5-1-1", good=2, okay=2, practice=0)            # 75: the best attempt of that lesson
        self.attempt("Mei", "5-1-1", good=1, okay=0, practice=3)            # 25: a worse try changes nothing
        self.attempt("Mei", "C5-5-1-I1-W001", good=2)                       # 100
        board = self.store.leaderboard()
        self.assertEqual(board["top"], [{"rank": 1, "name": "Mei", "points": 175, "lessons": 2, "is_me": False}])

    def test_own_text_and_attempts_without_a_good_syllable_do_not_count(self):
        self.attempt("Mei", "custom", good=1)                                # a single easy character must not give points
        self.attempt("Mei", "5-1-1", good=0, okay=5)                         # silence / a failed alignment is all Okay
        self.assertEqual(self.store.leaderboard()["top"], [])
        self.assertEqual(self.store.leaderboard()["players"], 0)

    def test_test_accounts_are_hidden(self):
        for name in ("mandarin-speaking-integration-test", "cal-log-123", "tts-check", "strict-check", "lvl-api2"):
            self.attempt(name, "5-1-1", good=3)
        self.attempt("Real Person", "5-1-1", good=3)
        self.assertEqual([row["name"] for row in self.store.leaderboard()["top"]], ["Real Person"])

    def test_the_top_ten_are_ordered_ties_share_a_rank_and_the_viewer_sees_their_own_row(self):
        ids = {}
        for n in range(12):
            ids[n] = self.attempt(f"Learner {n:02d}", "5-1-1", good=10 - (n // 2) if n < 11 else 1, okay=0, practice=(n // 2) if n < 11 else 9)
        board = self.store.leaderboard(viewer_id=ids[11])
        self.assertEqual(len(board["top"]), 10)
        points = [row["points"] for row in board["top"]]
        self.assertEqual(points, sorted(points, reverse=True))
        ranks = [row["rank"] for row in board["top"]]
        self.assertEqual(ranks[:2], [1, 1])                                   # equal points: the same rank
        self.assertEqual(ranks[2], 3)                                         # and the next rank skips
        self.assertEqual(board["players"], 12)
        self.assertEqual(board["me"]["name"], "Learner 11")                  # outside the top 10, but the learner still sees where they are
        self.assertGreater(board["me"]["rank"], 10)
        self.assertFalse(any(row["is_me"] for row in board["top"]))

    def test_the_viewer_is_marked_in_the_top_and_nobody_elses_id_is_given_out(self):
        mine = self.attempt("Mei", "5-1-1", good=3)
        other = self.attempt("Lin", "5-1-1", good=2, practice=1)
        board = self.store.leaderboard(viewer_id=mine)
        self.assertEqual([(row["name"], row["is_me"]) for row in board["top"]], [("Mei", True), ("Lin", False)])
        self.assertNotIn(other, repr(board))
        self.assertNotIn(mine, repr(board))

    def test_the_endpoint_answers_for_a_viewer_and_for_nobody(self):
        mine = self.attempt("Mei", "5-1-1", good=3)
        with mock.patch.object(main, "practice_store", self.store):
            anonymous = asyncio.run(main.get_leaderboard())
            known = asyncio.run(main.get_leaderboard(student_id=mine))
        self.assertIsNone(anonymous["me"])
        self.assertEqual(known["top"][0]["is_me"], True)


class PageTests(unittest.TestCase):
    def test_the_board_is_to_the_right_on_a_wide_screen(self):
        self.assertIn('class="page-grid"', SOURCE)
        self.assertIn('<aside id="leaderboardCard" class="card leaderboard"', SOURCE)
        self.assertRegex(SOURCE, r"@media \(min-width: 1280px\) \{\s*body \{ max-width: 1400px; \}\s*\.page-grid \{ display: grid; grid-template-columns: minmax\(0, 1040px\) 300px;")

    def test_every_language_has_the_board_texts(self):
        for key in ("leaderboardTitle", "leaderboardHelp", "leaderboardEmpty", "leaderboardPoints", "leaderboardLessons", "leaderboardMeRank", "leaderboardMeNone"):
            self.assertEqual(len(re.findall(r"\b%s: " % key, SOURCE)), 5, key)

    def test_the_board_is_refreshed_after_an_assessment_and_when_the_speaker_changes(self):
        self.assertIn("loadLeaderboard();                                   // the attempt may have moved the learner up", SOURCE)
        self.assertIn('loadLeaderboard();                                       // "You" is a different person now', SOURCE)
        self.assertIn("if (lastLeaderboard) renderLeaderboard(lastLeaderboard);", SOURCE)


if __name__ == "__main__":
    unittest.main()
