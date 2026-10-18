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


def attempt(sample, rating=None, comment="", created="2026-10-17T10:00:00+00:00"):
    base = {"id": sample, "sample_id": sample, "practice_mode": "book", "target_script": "文本", "created_at": created, "summary": {}}
    if rating is not None:
        base["feedback"] = {"rating": rating, "comment": comment, "created_at": created}
    return base


USERS = [
    {"name": "Mei", "attempts": [attempt("5-1-1", 5, "", "2026-10-17T10:00:00+00:00"), attempt("5-1-2", 2, "too hard", "2026-10-18T09:00:00+00:00"), attempt("5-1-3")]},
    {"name": "Lan Anh", "attempts": [attempt("5-1-1", 4, "great graph", "2026-10-16T08:00:00+00:00"), attempt("5-1-2", 3, "  ", "2026-10-18T11:00:00+00:00")]},
    {"name": "Nobody", "attempts": []},
]


def collect(filters):
    script = f"{function_source('collectFeedback')}\nconsole.log(JSON.stringify(collectFeedback({json.dumps(USERS)}, {json.dumps(filters)})));"
    result = json.loads(subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30, check=True).stdout)
    return [(row["user"]["name"], row["rating"], row["comment"]) for row in result["rows"]], result["summary"]


@unittest.skipUnless(NODE, "node is needed to run the page's JavaScript")
class CollectFeedbackTests(unittest.TestCase):
    def test_everything_is_listed_newest_first_and_attempts_without_feedback_are_left_out(self):
        rows, summary = collect({})
        self.assertEqual(rows, [("Lan Anh", 3, ""), ("Mei", 2, "too hard"), ("Mei", 5, ""), ("Lan Anh", 4, "great graph")])
        self.assertEqual(summary["count"], 4)
        self.assertAlmostEqual(summary["average"], 3.5)
        self.assertEqual(summary["withComments"], 2)                      # a comment of only spaces is no comment

    def test_only_comments(self):
        rows, summary = collect({"onlyComments": True})
        self.assertEqual(rows, [("Mei", 2, "too hard"), ("Lan Anh", 4, "great graph")])
        self.assertEqual(summary["withComments"], 2)

    def test_one_rating_or_three_stars_and_lower(self):
        self.assertEqual([r[1] for r in collect({"rating": "5"})[0]], [5])
        self.assertEqual(sorted(r[1] for r in collect({"rating": "low"})[0]), [2, 3])
        self.assertEqual(collect({"rating": "1"})[0], [])

    def test_a_user_filter_matches_part_of_a_name_whatever_the_case(self):
        self.assertEqual({r[0] for r in collect({"user": "  lan"})[0]}, {"Lan Anh"})
        self.assertEqual(collect({"user": "zzz"}), ([], {"count": 0, "average": None, "withComments": 0}))

    def test_the_filters_combine(self):
        self.assertEqual(collect({"onlyComments": True, "rating": "low", "user": "mei"})[0], [("Mei", 2, "too hard")])


class AdminPageTests(unittest.TestCase):
    def test_the_page_has_the_two_views_and_the_filters(self):
        for needle in ("setAdminView('feedback')", 'id="onlyComments"', 'id="ratingFilter"', 'id="userFilter"', 'id="feedbackList"', "function renderAdminView()"):
            self.assertIn(needle, SOURCE, needle)

    def test_the_chosen_view_is_remembered_and_the_dashboard_fills_both_views(self):
        self.assertIn('localStorage.setItem("adminView", adminView)', SOURCE)
        self.assertIn("adminData = data;\n                    renderAdminView();", SOURCE)


if __name__ == "__main__":
    unittest.main()
