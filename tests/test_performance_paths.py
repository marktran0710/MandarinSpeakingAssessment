import asyncio
import gzip
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main  # noqa: E402
from practice_store import PracticeStore  # noqa: E402

BODY = ("<p>" + "taiwan mandarin " * 400 + "</p>").encode("utf-8")           # 6 KB: above the size that is worth compressing


async def inner_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/html; charset=utf-8"), (b"content-length", str(len(BODY)).encode())]})
    await send({"type": "http.response.body", "body": BODY})


def ask(path, headers=(), method="GET"):
    """The (status, headers, body) the compression layer sends for one request to a page that always answers BODY."""
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "path": path, "raw_path": path.encode(), "query_string": b"",
             "headers": [(b"accept-encoding", b"gzip")] + list(headers), "server": ("test", 80), "client": ("test", 1)}
    asyncio.run(main.TextGzip(inner_app)(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], {k.decode().lower(): v.decode() for k, v in start["headers"]}, body


class CompressionTests(unittest.TestCase):
    def test_the_page_and_the_json_answers_are_compressed(self):
        for path in ("/", "/api/sample-lessons", "/api/leaderboard"):
            status, headers, body = ask(path)
            self.assertEqual(headers.get("content-encoding"), "gzip", path)
            self.assertEqual(gzip.decompress(body), BODY, path)
            self.assertLess(len(body), len(BODY) / 5, path)

    def test_audio_is_never_compressed(self):
        for path in ("/api/sample-lessons/5-1-1/audio", "/api/chapter5-word-items/C5-5-1-I1-W001/audio", "/api/tts-sample", "/api/calibration-sound/ma1",
                     "/api/admin/attempts/abc/audio"):
            status, headers, body = ask(path)
            self.assertNotIn("content-encoding", headers, path)
            self.assertEqual(body, BODY, path)

    def test_a_range_request_is_left_alone_so_the_players_can_seek(self):
        status, headers, body = ask("/api/something", headers=[(b"range", b"bytes=0-99")])
        self.assertNotIn("content-encoding", headers)

    def test_the_answer_still_says_it_varies_with_the_encoding(self):
        status, headers, body = ask("/")
        self.assertIn("accept-encoding", headers.get("vary", "").lower())


class StaticDataIsWorkedOutOnceTests(unittest.TestCase):
    def test_the_pinyin_of_a_sentence_is_worked_out_once_and_the_list_is_the_same(self):
        first = asyncio.run(main.get_sample_lessons())
        hits = main._sample_pinyin.cache_info().hits
        second = asyncio.run(main.get_sample_lessons())
        self.assertEqual(first, second)
        self.assertGreaterEqual(main._sample_pinyin.cache_info().hits - hits, len(first))

    def test_the_chapter_5_items_keep_their_readings_and_the_taxonomy_is_read_again_only_when_the_file_changes(self):
        items = asyncio.run(main.get_chapter5_word_items())
        again = asyncio.run(main.get_chapter5_word_items())
        self.assertEqual(items, again)
        self.assertTrue(all(item["character_pinyin"] for item in items))
        self.assertIs(main.load_chapter5_taxonomy(), main.load_chapter5_taxonomy())


class LeaderboardIsWorkedOutOncePerChangeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.store = PracticeStore(root / "practice.sqlite3", root / "recordings")
        self.audio = root / "a.webm"
        self.audio.write_bytes(b"x")

    def attempt(self, name, sample_id="5-1-1", good=3):
        student = self.store.create_or_get_student(name, "foreigner")
        self.store.save_attempt(student["id"], sample_id, "文本", self.audio, "a.webm",
                                {"summary": {"syllables": {"likely_correct": good, "insufficient_evidence": 0, "practice": 0}}})
        return student["id"]

    def test_a_second_ask_is_answered_from_the_first(self):
        self.attempt("Mei")
        self.assertIs(self.store.leaderboard(), self.store.leaderboard())

    def test_a_saved_attempt_shows_at_once(self):
        self.attempt("Mei")
        self.assertEqual([row["name"] for row in self.store.leaderboard()["top"]], ["Mei"])
        self.attempt("Lin", good=5)
        self.assertEqual([row["name"] for row in self.store.leaderboard()["top"]], ["Lin", "Mei"])

    def test_a_deleted_learner_disappears_at_once(self):
        mei = self.attempt("Mei")
        self.attempt("Lin", good=5)
        self.assertEqual(len(self.store.leaderboard()["top"]), 2)
        self.store.delete_student_data(mei)
        self.assertEqual([row["name"] for row in self.store.leaderboard()["top"]], ["Lin"])

    def test_each_viewer_gets_their_own_answer(self):
        mei, lin = self.attempt("Mei"), self.attempt("Lin", good=5)
        self.assertEqual([row["is_me"] for row in self.store.leaderboard(mei)["top"]], [False, True])
        self.assertEqual([row["is_me"] for row in self.store.leaderboard(lin)["top"]], [True, False])


if __name__ == "__main__":
    unittest.main()
