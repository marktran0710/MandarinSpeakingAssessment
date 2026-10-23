import tempfile
import unittest
from pathlib import Path

from practice_store import PracticeStore
from public_store import PublicStore


class PublicLearningStoreTests(unittest.TestCase):
    def test_accounts_sessions_and_quizzes_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            practice = PracticeStore(root / 'practice.sqlite3', root / 'recordings')
            store = PublicStore(practice)
            alice = store.account_for_google('google-sub-a', 'same@example.com', 'Alice', '')
            bob = store.account_for_google('google-sub-b', 'same@example.com', 'Bob', '')
            self.assertNotEqual(alice['student_id'], bob['student_id'])
            self.assertEqual(alice['id'], store.account_for_google('google-sub-a', 'new@example.com', 'Changed name', '')['id'])
            token = store.create_session(alice['id'])
            self.assertEqual(alice['student_id'], store.session(token)['student_id'])
            store.save_quiz(alice['student_id'], '5-1', 'listen', {'q':'a'}, 1, 3)
            self.assertEqual(1, len(store.quiz_rows(alice['student_id'])))
            self.assertEqual([], store.quiz_rows(bob['student_id']))
            store.revoke(token)
            self.assertIsNone(store.session(token))


if __name__ == '__main__':
    unittest.main()
