"""Public learning API boundaries that must hold before an Internet launch."""
import tempfile
import unittest
import io
import zipfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from practice_store import PracticeStore
from public_site import COOKIE, install_public_site


class PublicLearningApiTests(unittest.TestCase):
    def test_remote_assessment_is_saved_only_under_signed_in_website_account(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            practice = PracticeStore(root / 'practice.sqlite3', root / 'recordings')
            sample = {'id': '5-1-1', 'lesson': 5, 'story': 1, 'scene': 1,
                      'script': '你好', 'audio_available': False}
            calls = []
            fail_delete = [True]

            class Coordinator:
                @contextmanager
                def assessment(self, student_id):
                    yield

                @contextmanager
                def student(self, student_id):
                    yield

            class Remote:
                def create_speaker(self, site_student_id):
                    calls.append(('create', site_student_id))
                    return 'c' * 32

                def assess(self, speaker_id, file, **values):
                    calls.append(('assess', speaker_id, values['save_attempt']))
                    testcase.assertEqual(b'voice-bytes', file.file.read())
                    file.file.seek(0)
                    return {'summary': {'syllables': {'likely_correct': 1, 'practice': 0,
                                                      'insufficient_evidence': 0}, 'scoring_mode': 'fusion'}}

                def calibrate(self, speaker_id, file):
                    calls.append(('calibrate', speaker_id))
                    return {'voice_calibration': {'status': 'calibrated', 'pitch_floor_hz': 80,
                                                  'pitch_ceiling_hz': 400}, 'pace_hint': ''}

                def delete_speaker(self, speaker_id):
                    calls.append(('delete', speaker_id))
                    if fail_delete[0]:
                        raise HTTPException(503, 'Assessor unavailable')

            testcase = self
            app = FastAPI()
            store = install_public_site(app, samples={sample['id']: sample}, words={}, meanings={},
                                        practice=practice, coordinator=Coordinator(), assess=None,
                                        calibrate=None, pinyin=lambda s: s, speaking_service=Remote())
            alice = store.account_for_google('alice-sub', '', 'Same name', '')
            bob = store.account_for_google('bob-sub', '', 'Same name', '')
            with TestClient(app) as client:
                client.cookies.set(COOKIE, store.create_session(alice['id']))
                csrf = client.get('/api/me/session').json()['csrf_token']
                response = client.post('/api/me/speaking/assess',
                                       data={'sample_id': sample['id'], 'student_id': bob['student_id']},
                                       files={'file': ('voice.webm', b'voice-bytes', 'audio/webm')},
                                       headers={'X-CSRF-Token': csrf})
                self.assertEqual(200, response.status_code, response.text)
                attempt_id = response.json()['saved_attempt']['id']
                self.assertEqual([('create', alice['student_id']), ('assess', 'c' * 32, False)], calls)
                calibration = client.post('/api/me/voice-calibration',
                                          files={'file': ('calibration.webm', b'voice-bytes', 'audio/webm')},
                                          headers={'X-CSRF-Token': csrf})
                self.assertEqual(200, calibration.status_code, calibration.text)
                self.assertEqual('calibrated', practice.get_voice_calibration(alice['student_id'])['status'])
                self.assertEqual(('calibrate', 'c' * 32), calls[-1])
                self.assertEqual(1, len(store.speaking_rows(alice['student_id'])))
                self.assertEqual([], store.speaking_rows(bob['student_id']))
                self.assertEqual(b'voice-bytes', client.get(f'/api/me/attempts/{attempt_id}/audio').content)
                with zipfile.ZipFile(io.BytesIO(client.get('/api/me/export').content)) as archive:
                    self.assertIn('speaking.json', archive.namelist())
                    self.assertIn('voice_calibration.json', archive.namelist())
                    self.assertTrue(any(name.startswith('recordings/') for name in archive.namelist()))
                client.cookies.set(COOKIE, store.create_session(bob['id']))
                self.assertEqual(404, client.get(f'/api/me/attempts/{attempt_id}/audio').status_code)
                client.cookies.set(COOKIE, store.create_session(alice['id']))
                csrf = client.get('/api/me/session').json()['csrf_token']
                self.assertEqual(503, client.delete('/api/me/profile', headers={'X-CSRF-Token': csrf}).status_code)
                self.assertEqual(1, len(store.speaking_rows(alice['student_id'])))
                fail_delete[0] = False
                self.assertEqual(200, client.delete('/api/me/profile', headers={'X-CSRF-Token': csrf}).status_code)
                self.assertEqual(('delete', 'c' * 32), calls[-1])
                self.assertEqual([], store.speaking_rows(alice['student_id']))

    def test_speaking_profile_mapping_is_stable_and_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            practice = PracticeStore(root / 'practice.sqlite3', root / 'recordings')
            store = install_public_site(FastAPI(), samples={}, words={}, meanings={},
                                        practice=practice, coordinator=None, assess=None,
                                        calibrate=None, pinyin=lambda s: s)
            alice = store.account_for_google('alice-sub', '', 'Same name', '')
            bob = store.account_for_google('bob-sub', '', 'Same name', '')
            speaker_id = 'c' * 32
            self.assertEqual(speaker_id, store.bind_speaking_profile(alice['student_id'], speaker_id))
            self.assertEqual(speaker_id, store.bind_speaking_profile(alice['student_id'], speaker_id))
            self.assertEqual(speaker_id, store.speaking_profile(alice['student_id']))
            with self.assertRaises(ValueError):
                store.bind_speaking_profile(bob['student_id'], speaker_id)
            self.assertIsNone(store.speaking_profile(bob['student_id']))

    def test_speaking_uses_session_owner_and_csrf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            practice = PracticeStore(root / 'practice.sqlite3', root / 'recordings')
            sample = {'id': '5-1-01', 'lesson': 5, 'story': 1, 'scene': 1,
                      'script': '你好', 'audio_available': False}
            calls = []

            class Coordinator:
                @contextmanager
                def assessment(self, student_id):
                    calls.append(('lock', student_id))
                    yield

            def assess(file, script, sample_id, student_id, practice_mode, level, scoring_mode):
                calls.append(('assess', student_id, sample_id))
                return {'summary': {'likely_correct': 1}}

            app = FastAPI()
            store = install_public_site(app, samples={sample['id']: sample}, words={}, meanings={},
                                        practice=practice, coordinator=Coordinator(), assess=assess,
                                        calibrate=None, pinyin=lambda s: s)
            owner = store.account_for_google('owner-sub', '', 'Owner', '')
            with TestClient(app) as client:
                client.cookies.set(COOKIE, store.create_session(owner['id']))
                payload = {'sample_id': sample['id'], 'practice_mode': 'book',
                           'student_id': 'attacker-supplied-id'}
                audio = {'file': ('voice.webm', b'fake-audio', 'audio/webm')}
                self.assertEqual(403, client.post('/api/me/speaking/assess', data=payload, files=audio).status_code)
                csrf = client.get('/api/me/session').json()['csrf_token']
                response = client.post('/api/me/speaking/assess', data=payload, files=audio,
                                       headers={'X-CSRF-Token': csrf})
                self.assertEqual(200, response.status_code, response.text)
                self.assertEqual([('lock', owner['student_id']),
                                  ('assess', owner['student_id'], sample['id'])], calls)

    def test_https_launch_requires_persistent_auth_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            practice = PracticeStore(root / 'practice.sqlite3', root / 'recordings')
            with patch.dict('os.environ', {'MANDARIN_PUBLIC_HTTPS': '1', 'MANDARIN_SESSION_SECRET': '',
                                           'MANDARIN_PUBLIC_ORIGIN': '', 'GOOGLE_CLIENT_ID': '',
                                           'GOOGLE_CLIENT_SECRET': ''}):
                with self.assertRaisesRegex(RuntimeError, 'Public HTTPS configuration is incomplete'):
                    install_public_site(FastAPI(), samples={}, words={}, meanings={}, practice=practice,
                                        coordinator=None, assess=None, calibrate=None, pinyin=lambda s: s)

    def test_answers_and_story_review_are_private_to_skill_and_account(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            practice = PracticeStore(root / 'practice.sqlite3', root / 'recordings')
            samples = {
                f'5-1-0{i}': {'id': f'5-1-0{i}', 'lesson': 5, 'story': 1, 'scene': i,
                             'script': f'句子{i}', 'audio_available': False}
                for i in range(1, 5)
            }
            meanings = {key: {'en': f'Sentence {i}', 'vi': f'Câu {i}'}
                        for i, key in enumerate(samples, 1)}
            app = FastAPI()
            store = install_public_site(app, samples=samples, words={}, meanings=meanings,
                                        practice=practice, coordinator=None, assess=None,
                                        calibrate=None, pinyin=lambda script: script)
            alice = store.account_for_google('alice-sub', 'alice@example.com', 'Alice', '')
            bob = store.account_for_google('bob-sub', 'bob@example.com', 'Bob', '')
            with TestClient(app) as client:
                client.cookies.set(COOKIE, store.create_session(alice['id']))
                session = client.get('/api/me/session').json()
                exercise = client.get('/api/me/stories/5-1/listen')
                self.assertEqual(200, exercise.status_code)
                questions = exercise.json()['questions']
                self.assertEqual(3, len(questions))
                self.assertTrue(all('answer' not in q and 'explanation' not in q for q in questions))
                review_url = '/api/me/story-reviews/5-1'
                self.assertEqual(403, client.get(review_url, params={'skill': 'listen'}).status_code)
                self.assertEqual(403, client.post('/api/me/exercise-attempts', json={}).status_code)
                answers = {q['id']: q['options'][0]['id'] for q in questions}
                submitted = client.post('/api/me/exercise-attempts',
                                        headers={'X-CSRF-Token': session['csrf_token']},
                                        json={'story_id': '5-1', 'skill': 'listen', 'answers': answers})
                self.assertEqual(200, submitted.status_code, submitted.text)
                self.assertEqual(200, client.get(review_url, params={'skill': 'listen'}).status_code)
                self.assertEqual(403, client.get(review_url, params={'skill': 'read'}).status_code)
                client.cookies.set(COOKIE, store.create_session(bob['id']))
                self.assertEqual(403, client.get(review_url, params={'skill': 'listen'}).status_code)
                client.cookies.set(COOKIE, store.create_session(alice['id']))
                exported = client.get('/api/me/export')
                self.assertEqual(200, exported.status_code)
                with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
                    self.assertIn('profile.json', archive.namelist())
                    self.assertIn('comprehension.json', archive.namelist())
                    self.assertIn(b'5-1', archive.read('comprehension.json'))
                csrf = client.get('/api/me/session').json()['csrf_token']
                deleted = client.delete('/api/me/profile', headers={'X-CSRF-Token': csrf})
                self.assertEqual(200, deleted.status_code, deleted.text)
                self.assertEqual(401, client.get('/api/me/session').status_code)
                self.assertEqual([], store.quiz_rows(alice['student_id']))


if __name__ == '__main__':
    unittest.main()
