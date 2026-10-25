"""Contract checks for the private speaking-service boundary."""

import io
import unittest

import httpx
from fastapi import HTTPException, UploadFile

from speaking_service import SpeakingService


SITE_ID = "a" * 32
REMOTE_ID = "b" * 32


class SpeakingServiceTests(unittest.TestCase):
    def make_service(self, handler):
        client = httpx.Client(base_url="http://127.0.0.1:8011", transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        return SpeakingService("http://127.0.0.1:8011", "private-admin-secret", client)

    def test_opaque_profile_and_existing_assessment_contract(self):
        calls = []

        def respond(request):
            calls.append(request)
            if request.url.path == "/api/students":
                self.assertEqual({"name": f"site-{SITE_ID}", "group_type": "foreigner"},
                                 __import__("json").loads(request.content))
                return httpx.Response(200, json={"id": REMOTE_ID, "name": f"site-{SITE_ID}"})
            self.assertEqual("/api/assess-pronunciation", request.url.path)
            body = request.content
            for field in (b'name="student_id"', REMOTE_ID.encode(), b'name="sample_id"',
                          b"5-1-1", b'name="script"', b"voice-bytes"):
                self.assertIn(field, body)
            self.assertNotIn(b'name="save_attempt"', body)
            return httpx.Response(200, json={"summary": {"syllables": {"likely_correct": 1}},
                                             "saved_attempt": {"id": "e" * 32}})

        service = self.make_service(respond)
        speaker_id = service.create_speaker(SITE_ID)
        self.assertEqual(REMOTE_ID, speaker_id)
        upload = UploadFile(file=io.BytesIO(b"voice-bytes"), filename="voice.webm")
        result = service.assess(speaker_id, upload, script="你好", sample_id="5-1-1",
                                practice_mode="book", scoring_mode="fusion")
        self.assertEqual(1, result["summary"]["syllables"]["likely_correct"])
        self.assertIn("saved_attempt", result)
        self.assertEqual(0, upload.file.tell())
        self.assertEqual(2, len(calls))

    def test_existing_assessor_needs_no_stateless_capability_endpoint(self):
        calls = []

        def existing_service(request):
            calls.append(request.url.path)
            self.assertEqual("/api/assess-pronunciation", request.url.path)
            return httpx.Response(200, json={"summary": {"syllables": {"likely_correct": 1}}})

        service = self.make_service(existing_service)
        upload = UploadFile(file=io.BytesIO(b"voice-bytes"), filename="voice.webm")
        result = service.assess(REMOTE_ID, upload, script="你好", sample_id="5-1-1",
                                practice_mode="book", scoring_mode="fusion")
        self.assertIn("summary", result)
        self.assertEqual(["/api/assess-pronunciation"], calls)

    def test_rejects_non_loopback_target_and_invalid_remote_identity(self):
        for url in ("https://example.com:443", "http://example.com:8011", "http://127.0.0.1:8011/other"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                SpeakingService(url)
        service = self.make_service(lambda _: httpx.Response(200, json={"id": REMOTE_ID, "name": "Another learner"}))
        with self.assertRaises(HTTPException) as error:
            service.create_speaker(SITE_ID)
        self.assertEqual(502, error.exception.status_code)

    def test_unavailable_assessor_does_not_retry_upload(self):
        calls = []

        def unavailable(request):
            calls.append(request)
            raise httpx.ConnectError("offline", request=request)

        service = self.make_service(unavailable)
        upload = UploadFile(file=io.BytesIO(b"voice-bytes"), filename="voice.webm")
        with self.assertRaises(HTTPException) as error:
            service.assess(REMOTE_ID, upload, script="你好", sample_id="5-1-1",
                           practice_mode="book", scoring_mode="fusion")
        self.assertEqual(503, error.exception.status_code)
        self.assertEqual(1, len(calls))
        self.assertEqual(0, upload.file.tell())


if __name__ == "__main__":
    unittest.main()
