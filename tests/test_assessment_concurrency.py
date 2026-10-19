import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from assessment_concurrency import AssessmentBusy, AssessmentCoordinator, assessment_capacity


class AssessmentConcurrencyTests(unittest.TestCase):
    def test_two_api_assessments_enter_together_and_keep_their_own_results(self):
        import main

        both_entered = threading.Barrier(2)

        def assess(file, script, sample_id, student_id, mode, level, scoring_mode="rules"):
            both_entered.wait(timeout=5)
            return {"student_id": student_id, "script": script}

        with mock.patch.object(main, "ASSESSMENTS", AssessmentCoordinator(2)), \
                mock.patch.object(main, "_assess_pronunciation", side_effect=assess):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(main.assess_pronunciation, None, text, "custom", student, True, "custom", "")
                           for student, text in (("alice", "你好"), ("bob", "謝謝"))]
                results = [future.result(timeout=10) for future in futures]
        self.assertEqual(results, [{"student_id": "alice", "script": "你好"}, {"student_id": "bob", "script": "謝謝"}])

    def test_capacity_is_bounded_and_a_waiting_request_enters_after_release(self):
        gate = AssessmentCoordinator(2, wait_seconds=5)
        started = threading.Event()
        admitted = threading.Event()

        def third():
            started.set()
            with gate.assessment("third"):
                admitted.set()

        with ThreadPoolExecutor(max_workers=1) as executor:
            with gate.assessment("first"):
                with gate.assessment("second"):
                    future = executor.submit(third)
                    self.assertTrue(started.wait(2))
                    self.assertFalse(admitted.wait(0.1))
                self.assertTrue(admitted.wait(2))
            future.result(timeout=5)

    def test_same_student_waits_even_when_another_slot_is_free(self):
        gate = AssessmentCoordinator(2, wait_seconds=0)
        with gate.assessment("alice"):
            with self.assertRaises(AssessmentBusy):
                with gate.assessment("alice"):
                    self.fail("the same student was admitted twice")
            with gate.assessment("bob"):
                pass

    def test_deletion_waits_for_that_student_but_does_not_block_other_students(self):
        gate = AssessmentCoordinator(2, wait_seconds=5)
        started = threading.Event()
        deleting = threading.Event()

        def delete():
            started.set()
            with gate.student("alice"):
                deleting.set()

        with ThreadPoolExecutor(max_workers=1) as executor:
            with gate.assessment("alice"):
                future = executor.submit(delete)
                self.assertTrue(started.wait(2))
                self.assertFalse(deleting.wait(0.1))
                with gate.assessment("bob"):
                    pass
            future.result(timeout=5)
        self.assertTrue(deleting.is_set())

    def test_an_exception_releases_the_student_and_capacity(self):
        gate = AssessmentCoordinator(1, wait_seconds=0)
        with self.assertRaisesRegex(ValueError, "failed"):
            with gate.assessment("alice"):
                raise ValueError("failed")
        with gate.assessment("alice"):
            pass
        with gate.student("alice"):
            with self.assertRaises(AssessmentBusy):
                with gate.assessment("alice"):
                    self.fail("assessment entered during deletion")

    def test_saturation_times_out_and_api_response_allows_retry(self):
        import asyncio
        import main

        gate = AssessmentCoordinator(1, wait_seconds=0)
        with gate.assessment("alice"):
            with self.assertRaises(AssessmentBusy) as caught:
                with gate.assessment("bob"):
                    self.fail("capacity was exceeded")
        response = asyncio.run(main.explain_assessment_busy(None, caught.exception))
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["Retry-After"], "1")

    def test_capacity_setting_rejects_invalid_values(self):
        for value in ("0", "-1", "many", "1.5", ""):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"MANDARIN_ASSESSMENT_CONCURRENCY": value}):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    assessment_capacity()
        with mock.patch.dict(os.environ, {"MANDARIN_ASSESSMENT_CONCURRENCY": "3"}):
            self.assertEqual(assessment_capacity(), 3)


if __name__ == "__main__":
    unittest.main()
