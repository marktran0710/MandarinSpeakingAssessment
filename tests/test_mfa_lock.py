import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tone_pipeline as tp  # noqa: E402

CHILD = r'''
import sys, time
sys.path.insert(0, sys.argv[1])
import tone_pipeline as tp
tp.MFA_ROOT_DIR = sys.argv[2]
with tp._MFA_LOCK:
    open(sys.argv[3], "w").write(repr(time.time()))
'''


class CrossProcessLockTests(unittest.TestCase):
    def test_a_second_process_waits_until_the_first_has_finished_with_mfa(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(tp, "MFA_ROOT_DIR", folder):
            marker = Path(folder) / "child_acquired.txt"
            with tp._MFA_LOCK:
                child = subprocess.Popen([sys.executable, "-c", CHILD, str(ROOT), folder, str(marker)])
                time.sleep(4.0)                                  # the child imports the pipeline, then must be stuck waiting
                self.assertFalse(marker.exists(), "the second process got in while the first still held MFA")
                released = time.time()
            self.assertEqual(child.wait(timeout=60), 0)
            self.assertGreaterEqual(float(marker.read_text()), released - 0.01)

    def test_the_same_process_can_take_it_again_after_releasing(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(tp, "MFA_ROOT_DIR", folder):
            for _ in range(3):
                with tp._MFA_LOCK:
                    pass

    def test_if_the_lock_file_cannot_be_created_alignment_is_not_blocked(self):
        with tempfile.NamedTemporaryFile() as occupied:                 # a file where the folder should be
            with mock.patch.object(tp, "MFA_ROOT_DIR", occupied.name):
                with tp._MFA_LOCK:
                    pass


class AlignmentRetryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="mfa-retry-")
        self.patches = [mock.patch.object(tp, "MFA_ROOT_DIR", self.folder), mock.patch.object(tp, "MFA_EXE", sys.executable),
                        mock.patch.object(tp, "warm_aligner", lambda config=None: None), mock.patch.object(tp.time, "sleep", lambda s: None)]
        for patch in self.patches:
            patch.start()
        self.addCleanup(lambda: [patch.stop() for patch in self.patches])

    @staticmethod
    def run_with(outcomes):
        """subprocess.run stand-in: each outcome is 'fail', 'ok' or 'timeout'; returns the call list."""
        calls = []

        def fake(command, **kwargs):
            outcome = outcomes[min(len(calls), len(outcomes) - 1)]
            calls.append(outcome)
            if outcome == "timeout":
                raise subprocess.TimeoutExpired(command, 1)
            if outcome == "ok":
                Path(command[-1]).write_text(json.dumps({"tiers": {"words": {"entries": []}}}), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "", "")
            return subprocess.CompletedProcess(command, 1, "", "could not find the acoustic model")
        return fake, calls

    def align(self, outcomes):
        fake, calls = self.run_with(outcomes)
        with mock.patch.object(tp.subprocess, "run", fake):
            try:
                return tp.mfa_align(np.zeros(16000), 16000, ["你", "好"]), calls
            except tp.AlignmentError as exc:
                return exc, calls

    def test_a_transient_failure_is_retried_once_and_then_succeeds(self):
        tiers, calls = self.align(["fail", "ok"])
        self.assertEqual(tiers, {"words": {"entries": []}})
        self.assertEqual(calls, ["fail", "ok"])

    def test_a_persistent_failure_is_reported_after_exactly_two_tries(self):
        error, calls = self.align(["fail"])
        self.assertIsInstance(error, tp.AlignmentError)
        self.assertIn("MFA failed", str(error))
        self.assertEqual(calls, ["fail", "fail"])

    def test_a_timeout_is_not_retried(self):
        error, calls = self.align(["timeout", "ok"])
        self.assertIsInstance(error, tp.AlignmentError)
        self.assertEqual(calls, ["timeout"])

    def test_a_first_time_success_runs_once(self):
        _, calls = self.align(["ok"])
        self.assertEqual(calls, ["ok"])


if __name__ == "__main__":
    unittest.main()
