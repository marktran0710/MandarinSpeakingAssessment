import json
import os
import sys
import tempfile
import textwrap
import time
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mfa_warm  # noqa: E402

FAKE_WORKER = textwrap.dedent('''
    import json, os, sys
    MODE = os.environ.get("FAKE_MODE", "ok")
    def reply(message):
        sys.stdout.write(json.dumps(message) + chr(10)); sys.stdout.flush()
    if MODE == "no_start":
        sys.exit(3)
    reply({"ready": True})
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("op") == "load":
            reply({"loaded": True})
            continue
        if MODE == "error":
            reply({"error": "ValueError: boom"})
        elif MODE == "die":
            sys.exit(1)
        else:
            if MODE == "parallel":
                from pathlib import Path
                import time
                marker = Path(request["txt"])
                marker.write_text(str(os.getpid()))
                deadline = time.monotonic() + 5
                while len(list(marker.parent.glob("*.started"))) < 2:
                    if time.monotonic() > deadline:
                        reply({"error": "second worker did not run concurrently"})
                        sys.exit(1)
                    time.sleep(0.01)
            reply({"tiers": {"words": {"entries": [[0.0, 0.1, request["txt"]]]}, "pid": os.getpid()}})
        break
''')


class WarmAlignerTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="warm-test-")
        self.worker = Path(self.folder) / "fake_worker.py"
        self.worker.write_text(FAKE_WORKER, encoding="utf-8")
        self.fingerprint = ["v1"]
        self.created = []

    def tearDown(self):
        for aligner in self.created:
            aligner.close()

    def make(self, mode="ok", **kwargs):
        aligner = mfa_warm.WarmAligner(
            sys.executable, {**os.environ, "FAKE_MODE": mode}, "mandarin_mfa", lambda folder: "dictionary.dict",
            lambda: self.fingerprint[0], timeout_sec=20, load_timeout_sec=20, ready_wait_sec=20,
            worker_script=self.worker, **kwargs)
        self.created.append(aligner)
        return aligner

    def wait_for_spare(self, aligner):
        deadline = time.time() + 20
        while time.time() < deadline:
            spare = aligner._spare
            if spare is not None and spare.ready.is_set():
                return spare
            time.sleep(0.02)
        self.fail("the spare never became ready")

    def test_no_spare_yet_means_the_caller_runs_the_normal_command_and_a_spare_starts_loading(self):
        aligner = self.make()
        self.assertIsNone(aligner.align("a.wav", "b.txt"))
        self.wait_for_spare(aligner)

    def test_a_ready_spare_serves_one_request_and_is_replaced_by_a_new_process(self):
        aligner = self.make()
        aligner.prewarm()
        first = self.wait_for_spare(aligner)
        tiers = aligner.align("a.wav", "first.txt")
        self.assertEqual(tiers["words"]["entries"][0][2], "first.txt")
        second = self.wait_for_spare(aligner)
        self.assertIsNot(first, second)
        again = aligner.align("a.wav", "second.txt")
        self.assertEqual(again["words"]["entries"][0][2], "second.txt")
        self.assertNotEqual(tiers["pid"], again["pid"])                       # never a second request in one process

    def test_an_error_reply_falls_back_to_the_normal_command(self):
        aligner = self.make("error")
        aligner.prewarm()
        self.wait_for_spare(aligner)
        self.assertIsNone(aligner.align("a.wav", "b.txt"))

    def test_a_worker_that_dies_mid_request_falls_back(self):
        aligner = self.make("die")
        aligner.prewarm()
        self.wait_for_spare(aligner)
        self.assertIsNone(aligner.align("a.wav", "b.txt"))

    def test_a_changed_dictionary_discards_the_spare(self):
        aligner = self.make()
        aligner.prewarm()
        self.wait_for_spare(aligner)
        self.fingerprint[0] = "v2"                                            # the dictionary was rebuilt on disk
        self.assertIsNone(aligner.align("a.wav", "b.txt"))
        self.wait_for_spare(aligner)
        self.assertEqual(aligner.align("a.wav", "ok.txt")["words"]["entries"][0][2], "ok.txt")

    def test_repeated_failures_switch_warming_off_for_good(self):
        aligner = self.make("no_start", max_failures=2)
        for _ in range(4):
            aligner.prewarm()
            deadline = time.time() + 20
            while aligner._spare is not None and not aligner._spare.ready.is_set() and time.time() < deadline:
                time.sleep(0.02)
            self.assertIsNone(aligner.align("a.wav", "b.txt"))
        self.assertTrue(aligner.disabled)
        self.assertIsNone(aligner._spare)
        aligner.prewarm()
        self.assertIsNone(aligner._spare)                                      # and it does not start building again

    def test_the_switch_is_off_unless_asked_for(self):
        saved = os.environ.pop("TONE_MFA_WARM", None)
        try:
            self.assertFalse(mfa_warm.enabled_by_environment())
            for value in ("1", "true", "YES"):
                os.environ["TONE_MFA_WARM"] = value
                self.assertTrue(mfa_warm.enabled_by_environment(), value)
            os.environ["TONE_MFA_WARM"] = "0"
            self.assertFalse(mfa_warm.enabled_by_environment())
        finally:
            os.environ.pop("TONE_MFA_WARM", None)
            if saved is not None:
                os.environ["TONE_MFA_WARM"] = saved

    def test_mfa_align_uses_the_warm_tiers_when_available_and_the_command_otherwise(self):
        import numpy as np
        import tone_pipeline as tp
        calls = []

        class Stub:
            def __init__(self, result):
                self.result = result

            def align(self, wav, txt):
                calls.append((Path(wav).name, Path(txt).read_text(encoding="utf-8")))
                return self.result

        original = tp.warm_aligner
        try:
            tp.warm_aligner = lambda config=None: Stub({"words": {"entries": []}, "warm": True})
            tiers = tp.mfa_align(np.zeros(16000), 16000, ["你", "好"])
            self.assertTrue(tiers["warm"])
            self.assertEqual(calls, [("utterance.wav", "你 好")])
        finally:
            tp.warm_aligner = original

    def test_pool_uses_two_separate_worker_processes_at_the_same_time(self):
        pool = mfa_warm.WarmAlignerPool(
            2, sys.executable, {**os.environ, "FAKE_MODE": "parallel"}, "mandarin_mfa",
            lambda folder: "dictionary.dict", lambda: "v1", timeout_sec=15,
            load_timeout_sec=15, ready_wait_sec=15, worker_script=self.worker,
            load_lock=threading.Lock())
        self.addCleanup(pool.close)
        pool.prewarm()
        for aligner in pool._aligners:
            self.wait_for_spare(aligner)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(pool.align, "a.wav", str(Path(self.folder) / f"{i}.started")) for i in range(2)]
            results = [future.result(timeout=20) for future in futures]
        self.assertTrue(all(result is not None for result in results))
        self.assertNotEqual(results[0]["pid"], results[1]["pid"])

    def test_pool_returns_a_slot_after_a_failed_alignment(self):
        pool = mfa_warm.WarmAlignerPool(
            1, sys.executable, dict(os.environ), "mandarin_mfa", lambda folder: "dictionary.dict",
            lambda: "v1", timeout_sec=15, worker_script=self.worker)
        self.addCleanup(pool.close)
        from unittest import mock
        with mock.patch.object(pool._aligners[0], "align", side_effect=[RuntimeError("boom"), {"ok": True}]):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                pool.align("a.wav", "a.txt")
            self.assertEqual(pool.align("b.wav", "b.txt"), {"ok": True})


if __name__ == "__main__":
    unittest.main()
