"""
Keeps a pool of pre-loaded MFA aligners ready, so a learner does not wait for `mfa align_one` to start up (about 5 s of
a 6 s assessment). The score does not change: each spare serves exactly one request in a fresh process (see
mfa_oneshot_worker.py), so its alignment is bit-identical to the one `align_one` gives.

The warm path is only ever a shortcut. Anything unusual (no spare yet, a slow or dead worker, an error reply, a
dictionary that changed on disk) returns None and the caller runs the normal `align_one` command instead.
After a spare is used, its replacement is built in the background. Cost per slot: one Python process holding the model
(about 340 MB) while it waits.
"""
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

WORKER_SCRIPT = Path(__file__).with_name("mfa_oneshot_worker.py")
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class _Spare:
    """One worker process, being loaded in a background thread."""

    def __init__(self):
        self.ready = threading.Event()
        self.process = None            # None once ready means the build failed
        self.key = None
        self.dictionary = None

    def kill(self):
        process, self.process = self.process, None
        if process is None:
            return
        try:
            process.kill()
        except OSError:
            pass
        for pipe in (process.stdin, process.stdout):
            try:
                pipe.close()
            except OSError:
                pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


class WarmAligner:
    def __init__(self, python_exe, env, acoustic_model, prepare_dictionary, fingerprint, timeout_sec,
                 load_timeout_sec=120, ready_wait_sec=15, max_failures=3, worker_script=WORKER_SCRIPT, load_lock=None):
        # MFA unpacks the acoustic model into a shared folder every time a process loads it, so two loading
        # processes can see it half-written. `load_lock` is the lock that already serializes `mfa align_one` runs.
        self._load_lock = load_lock or threading.Lock()
        self._python = str(python_exe)
        self._env = dict(env)
        self._acoustic_model = acoustic_model
        self._prepare_dictionary = prepare_dictionary      # (folder) -> argument MFA accepts for the dictionary
        self._fingerprint = fingerprint                    # () -> short hash of the dictionary content
        self._timeout = timeout_sec
        self._load_timeout = load_timeout_sec
        self._ready_wait = ready_wait_sec
        self._max_failures = max_failures
        self._worker_script = str(worker_script)
        self._workdir = tempfile.mkdtemp(prefix="tonepipe-warm-")
        self._log = open(Path(self._workdir) / "worker.log", "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._spare = None
        self._failures = 0
        self.disabled = False

    # ------------------------------------------------------------ building ----

    def prewarm(self):
        """Start loading a spare in the background unless one is already there (or warming is switched off)."""
        with self._lock:
            if self.disabled or self._spare is not None:
                return
            spare = self._spare = _Spare()
        threading.Thread(target=self._build, args=(spare,), daemon=True, name="mfa-warm-build").start()

    def _build(self, spare):
        process = None
        timer = None
        try:
            if self.disabled:                              # closed (or given up) while this build was queued
                return
            spare.key = self._fingerprint()
            spare.dictionary = self._prepare_dictionary(self._workdir)
            with self._load_lock:
                process = subprocess.Popen(
                    [self._python, self._worker_script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=self._log, env=self._env, text=True, encoding="utf-8", creationflags=_NO_WINDOW)
                spare.process = process
                if self.disabled:                          # closed while the process was starting
                    spare.kill()
                    return
                timer = threading.Timer(self._load_timeout, process.kill)
                timer.start()
                if not json.loads(process.stdout.readline() or "{}").get("ready"):
                    raise RuntimeError("worker did not start")
                process.stdin.write(json.dumps({"op": "load", "dictionary": spare.dictionary,
                                                "acoustic_model": self._acoustic_model}) + "\n")
                process.stdin.flush()
                if not json.loads(process.stdout.readline() or "{}").get("loaded"):
                    raise RuntimeError("worker could not load the models")
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"[mfa-warm] could not build a spare: {exc}", file=sys.stderr)
            spare.kill()
            self._note_failure()
        finally:
            if timer is not None:
                timer.cancel()
            spare.ready.set()

    def _note_failure(self):
        with self._lock:
            self._failures += 1
            if self._failures >= self._max_failures and not self.disabled:
                self.disabled = True
                self._spare = None
                print("[mfa-warm] switched off after repeated failures; using `mfa align_one`.", file=sys.stderr)

    # ------------------------------------------------------------- serving ----

    def align(self, wav, txt):
        """The MFA tiers for this utterance, or None when the caller should run `mfa align_one` itself."""
        if self.disabled:
            return None
        with self._lock:
            spare, self._spare = self._spare, None
        if spare is None:                                  # first request, or the replacement is not started yet
            self.prewarm()
            return None
        if not spare.ready.wait(self._ready_wait):         # still loading: leave it, do not hold the learner up
            with self._lock:
                if self._spare is None:
                    self._spare = spare
                else:
                    spare.kill()
            return None
        if spare.process is None or spare.key != self._fingerprint():
            spare.kill()                                   # failed to build, or the dictionary changed on disk
            self.prewarm()
            return None
        reply = {}
        timer = threading.Timer(self._timeout, spare.kill)
        timer.start()
        try:
            spare.process.stdin.write(json.dumps({"dictionary": spare.dictionary, "acoustic_model": self._acoustic_model,
                                                  "wav": str(wav), "txt": str(txt)}) + "\n")
            spare.process.stdin.flush()
            reply = json.loads(spare.process.stdout.readline() or "{}")
        except (OSError, ValueError, AttributeError):
            reply = {}
        finally:
            timer.cancel()
            spare.kill()                                   # one-shot: it ends after its reply anyway
            self.prewarm()                                 # the next learner's spare, built while this one is scored
        tiers = reply.get("tiers")
        if tiers is None:
            self._note_failure()
            return None
        self._failures = 0
        return tiers

    def close(self):
        with self._lock:
            spare, self._spare = self._spare, None
            self.disabled = True
        if spare is not None:
            spare.kill()                                   # a worker still starting ends by itself once our pipe closes


class WarmAlignerPool:
    """Separate one-shot workers can align concurrently; each owns its dictionary files."""

    def __init__(self, size, *args, **kwargs):
        if size < 1:
            raise ValueError("MFA pool size must be positive")
        self._aligners = [WarmAligner(*args, **kwargs) for _ in range(size)]
        self._available = queue.Queue()
        for aligner in self._aligners:
            self._available.put(aligner)

    def prewarm(self):
        for aligner in self._aligners:
            aligner.prewarm()

    def align(self, wav, txt):
        try:
            aligner = self._available.get_nowait()
        except queue.Empty:
            return None  # callers outside the API may exceed the pool; use the locked fallback
        try:
            return aligner.align(wav, txt)
        finally:
            self._available.put(aligner)

    def close(self):
        for aligner in self._aligners:
            aligner.close()


def enabled_by_environment():
    return os.environ.get("TONE_MFA_WARM", "").strip().lower() in ("1", "true", "yes", "on")
