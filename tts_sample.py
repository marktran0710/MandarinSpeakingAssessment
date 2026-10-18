"""A computer-voice sample for the "My own text" practice: Windows' Taiwan Mandarin voice (Microsoft Hanhan), a little slower than normal speech.

Free and offline: the text typed by the learner never leaves this machine. The sample is only a guide to listen to (it is not used for scoring) and
its pronunciation is the voice's own, so a word can differ slightly from a teacher's reading. Samples are cached on disk (same text, same speed: one synthesis).
"""
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

VOICE = "Microsoft Hanhan Desktop"              # zh-TW; installed with Windows' Chinese (Traditional) speech support
# SAPI speed -10..10 (0 = the voice's normal speed). Measured 2026-10-18 on 16 textbook sentences through our own tone verifier: 0 -> Good 91% of the syllables,
# median syllable 240 ms; -2 -> Good 87%, 290 ms; -4 -> Good 90%, 360 ms (the teacher's recording: about 310 ms). -3 is a touch slower than the teacher.
RATE = -3
MAX_HAN = 80                                    # characters of a sample: a sentence or two
HAN = re.compile(r"[一-鿿]")
KEEP = re.compile(r"[一-鿿，。！？、；：]")
PUNCTUATION = {",": "，", ".": "。", "!": "！", "?": "？", ";": "；", ":": "：", "，": "，", "。": "。", "！": "！", "？": "？", "、": "、", "；": "；", "：": "："}

_LOCK = threading.Lock()                        # one synthesis at a time: each one starts a speech engine
_AVAILABLE = {}


class SampleUnavailable(RuntimeError):
    """No speech voice on this machine (not Windows, or the Taiwan Mandarin voice is not installed), or it failed."""


class BadText(ValueError):
    """Nothing to say, or too long."""


def cache_dir():
    base = os.environ.get("MANDARIN_PRACTICE_DATA_DIR") or Path(__file__).resolve().parent / "practice_data"
    folder = Path(base) / "tts_cache"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def clean_text(text):
    """The part of `text` that is spoken: Han characters and the punctuation that makes a pause (other symbols, digits and spaces are dropped)."""
    text = "".join(PUNCTUATION.get(c, c) for c in str(text or ""))
    spoken = "".join(KEEP.findall(text))
    spoken = re.sub(r"^[，。！？、；：]+", "", spoken)
    if not HAN.search(spoken):
        raise BadText("There is no Chinese text to read.")
    if len(HAN.findall(spoken)) > MAX_HAN:
        raise BadText(f"The text is too long for a sample (at most {MAX_HAN} characters).")
    return spoken


def available():
    """True when the Taiwan Mandarin voice can be used here (checked once)."""
    if "voice" not in _AVAILABLE:
        _AVAILABLE["voice"] = False
        if sys.platform == "win32":
            try:
                script = ("Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() "
                          "| ForEach-Object { $_.VoiceInfo.Name }")
                result = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace")
                _AVAILABLE["voice"] = VOICE in result.stdout
            except (OSError, subprocess.SubprocessError):
                pass
    return _AVAILABLE["voice"]


def sample_path(text, rate=RATE):
    key = hashlib.sha1(f"{VOICE}|{rate}|{text}".encode("utf-8")).hexdigest()[:24]
    return cache_dir() / f"{key}.wav"


def synthesize(text, rate=RATE):
    """Path of the WAV sample for `text` (made now, or taken from the cache). Raises BadText / SampleUnavailable."""
    spoken = clean_text(text)
    path = sample_path(spoken, rate)
    if path.is_file() and path.stat().st_size > 1000:
        return path
    if not available():
        raise SampleUnavailable("The Taiwan Mandarin voice is not available on this server.")
    with _LOCK:
        if path.is_file() and path.stat().st_size > 1000:        # another request made it while this one waited
            return path
        with tempfile.TemporaryDirectory(prefix="tts-") as folder:
            source, target = Path(folder) / "text.txt", Path(folder) / "sample.wav"
            source.write_text(spoken, encoding="utf-8")           # the text goes in through a file, never through the command line
            quote = lambda path: str(path).replace("'", "''")       # noqa: E731  (a PowerShell single-quoted string)
            script = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                      f"$s.SelectVoice('{VOICE}'); $s.Rate = {int(rate)}; $s.SetOutputToWaveFile('{quote(target)}'); "
                      f"$s.Speak([IO.File]::ReadAllText('{quote(source)}', [Text.Encoding]::UTF8)); $s.Dispose()")
            try:
                subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, timeout=60, check=True)
            except (OSError, subprocess.SubprocessError) as exc:
                raise SampleUnavailable(f"The voice could not make the sample: {type(exc).__name__}") from exc
            if not target.is_file() or target.stat().st_size < 1000:
                raise SampleUnavailable("The voice produced no sound.")
            path.write_bytes(target.read_bytes())
    return path
