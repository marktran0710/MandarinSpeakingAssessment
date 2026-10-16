"""
Audio preparation for the tone pipeline.

    recording (webm/mp3/m4a/ogg/wav) -> FFmpeg decode to 16 kHz mono -> quality check on the *raw* signal
    -> peak normalisation -> edge-silence trim

The quality check runs before normalisation (clipping and level are only meaningful on the raw signal) and
before forced alignment, so a bad recording is rejected with a specific reason instead of being analysed.
Functions take the config dict from tone_v1_config.yaml; this module never loads it itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import parselmouth
from scipy.signal import resample_poly

TARGET_RATE = 16000
HOP_SEC = 0.01
# Report order, and the order feedback messages are chosen in.
REASON_PRIORITY = ("clipped", "too_quiet", "noisy", "too_short", "too_long", "no_speech", "no_voicing")


def decode_audio(path, sample_rate=TARGET_RATE):
    """Decode any supported container to mono float32 at `sample_rate` (pydub shells out to FFmpeg)."""
    from pydub import AudioSegment

    with open(path, "rb") as handle:
        segment = AudioSegment.from_file(handle).set_channels(1).set_frame_rate(sample_rate)
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    return samples / float(2 ** (8 * segment.sample_width - 1)), sample_rate


def frame_rms(samples, sample_rate, frame_ms):
    size = max(1, int(sample_rate * frame_ms / 1000))
    hop = max(1, int(sample_rate * HOP_SEC))
    if len(samples) < size:
        return np.array([], dtype=float), hop, size
    frames = np.lib.stride_tricks.sliding_window_view(np.asarray(samples, dtype=np.float64), size)[::hop]
    return np.sqrt(np.mean(frames ** 2, axis=1)), hop, size


def clipped_fraction(samples, clip_level):
    samples = np.asarray(samples)
    return float(np.mean(np.abs(samples) >= clip_level)) if len(samples) else 0.0


def voicing_metrics(samples, sample_rate, min_hnr_db=0.0):
    """(voiced seconds, median harmonics-to-noise ratio over voiced frames); Praat HNR frames above `min_hnr_db`."""
    if len(samples) < sample_rate * 0.1:
        return 0.0, None
    sound = parselmouth.Sound(np.asarray(samples, dtype=np.float64), sampling_frequency=sample_rate)
    hnr = sound.to_harmonicity_cc(time_step=HOP_SEC, minimum_pitch=75.0).values[0]
    voiced = hnr[hnr > min_hnr_db]
    return len(voiced) * HOP_SEC, (float(np.median(voiced)) if len(voiced) else None)


def quality_metrics(samples, sample_rate, config):
    """Measurements the quality check decides on (raw, un-normalised signal at `sample_rate`)."""
    cfg = config["audio_quality"]
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    rms, hop, _ = frame_rms(samples, sample_rate, cfg["frame_ms"])
    loud = float(np.percentile(rms, 95)) if len(rms) else 0.0
    active = rms > cfg["speech_rms_ratio"] * loud if loud > 0 else np.zeros(len(rms), dtype=bool)
    snr_db = None
    if len(rms):
        # Crude and deliberately simple: noise floor ~ the quietest frames, speech level ~ the loud frames.
        noise = float(np.percentile(rms, cfg["snr_noise_percentile"]))
        speech = float(np.percentile(rms, cfg["snr_speech_percentile"]))
        snr_db = 20 * math.log10(max(speech, 1e-9) / max(noise, 1e-9))
    voiced_sec, median_hnr = voicing_metrics(samples, sample_rate, cfg["voiced_min_hnr_db"])
    return {
        "duration_sec": round(len(samples) / float(sample_rate), 3),
        "peak": round(float(np.max(np.abs(samples))) if len(samples) else 0.0, 4),
        "clipped_fraction": round(clipped_fraction(samples, cfg["clip_level"]), 5),
        "speech_sec": round(float(active.sum()) * hop / sample_rate, 3),
        "snr_db": None if snr_db is None else round(snr_db, 1),
        "voiced_sec": round(voiced_sec, 2),
        "median_hnr_db": None if median_hnr is None else round(median_hnr, 1),
    }


def evaluate_quality(metrics, config):
    """Reasons a recording is not usable (empty = passes), ordered by REASON_PRIORITY."""
    cfg = config["audio_quality"]
    reasons = set()
    if metrics["duration_sec"] < cfg["min_duration_sec"]:
        reasons.add("too_short")
    if metrics["duration_sec"] > cfg["max_duration_sec"]:
        reasons.add("too_long")
    if metrics["clipped_fraction"] > cfg["max_clipped_fraction"]:
        reasons.add("clipped")
    if metrics["peak"] < cfg["min_peak"]:
        reasons.add("too_quiet")
    signal_present = metrics["peak"] >= cfg["min_peak"]
    if metrics["speech_sec"] < cfg["min_speech_sec"]:
        reasons.add("no_speech")
    elif signal_present and metrics["voiced_sec"] < cfg["min_voiced_sec"]:
        reasons.add("no_voicing")
    elif signal_present and metrics["median_hnr_db"] is not None             and metrics["voiced_sec"] >= cfg["hnr_min_voiced_sec"] and metrics["median_hnr_db"] < cfg["min_median_hnr_db"]:
        reasons.add("noisy")
    # Noise is only meaningful when there is a signal to be noisy: a silent file is "too_quiet", not "noisy".
    if signal_present and metrics["snr_db"] is not None and metrics["snr_db"] < cfg["min_snr_db"]:
        reasons.add("noisy")
    return [r for r in REASON_PRIORITY if r in reasons]


def normalize_and_trim(samples, sample_rate, config):
    """Peak-normalise and trim long leading/trailing silence (keeps `trim_padding_sec` around the speech)."""
    cfg = config["audio_quality"]
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    if peak > 0:
        samples = samples * (cfg["target_peak"] / peak)
    rms, hop, size = frame_rms(samples, sample_rate, cfg["frame_ms"])
    loud = float(np.percentile(rms, 95)) if len(rms) else 0.0
    if loud <= 0:
        return samples, 0.0, len(samples) / float(sample_rate)
    active = np.flatnonzero(rms > cfg["trim_rms_ratio"] * loud)
    if not len(active):
        return samples, 0.0, len(samples) / float(sample_rate)
    pad = int(cfg["trim_padding_sec"] * sample_rate)
    start = max(0, int(active[0]) * hop - pad)
    end = min(len(samples), int(active[-1]) * hop + size + pad)
    return samples[start:end], start / float(sample_rate), end / float(sample_rate)


@dataclass
class PreparedAudio:
    samples: np.ndarray
    sample_rate: int
    metrics: dict
    reasons: list = field(default_factory=list)
    trim_start_sec: float = 0.0
    trim_end_sec: float = 0.0

    @property
    def passed(self):
        return not self.reasons

    @property
    def rejected_reasons(self):
        return list(self.reasons) or None

    def report(self):
        return {"passed": self.passed, "reasons": list(self.reasons), "metrics": self.metrics,
                "trim_start_sec": round(self.trim_start_sec, 3), "trim_end_sec": round(self.trim_end_sec, 3)}


def slice_prepared(prepared, start_sec, end_sec):
    """A PreparedAudio for part of a prepared recording (same quality verdict, time offset kept)."""
    a, b = int(start_sec * prepared.sample_rate), int(end_sec * prepared.sample_rate)
    return PreparedAudio(prepared.samples[a:b], prepared.sample_rate, prepared.metrics, list(prepared.reasons),
                         prepared.trim_start_sec + start_sec, prepared.trim_start_sec + end_sec)


def speech_chunks(prepared, config):
    """Separate readings inside one recording, as (start_sec, end_sec) in the prepared audio's timeline.

    A pause of at least `variants.split_on_pause_ms` between stretches of speech separates two readings (a clip that
    says both 裡面 and 裡). Stretches shorter than `min_chunk_ms` are ignored; the rest get a little padding."""
    cfg, quality = config["variants"], config["audio_quality"]
    rms, hop, size = frame_rms(prepared.samples, prepared.sample_rate, quality["frame_ms"])
    if not len(rms):
        return []
    loud = float(np.percentile(rms, 95))
    if loud <= 0:
        return []
    active = rms > cfg["chunk_rms_ratio"] * loud
    frame_sec = hop / float(prepared.sample_rate)
    merge_frames = int(round(cfg["split_on_pause_ms"] / 1000.0 / frame_sec))
    stretches = []
    for index in np.flatnonzero(active):
        if stretches and index - stretches[-1][1] <= merge_frames:
            stretches[-1][1] = index
        else:
            stretches.append([index, index])
    spans = [(a * frame_sec, b * frame_sec + size / float(prepared.sample_rate)) for a, b in stretches]
    spans = [(a, b) for a, b in spans if b - a >= cfg["min_chunk_ms"] / 1000.0]
    pad, total = cfg["chunk_padding_ms"] / 1000.0, len(prepared.samples) / float(prepared.sample_rate)
    chunks = []
    for number, (a, b) in enumerate(spans):
        low = (spans[number - 1][1] + a) / 2 if number else 0.0                      # never reach into the neighbour
        high = (b + spans[number + 1][0]) / 2 if number + 1 < len(spans) else total
        chunks.append((max(low, a - pad), min(high, b + pad)))
    longest = sorted(chunks, key=lambda c: c[1] - c[0], reverse=True)[: cfg["max_chunks"]]
    return sorted(longest)


def prepare_samples(samples, sample_rate, config):
    """Resample to 16 kHz, check quality on the raw signal, then normalise and trim."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if int(sample_rate) != TARGET_RATE:
        divisor = math.gcd(TARGET_RATE, int(sample_rate))
        samples = resample_poly(samples, TARGET_RATE // divisor, int(sample_rate) // divisor).astype(np.float32)
    metrics = quality_metrics(samples, TARGET_RATE, config)
    reasons = evaluate_quality(metrics, config)
    prepared, start, end = normalize_and_trim(samples, TARGET_RATE, config)
    return PreparedAudio(prepared, TARGET_RATE, metrics, reasons, start, end)


def prepare_recording(path, config):
    samples, rate = decode_audio(path)
    return prepare_samples(samples, rate, config)
