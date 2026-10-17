"""
Tone verification pipeline v1.

    student calibration -> personalised Praat floor/ceiling -> Praat AC + CC extraction -> sanity checks /
    adaptive retries -> forced alignment (MFA) -> F0 cleanup -> semitone normalisation ->
    lexical -> surface tone -> expected-tone verifier -> confidence gating

The system already knows the transcript, so the verifier asks "is this contour compatible with the surface
tone expected here?" and never "which tone class is this audio?". It is deliberately harder to accuse the
learner of an error than to accept a plausible pronunciation: everything that is not clearly good or clearly
wrong is `insufficient_evidence`. All thresholds live in tone_v1_config.yaml.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import parselmouth
import yaml
from pypinyin import Style, lazy_pinyin
from scipy.io import wavfile
from scipy.signal import medfilt, resample_poly

import audio_prep
import capt_core
import mfa_warm

CONFIG_PATH = Path(__file__).with_name("tone_v1_config.yaml")
MFA_EXE = os.environ.get("MFA_EXE", r"D:\CodexData\mfa-runtime\env\Scripts\mfa.exe")
MFA_ROOT_DIR = os.environ.get("MFA_ROOT_DIR", r"D:\CodexData\mfa-runtime\models")
TONE_LETTERS = set("˥˦˧˨˩")
SILENCE_LABELS = {"", "<eps>", "sil", "sp"}
METHOD = "tone_pipeline_v1"
STATUSES = ("likely_correct", "practice", "insufficient_evidence")


class _MfaExclusive:
    """One MFA process at a time on the whole machine, not just in this process.

    MFA keeps shared state under MFA_ROOT_DIR (the unpacked acoustic model, command_history.yaml). Two MFA processes
    starting together, a second server, a batch evaluation next to the app, can find the model half-written or corrupt the
    history, and the aligner then fails (`alignment_failed`: no pitch chart, every phrase "Okay"). A thread lock cannot see
    another process, so this also takes an operating-system file lock; the OS drops it if the holder dies, so it cannot go
    stale. Use as a context manager."""

    POLL_SEC = 0.05
    COURTESY_SEC = 0.12          # after releasing, give a waiting process a turn (a batch job would otherwise re-take it at once)

    def __init__(self):
        self._thread_lock = threading.Lock()
        self._handle = None

    @staticmethod
    def lock_path():
        return Path(MFA_ROOT_DIR) / ".tonepipe_mfa.lock"

    def __enter__(self):
        self._thread_lock.acquire()
        try:
            path = self.lock_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(path, "a+b")
        except OSError:
            return self                                  # no lock file possible: the thread lock still serialises this process
        try:
            while not self._try_lock(handle):
                time.sleep(self.POLL_SEC)
        except BaseException:
            handle.close()
            self._thread_lock.release()
            raise
        self._handle = handle
        return self

    def __exit__(self, *exc_info):
        handle, self._handle = self._handle, None
        try:
            if handle is not None:
                self._unlock(handle)
                handle.close()
                time.sleep(self.COURTESY_SEC)
        finally:
            self._thread_lock.release()
        return False

    @staticmethod
    def _try_lock(handle):
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    @staticmethod
    def _unlock(handle):
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_UN)
        except OSError:
            pass


_MFA_LOCK = _MfaExclusive()


class AlignmentError(ValueError):
    """The forced alignment could not be trusted (OOV character, <unk>, spn, word mismatch...)."""


class CalibrationError(ValueError):
    """The calibration recording is not good enough to build a speaker profile."""

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


def load_config(path=None, overrides=None):
    with open(path or CONFIG_PATH, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for section, values in (overrides or {}).items():
        config.setdefault(section, {}).update(values)
    return config


def _round_to(value, step, mode):
    ratio = value / step
    return step * {"nearest": round, "ceil": math.ceil}[mode](ratio)


A4_HZ, A4_MIDI = 440.0, 69.0


def hz_to_midi(f0):
    """MIDI note number: 69 + 12 * log2(F0 / 440). One unit is one semitone; A4 (440 Hz) is 69."""
    return A4_MIDI + 12.0 * np.log2(np.asarray(f0, dtype=float) / A4_HZ)


def hz_to_st(f0, reference_hz):
    """Semitones above a reference pitch (the speaker's median) = MIDI(F0) - MIDI(reference) = 12 * log2(F0 / reference).

    The 69 and the 440 of the MIDI formula cancel, and every rule measures differences (a fall, a rise, a range), so the
    rules are the same on the MIDI scale and on this speaker-relative scale. It is computed as the ratio, not as two MIDI
    numbers: adding and cancelling the offsets leaves rounding noise of about 1e-14, which flipped a tie between two
    equally good boundaries in the boundary search on one native syllable. Absolute pitch is reported with hz_to_midi."""
    return 12.0 * np.log2(np.asarray(f0, dtype=float) / float(reference_hz))


def _st_diff(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return 12.0 * np.log2(a / b)


# ------------------------------------------------------------ Praat AC + CC ----

def run_praat_ac_cc(samples, sample_rate, floor, ceiling, config):
    """Run To Pitch (ac) and (cc) with the frozen config; CC is resampled onto the AC time grid."""
    pitch_cfg = config["pitch"]
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    if len(samples) == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty
    sound = parselmouth.Sound(samples, sampling_frequency=sample_rate)
    tracks = {}
    for method in ("ac", "cc"):
        pitch = getattr(sound, f"to_pitch_{method}")(
            time_step=float(pitch_cfg["time_step"]),
            pitch_floor=float(floor),
            max_number_of_candidates=int(pitch_cfg["max_candidates"]),
            very_accurate=bool(pitch_cfg["very_accurate"]),
            silence_threshold=float(pitch_cfg["silence_threshold"]),
            voicing_threshold=float(pitch_cfg["voicing_threshold"]),
            octave_cost=float(pitch_cfg["octave_cost"]),
            octave_jump_cost=float(pitch_cfg["octave_jump_cost"]),
            voiced_unvoiced_cost=float(pitch_cfg["voiced_unvoiced_cost"]),
            pitch_ceiling=float(ceiling),
        )
        f0 = np.asarray(pitch.selected_array["frequency"], dtype=float)
        f0[~(np.isfinite(f0) & (f0 > 0))] = np.nan
        tracks[method] = (np.asarray(pitch.xs(), dtype=float), f0)
    times, ac = tracks["ac"]
    cc = _resample_nearest(times, *tracks["cc"], max_gap=float(pitch_cfg["time_step"]) * 0.55)
    return times, ac, cc


def _resample_nearest(target_times, times, values, max_gap):
    out = np.full(len(target_times), np.nan)
    if not len(times) or not len(target_times):
        return out
    index = np.clip(np.searchsorted(times, target_times), 1, max(len(times) - 1, 1))
    index = np.minimum(index, len(times) - 1)
    left_closer = np.abs(target_times - times[np.maximum(index - 1, 0)]) <= np.abs(times[index] - target_times)
    nearest = np.where(left_closer, np.maximum(index - 1, 0), index)
    close = np.abs(times[nearest] - target_times) <= max_gap
    out[close] = values[nearest[close]]
    return out


# ------------------------------------------------------------- calibration ----

def build_profile(stable_hz, config):
    """Percentiles of agreed F0 -> personal floor/ceiling. No recording-level gates (see calibrate_voice)."""
    cal = config["calibration"]
    p10, median, p90, p95 = [float(v) for v in np.percentile(stable_hz, [10, 50, 90, 95])]
    floor = float(np.clip(
        _round_to(cal["floor_scale"] * median, cal["floor_round_to"], "nearest"),
        cal["floor_min"], cal["floor_max"]))
    ceiling = float(np.clip(
        _round_to(cal["ceiling_scale"] * p95, cal["ceiling_round_to"], "ceil"),
        cal["ceiling_min"], cal["ceiling_max"]))
    return {
        "status": "calibrated",
        "calibration_version": config["calibration_version"],
        "method": f"{METHOD}.calibration",
        "f0_p10": round(p10, 1),
        "f0_median": round(median, 1),
        "f0_p90": round(p90, 1),
        "f0_p95": round(p95, 1),
        "pitch_floor": floor,
        "pitch_ceiling": ceiling,
        "acoustic_profile": acoustic_profile(median, config),
        # Aliases kept for practice_store and the existing UI.
        "pitch_floor_hz": floor,
        "pitch_ceiling_hz": ceiling,
        "f0_hz": {"p10": round(p10, 1), "p50": round(median, 1), "p90": round(p90, 1),
                  "p95": round(p95, 1), "baseline": round(median, 1)},
    }


def acoustic_profile(median_hz, config):
    """low / mid / high / very_high by calibrated median F0 (an acoustic label, not a sex/gender label)."""
    profiles = config["profiles"]
    if median_hz < profiles["low_max_median_hz"]:
        return "low"
    if median_hz < profiles["mid_max_median_hz"]:
        return "mid"
    if median_hz < profiles["high_max_median_hz"]:
        return "high"
    return "very_high"


def wide_pass_stable_frames(samples, sample_rate, config):
    """Wide AC+CC pass. Returns the agreed frames (Hz) plus the raw material for the quality gate."""
    cal, tol = config["calibration"], config["sanity"]["ac_cc_tolerance_st"]
    times, ac, cc = run_praat_ac_cc(samples, sample_rate, cal["wide_floor"], cal["wide_ceiling"], config)
    both = np.isfinite(ac) & np.isfinite(cc)
    agree = both & (np.abs(_st_diff(ac, cc)) <= tol)
    stable = np.full(len(ac), np.nan)
    stable[agree] = np.sqrt(ac[agree] * cc[agree])
    return {"times": times, "ac": ac, "cc": cc, "both": both, "agree": agree, "stable": stable}


def reject_octave_candidates(stable_hz, config):
    """Drop agreed frames that sit an octave-ish distance from the speaker's first-pass median."""
    stable_hz = np.asarray(stable_hz, dtype=float)
    if not len(stable_hz):
        return stable_hz, 0
    first_median = float(np.median(stable_hz))
    keep = np.abs(hz_to_st(stable_hz, first_median)) <= config["calibration"]["octave_reject_st"]
    return stable_hz[keep], int((~keep).sum())


def analyze_calibration_tokens(times, voiced, config):
    """Report on the first N voiced stretches (isolated tokens). Advisory: only produces warnings."""
    cal = config["calibration"]
    step = float(np.median(np.diff(times))) if len(times) > 1 else config["pitch"]["time_step"]
    bridge = int(round(cal["token_gap_sec"] / step))
    segments = []
    for start, end in _voiced_runs(voiced):
        if segments and start - segments[-1][1] <= bridge:
            segments[-1][1] = end
        else:
            segments.append([start, end])
    tokens = []
    for start, end in segments[: cal["token_count"]]:
        frames = int(voiced[start:end].sum())
        span = float((end - start) * step)
        tokens.append({      # plain Python types only: the profile is stored as JSON
            "start_sec": round(float(times[start]), 3), "voiced_span_sec": round(span, 3), "voiced_frames": frames,
            "meets_minimum": bool(span >= cal["token_min_voiced_sec"] and frames >= cal["token_min_frames"]),
            "meets_preferred": bool(span >= cal["token_preferred_voiced_sec"] and frames >= cal["token_preferred_frames"]),
        })
    warnings = []
    if len(segments) < cal["token_count"]:
        warnings.append(f"Only {len(segments)} separate voiced stretches found; expected {cal['token_count']} isolated tokens.")
    short = [i + 1 for i, t in enumerate(tokens) if not t["meets_minimum"]]
    if short:
        warnings.append(f"Calibration tokens {short} are shorter than the minimum usable voicing.")
    return {"tokens": tokens, "warnings": warnings}


def calibration_pace_hint(profile, config=None):
    """"slower" when the calibration reading was rushed (fewer separate voiced stretches than sounds, or at least two of the first sounds shorter
    than the minimum usable voicing): the learner is told to read slower and stretch each sound. "" when the reading was fine."""
    cal = (config or load_config())["calibration"]
    tokens = ((profile.get("quality") or {}).get("tokens")) or []
    short = sum(1 for token in tokens if not token.get("meets_minimum"))
    return "slower" if len(tokens) < cal["token_count"] or short >= 2 else ""


def calibrate_voice(samples, sample_rate, config=None):
    """Quality-gated speaker F0 profile. Raises CalibrationError instead of creating a poor profile."""
    config = config or load_config()
    cal = config["calibration"]
    raw = np.asarray(samples, dtype=np.float64).reshape(-1)
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    duration = len(raw) / float(sample_rate)
    if duration < cal["min_duration_sec"]:
        raise CalibrationError("too_short", "Record a calibration sample of at least 5 seconds.")
    if duration > cal["max_duration_sec"]:
        raise CalibrationError("too_long", "Record a calibration sample of at most about 10 seconds.")
    clipped = audio_prep.clipped_fraction(raw, cal["clip_level"])
    if clipped > cal["max_clipped_fraction"]:
        raise CalibrationError("clipped", "The recording is clipped. Move back from the microphone and try again.")

    audio = capt_core.normalize_audio(raw)
    wide = wide_pass_stable_frames(audio, sample_rate, config)
    step = float(config["pitch"]["time_step"])

    voiced_any = np.isfinite(wide["ac"]) | np.isfinite(wide["cc"])
    edge_hits = lambda track: (  # noqa: E731
        np.isfinite(track) & ((track <= cal["wide_floor"] + cal["boundary_margin_hz"])
                              | (track >= cal["wide_ceiling"] - cal["boundary_margin_hz"])))
    boundary_rate = float((edge_hits(wide["ac"]) | edge_hits(wide["cc"])).sum() / voiced_any.sum()) if voiced_any.any() else 0.0
    if boundary_rate > cal["boundary_hit_rate"]:
        raise CalibrationError(
            "boundary_hits", "Pitch kept hitting the tracker limits; record again in a quieter place.")

    agreement = float(wide["agree"].sum() / wide["both"].sum()) if wide["both"].any() else 0.0
    if agreement < cal["hard_min_agreement"]:
        raise CalibrationError(
            "ac_cc_disagree", "Pitch tracking was not reliable. Read the prompt clearly and try again.")

    stable_hz, octave_rejected = reject_octave_candidates(wide["stable"][np.isfinite(wide["stable"])], config)
    voiced_sec = len(stable_hz) * step
    if len(stable_hz) < cal["min_stable_frames"] or voiced_sec < cal["min_voiced_sec"]:
        raise CalibrationError(
            "too_little_voicing",
            "Not enough voiced speech was found (a whisper cannot be used). Speak the full prompt aloud.")

    _, median_hnr = audio_prep.voicing_metrics(audio, sample_rate, config["audio_quality"]["voiced_min_hnr_db"])
    median_hnr = float("-inf") if median_hnr is None else median_hnr
    if median_hnr < cal["min_median_hnr_db"]:
        raise CalibrationError("noisy", "The recording is too noisy or breathy. Record again in a quiet place.")

    profile = build_profile(stable_hz, config)
    token_report = analyze_calibration_tokens(wide["times"], np.isfinite(wide["stable"]), config)
    warnings = list(token_report["warnings"])
    reliable = agreement >= cal["preferred_agreement"]
    if not reliable:
        warnings.append("AC/CC agreement is below the preferred level; the profile may be less precise.")
    profile.update({
        "sample_duration_sec": round(duration, 3),
        "stable_frame_count": int(len(stable_hz)),
        "ac_cc_agreement": round(agreement, 4),
        "reliable": reliable,
        "quality": {
            "voiced_sec": round(voiced_sec, 2), "clipped_fraction": round(clipped, 5),
            "median_hnr_db": round(median_hnr, 1), "boundary_hit_rate": round(boundary_rate, 4),
            "octave_candidates_rejected": octave_rejected, "tokens": token_report["tokens"],
        },
        "warnings": warnings,
    })
    return profile


def pooled_calibration(recordings, config=None):
    """Speaker profile from several recordings (benchmarks only: no recording-level gates)."""
    config = config or load_config()
    frames = []
    for samples, sample_rate in recordings:
        wide = wide_pass_stable_frames(capt_core.normalize_audio(samples), sample_rate, config)
        frames.append(wide["stable"][np.isfinite(wide["stable"])])
    stable, _ = reject_octave_candidates(np.concatenate(frames) if frames else np.array([]), config)
    if len(stable) < config["calibration"]["min_stable_frames"]:
        raise CalibrationError("too_little_voicing", "Not enough voiced frames to pool a speaker profile.")
    profile = build_profile(stable, config)
    profile["pooled_from_recordings"] = len(recordings)
    return profile


# --------------------------------------------------- merge + sanity checks ----

def merge_ac_cc(ac, cc, tolerance_st):
    """Agreeing frames -> geometric mean; single-tracker frames kept; disagreements left to `resolve`."""
    merged = np.full(len(ac), np.nan)
    both = np.isfinite(ac) & np.isfinite(cc)
    agree = both & (np.abs(_st_diff(ac, cc)) <= tolerance_st)
    merged[agree] = np.sqrt(ac[agree] * cc[agree])
    only_ac = np.isfinite(ac) & ~np.isfinite(cc)
    only_cc = np.isfinite(cc) & ~np.isfinite(ac)
    merged[only_ac] = ac[only_ac]
    merged[only_cc] = cc[only_cc]
    return merged, both & ~agree


def _local_reference(values, index, radius):
    lo = max(0, index - radius)
    window = values[lo: index + radius + 1].copy()
    window[index - lo] = np.nan
    window = window[np.isfinite(window)]
    return float(np.median(window)) if len(window) >= 3 else None


def resolve_disagreements(merged, ac, cc, disagree, config):
    """Where AC and CC disagree keep whichever tracker is continuous with its neighbours, else drop."""
    radius = int(config["cleanup"]["octave_context_frames"])
    out = merged.copy()
    for index in np.flatnonzero(disagree):
        reference = _local_reference(merged, index, radius)
        if reference is None:
            continue
        best = min((ac[index], cc[index]), key=lambda v: abs(_st_diff(v, reference)))
        if abs(_st_diff(best, reference)) <= 3.0:
            out[index] = best
    return out


def count_octave_jumps(values, threshold_st, max_gap_frames=2, in_semitones=False):
    """Adjacent-frame jumps >= threshold. `values` are Hz unless `in_semitones` is set."""
    idx = np.flatnonzero(np.isfinite(values))
    if len(idx) < 2:
        return 0
    near = np.diff(idx) <= max_gap_frames
    if in_semitones:
        steps = np.abs(values[idx[1:]] - values[idx[:-1]])
    else:
        steps = np.abs(_st_diff(values[idx[1:]], values[idx[:-1]]))
    return int(np.count_nonzero(near & (steps >= threshold_st)))


def trim_jump_fragments(st, step_sec, jump_st, max_fragment_ms, max_gap_frames=2):
    """Blank the short voiced fragment that lies behind an adjacent-frame jump >= `jump_st` at either END of `st`.

    A jump that size is a tracking error (an octave slip on a creaky tail), never a tone movement, and in learner speech it is mostly a
    fragment of a few frames at the very end of the syllable. When that fragment lasts at most `max_fragment_ms` it is dropped and the
    rest of the syllable is judged; a jump in the middle, or behind a long fragment, is left alone (the syllable is still gated).
    Returns (trimmed copy, number of frames dropped)."""
    out = np.array(st, dtype=float)
    dropped = 0
    max_frames = int(max_fragment_ms / 1000.0 / step_sec + 1e-9)
    for _ in range(4):
        idx = np.flatnonzero(np.isfinite(out))
        if len(idx) < 3 or max_frames < 1:
            break
        jumps = np.flatnonzero((np.diff(idx) <= max_gap_frames) & (np.abs(out[idx[1:]] - out[idx[:-1]]) >= jump_st))
        if not len(jumps):
            break
        tail, head = idx[jumps[-1] + 1:], idx[: jumps[0] + 1]
        if len(tail) <= max_frames and len(tail) < len(idx) - len(tail):
            out[tail] = np.nan
            dropped += len(tail)
        elif len(head) <= max_frames and len(head) < len(idx) - len(head):
            out[head] = np.nan
            dropped += len(head)
        else:
            break
    return out, dropped


def segment_pitch(clean, seg_start, seg_end, config):
    """(times, st, frames dropped) of one syllable's analysis segment, with the short fragments behind tracking jumps dropped when
    `sanity.jump_fragment_trim_ms` asks for it (0 = off), and how many frames were dropped."""
    mask = (clean["times"] >= seg_start) & (clean["times"] < seg_end)
    times, st = clean["times"][mask], clean["st"][mask]
    limit = config["sanity"].get("jump_fragment_trim_ms", 0)
    dropped = 0
    if limit:
        st, dropped = trim_jump_fragments(st, clean.get("step") or config["pitch"]["time_step"], config["sanity"]["octave_jump_st"], limit)
    return times, st, dropped


def sanity_metrics(merged, ac, cc, floor, ceiling, config):
    san = config["sanity"]
    both = np.isfinite(ac) & np.isfinite(cc)
    agree = both & (np.abs(_st_diff(ac, cc)) <= san["ac_cc_tolerance_st"])
    agreement = float(agree.sum() / both.sum()) if both.any() else 0.0
    voiced = np.isfinite(merged)
    n_voiced = int(voiced.sum())
    either = int((np.isfinite(ac) | np.isfinite(cc)).sum())
    margin = san["boundary_margin"]
    near_ceiling = voiced & (np.nan_to_num(merged) >= ceiling * (1 - margin))
    near_floor = voiced & (np.nan_to_num(merged, nan=1e9) <= floor * (1 + margin))
    ceiling_rate = float(near_ceiling.sum() / n_voiced) if n_voiced else 0.0
    floor_rate = float(near_floor.sum() / n_voiced) if n_voiced else 0.0
    jumps = count_octave_jumps(merged, san["octave_jump_st"])
    coverage_ratio = n_voiced / either if either else 1.0
    reasons = []
    if agreement < san["min_agreement"]:
        reasons.append("ac_cc_disagree")
    if ceiling_rate > san["boundary_hit_rate"]:
        reasons.append("ceiling_hit")
    if floor_rate > san["boundary_hit_rate"]:
        reasons.append("floor_hit")
    if jumps:
        reasons.append("octave_jump")
    if either >= 20 and coverage_ratio < san["min_coverage_ratio"]:
        reasons.append("coverage_collapse")
    return {
        "ac_cc_agreement": round(agreement, 4),
        "ceiling_hit_rate": round(ceiling_rate, 4),
        "floor_hit_rate": round(floor_rate, 4),
        "octave_jumps": jumps,
        "voiced_frames": n_voiced,
        "coverage_ratio": round(coverage_ratio, 4),
        "suspicious": reasons,
    }


def _penalty(metrics, jump_penalty=0.5):
    return ((1 - metrics["ac_cc_agreement"]) + metrics["ceiling_hit_rate"] + metrics["floor_hit_rate"]
            + min(metrics["octave_jumps"], 2) * jump_penalty)


def octave_up_fraction(f0_a, f0_b, config):
    """Fraction of co-voiced frames where A sits about one octave above B."""
    both = np.isfinite(f0_a) & np.isfinite(f0_b)
    if both.sum() < 6:
        return 0.0
    tol = config["cleanup"]["octave_tolerance_st"]
    return float(np.mean(np.abs(_st_diff(f0_a[both], f0_b[both]) - 12.0) <= tol))


def _retry_candidates(floor, ceiling, reasons, config):
    """Only widen the direction that looks suspicious; never the full 3x3 grid."""
    san = config["sanity"]
    cstep, fstep = san["ceiling_retry_step_hz"], san["floor_retry_step_hz"]
    lo, hi = san["ceiling_retry_min"], san["ceiling_retry_max"]
    out = []
    if "ceiling_hit" in reasons:
        out.append((floor, min(hi, ceiling + cstep)))
    if "floor_hit" in reasons:
        out.append((max(san["floor_retry_min"], floor - fstep), ceiling))
    if {"octave_jump", "ac_cc_disagree", "coverage_collapse"} & set(reasons):
        out.append((floor, max(lo, ceiling - cstep)))
        out.append((floor + fstep, ceiling))
    seen, unique = {(floor, ceiling)}, []
    for candidate in out:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def select_pass(passes, config):
    """Best pass by agreement, boundary hits, octave jumps, coverage and consistency with the other passes."""
    san = config["sanity"]
    max_voiced = max(p["metrics"]["voiced_frames"] for p in passes) or 1
    scored = []
    for index, p in enumerate(passes):
        score = _penalty(p["metrics"], san["octave_jump_penalty"])
        score += san["coverage_weight"] * (1 - p["metrics"]["voiced_frames"] / max_voiced)
        for other in passes:
            if other is p:
                continue
            # Passes with different floors have slightly different frame grids.
            other_f0 = _resample_nearest(p["times"], other["times"], other["f0"], config["pitch"]["time_step"] * 0.55)
            if octave_up_fraction(p["f0"], other_f0, config) >= san["octave_branch_fraction"]:
                score += san["octave_branch_penalty"]
                p.setdefault("flags", []).append("octave_above_other_pass")
        scored.append((score, index))
    return passes[min(scored)[1]]


def extract_f0(samples, sample_rate, floor, ceiling, config=None):
    """Adaptive AC+CC extraction with the sanity check / directional retry."""
    config = config or load_config()
    tol = config["sanity"]["ac_cc_tolerance_st"]

    def one_pass(f, c):
        times, ac, cc = run_praat_ac_cc(samples, sample_rate, f, c, config)
        merged, disagree = merge_ac_cc(ac, cc, tol)
        merged = resolve_disagreements(merged, ac, cc, disagree, config)
        return {"floor": f, "ceiling": c, "times": times, "ac": ac, "cc": cc, "f0": merged,
                "metrics": sanity_metrics(merged, ac, cc, f, c, config)}

    first = one_pass(floor, ceiling)
    passes = [first]
    if first["metrics"]["suspicious"]:
        for f, c in _retry_candidates(floor, ceiling, first["metrics"]["suspicious"], config):
            passes.append(one_pass(f, c))
    best = select_pass(passes, config) if len(passes) > 1 else first
    support = [
        _resample_nearest(best["times"], p["times"], p["f0"], config["pitch"]["time_step"] * 0.55)
        for p in passes if p is not best
    ]
    unresolved = {"ac_cc_disagree", "coverage_collapse"} & set(best["metrics"]["suspicious"])
    return {
        "times": best["times"], "f0": best["f0"], "ac": best["ac"], "cc": best["cc"],
        "support_tracks": support,
        "pitch_floor": best["floor"], "pitch_ceiling": best["ceiling"],
        "diagnostics": {
            **best["metrics"],
            "retried": len(passes) > 1,
            "stable": not unresolved,
            "passes": [{"floor": p["floor"], "ceiling": p["ceiling"], **p["metrics"], "flags": p.get("flags", [])}
                       for p in passes],
        },
    }


# ----------------------------------------------- cleanup + semitone normalisation ----

def correct_octaves(f0, sources, config):
    """Halve/double a frame only when it is an octave off its neighbours AND another source supports it."""
    clean = config["cleanup"]
    radius = int(clean["octave_context_frames"])
    tol = clean["octave_tolerance_st"]
    out = f0.copy()
    corrected = np.zeros(len(f0), dtype=bool)
    for index in np.flatnonzero(np.isfinite(f0)):
        reference = _local_reference(f0, index, radius)
        if reference is None:
            continue
        offset = _st_diff(f0[index], reference)
        if not 12.0 - tol <= abs(offset) <= 12.0 + tol:
            continue
        candidate = f0[index] / 2.0 if offset > 0 else f0[index] * 2.0
        supported = any(
            np.isfinite(src[index]) and abs(_st_diff(src[index], candidate)) <= tol for src in sources
        )
        if supported and abs(_st_diff(candidate, reference)) < abs(offset):
            out[index] = candidate
            corrected[index] = True
    return out, corrected


def _voiced_runs(mask):
    padded = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(padded == 1), np.flatnonzero(padded == -1)))


def smooth_and_interpolate(st, step_sec, config):
    """Median filter inside voiced runs; interpolate only gaps <= interpolate_gap_ms."""
    clean = config["cleanup"]
    window = int(clean["median_filter_frames"]) | 1
    out = st.copy()
    for start, end in _voiced_runs(np.isfinite(st)):
        run = st[start:end]
        w = min(window, (len(run) // 2) * 2 + 1)
        if w >= 3:
            padded = np.pad(run, w // 2, mode="edge")
            out[start:end] = medfilt(padded, w)[w // 2: w // 2 + len(run)]
    max_gap = int(round(clean["interpolate_gap_ms"] / 1000.0 / step_sec))
    runs = _voiced_runs(np.isfinite(out))
    for (_, left_end), (right_start, _) in zip(runs, runs[1:]):
        if 0 < right_start - left_end <= max_gap:
            out[left_end - 1: right_start + 1] = np.interp(
                np.arange(left_end - 1, right_start + 1), [left_end - 1, right_start],
                [out[left_end - 1], out[right_start]])
    return out


def _frame_confidence(f0, ac, cc, extra_tracks, tolerance_st):
    """Per-frame trust in the merged F0: 1.0 when AC and CC agree, 0.6 when one tracker is confirmed by another
    pass, 0.3 for a single unconfirmed tracker or a disagreement that had to be resolved."""
    confidence = np.zeros(len(f0))
    for index in np.flatnonzero(np.isfinite(f0)):
        a, c = ac[index], cc[index]
        if np.isfinite(a) and np.isfinite(c) and abs(_st_diff(a, c)) <= tolerance_st:
            confidence[index] = 1.0
        elif any(np.isfinite(t[index]) and abs(_st_diff(t[index], f0[index])) <= tolerance_st for t in extra_tracks):
            confidence[index] = 0.6
        else:
            confidence[index] = 0.3
    return confidence


def _split_at_jumps(f0, jump_st, max_gap_frames=2):
    """Runs of consecutive voiced frames; a gap longer than max_gap_frames or a jump >= jump_st starts a new run."""
    runs, current = [], []
    last = None
    for index in np.flatnonzero(np.isfinite(f0)):
        if last is not None and (index - last > max_gap_frames or abs(_st_diff(f0[index], f0[last])) >= jump_st):
            runs.append(current)
            current = []
        current.append(index)
        last = index
    if current:
        runs.append(current)
    return runs


def correct_octave_runs(f0, ac, cc, extra_tracks, baseline_hz, config):
    """Fix a sustained octave error: two neighbouring runs exactly an octave apart, one of them mis-tracked.

    The run to move is the less trustworthy one: fewer frames, less AC/CC agreement, no confirmation from the other
    tracker or passes, and (for the speaker) a pitch far outside their usual range. Nothing is moved unless one run
    is clearly weaker than the other and the junction is smooth after the move. Jumps that are not an octave
    (creaky voice, a new syllable) are left alone."""
    clean = config["cleanup"]
    out = f0.copy()
    corrected = np.zeros(len(f0), dtype=bool)
    if not clean["octave_run_correction"]:
        return out, corrected
    tol, jump = clean["octave_tolerance_st"], config["sanity"]["octave_jump_st"]
    confidence = _frame_confidence(f0, ac, cc, extra_tracks, config["sanity"]["ac_cc_tolerance_st"])

    def score(run):
        median_st = float(np.median(hz_to_st(out[run], baseline_hz)))
        in_range = abs(median_st) <= clean["octave_range_st"]
        return float(confidence[run].sum()) * (1.0 if in_range else 0.5)

    for _ in range(4):                                   # shifting one run can expose the next junction
        changed = False
        runs = _split_at_jumps(out, jump)
        for left, right in zip(runs, runs[1:]):
            if right[0] - left[-1] > 2:
                continue                                 # separated by a gap: different syllables, not a jump
            # Compare the two frames at the junction (the pair that made the jump), not run medians: a glitch run
            # can sit an octave off its neighbour at the join while a long, sloping neighbour has another median.
            step = float(_st_diff(out[right[0]], out[left[-1]]))
            if not 12.0 - tol <= abs(step) <= 12.0 + tol:
                continue                                 # not an octave: creak or a syllable change, leave it
            weaker, stronger = (right, left) if score(right) < score(left) else (left, right)
            if score(stronger) < clean["octave_run_decisive_ratio"] * score(weaker):
                continue                                 # neither run is clearly the mis-tracked one
            # step > 0: the right run is an octave above the left one. Move the weaker run onto the stronger.
            move = -12.0 if (weaker is right) == (step > 0) else 12.0
            candidate = out[weaker] * 2.0 ** (move / 12.0)
            moved_edge = candidate[-1] if weaker is left else candidate[0]        # frame at the junction
            other_edge = out[right[0]] if weaker is left else out[left[-1]]
            if abs(_st_diff(moved_edge, other_edge)) > clean["octave_run_continuity_st"]:
                continue                                 # moving it does not restore a smooth contour
            out[weaker] = candidate
            corrected[weaker] = True
            changed = True
            break
        if not changed:
            break
    return out, corrected


def clean_contour(extraction, baseline_hz, config):
    """Octave correction -> median filter -> short-gap interpolation -> semitones re. speaker median."""
    sources = [extraction["ac"], extraction["cc"], *extraction["support_tracks"]]
    f0, corrected = correct_octaves(extraction["f0"], sources, config)
    f0, run_corrected = correct_octave_runs(
        f0, extraction["ac"], extraction["cc"], extraction["support_tracks"], baseline_hz, config)
    corrected = corrected | run_corrected
    times = extraction["times"]
    step = float(np.median(np.diff(times))) if len(times) > 1 else config["pitch"]["time_step"]
    with np.errstate(invalid="ignore", divide="ignore"):
        st = hz_to_st(f0, baseline_hz)
    return {
        "times": times,
        "st": smooth_and_interpolate(st, step, config),
        "observed": np.isfinite(extraction["f0"]),
        "octave_corrected": corrected,
        "step": step,
    }


# --------------------------------------------------------------- alignment ----

def _write_wav_16k_mono(samples, sample_rate, path):
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    if int(sample_rate) != 16000:
        divisor = math.gcd(16000, int(sample_rate))
        samples = resample_poly(samples, 16000 // divisor, int(sample_rate) // divisor)
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    if peak > 1.0:
        samples = samples / peak
    wavfile.write(path, 16000, (samples * 32767).astype(np.int16))


def dictionary_fingerprint(config):
    """Short hash of the MFA dictionary content (or its name when it is an installed dictionary).

    Anything that caches an alignment must include this: a cached alignment (or cached failure) made with an
    older dictionary is wrong once characters have been added."""
    name = config["alignment"]["mfa_dictionary"]
    path = CONFIG_PATH.parent / name
    return hashlib.sha1(path.read_bytes() if path.is_file() else name.encode("utf-8")).hexdigest()[:10]


def _dictionary_argument(dictionary, workdir):
    """MFA caches a compiled dictionary by *file name*, so a rebuilt dictionary with an unchanged name silently
    keeps the old words (every new character aligns as <unk>). Hand MFA a copy whose name carries a content hash."""
    path = CONFIG_PATH.parent / dictionary
    if not path.is_file():
        return dictionary                      # an installed MFA dictionary name
    digest = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    named = Path(workdir) / f"{path.stem}_{digest}.dict"
    shutil.copyfile(path, named)
    return str(named)


_WARM = {}
_WARM_LOCK = threading.Lock()


def warm_aligner(config=None):
    """The pre-loaded aligner (see mfa_warm.py), or None when it is switched off (TONE_MFA_WARM=1 turns it on)."""
    if not mfa_warm.enabled_by_environment():
        return None
    config = config or load_config()
    align = config["alignment"]
    python = Path(os.environ.get("MFA_PYTHON") or Path(MFA_EXE).parent.parent / "python.exe")
    if not python.exists():
        return None
    key = (align["mfa_dictionary"], align["mfa_acoustic_model"])
    with _WARM_LOCK:
        if key not in _WARM:
            if _WARM:                                          # one spare only: another model would double the memory
                return None
            aligner = mfa_warm.WarmAligner(
                python, {**os.environ, "MFA_ROOT_DIR": MFA_ROOT_DIR, "PYTHONIOENCODING": "utf-8"},
                align["mfa_acoustic_model"], lambda folder: _dictionary_argument(align["mfa_dictionary"], folder),
                lambda: dictionary_fingerprint(config), align["mfa_timeout_sec"], load_lock=_MFA_LOCK)
            atexit.register(aligner.close)
            _WARM[key] = aligner
        return _WARM[key]


def prewarm_mfa(config=None):
    """Start loading the spare now (called once when the server starts) so even the first learner is fast."""
    aligner = warm_aligner(config)
    if aligner is not None:
        aligner.prewarm()
    return aligner is not None


def mfa_align(samples, sample_rate, chars, config=None):
    """Align the fixed script (no ASR, no tokenization) and return the MFA tiers.

    Uses the pre-loaded aligner when it is on and ready, otherwise runs `mfa align_one`; both give the same tiers."""
    config = config or load_config()
    align = config["alignment"]
    if not Path(MFA_EXE).exists():
        raise AlignmentError(f"MFA executable not found: {MFA_EXE}")
    workdir = Path(tempfile.mkdtemp(prefix="tonepipe-mfa-"))
    try:
        wav, txt, result_path = workdir / "utterance.wav", workdir / "utterance.txt", workdir / "result.json"
        _write_wav_16k_mono(samples, sample_rate, wav)
        txt.write_text(" ".join(chars), encoding="utf-8")
        warm = warm_aligner(config)
        if warm is not None:
            tiers = warm.align(wav, txt)
            if tiers is not None:
                return tiers
        # align_one's last argument is the output *file*; a directory exits 0 and writes nothing.
        command = [
            MFA_EXE, "align_one", "--output_format", "json", "--no_tokenization",
            "--no_textgrid_cleanup", "--no_use_mp", "--no_use_threading", "--num_jobs", "1",
            "--single_speaker", "--quiet", str(wav), str(txt),
            align["mfa_dictionary"], align["mfa_acoustic_model"], str(result_path),
        ]
        command[command.index(align["mfa_dictionary"])] = _dictionary_argument(align["mfa_dictionary"], workdir)
        env = {**os.environ, "MFA_ROOT_DIR": MFA_ROOT_DIR}
        for attempt in (1, 2):
            with _MFA_LOCK:  # one MFA process at a time on this machine (see _MfaExclusive)
                try:
                    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                            env=env, timeout=align["mfa_timeout_sec"])
                except subprocess.TimeoutExpired:
                    raise AlignmentError("MFA timed out.")
            if result.returncode == 0 and result_path.exists():
                return json.loads(result_path.read_text(encoding="utf-8"))["tiers"]
            if attempt == 2:
                raise AlignmentError(f"MFA failed (exit {result.returncode}): {(result.stderr or result.stdout)[-400:]}")
            time.sleep(0.5)                              # a failed start is usually transient (memory pressure, a racing process): once more
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def syllables_from_tiers(tiers, chars, config=None):
    """Turn MFA word/phone tiers into one syllable per character, with the vowel nucleus located."""
    align = (config or load_config())["alignment"]
    words = [e for e in tiers["words"]["entries"] if e[2] not in SILENCE_LABELS]
    phones = tiers["phones"]["entries"]
    if len(words) != len(chars):
        raise AlignmentError(f"MFA returned {len(words)} words for {len(chars)} characters.")
    syllables = []
    for char, (start, end, label) in zip(chars, words):
        if label == "<unk>" and not align["allow_unk"]:
            raise AlignmentError(f"Character {char} is out of vocabulary (<unk>); fix the dictionary.")
        if label != char and label != "<unk>":
            raise AlignmentError(f"MFA word {label!r} does not match script character {char!r}.")
        inside = [p for p in phones if start - 1e-6 <= (p[0] + p[1]) / 2 <= end + 1e-6
                  and p[2] not in SILENCE_LABELS]
        if any(p[2] == "spn" for p in inside) and not align["allow_spn"]:
            raise AlignmentError(f"Character {char} aligned to spn; fix the dictionary.")
        nucleus = next((i for i, p in enumerate(inside) if TONE_LETTERS & set(p[2])), None)
        syllables.append({
            "char": char, "start": float(start), "end": float(end),
            "phones": [(float(p[0]), float(p[1]), p[2]) for p in inside],
            "nucleus": None if nucleus is None else (float(inside[nucleus][0]), float(inside[-1][1])),
        })
    return syllables


def _mark_alignment_sanity(syllables):
    """Flag syllables with impossible boundaries (empty or overlapping the previous one)."""
    previous_end = -math.inf
    for syllable in syllables:
        ok = syllable["end"] > syllable["start"] and syllable["start"] >= previous_end - 0.02
        syllable["alignment_ok"] = bool(syllable.get("alignment_ok", True) and ok)
        previous_end = max(previous_end, syllable["end"])
    return syllables


# ---------------------------------------------------- features + verifier ----

def syllable_features(times, st, start, end, syllable_ms=None, followed_by_t4=False):
    """Per-syllable acoustic features (the 12 core ones plus mean / core range) on a segment [start, end).

    `syllable_ms` is the length of the whole aligned syllable (the segment may be only its vowel): rules that depend on
    how long the syllable is read it from here."""
    span = max(end - start, 1e-6)
    voiced = np.isfinite(st)
    n = int(voiced.sum())
    feats = {"duration": float(end - start), "voiced_ratio": n / len(st) if len(st) else 0.0,
             "voiced_frames": n}
    if syllable_ms is not None:
        feats["syllable_ms"] = float(syllable_ms)
    if followed_by_t4:
        feats["followed_by_t4"] = True               # the next syllable is also a tone 4 (see tone.t4_before_t4)
    if n == 0:
        return feats
    v_st, v_pos = st[voiced], (times[voiced] - start) / span
    edge = max(2, int(math.ceil(0.2 * n)))
    third = max(1, n // 3)
    low = int(np.argmin(v_st))
    feats.update({
        "start_ST": float(v_st[:edge].mean()), "end_ST": float(v_st[-edge:].mean()),
        "min_ST": float(v_st[low]), "max_ST": float(v_st.max()),
        "range_ST": float(v_st.max() - v_st[low]),
        "range_core_ST": float(np.percentile(v_st, 90) - np.percentile(v_st, 10)),
        "mean_ST": float(v_st.mean()),
        "min_position": float(v_pos[low]), "max_position": float(v_pos[int(np.argmax(v_st))]),
        "early_to_late_delta": float(v_st[:third].mean() - v_st[-third:].mean()),
        "fall_before_min": float(v_st[: low + 1].max() - v_st[low]),
        "rise_after_min": float(v_st[low:].max() - v_st[low]),
    })
    return feats


def _ge(x, threshold):
    """1 at/above the threshold, falling linearly to 0 at 40% of it."""
    return float(np.clip((x - 0.4 * threshold) / (0.6 * threshold), 0.0, 1.0))


def _le(x, threshold, width=None):
    """1 at/below the threshold, falling linearly to 0 after `width` more."""
    return float(np.clip(1.0 - (x - threshold) / (width or threshold), 0.0, 1.0))


def _window(x, low, high, margin=0.1):
    if low <= x <= high:
        return 1.0
    return float(np.clip(1.0 - (low - x if x < low else x - high) / margin, 0.0, 1.0))


def t4_min_fall(f, tone_config):
    """The fall a tone 4 needs: `t4_min_fall_st`, or, when `tone.t4_by_duration` is configured, a length-dependent value
    (reduced tone 4s in running speech are short and fall less): short / medium / long syllables by `short_ms` and
    `medium_ms`. Without the setting, or without a syllable length, the single value applies."""
    needed = _t4_fall_by_duration(f, tone_config)
    before_t4 = tone_config.get("t4_before_t4")
    if before_t4 and f.get("followed_by_t4"):
        # the first of two tone 4s may fall less (the second keeps the normal rule): `relaxation_st` lower than the length rule
        # says (a 220 ms syllable: 2.0 -> 1.5), or the absolute `min_fall_st`
        if "relaxation_st" in before_t4:
            needed = max(needed - before_t4["relaxation_st"], 0.5)
        else:
            needed = min(needed, before_t4["min_fall_st"])
    return needed


def _t4_fall_by_duration(f, tone_config):
    rule = tone_config.get("t4_by_duration")
    ms = f.get("syllable_ms")
    if not rule or ms is None:
        return tone_config["t4_min_fall_st"]
    if ms < rule["short_ms"]:
        return rule["min_fall_short_st"]
    if ms < rule["medium_ms"]:
        return rule["min_fall_medium_st"]
    return rule["min_fall_long_st"]


# Tone settings that change verdicts only; the boundary search must not see them (it would move boundaries to suit them).
VERDICT_ONLY_TONE_RULES = ("t4_by_duration", "t4_before_t4", "final_tail_guard", "carryover_trim", "left_context", "t3_low_register")


def combine_factors(parts, config):
    """One tone's factors -> one match score. `scoring.combine_features`: "product" (default) or "geometric_mean" (the n-th
    root of the product: three factors of 0.85 give 0.85, not 0.61; a factor of 0 still gives 0)."""
    product = float(np.prod(parts))
    if (config.get("scoring") or {}).get("combine_features", "product") == "geometric_mean":
        return product ** (1.0 / len(parts))
    return product


def tone_scores(f, allow_half_t3, config):
    """Soft 0..1 fit of a contour to each of T1-T4 (T3 = max of full and, if the context allows, half)."""
    t = config["tone"]
    t1 = combine_factors([_le(abs(f["early_to_late_delta"]), t["t1_max_delta_st"]),
                          _le(f["range_core_ST"], t["t1_max_range_st"], width=2.0)], config)
    t2 = combine_factors([_ge(f["rise_after_min"], t["t2_min_rise_st"]), _le(f["min_position"], t["t2_max_min_position"], 0.2)], config)
    t3_full = combine_factors([_ge(f["fall_before_min"], t["t3_min_fall_st"]), _ge(f["rise_after_min"], t["t3_min_full_rise_st"]),
                               _window(f["min_position"], t["t3_min_position"], t["t3_max_position"])], config)
    t3_half = combine_factors([_ge(f["fall_before_min"], t["t3_min_fall_st"]), _le(f["rise_after_min"], t["t3_min_full_rise_st"], 1.0),
                               _ge(f["min_position"], 0.5)], config) if allow_half_t3 else 0.0
    # A tone 3 is first of all LOW. `t3_low_register` (sentence practice) also accepts a syllable that starts low, reaches a deep low point and never
    # gets high, with or without a rise at the end: said after a pause or after a low syllable it has no fall to measure, so the two forms above
    # (a fall into the low point is required by both) scored a correct low-rising tone 3 as 0.
    low = t.get("t3_low_register")
    t3_low = combine_factors([_le(f["start_ST"], low["max_start_st"], 2.0), _ge(-f["min_ST"], low["min_depth_st"]),
                              _le(f["max_ST"], low["max_st"], 2.0)], config) if low and allow_half_t3 else 0.0
    t4 = _ge(f["early_to_late_delta"], t4_min_fall(f, t))
    rule = t.get("t4_by_duration")
    if rule and rule.get("require_end_below_start") and not f["end_ST"] < f["start_ST"]:
        t4 = 0.0                                        # a tone 4 must end lower than it starts
    return {1: t1, 2: t2, 3: max(t3_full, t3_half, t3_low), 4: t4, "t3_full": t3_full, "t3_half": t3_half, "t3_low": t3_low}


# ------------------------------------------------------- confidence gating ----

def acoustic_quality(components, config):
    weights = config["confidence"]["quality_weights"]
    return float(sum(weights[k] * float(np.clip(components[k], 0, 1)) for k in weights))


def decide(quality, match, config, allow_practice=True):
    """likely_correct needs decent quality and match; practice needs high quality and mismatch (= 1 - match)."""
    conf = config["confidence"]
    mismatch = 1.0 - match
    if quality >= conf["likely_correct_quality"] and match >= conf["likely_correct_match"]:
        return "likely_correct"
    if allow_practice and quality >= conf["practice_quality"] and mismatch >= conf["practice_mismatch"]:
        return "practice"
    return "insufficient_evidence"


def student_feedback(status, expected_tone, config):
    text = config["feedback"]
    if status == "practice":
        return text["practice"].get(expected_tone, text["insufficient_evidence"])
    return text[status]


def _segment_for(syllable, config):
    clean = config["cleanup"]
    if syllable["nucleus"] is not None:
        return syllable["nucleus"][0], syllable["nucleus"][1], syllable.get("segment_source", "vowel_nucleus")
    span = syllable["end"] - syllable["start"]
    return (syllable["start"] + clean["fallback_onset_trim"] * span,
            syllable["end"] - clean["fallback_offset_trim"] * span, "syllable_trimmed")


def _pinyin(char, lexical_tone):
    base = lazy_pinyin(char, style=Style.NORMAL)[0]
    return f"{base}{lexical_tone}" if lexical_tone else base


def _empty_result(context, char, config, speaker, status="insufficient_evidence", reasons=()):
    result = {
        "index": context["index"], "character": char, "pinyin": _pinyin(char, context["lexical_tone"]),
        "lexical_tone": context["lexical_tone"], "expected_surface_tone": context["target_tone"],
        "tone_rule": context["tone_rule"],
        "speaker_f0_median": speaker.get("baseline_hz"), "pitch_floor": speaker.get("pitch_floor"),
        "pitch_ceiling": speaker.get("pitch_ceiling"),
        "tone_match": None, "tone_mismatch": None, "detected_alternative": None,
        "acoustic_quality": 0.0, "status": status, "reasons": list(reasons),
    }
    result["feedback"] = student_feedback(status, context["target_tone"], config)
    return result


def _rule_applies(when, context):
    """Every condition in `when` holds for this syllable's neighbourhood (all must; an empty `when` always applies)."""
    tone = when.get("tone")
    if tone is not None and context["target_tone"] not in ([tone] if isinstance(tone, int) else tone):
        return False
    for key, neighbour in (("previous", context.get("adjacent_previous_tone")), ("next", context.get("adjacent_next_tone"))):
        wanted = when.get(key)
        if wanted is not None:
            wanted = [wanted] if not isinstance(wanted, (list, tuple)) else list(wanted)
            if (neighbour if neighbour is not None else "none") not in wanted:
                return False
    if "phrase_final" in when and bool(context.get("is_clause_final")) != bool(when["phrase_final"]):
        return False
    end = context.get("previous_end_st")
    if "previous_end_min_st" in when and (end is None or end < when["previous_end_min_st"]):
        return False
    if "previous_end_max_st" in when and (end is None or end > when["previous_end_max_st"]):
        return False
    return True


def context_config(config, context):
    """`tone.left_context`: tolerance that depends on the neighbours, as a list of {when: {...}, set: {...}} rules.

    A syllable has no tone shape of its own: the tone before it drags its start (a tone 2 after a high tone dips first and
    rises less), the tone after it can pull its end, and a clause end lowers it. A matching rule overrides tone
    thresholds for that syllable only (`set` may only name keys of `tone`); several matching rules apply in order.
    `when` may name: tone, previous / next (the neighbouring SURFACE tone in the clause, even across a word end; "none"
    = no neighbour), phrase_final, previous_end_min_st / previous_end_max_st (where the previous syllable's pitch ended,
    in semitones from the speaker's median). Verdict-only: the boundary search never sees it."""
    rules = config["tone"].get("left_context")
    if not rules:
        return config
    changes = {}
    for rule in rules:
        if _rule_applies(rule.get("when") or {}, context):
            changes.update(rule["set"])
    if not changes:
        return config
    unknown = set(changes) - set(config["tone"])
    if unknown:
        raise ValueError(f"left_context sets unknown tone setting(s): {sorted(unknown)}")
    return {**config, "tone": {**config["tone"], **changes}}


def previous_end_level(syllable, clean, config):
    """Where a syllable's pitch ended (mean of the last 20% of its voiced frames, semitones), or None if it has none."""
    seg_start, seg_end, _ = _segment_for(syllable, config)
    mask = (clean["times"] >= seg_start) & (clean["times"] < seg_end)
    feats = syllable_features(clean["times"][mask], clean["st"][mask], seg_start, seg_end)
    return feats.get("end_ST")


def scoring_window(config, context, seg_start, seg_end):
    """The part of the analysis segment the tone is judged on (verdict only; quality and gates use the whole segment).

    `tone.final_tail_guard`: at the end of a clause the pitch is pulled down (final lowering), so the last `trim_frac` of
    a clause-final tone 1/2/3 is left out. `tone.carryover_trim`: the first part of a syllable still carries the pitch
    of the syllable before it, so up to `max_ms` (and `frac` of the segment) is left out when there is a neighbour on
    the left. Both are off unless configured."""
    tone = config["tone"]
    span = seg_end - seg_start
    start, end = seg_start, seg_end
    guard = tone.get("final_tail_guard")
    if guard and context.get("is_clause_final") and context["target_tone"] in guard["tones"]:
        end -= guard["trim_frac"] * span
    carry = tone.get("carryover_trim")
    if carry and context.get("adjacent_previous_tone") is not None and (
            not carry.get("tones") or context["target_tone"] in carry["tones"]):
        start += min(carry["frac"] * span, carry["max_ms"] / 1000.0)
    return start, end


def track_stability(track_ok, jumps_by_syllable, scope="utterance"):
    """(is the recording's pitch track stable, [is each syllable's]). A jump left inside a syllable's segment makes the recording
    unstable; with scope "utterance" (default) that cuts the quality of EVERY syllable, with scope "syllable" only the syllables that
    contain a jump (they are gated as octave_jump_in_segment anyway). A track that is unreliable as a whole (`track_ok` False)
    makes every syllable unstable in both scopes."""
    stable = bool(track_ok) and not any(jumps_by_syllable)
    if scope == "syllable":
        return stable, [bool(track_ok) and not jumps for jumps in jumps_by_syllable]
    return stable, [stable] * len(jumps_by_syllable)


def verify_syllable(syllable, context, extraction, clean, config, speaker, stable_track=True, accuse_ok=True, doubtful_alignment=False):
    align = config["alignment"]
    seg_start, seg_end, source = _segment_for(syllable, config)
    mask = (clean["times"] >= seg_start) & (clean["times"] < seg_end)
    times, st, trimmed_frames = segment_pitch(clean, seg_start, seg_end, config)
    syllable_sec = syllable["end"] - syllable["start"]
    followed_by_t4 = context.get("adjacent_next_tone") == 4
    feats = syllable_features(times, st, seg_start, seg_end, syllable_ms=syllable_sec * 1000.0, followed_by_t4=followed_by_t4)

    ac, cc = extraction["ac"][mask], extraction["cc"][mask]
    either, both = np.isfinite(ac) | np.isfinite(cc), np.isfinite(ac) & np.isfinite(cc)
    agree = both & (np.abs(_st_diff(ac, cc)) <= config["sanity"]["ac_cc_tolerance_st"])
    n_voiced = max(feats["voiced_frames"], 1)
    remaining_jumps = count_octave_jumps(st, config["sanity"]["octave_jump_st"], in_semitones=True)
    unstable = int(clean["octave_corrected"][mask].sum()) + 2 * remaining_jumps
    components = {
        "ac_cc_agreement": float(agree.sum() / either.sum()) if either.any() else 0.0,
        "voiced_coverage": feats["voiced_ratio"],
        "octave_stability": max(0.0, 1.0 - unstable / n_voiced),
        "alignment_quality": (0.5 * min(1.0, feats["voiced_frames"] / (2 * align["min_voiced_frames"]))
                              + 0.5 * min(1.0, syllable_sec * 1000 / (2 * align["min_syllable_ms"]))
                              if syllable.get("alignment_ok", True) else 0.0),
    }
    quality = acoustic_quality(components, config)
    reasons = []
    if not stable_track:
        quality *= config["confidence"]["unstable_pitch_quality_factor"]
        reasons.append("unstable_pitch_track")

    result = _empty_result(context, syllable["char"], config, speaker, reasons=reasons)
    if "boundary_shift_ms" in syllable:
        result["boundary_shift_ms"] = syllable["boundary_shift_ms"]
    result.update({
        "start_ms": round(syllable["start"] * 1000), "end_ms": round(syllable["end"] * 1000),
        "duration_ms": round(syllable_sec * 1000),
        "analysis_segment": {"start_ms": round(seg_start * 1000), "end_ms": round(seg_end * 1000), "source": source},
        "ac_cc_agreement": round(components["ac_cc_agreement"], 3),
        "voiced_ratio": round(feats["voiced_ratio"], 3),
        "features": {k: round(v, 3) for k, v in feats.items()},
        "acoustic_quality": round(quality, 3),
        "quality_components": {k: round(v, 3) for k, v in components.items()},
        "_syllable_sec": syllable_sec, "_feats": feats,
    })
    if feats["voiced_frames"]:
        if speaker.get("baseline_hz"):                                 # the same points on the absolute MIDI scale
            reference = float(hz_to_midi(speaker["baseline_hz"]))
            result.update({"start_midi": round(reference + feats["start_ST"], 2), "end_midi": round(reference + feats["end_ST"], 2),
                           "min_midi": round(reference + feats["min_ST"], 2), "max_midi": round(reference + feats["max_ST"], 2)})
        result.update({
            "start_st": round(feats["start_ST"], 2), "end_st": round(feats["end_ST"], 2),
            "min_st": round(feats["min_ST"], 2), "max_st": round(feats["max_ST"], 2),
            "range_st": round(feats["range_ST"], 2), "min_position": round(feats["min_position"], 2),
            "max_position": round(feats["max_position"], 2),
            "early_to_late_delta_st": round(feats["early_to_late_delta"], 2),
            "fall_before_min_st": round(feats["fall_before_min"], 2),
            "rise_after_min_st": round(feats["rise_after_min"], 2),
        })

    gate = []
    if not syllable.get("alignment_ok", True):
        gate.append("bad_alignment")
    if syllable_sec * 1000 < align["min_syllable_ms"]:
        gate.append("syllable_too_short")
    if feats["voiced_frames"] < align["min_voiced_frames"]:
        gate.append("too_few_voiced_frames")
    if not gate and remaining_jumps:  # a jump >7 st between adjacent frames is a tracking error, not a tone movement
        gate.append("octave_jump_in_segment")
    reasons.extend(gate)
    result["reasons"] = reasons
    result["_gated"] = bool(gate)
    if gate:
        result["feedback"] = student_feedback("insufficient_evidence", context["target_tone"], config)
        return result

    if context["target_tone"] == 5:
        result["_pending_neutral"] = True   # needs the rest of the utterance for the duration reference
        return result

    expected = context["target_tone"]
    accepted = [expected, *context.get("also_accept_tones", ())]        # sentence practice may accept a sandhi alternative
    config = context_config(config, context)                            # tolerance that depends on the neighbours (off by default)
    score_start, score_end = scoring_window(config, context, seg_start, seg_end)
    score_times, score_st, score_feats = times, st, feats
    if (score_start, score_end) != (seg_start, seg_end):
        keep = (times >= score_start) & (times < score_end)
        trimmed_feats = syllable_features(times[keep], st[keep], score_start, score_end, syllable_ms=syllable_sec * 1000.0,
                                          followed_by_t4=followed_by_t4)
        if score_end > score_start and trimmed_feats["voiced_frames"] >= align["min_voiced_frames"]:
            score_times, score_st, score_feats = times[keep], st[keep], trimmed_feats
            result["scoring_window_ms"] = [round(score_start * 1000), round(score_end * 1000)]
        else:
            score_start, score_end = seg_start, seg_end                  # too little would be left: judge the whole syllable
    scores = tone_scores(score_feats, allows_half_third(context, config), config)
    match = max(scores[tone] for tone in accepted)
    others = {tone: scores[tone] for tone in (1, 2, 3, 4) if tone not in accepted}
    status = decide(quality, match, config)
    # The second character of a repeated pair (看看, 謝謝) is read lightly, often as a near-neutral: it may not be accused.
    softened = bool(status == "practice" and config["tone"].get("soften_reduplication") and context.get("is_reduplicated"))
    if softened:
        status = "insufficient_evidence"
    # A tone 3 where a half-third is not accepted as correct (the end of a single word) must be said in full. A low fall
    # with no rise is not a wrong tone though: it is Okay, with the advice to say it in full, never "Needs practice".
    half_only = False
    if (status == "practice" and expected == 3 and config["tone"].get("half_t3_end_as_okay")
            and not allows_half_third(context, config)):
        half_only = tone_scores(score_feats, True, config)["t3_half"] >= config["confidence"]["likely_correct_match"]
        if half_only:
            status = "insufficient_evidence"
    # A window this short (a reduced function word) cannot support an accusation; 0 = off, as for single words.
    short_window = status == "practice" and (seg_end - seg_start) * 1000.0 < config["confidence"].get("practice_min_window_ms", 0)
    if short_window:
        status = "insufficient_evidence"
    # `unstable_scope: syllable` spares a clean syllable of a recording whose track has a jump elsewhere from the quality penalty, so
    # it can be Good; but accusing needs more certainty than accepting: that recording may not produce a "Needs practice".
    # The same goes for a syllable whose tail had to be cut off behind a tracking jump (`sanity.jump_fragment_trim_ms`): judged on what
    # is left, it can be accepted but not accused.
    held_back = status == "practice" and (not accuse_ok or trimmed_frames > 0 or doubtful_alignment)
    if held_back:
        status = "insufficient_evidence"
    edge_step = status != "insufficient_evidence" and edge_step_changes_verdict(
        score_times, score_st, score_start, score_end, expected, allows_half_third(context, config), status, quality, config,
        also_accept=accepted[1:], syllable_ms=syllable_sec * 1000.0, followed_by_t4=followed_by_t4)
    result.update({
        "tone_scores": {str(k): round(scores[k], 3) for k in (1, 2, 3, 4)},
        "tone_match": round(match, 3), "tone_mismatch": round(1.0 - match, 3),
        "detected_alternative": max(others, key=others.get),   # diagnostic only; never used for gating
        "status": "insufficient_evidence" if edge_step else status,
    })
    if trimmed_frames:
        result["jump_fragment_trimmed_ms"] = round(trimmed_frames * float(clean.get("step") or config["pitch"]["time_step"]) * 1000)
    if len(accepted) > 1:
        result["also_accepted_tones"] = accepted[1:]
    if (config["tone"].get("clause_final_half_t3") and context["is_clause_final"]
            and context["lexical_tone"] == 3 and expected == 3):
        result["clause_final_half_t3"] = True                  # this verdict allowed a low fall as a T3
    if edge_step:
        result["reasons"].append("edge_pitch_step")
    elif softened:
        result["reasons"].append("reduplicated_second_syllable")
    elif half_only:
        result["reasons"].append("half_third_only")
    elif short_window:
        result["reasons"].append("short_analysis_window")
    elif held_back:
        result["reasons"].append("bad_alignment" if doubtful_alignment else "unstable_pitch_track" if not accuse_ok else "octave_jump_in_segment")
    elif status == "insufficient_evidence":
        result["reasons"].append(
            "low_acoustic_quality" if quality < config["confidence"]["likely_correct_quality"] else "ambiguous_contour")
    result["feedback"] = student_feedback(result["status"], expected, config)
    return result


def edge_fragment_mask(times, st, step_st, fragment_sec):
    """Frames of a short voiced run at either edge of `st` that lies beyond an implausible one-frame step, or None.

    Real tone movements are smooth (about 1 st per 5 ms at the very most); a jump of `step_st` or more between adjacent
    frames, followed by only a brief plateau before the segment ends, is a tracking artefact (a harmonic jump, creak,
    the tail of the voicing) rather than something the speaker did."""
    idx = np.flatnonzero(np.isfinite(st))
    if len(idx) < 3:
        return None
    frame = float(np.median(np.diff(times))) if len(times) > 1 else 0.005
    steps = np.flatnonzero((np.diff(idx) <= 2) & (np.abs(np.diff(st[idx])) >= step_st))
    if not len(steps):
        return None
    drop = np.zeros(len(st), dtype=bool)
    last = steps[-1]
    if (idx[-1] - idx[last + 1] + 1) * frame <= fragment_sec:
        drop[idx[last + 1:]] = True
    first = steps[0]
    if (idx[first] - idx[0] + 1) * frame <= fragment_sec:
        drop[idx[: first + 1]] = True
    return drop if drop.any() else None


def edge_step_changes_verdict(times, st, seg_start, seg_end, expected, allow_half_t3, status, quality, config,
                              also_accept=(), syllable_ms=None, followed_by_t4=False):
    """True when the verdict rests on a short edge fragment behind an implausible pitch step.

    The syllable is scored again without that fragment; if the status differs, the evidence is not solid. This can only
    turn a decision into an abstention, never the other way round."""
    clean = config["cleanup"]
    drop = edge_fragment_mask(times, st, clean["edge_step_st"], clean["edge_fragment_ms"] / 1000.0)
    if drop is None:
        return False
    trimmed = np.where(drop, np.nan, st)
    if int(np.isfinite(trimmed).sum()) < config["alignment"]["min_voiced_frames"]:
        return True
    scores = tone_scores(syllable_features(times, trimmed, seg_start, seg_end, syllable_ms=syllable_ms,
                                           followed_by_t4=followed_by_t4), allow_half_t3, config)
    return decide(quality, max(scores[t] for t in (expected, *also_accept)), config) != status


def allows_half_third(context, config=None):
    """A surface T3 that is not clause-final may be a half-third (fall -> low), even at a word boundary.

    Sentence practice (`tone.clause_final_half_t3`) allows it at the end of a clause too: running speech reads that T3
    as a low fall without the final rise. Known trade-off: a T4 also falls to low, so at that position a T4 where a T3
    is expected is not told apart from a half-third."""
    if (config and config["tone"].get("clause_final_half_t3")
            and context["lexical_tone"] == 3 and context["target_tone"] == 3):
        return True
    return bool(context["allow_half_t3"]
                or (context["lexical_tone"] == 3 and context["target_tone"] == 3 and not context["is_clause_final"]))


def finalize_neutral(results, config):
    """Neutral tone in v1.

    The script already says which syllables are neutral, so nothing is classified as T1-T4 and no duration or pitch-
    range rule is applied (native isolated words showed pitch spreads of 6-12 st, far beyond any fixed ceiling).
    A neutral syllable is `likely_correct` when its audio is usable (it passed the same gates as every syllable: a
    trusted alignment, enough voiced frames, long enough, acoustic quality at the likely_correct level) and
    `insufficient_evidence` when the pitch or alignment is too poor. It is never `practice`.
    tone_match / tone_mismatch stay null: the tone itself was not scored."""
    for result in results:
        if not result.get("_pending_neutral"):
            continue
        usable = result["acoustic_quality"] >= config["confidence"]["likely_correct_quality"]
        result["status"] = "likely_correct" if usable else "insufficient_evidence"
        if not usable:
            result["reasons"].append("low_acoustic_quality")
        result["feedback"] = student_feedback(result["status"], 5, config)


def enforce_neutral_policy(results, config):
    """v1 policy: a neutral-tone syllable is `likely_correct` or `insufficient_evidence`, never `practice`.

    An F0 shape alone must not accuse the learner of a neutral-tone error. finalize_neutral already behaves this
    way; this is the single place that guarantees it. It is a silent
    guard: the downgrade leaves no trace in the reasons, payload or UI (only a private flag, dropped by _public)."""
    if config["neutral"]["allow_practice"]:
        return
    for result in results:
        if result["expected_surface_tone"] == 5 and result["status"] == "practice":
            result["status"] = "insufficient_evidence"
            result["feedback"] = student_feedback("insufficient_evidence", 5, config)
            result["_neutral_policy_downgrade"] = True


def _public(result):
    return {k: v for k, v in result.items() if not k.startswith("_")}


def aggregate_status(syllables):
    """Status of a group of syllables (word / phrase / file).

    Any `practice` wins. Otherwise the group is `likely_correct` only if every scored (non-neutral)
    syllable is; neutral syllables never block it, because v1 only weakly checks them.
    """
    if any(s["status"] == "practice" for s in syllables):
        return "practice"
    scored = [s for s in syllables if s["expected_surface_tone"] != 5]
    pool = scored or list(syllables)
    if pool and all(s["status"] == "likely_correct" for s in pool):
        return "likely_correct"
    return "insufficient_evidence"


def _contour_payload(clean):
    """Semitone contour (re. speaker median) for charts; unvoiced frames are None."""
    return {
        "step_sec": round(float(clean["step"]), 4),
        "times": [round(float(t), 3) for t in clean["times"]],
        "st": [round(float(v), 2) if np.isfinite(v) else None for v in clean["st"]],
    }


def _resolve_pitch_range(calibration, pitch_range):
    calibrated = isinstance(calibration, dict) and calibration.get("status") == "calibrated"
    baseline = None
    floor, ceiling = 75.0, 450.0
    if calibrated:
        floor = float(calibration.get("pitch_floor", calibration.get("pitch_floor_hz")))
        ceiling = float(calibration.get("pitch_ceiling", calibration.get("pitch_ceiling_hz")))
        baseline = float(calibration.get("f0_median", (calibration.get("f0_hz") or {}).get("baseline")))
    if pitch_range:
        floor, ceiling = (float(v) for v in pitch_range)
    return calibrated, floor, ceiling, baseline


def syllable_count(script):
    return len(capt_core.expected_tone_contexts(capt_core.prepare_target(script)))


def needs_mfa(syllable_count, config):
    """A one-syllable script needs no forced alignment: the syllable is simply the voiced run."""
    return not (syllable_count == 1 and config["alignment"]["skip_mfa_single_syllable"])


def single_syllable_alignment(extraction, char, config):
    """The longest voiced run (gaps up to single_syllable_bridge_ms bridged) as the one syllable; [] if none."""
    times = extraction["times"]
    if not len(times):
        return []
    step = float(np.median(np.diff(times))) if len(times) > 1 else config["pitch"]["time_step"]
    bridge = int(round(config["alignment"]["single_syllable_bridge_ms"] / 1000.0 / step))
    runs = []
    for start, end in _voiced_runs(np.isfinite(extraction["f0"])):
        if runs and start - runs[-1][1] <= bridge:
            runs[-1][1] = end
        else:
            runs.append([start, end])
    if not runs:
        return []
    start, end = max(runs, key=lambda run: run[1] - run[0])
    first, last = float(times[start]), float(times[end - 1]) + step
    return [{"char": char, "start": first, "end": last, "phones": [], "nucleus": (first, last),
             "segment_source": "voiced_run"}]


def _shifted_segment(syllable, shift_start, shift_end, config):
    """Analysis segment of a syllable whose start / end boundaries moved by the given seconds.

    The nucleus start moves with the syllable start (the onset keeps its length); the end is shared."""
    start, end = syllable["start"] + shift_start, syllable["end"] + shift_end
    if syllable["nucleus"] is not None:
        return syllable["nucleus"][0] + shift_start, end
    span = end - start
    cleanup = config["cleanup"]
    return start + cleanup["fallback_onset_trim"] * span, end - cleanup["fallback_offset_trim"] * span


def search_boundaries(syllables, contexts, clean, config):
    """Choose syllable boundaries within +-max_shift_ms of the aligner's so the expected tones fit best.

    MFA boundaries between sonorants are uncertain, and a boundary that is 40 ms off moves F0 from one syllable
    into its neighbour. Rather than trust one segmentation, look for a plausible one (dynamic programming over
    the shared boundaries, a small penalty for moving them) in which each syllable fits its expected tone. A
    syllable is then only called `practice` if NO plausible segmentation makes its expected tone fit.
    Returns (syllables, info); the syllables are unchanged when the search is off or not applicable."""
    cfg = config["boundary_search"]
    # Verdict-only rules never move boundaries: a looser tone-4 rule would pull a boundary into the neighbouring syllable
    # (taking the rising tail off a tone 2 beside it) and make the neighbour look wrong. Like the half-third flag below,
    # The rules in VERDICT_ONLY_TONE_RULES therefore apply to verdicts only; boundaries are placed as without them.
    # The boundary search also keeps the plain product of factors: how factors are combined is a verdict question.
    search_config = {**config, "tone": {k: v for k, v in config["tone"].items() if k not in VERDICT_ONLY_TONE_RULES},
                     "scoring": {"combine_features": "product"}}
    count = len(syllables)
    info = {"enabled": bool(cfg["enabled"]), "applied": False, "shifts_ms": [0] * max(count - 1, 0),
            "mean_match_before": None, "mean_match_after": None}
    if not cfg["enabled"] or count < 2 or any(not s.get("alignment_ok", True) for s in syllables):
        return syllables, info
    step, width = cfg["step_ms"] / 1000.0, cfg["max_shift_ms"] / 1000.0
    reach = int(round(width / step))
    offsets = [round(k * step, 4) for k in range(-reach, reach + 1)]
    zero = offsets.index(0.0)
    min_length = config["alignment"]["min_syllable_ms"] / 1000.0
    min_frames = config["alignment"]["min_voiced_frames"]

    def fit(index, shift_start, shift_end):
        syllable, context = syllables[index], contexts[index]
        if syllable["end"] + shift_end - (syllable["start"] + shift_start) < min_length:
            return None
        seg_start, seg_end = _shifted_segment(syllable, shift_start, shift_end, config)
        if seg_end <= seg_start:
            return None
        if context["target_tone"] == 5:
            return 0.5                                  # neutral: no preference, neighbours decide the boundary
        mask = (clean["times"] >= seg_start) & (clean["times"] < seg_end)
        feats = syllable_features(clean["times"][mask], clean["st"][mask], seg_start, seg_end,
                                  syllable_ms=(syllable["end"] + shift_end - (syllable["start"] + shift_start)) * 1000.0)
        if feats["voiced_frames"] < min_frames:
            return 0.0
        # The boundary search keeps the strict reading (no clause-final half-third): `clause_final_half_t3` changes
        # verdicts only, never where the syllable boundaries are put.
        return tone_scores(feats, allows_half_third(context), search_config)[context["target_tone"]]

    best = {zero: (0.0, [zero])}                        # outer boundaries stay where the aligner put them
    for index in range(count):
        following = {}
        for b in ([zero] if index == count - 1 else range(len(offsets))):
            for a, (score, path) in best.items():
                value = fit(index, offsets[a], offsets[b])
                if value is None:
                    continue
                total = score + value - cfg["shift_penalty"] * abs(offsets[b]) / width
                if b not in following or total > following[b][0]:
                    following[b] = (total, path + [b])
        if not following:
            return syllables, info                      # no feasible segmentation: keep the aligner's
        best = following
    path = best[zero][1]
    shifts = [offsets[k] for k in path]

    def mean_match(chosen):
        values = [fit(i, chosen[i], chosen[i + 1]) for i in range(count) if contexts[i]["target_tone"] != 5]
        values = [0.0 if v is None else v for v in values]
        return float(np.mean(values)) if values else None

    before, after = mean_match([0.0] * (count + 1)), mean_match(shifts)
    adjusted = []
    for i, syllable in enumerate(syllables):
        a, b = shifts[i], shifts[i + 1]
        moved = dict(syllable, start=syllable["start"] + a, end=syllable["end"] + b,
                     boundary_shift_ms={"start": round(a * 1000), "end": round(b * 1000)})
        if syllable["nucleus"] is not None:
            moved["nucleus"] = (syllable["nucleus"][0] + a, syllable["nucleus"][1] + b)
        adjusted.append(moved)
    info.update({"applied": any(shifts), "shifts_ms": [round(v * 1000) for v in shifts[1:-1]],
                 "mean_match_before": None if before is None else round(before, 3),
                 "mean_match_after": None if after is None else round(after, 3)})
    return adjusted, info


def verify_utterance(samples, sample_rate, script, calibration=None, *, alignment=None, config=None,
                     pitch_range=None, contexts=None, rejected_reasons=None, force_mfa=False):
    """Full pipeline on one recording of a known script.

    `alignment` can inject syllables (tests, cached alignments); `pitch_range=(floor, ceiling)` overrides the
    calibrated range while keeping the calibrated median for semitone normalisation (fixed-range comparison);
    `contexts` replaces the default surface-tone contexts (word-boundary sandhi, catalog pinyin overrides).
    A failed alignment never raises: every syllable becomes insufficient_evidence. `rejected_reasons` (from
    audio_prep: clipped, too_quiet, noisy, ...) short-circuits before alignment with the same outcome.
    A one-syllable script is aligned by its voiced run instead of MFA unless `force_mfa` is set; set it when the
    caller is choosing between script variants of different length (a voiced run cannot tell 裡 from 裡面).
    """
    config = config or load_config()
    if contexts is None:
        contexts = capt_core.expected_tone_contexts(capt_core.prepare_target(script))
    chars = [c["char"] for c in contexts]
    if not chars:
        raise ValueError("The script contains no Chinese characters.")
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    calibrated, floor, ceiling, baseline = _resolve_pitch_range(calibration, pitch_range)
    speaker = {"baseline_hz": baseline, "pitch_floor": floor, "pitch_ceiling": ceiling}

    base = {"method": METHOD, "script": "".join(chars), "calibrated": calibrated, "aligner": None,
            "calibration_version": (calibration or {}).get("calibration_version") if calibrated else None,
            "acoustic_profile": (calibration or {}).get("acoustic_profile") if calibrated else None}
    if rejected_reasons:
        results = [_empty_result(c, c["char"], config, speaker, reasons=["audio_quality_failed", *rejected_reasons])
                   for c in contexts]
        message = config["feedback"]["audio_quality"].get(
            rejected_reasons[0], config["feedback"]["insufficient_evidence"])
        for result in results:
            result["feedback"] = message
        return {**base, "alignment_error": None, "audio_quality_failed": list(rejected_reasons),
                "f0_extraction": None, "contour": None, "boundary_search": None, "syllables": results,
                "summary": {k: sum(r["status"] == k for r in results) for k in STATUSES},
                "feedback": message, "config": deepcopy(config)}

    alignment_error = None
    extraction = None
    base["aligner"] = "injected" if alignment is not None else "mfa"
    try:
        if alignment is None and not force_mfa and not needs_mfa(len(contexts), config):
            extraction = extract_f0(samples, sample_rate, floor, ceiling, config)
            alignment = single_syllable_alignment(extraction, chars[0], config)
            base["aligner"] = "voiced_run"
            if not alignment:
                raise AlignmentError("No voiced speech was found in the recording.")
        syllables = alignment if alignment is not None else syllables_from_tiers(
            mfa_align(samples, sample_rate, chars, config), chars, config)
        if len(syllables) != len(contexts):
            raise AlignmentError("Alignment length does not match the script.")
        _mark_alignment_sanity(syllables)
    except AlignmentError as exc:
        alignment_error = str(exc)

    if alignment_error:
        results = [_empty_result(c, c["char"], config, speaker, reasons=["alignment_failed"]) for c in contexts]
        return {**base, "alignment_error": alignment_error, "audio_quality_failed": None,
                "f0_extraction": None, "contour": None, "boundary_search": None,
                "syllables": results,
                "summary": {k: sum(r["status"] == k for r in results) for k in STATUSES},
                "feedback": config["feedback"]["insufficient_evidence"], "config": deepcopy(config)}

    if extraction is None:
        extraction = extract_f0(samples, sample_rate, floor, ceiling, config)
    # Voiced frames that no aligned syllable covers mean the script does not account for what was said
    # (a missing word, an omitted optional character, a wrong variant): the alignment cannot be trusted.
    voiced_frames = np.isfinite(extraction["f0"])
    covered = np.zeros(len(voiced_frames), dtype=bool)
    for syllable in syllables:
        covered |= (extraction["times"] >= syllable["start"]) & (extraction["times"] < syllable["end"])
    outside = float((voiced_frames & ~covered).sum() / max(int(voiced_frames.sum()), 1))
    # Up to `max_unaligned_voiced_fraction` all is well. Above `hard_unaligned_voiced_fraction` (default: the same number) the alignment is not trusted and
    # no syllable is judged. In between the alignment is DOUBTFUL (a hesitation, a repeated word, loose boundaries): syllables are judged as usual but the
    # recording cannot accuse anyone (a syllable can be Good or Okay, never Needs practice).
    soft_limit = config["alignment"]["max_unaligned_voiced_fraction"]
    unaligned = outside > max(soft_limit, config["alignment"].get("hard_unaligned_voiced_fraction", soft_limit))
    doubtful = soft_limit < outside and not unaligned
    if unaligned:
        for syllable in syllables:
            syllable["alignment_ok"] = False
    extraction["diagnostics"].update({"voiced_outside_syllables": round(outside, 3), "speech_outside_alignment": unaligned,
                                      "alignment_doubtful": doubtful})
    if baseline is None:
        voiced = extraction["f0"][np.isfinite(extraction["f0"])]
        if not len(voiced):
            raise ValueError("No voiced pitch found in the recording.")
        baseline = float(np.median(voiced))
        speaker["baseline_hz"] = baseline
    clean = clean_contour(extraction, baseline, config)

    syllables, boundary_search = search_boundaries(syllables, contexts, clean, config)
    # A jump counts against stability only if it survives cleanup (median filter, octave fixes) AND lies inside a
    # syllable's analysis segment. A step between two syllables is an ordinary transition, not a tracking error.
    jump_st = config["sanity"]["octave_jump_st"]
    jumps_left = count_octave_jumps(clean["st"], jump_st, in_semitones=True)
    jumps_in_segments = 0
    jumps_by_syllable = []
    trimmed_anywhere = False
    for syllable in syllables:
        seg_start, seg_end, _ = _segment_for(syllable, config)
        _, window, dropped = segment_pitch(clean, seg_start, seg_end, config)
        trimmed_anywhere = trimmed_anywhere or dropped > 0
        jumps_by_syllable.append(count_octave_jumps(window, jump_st, in_semitones=True))
        jumps_in_segments += jumps_by_syllable[-1]
    # False: AC and CC still disagree / coverage collapsed (the whole track)
    stable, syllable_stable = track_stability(extraction["diagnostics"]["stable"], jumps_by_syllable,
                                              config["confidence"].get("unstable_scope", "utterance"))
    extraction["diagnostics"].update({"stable": stable, "octave_jumps_after_cleanup": jumps_left,
                                      "octave_jumps_in_segments": jumps_in_segments,
                                      "octave_frames_corrected": int(clean["octave_corrected"].sum())})
    raw = []
    for index, (s, ctx) in enumerate(zip(syllables, contexts)):
        if index and ctx.get("adjacent_previous_tone") is not None and config["tone"].get("left_context"):
            end = previous_end_level(syllables[index - 1], clean, config)
            if end is not None and np.isfinite(end):
                ctx = {**ctx, "previous_end_st": float(end)}
        raw.append(verify_syllable(s, ctx, extraction, clean, config, speaker, syllable_stable[index],
                                  accuse_ok=stable and not trimmed_anywhere,   # a recording that needed a cut cannot accuse anyone
                                  doubtful_alignment=doubtful))
    finalize_neutral(raw, config)
    enforce_neutral_policy(raw, config)
    results = [_public(r) for r in raw]
    counts = {k: sum(r["status"] == k for r in results) for k in STATUSES}
    overall_status = aggregate_status(results)
    if overall_status == "practice":
        overall = config["feedback"]["practice"].get(
            next(r["expected_surface_tone"] for r in results if r["status"] == "practice"))
    else:
        overall = config["feedback"][overall_status]
    return {
        **base,
        "alignment_error": None,
        "audio_quality_failed": None,
        "f0_extraction": {"pitch_floor": extraction["pitch_floor"], "pitch_ceiling": extraction["pitch_ceiling"],
                          "speaker_f0_median": round(baseline, 1), "speaker_f0_median_midi": round(float(hz_to_midi(baseline)), 2),
                          **extraction["diagnostics"]},
        "syllables": results,
        "summary": counts,
        "feedback": overall,
        "contour": {**_contour_payload(clean), "baseline_midi": round(float(hz_to_midi(baseline)), 2)},
        "boundary_search": boundary_search,
        "config": deepcopy(config),
    }
