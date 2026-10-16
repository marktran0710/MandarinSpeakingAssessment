import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

import audio_prep
import tone_assessment as ta
import tone_pipeline as tp
from tests.test_tone_pipeline import tone_audio

CONFIG = tp.load_config()
SR = 16000


def voice(seconds=2.0, f0=200.0, amplitude=1.0):
    """A voiced signal with 0.4 s of silence either side (so SNR can be judged)."""
    t = np.arange(int(seconds * SR)) / SR
    phase = 2 * np.pi * f0 * t
    body = sum(np.sin(h * phase) / h for h in range(1, 7)) * 0.18 * amplitude
    pad = np.zeros(int(0.4 * SR))
    return np.concatenate([pad, body, pad]).astype(np.float32)


class QualityTests(unittest.TestCase):
    def reasons(self, samples, sample_rate=SR):
        return audio_prep.prepare_samples(samples, sample_rate, CONFIG).reasons

    def test_clean_recording_passes(self):
        prepared = audio_prep.prepare_samples(voice(), SR, CONFIG)
        self.assertEqual(prepared.reasons, [])
        self.assertTrue(prepared.passed)
        self.assertIsNone(prepared.rejected_reasons)
        self.assertGreater(prepared.metrics["snr_db"], 40)
        self.assertGreater(prepared.metrics["voiced_sec"], 1.5)

    def test_clipped_recording_is_rejected(self):
        self.assertIn("clipped", self.reasons(np.clip(voice() * 20, -1, 1)))

    def test_quiet_recording_is_rejected(self):
        self.assertIn("too_quiet", self.reasons(voice(amplitude=0.01)))

    def test_short_recording_is_rejected(self):
        self.assertIn("too_short", self.reasons(voice()[: int(0.2 * SR)]))

    def test_a_very_short_loud_syllable_is_not_mistaken_for_no_speech(self):
        # A neutral-tone syllable such as 吧 is about 0.15-0.2 s of speech in a clip with silence around it.
        body = voice(seconds=0.19)[int(0.4 * SR): -int(0.4 * SR)]
        clip = np.concatenate([np.zeros(int(0.3 * SR), dtype=np.float32), body, np.zeros(int(0.3 * SR), dtype=np.float32)])
        self.assertEqual(self.reasons(clip), [])

    def test_harmonicity_is_not_trusted_on_a_very_short_stretch_of_voicing(self):
        metrics = {"duration_sec": 0.75, "peak": 0.97, "clipped_fraction": 0.0, "speech_sec": 0.19, "snr_db": 79.0,
                   "voiced_sec": 0.16, "median_hnr_db": 5.5}
        self.assertEqual(audio_prep.evaluate_quality(metrics, CONFIG), [])                # clean, just brief
        self.assertEqual(audio_prep.evaluate_quality({**metrics, "voiced_sec": 1.0}, CONFIG), ["noisy"])
        self.assertEqual(audio_prep.evaluate_quality({**metrics, "snr_db": 3.0}, CONFIG), ["noisy"])  # SNR still applies

    def test_silence_has_no_speech(self):
        self.assertIn("no_speech", self.reasons(np.zeros(SR * 2, dtype=np.float32)))

    def test_noise_without_voice_has_no_voicing(self):
        noise = (np.random.default_rng(1).standard_normal(SR * 2) * 0.05).astype(np.float32)
        reasons = self.reasons(noise)
        self.assertTrue({"no_voicing", "noisy"} & set(reasons), reasons)

    def test_voice_buried_in_noise_is_noisy(self):
        rng = np.random.default_rng(2)
        mixed = voice(amplitude=1.0) + (rng.standard_normal(len(voice())) * 0.08).astype(np.float32)
        self.assertIn("noisy", self.reasons(mixed))

    def test_quality_is_judged_before_normalisation(self):
        # Peak-normalising a too-quiet recording must not hide that it was too quiet.
        prepared = audio_prep.prepare_samples(voice(amplitude=0.01), SR, CONFIG)
        self.assertIn("too_quiet", prepared.reasons)
        self.assertAlmostEqual(float(np.max(np.abs(prepared.samples))), CONFIG["audio_quality"]["target_peak"], places=2)

    def test_reasons_follow_the_priority_order(self):
        reasons = audio_prep.evaluate_quality(
            {"duration_sec": 0.1, "peak": 0.5, "clipped_fraction": 0.2, "speech_sec": 0.0,
             "snr_db": 3.0, "voiced_sec": 0.0, "median_hnr_db": None}, CONFIG)
        self.assertEqual(reasons, ["clipped", "noisy", "too_short", "no_speech"])

    def test_a_silent_file_is_too_quiet_not_noisy(self):
        reasons = self.reasons(np.zeros(SR * 2, dtype=np.float32))
        self.assertIn("too_quiet", reasons)
        self.assertNotIn("noisy", reasons)


class PrepareTests(unittest.TestCase):
    def test_other_sample_rates_are_resampled_to_16k(self):
        samples = np.interp(np.arange(int(44100 * 2.8)) / 44100 * SR, np.arange(len(voice())), voice())
        prepared = audio_prep.prepare_samples(samples.astype(np.float32), 44100, CONFIG)
        self.assertEqual(prepared.sample_rate, SR)
        self.assertLess(len(prepared.samples), len(samples))

    def test_long_edge_silence_is_trimmed_but_padding_is_kept(self):
        prepared = audio_prep.prepare_samples(voice(), SR, CONFIG)
        pad = CONFIG["audio_quality"]["trim_padding_sec"]
        self.assertAlmostEqual(prepared.trim_start_sec, 0.4 - pad, delta=0.08)
        self.assertLess(len(prepared.samples) / SR, 2.8)
        self.assertGreater(len(prepared.samples) / SR, 2.0 + 2 * pad - 0.15)

    def test_decode_audio_reads_a_wav_file_at_16k_mono(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.wav"
            wavfile.write(path, SR, (voice() * 32767).astype(np.int16))
            samples, rate = audio_prep.decode_audio(path)
            prepared = audio_prep.prepare_recording(path, CONFIG)
        self.assertEqual(rate, SR)
        self.assertEqual(samples.ndim, 1)
        self.assertAlmostEqual(len(samples) / SR, 2.8, delta=0.01)
        self.assertTrue(prepared.passed)


class SpeechChunkTests(unittest.TestCase):
    def chunks(self, contours, gap):
        audio, _ = tone_audio(contours, gap_sec=gap)
        prepared = audio_prep.prepare_samples(audio, SR, CONFIG)
        return prepared, audio_prep.speech_chunks(prepared, CONFIG)

    def test_one_reading_is_one_chunk(self):
        _, chunks = self.chunks([[200] * 4], 0.05)
        self.assertEqual(len(chunks), 1)

    def test_two_readings_separated_by_a_long_pause_are_two_chunks(self):
        prepared, chunks = self.chunks([[200] * 4, [220] * 4], 0.25)
        self.assertEqual(len(chunks), 2)
        (a1, b1), (a2, b2) = chunks
        self.assertLessEqual(b1, a2)                                   # the readings do not overlap
        self.assertAlmostEqual((b1 - a1), (b2 - a2), delta=0.1)
        self.assertLessEqual(b2, len(prepared.samples) / SR + 1e-6)

    def test_a_short_pause_inside_a_word_does_not_split_it(self):
        _, chunks = self.chunks([[200] * 4, [220] * 4], 0.10)
        self.assertEqual(len(chunks), 1)

    def test_at_most_max_chunks_readings_are_returned_in_time_order(self):
        _, chunks = self.chunks([[200] * 4] * 5, 0.3)
        self.assertLessEqual(len(chunks), CONFIG["variants"]["max_chunks"])
        self.assertEqual(chunks, sorted(chunks))

    def test_silence_has_no_chunks(self):
        prepared = audio_prep.prepare_samples(np.zeros(SR, dtype=np.float32), SR, CONFIG)
        self.assertEqual(audio_prep.speech_chunks(prepared, CONFIG), [])

    def test_slicing_keeps_the_quality_verdict_and_the_time_offset(self):
        prepared, chunks = self.chunks([[200] * 4, [220] * 4], 0.25)
        piece = audio_prep.slice_prepared(prepared, *chunks[1])
        self.assertAlmostEqual(len(piece.samples) / SR, chunks[1][1] - chunks[1][0], delta=0.001)
        self.assertAlmostEqual(piece.trim_start_sec, prepared.trim_start_sec + chunks[1][0], places=3)
        self.assertEqual(piece.reasons, prepared.reasons)
        self.assertEqual(piece.sample_rate, prepared.sample_rate)


class RejectionFlowTests(unittest.TestCase):
    CALIBRATION = {"status": "calibrated", "pitch_floor": 90.0, "pitch_ceiling": 450.0, "f0_median": 200.0}

    def test_rejected_recording_skips_alignment_and_gets_a_specific_message(self):
        result = tp.verify_utterance(np.zeros(SR), SR, "媽麻", self.CALIBRATION, config=CONFIG,
                                     rejected_reasons=["clipped", "noisy"])   # no alignment injected: MFA must not run
        self.assertEqual({s["status"] for s in result["syllables"]}, {"insufficient_evidence"})
        self.assertEqual(result["syllables"][0]["reasons"], ["audio_quality_failed", "clipped", "noisy"])
        self.assertEqual(result["feedback"], CONFIG["feedback"]["audio_quality"]["clipped"])
        self.assertEqual(result["audio_quality_failed"], ["clipped", "noisy"])
        self.assertIsNone(result["f0_extraction"])

    def test_payload_carries_the_quality_report(self):
        prepared = audio_prep.prepare_samples(np.clip(voice() * 20, -1, 1), SR, CONFIG)
        result = tp.verify_utterance(prepared.samples, prepared.sample_rate, "媽", self.CALIBRATION, config=CONFIG,
                                     rejected_reasons=prepared.rejected_reasons)
        payload = ta.build_payload(
            result, groups=[("媽", 0, 1)], word_annotation_status="missing", boundaries_from="punctuation clauses",
            practice_mode="custom", sample_id="custom", target_script="媽", audio_quality=prepared.report())
        self.assertFalse(payload["audio_quality"]["passed"])
        self.assertIn("clipped", payload["audio_quality"]["reasons"])
        self.assertEqual(payload["phrases"][0]["status"], "insufficient_evidence")
        self.assertEqual(payload["feedback"], CONFIG["feedback"]["audio_quality"]["clipped"])

    def test_a_good_synthetic_utterance_passes_the_gate_and_is_analysed(self):
        audio, spans = tone_audio([[200, 200, 200, 200]])
        prepared = audio_prep.prepare_samples(audio, SR, CONFIG)
        self.assertTrue(prepared.passed, prepared.reasons)


if __name__ == "__main__":
    unittest.main()
