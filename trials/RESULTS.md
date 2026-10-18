
## Can the system tell that a syllable is the wrong sound? (ma4 said as he4, shi/si ...) (2026-10-18, measured, nothing switched on)

Today it cannot: tones only (no ASR since 2026-10-16; a one-syllable script is not even aligned). Test: 454 utterances (97 teacher sentences, 160 LATIC L2 speakers, 197 Vietnamese learner
attempts), each read against its CORRECT script (negative) and against 569 variants in which ONE syllable is replaced by a confusable one of the same tone (positive). MFA = a scratch
scorer on the same aligner (per-frame acoustic log-likelihood per syllable/phone; 0.04-0.05 s per alignment in a long-lived worker); "vs rivals" = the same audio re-aligned with each confusable
syllable in that place (about 11 rivals: every confusion kind + 5 random of the same tone) and the script's syllable compared with the best rival; Whisper = faster-whisper small int8 (1.4 s per
clip on 4 CPU threads), toneless pinyin of what it heard matched against the script. Caveats: the positives use CORRECT audio against a wrong script, which favours Whisper (it simply hears the
right word) and says nothing about near-miss learner sounds that its language model may "correct"; the rivals include the true syllable for the swap kinds (best case); a "correct syllable
flagged" on learners may be a real learner error, so false alarms are an upper bound. Wrong syllables flagged (%), at the false-alarm rate Whisper has on each audience:

| audience (correct syllables flagged) | Whisper | MFA fit | MFA fit, phones | MFA vs rivals |
|---|---|---|---|---|
| teacher (1%) | 100 | 3 | 6 | 22 |
| L2 learners (15%) | 99 | 33 | 40 | 80 |
| Vietnamese learners (11%) | 98 | 19 | 42 | 65 |

MFA held to 5% of correct syllables flagged (pooled learners): fit 12-13%, fit phones 23-30%, vs rivals 43-52% of wrong syllables flagged. By kind, Vietnamese learners, same false-alarm rate as Whisper:
shi/si 60% (rivals) vs 12% (fit), n/l 53 vs 20, f/h 74 vs 21, ba/pa 80 vs 24, an/ang 24 vs 0, vowels 84 vs 8, another syllable (ma4/he4) 80 vs 52. The fit score alone separates only a completely
different syllable (AUC 75-87%); with rivals AUC is 76-95% for every kind.
