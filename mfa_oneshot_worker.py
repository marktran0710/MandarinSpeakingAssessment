"""
A pre-loaded MFA aligner that serves exactly ONE alignment and then exits (run with the MFA environment's python).

`mfa align_one` spends about 2.7 s of every call starting Python and loading the acoustic model. This process
does that work in advance (op "load"), then performs the same steps align_one performs for one utterance (same
lexicon compiler, tokenizer, MFCC + CMVN, KalpyAligner and options, same JSON export). Because it never serves a
second request it is in the same fresh-process state as align_one, so the alignment is bit-identical: a
long-lived process is not (MFCC dither draws from a random state that moves on after every utterance).

Protocol: one JSON object per line on stdin and on stdout; everything the libraries print goes to stderr.
    <- {"ready": true}                                               (imports done)
    -> {"op": "load", "dictionary": "...", "acoustic_model": "mandarin_mfa"}
    <- {"loaded": true}
    -> {"dictionary": "...", "acoustic_model": "...", "wav": "...", "txt": "..."}
    <- {"tiers": {...}}   or   {"error": "..."}                      (then the process ends)
Closing stdin ends the process too, so a worker never outlives the server that started it.
"""
import json
import sys
import tempfile
import traceback
from pathlib import Path

PROTOCOL = sys.stdout
sys.stdout = sys.stderr

import pywrapfst  # noqa: E402
from kalpy.aligner import KalpyAligner  # noqa: E402
from kalpy.feat.cmvn import CmvnComputer  # noqa: E402
from kalpy.fstext.lexicon import HierarchicalCtm, LexiconCompiler  # noqa: E402
from kalpy.utterance import Segment  # noqa: E402
from kalpy.utterance import Utterance as KalpyUtterance  # noqa: E402
from montreal_forced_aligner import config  # noqa: E402
from montreal_forced_aligner.alignment import PretrainedAligner  # noqa: E402
from montreal_forced_aligner.command_line.utils import validate_model_arg  # noqa: E402
from montreal_forced_aligner.corpus.classes import FileData  # noqa: E402
from montreal_forced_aligner.data import BRACKETED_WORD, CUTOFF_WORD, LAUGHTER_WORD, OOV_WORD  # noqa: E402
from montreal_forced_aligner.dictionary.mixins import (  # noqa: E402
    DEFAULT_BRACKETS, DEFAULT_CLITIC_MARKERS, DEFAULT_COMPOUND_MARKERS, DEFAULT_PUNCTUATION, DEFAULT_WORD_BREAK_MARKERS)
from montreal_forced_aligner.online.alignment import tokenize_utterance_text  # noqa: E402
from montreal_forced_aligner.tokenization.simple import SimpleTokenizer  # noqa: E402

ALIGN_KEYS = ("beam", "retry_beam", "acoustic_scale", "transition_scale", "self_loop_scale", "boost_silence")


class Loaded:
    """One (acoustic model, dictionary) pair, ready to align."""

    def __init__(self, acoustic_name, dictionary_path):
        self.acoustic_model = validate_model_arg(acoustic_name, "acoustic")
        dictionary = validate_model_arg(str(dictionary_path), "dictionary")
        c = PretrainedAligner.parse_parameters(None, {}, [])
        directory = config.TEMPORARY_DIRECTORY.joinpath("extracted_models", "dictionary", dictionary.path.stem)
        directory.mkdir(parents=True, exist_ok=True)
        compiler = LexiconCompiler(
            disambiguation=False,
            silence_probability=self.acoustic_model.parameters["silence_probability"],
            initial_silence_probability=self.acoustic_model.parameters["initial_silence_probability"],
            final_silence_correction=self.acoustic_model.parameters["final_silence_correction"],
            final_non_silence_correction=self.acoustic_model.parameters["final_non_silence_correction"],
            silence_phone=self.acoustic_model.parameters["optional_silence_phone"],
            oov_phone=self.acoustic_model.parameters["oov_phone"],
            position_dependent_phones=self.acoustic_model.parameters["position_dependent_phones"],
            phones=self.acoustic_model.parameters["non_silence_phones"],
            ignore_case=c.get("ignore_case", True),
        )
        l_fst, l_align, words, phones = (directory / n for n in ("L.fst", "L_align.fst", "words.txt", "phones.txt"))
        if l_fst.exists() and not config.CLEAN:                       # the compiled dictionary MFA cached earlier
            compiler.load_l_from_file(l_fst)
            compiler.load_l_align_from_file(l_align)
            compiler.word_table = pywrapfst.SymbolTable.read_text(words)
            compiler.phone_table = pywrapfst.SymbolTable.read_text(phones)
        else:
            compiler.load_pronunciations(dictionary.path)
            compiler.create_fsts()
            compiler.clear()
            compiler.fst.write(str(l_fst))
            compiler.align_fst.write(str(l_align))
            compiler.word_table.write_text(words)
            compiler.phone_table.write_text(phones)
        self.compiler = compiler
        self.tokenizer = SimpleTokenizer(
            word_table=compiler.word_table,
            word_break_markers=c.get("word_break_markers", DEFAULT_WORD_BREAK_MARKERS),
            punctuation=c.get("punctuation", DEFAULT_PUNCTUATION),
            clitic_markers=c.get("clitic_markers", DEFAULT_CLITIC_MARKERS),
            compound_markers=c.get("compound_markers", DEFAULT_COMPOUND_MARKERS),
            brackets=c.get("brackets", DEFAULT_BRACKETS),
            laughter_word=c.get("laughter_word", LAUGHTER_WORD),
            oov_word=c.get("oov_word", OOV_WORD),
            bracketed_word=c.get("bracketed_word", BRACKETED_WORD),
            cutoff_word=c.get("cutoff_word", CUTOFF_WORD),
            ignore_case=c.get("ignore_case", True),
        )
        self.aligner = KalpyAligner(self.acoustic_model, compiler, **{k: v for k, v in c.items() if k in ALIGN_KEYS})

    def align(self, wav, txt):
        wav, txt = Path(wav), Path(txt)
        file = FileData.parse_file(wav.stem, wav, txt, "", 0)
        utterances = []
        for utterance in file.utterances:
            segment = Segment(wav, utterance.begin, utterance.end, utterance.channel)
            text = tokenize_utterance_text(utterance.text, self.compiler, self.tokenizer, None)
            utt = KalpyUtterance(segment, text)
            utt.generate_mfccs(self.acoustic_model.mfcc_computer)
            utterances.append(utt)
        cmvn = CmvnComputer().compute_cmvn_from_features([utt.mfccs for utt in utterances])
        ctm_all = HierarchicalCtm([])
        for utt in utterances:
            utt.apply_cmvn(cmvn)
            ctm_all.word_intervals.extend(self.aligner.align_utterance(utt).word_intervals)
        with tempfile.TemporaryDirectory(prefix="mfa-oneshot-") as folder:
            out = Path(folder) / "result.json"
            ctm_all.export_textgrid(out, file_duration=file.wav_info.duration, output_format="json")
            return json.loads(out.read_text(encoding="utf-8"))["tiers"]


def reply(message):
    PROTOCOL.write(json.dumps(message, ensure_ascii=False) + "\n")
    PROTOCOL.flush()


def main():
    reply({"ready": True})
    loaded = {}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            key = (request.get("acoustic_model", "mandarin_mfa"), request["dictionary"])
            if key not in loaded:
                loaded[key] = Loaded(*key)
            if request.get("op") == "load":                       # warm-up: models in memory, nothing aligned yet
                reply({"loaded": True})
                continue
            reply({"tiers": loaded[key].align(request["wav"], request["txt"])})
        except Exception as exc:                                  # the caller falls back to `mfa align_one`
            traceback.print_exc()
            reply({"error": f"{type(exc).__name__}: {exc}"})
        break                                                     # one alignment per process: always a fresh state


if __name__ == "__main__":
    main()
