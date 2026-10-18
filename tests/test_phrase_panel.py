import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE = (ROOT / "main.py").read_text(encoding="utf-8")
RESULT_CARD = SOURCE[SOURCE.index('<div id="resultCard"'): SOURCE.index('<div id="resultExtras">') + 40]


class WordCardsPickTheGraphTests(unittest.TestCase):
    def test_every_language_has_the_arrow_labels(self):
        for key in ("phrasePrev", "phraseNext"):
            self.assertEqual(len(re.findall(r"\b%s: " % key, SOURCE)), 5, key)

    def test_the_graph_sits_right_under_the_row_of_cards_not_in_a_list_at_the_bottom(self):
        self.assertLess(RESULT_CARD.index('<div id="pronunciationResults">'), RESULT_CARD.index('id="pitchPanel"'))
        self.assertLess(RESULT_CARD.index('id="pitchPanel"'), RESULT_CARD.index('id="resultExtras"'))       # the JSON output is below, not between cards and graph

    def test_the_panel_has_arrows_a_title_one_chart_and_the_controls(self):
        panel = RESULT_CARD[RESULT_CARD.index('<section id="pitchPanel"'): RESULT_CARD.index('<div id="resultExtras">')]
        for part in ('id="phrasePrevBtn"', 'onclick="movePhrase(-1)"', 'id="pitchPanelTitle"', 'id="pitchChart"', 'id="phraseNextBtn"', 'onclick="movePhrase(1)"',
                     'id="pitchScale"', 'id="showExpectedTone"'):
            self.assertIn(part, panel, part)

    def test_the_stacked_list_of_graphs_and_the_big_title_are_gone(self):
        for gone in ("clausesContainer", "pitchDetails", "openPhraseChart", "pitch-clause", 'id="phrase-chart-'):
            self.assertNotIn(gone, SOURCE, gone)
        self.assertIn('<h2 class="sr-only" data-i18n="resultsTitle">', RESULT_CARD)                  # still there for screen readers

    def test_a_word_card_selects_its_graph_and_keeps_the_full_text(self):
        self.assertIn('aria-controls="pitchChart" aria-pressed="false" onclick="selectPhrase(${index})"', SOURCE)
        for kept in ("word-card-label", "word-card-tone-sequence", "word-card-status", "word-card-advice"):
            self.assertIn(kept, SOURCE, kept)

    def test_a_new_result_opens_at_the_first_word_that_is_not_good_and_a_redraw_keeps_the_choice(self):
        self.assertIn('if (renderedResult !== data) selectedPhrase = Math.max(0, phrases.findIndex(phrase => phrase.status !== "likely_correct"));', SOURCE)

    def test_the_left_and_right_keys_move_between_words_but_leave_form_controls_alone(self):
        handler = SOURCE[SOURCE.index('document.getElementById("resultCard").addEventListener("keydown"'): SOURCE.index("function renderVisualResults(")]
        self.assertIn('"ArrowLeft"', handler)
        self.assertIn('"ArrowRight"', handler)
        self.assertIn("input, select, textarea, audio", handler)
        self.assertIn("movePhrase(", handler)

    def test_the_page_only_scrolls_when_the_graph_is_out_of_view(self):
        select = SOURCE[SOURCE.index("function selectPhrase(index)"): SOURCE.index("function movePhrase(")]
        self.assertIn('block: "nearest"', select)


if __name__ == "__main__":
    unittest.main()
