import unittest

from assistant_engine import respond


class AssistantEngineTests(unittest.TestCase):
    def test_starts_with_independent_l0_prompt(self):
        result = respond("", topic="restaurant", menu_id="menu_a")
        self.assertEqual(result["support"]["level"], 0)
        self.assertEqual(result["reply"], "What would you like to order?")

    def test_help_request_increases_support_without_forcing_all_levels(self):
        result = respond(
            "I don't know",
            history=[{"role": "user", "content": "I don't know"}],
            requested_level=2,
        )
        self.assertEqual(result["support"]["level"], 2)
        self.assertIn("I’d like", result["support"]["prompt"])

    def test_first_help_request_adapts_from_l0(self):
        result = respond(
            "I don't know",
            history=[{"role": "user", "content": "I don't know"}],
            requested_level=0,
        )
        self.assertEqual(result["support"]["level"], 2)

    def test_menu_b_explains_sold_out_item_and_offers_substitute(self):
        result = respond("I want the chicken wrap", menu_id="menu_b")
        self.assertIn("sold out", result["reply"])
        self.assertIn("vegetable sandwich", result["reply"])

    def test_free_topic_is_available(self):
        result = respond("hello", topic="free")
        self.assertIn("I like", result["reply"])


if __name__ == "__main__":
    unittest.main()
