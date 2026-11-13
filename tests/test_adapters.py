import unittest

from tokenledger.adapters import (
    from_anthropic_usage,
    from_openai_usage,
    from_records,
)


class TestAnthropicAdapter(unittest.TestCase):
    def test_total_input_sums_fresh_plus_cache(self):
        # Anthropic reports fresh input separately from cached input.
        rec = {
            "usage": {
                "input_tokens": 500,
                "cache_creation_input_tokens": 1000,
                "cache_read_input_tokens": 20000,
                "output_tokens": 120,
            }
        }
        r = from_anthropic_usage(rec, turn=1, step=0)
        self.assertEqual(r.input_tokens, 500 + 1000 + 20000)  # 21500 total context
        self.assertEqual(r.cache_read_tokens, 20000)
        self.assertEqual(r.output_tokens, 120)
        self.assertEqual(r.fresh_tokens, 1500)  # input - cache_read

    def test_usage_at_top_level(self):
        rec = {"input_tokens": 10, "cache_read_input_tokens": 90, "output_tokens": 5}
        r = from_anthropic_usage(rec, turn=2, step=3)
        self.assertEqual(r.input_tokens, 100)
        self.assertEqual(r.cache_read_tokens, 90)
        self.assertEqual(r.turn, 2)
        self.assertEqual(r.step, 3)

    def test_missing_cache_fields_default_zero(self):
        r = from_anthropic_usage({"usage": {"input_tokens": 42}}, turn=1, step=0)
        self.assertEqual(r.input_tokens, 42)
        self.assertEqual(r.cache_read_tokens, 0)


class TestOpenAIAdapter(unittest.TestCase):
    def test_prompt_tokens_is_total(self):
        rec = {
            "usage": {
                "prompt_tokens": 20500,
                "completion_tokens": 120,
                "total_tokens": 20620,
                "prompt_tokens_details": {"cached_tokens": 20000},
            }
        }
        r = from_openai_usage(rec, turn=1, step=0)
        self.assertEqual(r.input_tokens, 20500)  # already total, not summed
        self.assertEqual(r.cache_read_tokens, 20000)
        self.assertEqual(r.output_tokens, 120)

    def test_no_cached_details(self):
        r = from_openai_usage({"usage": {"prompt_tokens": 100, "completion_tokens": 10}}, turn=1, step=0)
        self.assertEqual(r.input_tokens, 100)
        self.assertEqual(r.cache_read_tokens, 0)


class TestFromRecords(unittest.TestCase):
    def test_turn_step_keys(self):
        recs = [
            {"t": 1, "s": 0, "usage": {"prompt_tokens": 100}},
            {"t": 1, "s": 1, "usage": {"prompt_tokens": 200}},
            {"t": 2, "s": 0, "usage": {"prompt_tokens": 150}},
        ]
        reqs = from_records(recs, "openai", turn_key="t", step_key="s")
        self.assertEqual([(r.turn, r.step) for r in reqs], [(1, 0), (1, 1), (2, 0)])
        self.assertEqual([r.input_tokens for r in reqs], [100, 200, 150])

    def test_default_single_turn_sequential_steps(self):
        recs = [{"usage": {"prompt_tokens": 10}}, {"usage": {"prompt_tokens": 20}}]
        reqs = from_records(recs, "openai")
        self.assertEqual([(r.turn, r.step) for r in reqs], [(1, 0), (1, 1)])

    def test_turn_fn(self):
        recs = [{"sid": "a", "usage": {"prompt_tokens": 10}},
                {"sid": "a", "usage": {"prompt_tokens": 10}}]
        reqs = from_records(recs, "openai", turn_fn=lambda r: 7 if r["sid"] == "a" else 0)
        self.assertEqual([r.turn for r in reqs], [7, 7])

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            from_records([], "cohere")


if __name__ == "__main__":
    unittest.main()
