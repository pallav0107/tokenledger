import unittest

from tokenledger.account import account, what_if_remove_step
from tokenledger.parse import Request


class TestAccount(unittest.TestCase):
    def setUp(self):
        self.reqs = [
            Request(turn=1, step=0, input_tokens=1000, cache_read_tokens=900,
                    priming_tokens=800, conversation_tokens=0, tool_result_tokens=100, output_tokens=50),
            Request(turn=1, step=1, input_tokens=1200, cache_read_tokens=1100,
                    priming_tokens=800, conversation_tokens=300, tool_result_tokens=100, output_tokens=50),
            Request(turn=2, step=0, input_tokens=1000, cache_read_tokens=900,
                    priming_tokens=800, conversation_tokens=0, tool_result_tokens=100, output_tokens=50),
        ]

    def test_totals(self):
        m = account(self.reqs)
        self.assertEqual(m.total_input, 3200)
        self.assertEqual(m.total_cache_read, 2900)
        self.assertAlmostEqual(m.cache_read_pct, 2900 / 3200)
        self.assertEqual(m.n_turns, 2)
        self.assertEqual(m.steps_per_turn, {1: 2, 2: 1})
        self.assertEqual(m.priming_total, 2400)
        self.assertAlmostEqual(m.priming_pct, 2400 / 3200)

    def test_fresh_tokens(self):
        self.assertEqual(self.reqs[0].fresh_tokens, 100)

    def test_what_if_values(self):
        self.assertEqual(what_if_remove_step(self.reqs, 1, 0), 1000 + 100 * 1)
        self.assertEqual(what_if_remove_step(self.reqs, 1, 1), 1200 + 100 * 0)

    def test_superlinear_earlier_step_saves_more(self):
        t = [
            Request(turn=9, step=0, input_tokens=500, tool_result_tokens=100),
            Request(turn=9, step=1, input_tokens=500, tool_result_tokens=100),
            Request(turn=9, step=2, input_tokens=500, tool_result_tokens=100),
        ]
        self.assertEqual(what_if_remove_step(t, 9, 0), 500 + 100 * 2)
        self.assertEqual(what_if_remove_step(t, 9, 2), 500 + 100 * 0)
        self.assertGreater(what_if_remove_step(t, 9, 0), what_if_remove_step(t, 9, 2))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            account([])


if __name__ == "__main__":
    unittest.main()
