import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from quality.metrics.perplexity import evaluate_perplexity


class _Tokenizer:
    bos_token = ""

    def encode(self, text, **kwargs):
        length = int(text)
        return torch.arange(length).unsqueeze(0)


class _LossModel:
    def __init__(self, losses):
        self._losses = iter(losses)

    def __call__(self, input_ids, labels):
        return (torch.tensor(next(self._losses), dtype=torch.float64),)


class TestPerplexity(unittest.TestCase):
    @patch("quality.metrics.common.device", "cpu")
    def test_gen_ppl_aggregates_nll_over_predicted_tokens(self):
        # The documents have two and one causal predictions, respectively.
        # Averaging their PPLs would give 5; corpus PPL is exp(weighted NLL).
        model = _LossModel([math.log(2), math.log(8)])

        result = evaluate_perplexity(model, _Tokenizer(), ["3", "2"])

        expected = math.exp((2 * math.log(2) + math.log(8)) / 3)
        self.assertAlmostEqual(result["gen_ppl"], expected, places=6)
        self.assertEqual(result["gen_ppl_per_sample"].tolist(), [2.0, 8.0])
        self.assertEqual(result["gen_ppl_scored_tokens_per_sample"].tolist(), [2, 1])
        self.assertEqual(result["gen_ppl_scored_tokens"], 3)
        np.testing.assert_allclose(
            result["gen_ppl_total_nll_per_sample"],
            [2 * math.log(2), math.log(8)],
        )
        self.assertAlmostEqual(result["gen_ppl_total_nll"], math.log(32), places=6)

    @patch("quality.metrics.common.device", "cpu")
    def test_prediction_free_documents_do_not_enter_corpus_denominator(self):
        model = _LossModel([math.log(2), math.log(100)])

        result = evaluate_perplexity(model, _Tokenizer(), ["3", "1"])

        self.assertAlmostEqual(result["gen_ppl"], 2.0, places=6)
        self.assertEqual(result["gen_ppl_scored_tokens_per_sample"].tolist(), [2, 0])
        self.assertEqual(result["gen_ppl_scored_tokens"], 2)
        self.assertTrue(math.isnan(result["gen_ppl_total_nll_per_sample"][1]))


if __name__ == "__main__":
    unittest.main()
