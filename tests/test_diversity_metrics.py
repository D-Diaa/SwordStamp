import math
import unittest

import numpy as np

from quality.metrics.diversity import (
    compute_diversity_metrics,
    distinct_n,
    evaluate_diversity,
    ngram_repeat_pct,
    sentence_duplicate_pct,
)


class DiversityMetricTests(unittest.TestCase):
    def test_word_metrics_match_whitespace_tokenization_spec(self):
        text = "A b c d a b c d"
        self.assertAlmostEqual(distinct_n(text, 2), 4 / 7)
        self.assertAlmostEqual(ngram_repeat_pct(text, 4), 20.0)

    def test_word_ngrams_cross_sentence_boundaries(self):
        # Global whitespace tokenization includes the "b. a" boundary bigram.
        self.assertAlmostEqual(distinct_n("a b. a b.", 2), 2 / 3)

    def test_zero_denominators_map_to_zero(self):
        self.assertEqual(distinct_n("one", 2), 0.0)
        self.assertEqual(ngram_repeat_pct("one two three", 4), 0.0)
        self.assertEqual(sentence_duplicate_pct(""), 0.0)

    def test_sentence_duplicates_are_trimmed_and_lowercased(self):
        text = "Hello world. A unique sentence! hello WORLD."
        self.assertAlmostEqual(sentence_duplicate_pct(text), 100 / 3)

    def test_sample_schema_contains_only_paper_metrics(self):
        metrics = compute_diversity_metrics("A sentence. A sentence.")
        self.assertEqual(
            set(metrics), {"sent_dup_pct", "distinct_2", "4g_repeat_pct"}
        )
        self.assertEqual(metrics["sent_dup_pct"], 50.0)

    def test_evaluation_averages_samples_and_retains_arrays(self):
        results = evaluate_diversity(["a b c d a b c d", "short"])
        np.testing.assert_allclose(results["distinct_2_per_sample"], [4 / 7, 0.0])
        np.testing.assert_allclose(results["4g_repeat_pct_per_sample"], [20.0, 0.0])
        self.assertAlmostEqual(results["distinct_2"], 2 / 7)
        self.assertAlmostEqual(results["4g_repeat_pct"], 10.0)
        self.assertTrue(math.isfinite(results["distinct_2_ci"]))

    def test_empty_evaluation_has_stable_schema(self):
        results = evaluate_diversity([])
        self.assertTrue(math.isnan(results["sent_dup_pct"]))
        self.assertTrue(math.isnan(results["sent_dup_pct_ci"]))
        self.assertEqual(results["sent_dup_pct_per_sample"].shape, (0,))


if __name__ == "__main__":
    unittest.main()
