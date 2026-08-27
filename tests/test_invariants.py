"""Test CUDA watermark invariants without loading models."""
import random
import unittest

import torch

REQUIRE_CUDA = unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")


@REQUIRE_CUDA
class TestKeySymmetryLSH(unittest.TestCase):
    """Same key at generation and detection time must produce the same mask."""

    def _mask(self, seed, key):
        from watermarking.primitives import get_mask_from_seed
        return set(get_mask_from_seed(8, 0.25, seed, key=key).cpu().tolist())

    def test_same_seed_same_key_gives_identical_mask(self):
        from watermarking.primitives import hash_key
        seed = 42
        self.assertEqual(self._mask(seed, hash_key), self._mask(seed, hash_key))

    def test_different_key_gives_different_mask(self):
        from watermarking.primitives import hash_key
        seed = 1234
        mask_a = self._mask(seed, hash_key)
        mask_b = self._mask(seed, hash_key + 1)
        self.assertNotEqual(mask_a, mask_b,
                            "Different keys should produce different masks")

    def test_wrong_key_at_detection_time_changes_mask(self):
        """Verify that changing the key changes the mask."""
        from watermarking.primitives import get_mask_from_seed
        from watermarking.primitives import hash_key

        lsh_dim = 8
        lmbd = 0.25
        seed = 9999

        mask_correct = get_mask_from_seed(lsh_dim, lmbd, seed, key=hash_key)
        mask_wrong = get_mask_from_seed(lsh_dim, lmbd, seed, key=hash_key + 1)

        self.assertNotEqual(
            set(mask_correct.cpu().tolist()),
            set(mask_wrong.cpu().tolist()),
            "Different keys must produce different masks (same seed)",
        )


@REQUIRE_CUDA
class TestKeySymmetryKMeans(unittest.TestCase):

    def _mask(self, cluster_id, key):
        from watermarking.primitives import get_cluster_mask
        cid = torch.tensor(cluster_id)
        return set(get_cluster_mask(cid, 8, 0.25, key=key).cpu().tolist())

    def test_same_cluster_same_key_gives_identical_mask(self):
        from watermarking.primitives import hash_key
        self.assertEqual(self._mask(0, hash_key), self._mask(0, hash_key))

    def test_different_key_gives_different_mask(self):
        from watermarking.primitives import hash_key
        # cluster_id=0 makes seed=0 regardless of key (0*key=0); use cluster_id=1.
        mask_a = self._mask(1, hash_key)
        mask_b = self._mask(1, hash_key + 1)
        self.assertNotEqual(mask_a, mask_b)


@REQUIRE_CUDA
class TestLambdaConservationLSH(unittest.TestCase):
    """For uniform random hash values, the acceptance rate should converge to lmbd."""

    def test_acceptance_rate_converges_to_lmbd(self):
        from watermarking.primitives import get_mask_from_seed
        from watermarking.primitives import hash_key

        lsh_dim = 8
        lmbd = 0.25
        n_bins = 2 ** lsh_dim
        n_trials = 200
        rng = random.Random(42)

        accepted = 0
        for seed_val in range(n_trials):
            hash_val = rng.randint(0, n_bins - 1)
            mask = get_mask_from_seed(lsh_dim, lmbd, seed_val, key=hash_key)
            if hash_val in mask:
                accepted += 1

        rate = accepted / n_trials
        # Allow four null standard deviations.
        self.assertAlmostEqual(rate, lmbd, delta=0.08,
                               msg=f"Acceptance rate {rate:.3f} is too far from lmbd={lmbd}")

    def test_mask_size_is_exact_for_all_seeds(self):
        from watermarking.primitives import get_mask_from_seed
        from watermarking.primitives import hash_key

        lsh_dim = 8
        lmbd = 0.25
        expected_size = int(2 ** lsh_dim * lmbd)

        for seed in range(200):
            mask = get_mask_from_seed(lsh_dim, lmbd, seed, key=hash_key)
            self.assertEqual(
                len(mask), expected_size,
                f"seed={seed}: wrong mask size {len(mask)} (expected {expected_size})"
            )


@REQUIRE_CUDA
class TestLambdaConservationKMeans(unittest.TestCase):

    def test_mask_size_is_exact_for_all_cluster_ids(self):
        from watermarking.primitives import get_cluster_mask
        from watermarking.primitives import hash_key

        k_dim = 8
        lmbd = 0.25
        expected_size = int(k_dim * lmbd)

        for cid in range(k_dim):
            mask = get_cluster_mask(torch.tensor(cid), k_dim, lmbd, key=hash_key)
            self.assertEqual(len(mask), expected_size,
                             f"cluster_id={cid}: expected size {expected_size}")


@REQUIRE_CUDA
class TestFixedVsContextDependentMaskIdentity(unittest.TestCase):
    """Verify that fixed mode reuses one mask."""

    def test_fixed_seed_gives_same_mask_each_call(self):
        from watermarking.primitives import get_mask_from_seed
        from watermarking.primitives import hash_key

        lsh_dim = 8
        lmbd = 0.25
        fixed_seed = 777  # derived from hash(secret_message)

        masks = [
            set(get_mask_from_seed(lsh_dim, lmbd, fixed_seed, key=hash_key).cpu().tolist())
            for _ in range(10)
        ]
        self.assertEqual(len(set(frozenset(m) for m in masks)), 1,
                         "All 10 calls with the same seed must give the same mask")

    def test_context_dependent_seeds_give_varying_masks(self):
        from watermarking.primitives import get_mask_from_seed
        from watermarking.primitives import hash_key

        # Simulate 10 unique seeds (different previous-sentence hashes)
        seeds = list(range(10))
        masks = [
            frozenset(get_mask_from_seed(8, 0.25, s, key=hash_key).cpu().tolist())
            for s in seeds
        ]
        unique_masks = set(masks)
        self.assertGreater(len(unique_masks), 1,
                           "Different seeds should yield different masks")


@REQUIRE_CUDA
class TestMaskOverlapStatistics(unittest.TestCase):
    """Verify expected overlap for independent masks."""

    def test_pairwise_overlap_near_expected(self):
        from watermarking.primitives import get_mask_from_seed
        from watermarking.primitives import hash_key

        lsh_dim = 8
        lmbd = 0.25
        n_bins = 2 ** lsh_dim
        n_masks = 50
        expected_overlap = lmbd ** 2 * n_bins  # 16 for lsh_dim=8, lmbd=0.25

        masks = [
            set(get_mask_from_seed(lsh_dim, lmbd, seed, key=hash_key).cpu().tolist())
            for seed in range(n_masks)
        ]

        overlaps = []
        for i in range(n_masks):
            for j in range(i + 1, n_masks):
                overlaps.append(len(masks[i] & masks[j]))

        mean_overlap = sum(overlaps) / len(overlaps)
        # Allow ±50% relative tolerance; this is a statistical test
        self.assertAlmostEqual(
            mean_overlap, expected_overlap, delta=expected_overlap * 0.5,
            msg=f"Mean pairwise mask overlap {mean_overlap:.2f} far from expected {expected_overlap:.2f}"
        )


if __name__ == "__main__":
    unittest.main()
