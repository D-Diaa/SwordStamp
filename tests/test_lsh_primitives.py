"""Test CUDA LSH masks and margin rejection."""
import unittest
import numpy as np
import torch

REQUIRE_CUDA = unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")


@REQUIRE_CUDA
class TestGetMaskFromSeed(unittest.TestCase):

    def _get_mask(self, lsh_dim=8, lmbd=0.25, seed=12345, key=None):
        from watermarking.primitives import get_mask_from_seed
        from watermarking.primitives import hash_key
        return get_mask_from_seed(lsh_dim, lmbd, seed, key=key or hash_key)

    # Size.

    def test_size_equals_floor_n_bins_times_lmbd(self):
        for lsh_dim in [2, 4, 8]:
            for lmbd in [0.25, 0.5, 0.1]:
                with self.subTest(lsh_dim=lsh_dim, lmbd=lmbd):
                    n_bins = 2 ** lsh_dim
                    expected = int(n_bins * lmbd)
                    mask = self._get_mask(lsh_dim=lsh_dim, lmbd=lmbd)
                    self.assertEqual(len(mask), expected)

    # Range.

    def test_all_values_in_valid_bin_range(self):
        for lsh_dim in [2, 4, 8]:
            with self.subTest(lsh_dim=lsh_dim):
                n_bins = 2 ** lsh_dim
                mask = self._get_mask(lsh_dim=lsh_dim)
                self.assertGreaterEqual(mask.min().item(), 0)
                self.assertLess(mask.max().item(), n_bins)

    # Uniqueness.

    def test_no_duplicate_entries(self):
        for lsh_dim in [4, 8]:
            with self.subTest(lsh_dim=lsh_dim):
                mask = self._get_mask(lsh_dim=lsh_dim)
                values = mask.cpu().tolist()
                self.assertEqual(len(values), len(set(values)))

    # Determinism.

    def test_same_seed_and_key_reproduce_identical_mask(self):
        seed = 42
        mask1 = self._get_mask(seed=seed)
        mask2 = self._get_mask(seed=seed)
        self.assertEqual(set(mask1.cpu().tolist()), set(mask2.cpu().tolist()))

    def test_different_seeds_give_different_masks(self):
        # A 64-of-256 mask makes random collision unlikely.
        mask_a = self._get_mask(lsh_dim=8, lmbd=0.25, seed=1)
        mask_b = self._get_mask(lsh_dim=8, lmbd=0.25, seed=2)
        self.assertNotEqual(
            set(mask_a.cpu().tolist()), set(mask_b.cpu().tolist()),
        )

    def test_different_keys_give_different_masks(self):
        from watermarking.primitives import hash_key
        # lsh_dim=8, lmbd=0.25 → 64-element mask out of 256 bins → collision prob ≈ 0
        seed = 9999
        mask_a = self._get_mask(lsh_dim=8, lmbd=0.25, seed=seed, key=hash_key)
        mask_b = self._get_mask(lsh_dim=8, lmbd=0.25, seed=seed, key=hash_key + 1)
        self.assertNotEqual(
            set(mask_a.cpu().tolist()), set(mask_b.cpu().tolist()),
        )

    # Boundary lambda values.

    def test_zero_lmbd_gives_empty_mask(self):
        mask = self._get_mask(lsh_dim=8, lmbd=0.0)
        self.assertEqual(len(mask), 0)

    def test_full_lmbd_gives_all_bins(self):
        lsh_dim = 4
        mask = self._get_mask(lsh_dim=lsh_dim, lmbd=1.0)
        self.assertEqual(len(mask), 2 ** lsh_dim)


@REQUIRE_CUDA
class TestComputeLSHMargins(unittest.TestCase):
    """Verify minimum hyperplane-distance margins."""

    def _make_model_with_normals(self, normals_np):
        """Build a minimal stub with hasher.normals set to normals_np."""

        class _FakeHasher:
            pass

        class _FakeModel:
            pass

        model = _FakeModel()
        model.hasher = _FakeHasher()
        model.hasher.normals = normals_np
        return model

    def test_embed_on_hyperplane_is_rejected(self):
        """Reject embeddings on the hyperplane."""
        from watermarking.primitives import compute_lsh_margins

        rng = np.random.RandomState(1)
        dim = 64
        normal = rng.randn(dim).astype("float32")
        normal /= np.linalg.norm(normal)

        # Construct embed orthogonal to normal
        embed = rng.randn(dim).astype("float32")
        embed -= np.dot(embed, normal) * normal  # project out normal component
        embed /= np.linalg.norm(embed)

        model = self._make_model_with_normals(normal[None])  # shape (1, dim)
        embed_t = torch.tensor(embed[None], device="cuda")

        margins = compute_lsh_margins(model, ["s"], embeds=embed_t)
        self.assertLess(margins[0].item(), 0.05)

    def test_embed_parallel_to_normal_is_accepted(self):
        """Accept embeddings far from the hyperplane."""
        from watermarking.primitives import compute_lsh_margins

        rng = np.random.RandomState(2)
        dim = 64
        normal = rng.randn(dim).astype("float32")
        normal /= np.linalg.norm(normal)

        embed = normal.copy()  # exactly parallel → |cos| = 1
        model = self._make_model_with_normals(normal[None])
        embed_t = torch.tensor(embed[None], device="cuda")

        margins = compute_lsh_margins(model, ["s"], embeds=embed_t)
        self.assertGreaterEqual(margins[0].item(), 0.3)

    def test_margin_threshold_distinguishes_acceptance(self):
        """Same embed: accepted at a low margin, rejected at a high margin."""
        from watermarking.primitives import compute_lsh_margins

        rng = np.random.RandomState(3)
        dim = 64
        normal = rng.randn(dim).astype("float32")
        normal /= np.linalg.norm(normal)

        # |cos| = 0.5 → accepted when margin < 0.5, rejected when margin > 0.5
        orth = rng.randn(dim).astype("float32")
        orth -= np.dot(orth, normal) * normal
        orth /= np.linalg.norm(orth)
        cos_val = 0.5
        embed = cos_val * normal + np.sqrt(1 - cos_val ** 2) * orth
        embed /= np.linalg.norm(embed)
        embed_t = torch.tensor(embed[None], device="cuda")

        model = self._make_model_with_normals(normal[None])

        margin = compute_lsh_margins(model, ["s"], embeds=embed_t)[0].item()
        self.assertGreaterEqual(margin, 0.3)
        self.assertLess(margin, 0.7)

    def test_batch_of_sentences_filters_correctly(self):
        """With multiple candidates, only those with min |cos| >= margin pass."""
        from watermarking.primitives import compute_lsh_margins

        rng = np.random.RandomState(4)
        dim = 64
        normal = rng.randn(dim).astype("float32")
        normal /= np.linalg.norm(normal)

        # embed0: orthogonal to normal (on hyperplane) → rejected
        orth = rng.randn(dim).astype("float32")
        orth -= np.dot(orth, normal) * normal
        orth /= np.linalg.norm(orth)
        embed0 = orth

        # embed1: parallel to normal → accepted
        embed1 = normal.copy()

        embeds = torch.tensor(np.stack([embed0, embed1]), device="cuda")
        model = self._make_model_with_normals(normal[None])

        margins = compute_lsh_margins(model, ["a", "b"], embeds=embeds)
        self.assertLess(margins[0].item(), 0.2)
        self.assertGreaterEqual(margins[1].item(), 0.2)


if __name__ == "__main__":
    unittest.main()
