"""Test CPU and CUDA KMeans watermark primitives."""
import unittest
import torch

REQUIRE_CUDA = unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")



class TestPairwiseCosine(unittest.TestCase):

    def setUp(self):
        from watermarking.primitives import pairwise_cosine
        self.pairwise_cosine = pairwise_cosine

    def _pc(self, a, b):
        return self.pairwise_cosine(
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(b, dtype=torch.float32),
            device=torch.device("cpu"),
        )

    def test_identical_vectors_distance_zero(self):
        a = [[1.0, 0.0, 0.0]]
        dist = self._pc(a, a)
        self.assertAlmostEqual(dist.item(), 0.0, places=5)

    def test_orthogonal_vectors_distance_one(self):
        a = [[1.0, 0.0]]
        b = [[0.0, 1.0]]
        dist = self._pc(a, b)
        self.assertAlmostEqual(dist.item(), 1.0, places=5)

    def test_opposite_vectors_distance_two(self):
        a = [[1.0, 0.0]]
        b = [[-1.0, 0.0]]
        dist = self._pc(a, b)
        self.assertAlmostEqual(dist.item(), 2.0, places=5)

    def test_output_shape_n_by_m(self):
        N, M, D = 3, 5, 8
        a = torch.randn(N, D)
        b = torch.randn(M, D)
        result = self.pairwise_cosine(a, b, device=torch.device("cpu"))
        self.assertEqual(result.shape, (N, M))

    def test_symmetry(self):
        a = torch.randn(4, 16)
        b = torch.randn(4, 16)
        dist_ab = self.pairwise_cosine(a, b, device=torch.device("cpu"))
        dist_ba = self.pairwise_cosine(b, a, device=torch.device("cpu"))
        # Swapping inputs transposes the distance matrix.
        torch.testing.assert_close(dist_ab, dist_ba.T, atol=1e-5, rtol=0)

    def test_non_negative(self):
        a = torch.randn(5, 32)
        b = torch.randn(7, 32)
        result = self.pairwise_cosine(a, b, device=torch.device("cpu"))
        self.assertTrue((result >= -1e-5).all(), "Cosine distance should be non-negative")

    def test_upper_bound_two(self):
        a = torch.randn(5, 32)
        b = torch.randn(7, 32)
        result = self.pairwise_cosine(a, b, device=torch.device("cpu"))
        self.assertTrue((result <= 2.0 + 1e-5).all())



@REQUIRE_CUDA
class TestGetClusterMask(unittest.TestCase):

    def _get_mask(self, cluster_id, k_dim=8, lmbd=0.25, key=None):
        from watermarking.primitives import get_cluster_mask
        from watermarking.primitives import hash_key
        cid = torch.tensor(cluster_id)
        return get_cluster_mask(cid, k_dim, lmbd, key=key or hash_key)

    def test_size_equals_floor_k_times_lmbd(self):
        for k_dim in [4, 8, 16]:
            for lmbd in [0.25, 0.5, 0.1]:
                with self.subTest(k_dim=k_dim, lmbd=lmbd):
                    expected = int(k_dim * lmbd)
                    mask = self._get_mask(0, k_dim=k_dim, lmbd=lmbd)
                    self.assertEqual(len(mask), expected)

    def test_all_values_in_valid_range(self):
        for k_dim in [4, 8]:
            with self.subTest(k_dim=k_dim):
                mask = self._get_mask(0, k_dim=k_dim)
                self.assertGreaterEqual(mask.min().item(), 0)
                self.assertLess(mask.max().item(), k_dim)

    def test_no_duplicates(self):
        for k_dim in [4, 8]:
            with self.subTest(k_dim=k_dim):
                mask = self._get_mask(2, k_dim=k_dim)
                values = mask.cpu().tolist()
                self.assertEqual(len(values), len(set(values)))

    def test_deterministic_same_cluster_id(self):
        mask1 = self._get_mask(3, k_dim=8, lmbd=0.25)
        mask2 = self._get_mask(3, k_dim=8, lmbd=0.25)
        self.assertEqual(set(mask1.cpu().tolist()), set(mask2.cpu().tolist()))

    def test_different_cluster_ids_give_different_masks(self):
        mask_a = self._get_mask(0, k_dim=8, lmbd=0.25)
        mask_b = self._get_mask(1, k_dim=8, lmbd=0.25)
        self.assertNotEqual(set(mask_a.cpu().tolist()), set(mask_b.cpu().tolist()))

    def test_different_keys_give_different_masks(self):
        from watermarking.primitives import hash_key
        # Avoid cluster zero, whose seed ignores the key.
        mask_a = self._get_mask(1, k_dim=32, lmbd=0.5, key=hash_key)
        mask_b = self._get_mask(1, k_dim=32, lmbd=0.5, key=hash_key + 1)
        self.assertNotEqual(set(mask_a.cpu().tolist()), set(mask_b.cpu().tolist()))

    def test_zero_lmbd_gives_empty_mask(self):
        mask = self._get_mask(0, k_dim=8, lmbd=0.0)
        self.assertEqual(len(mask), 0)



@REQUIRE_CUDA
class TestComputeKMeansMargins(unittest.TestCase):

    def _run(self, embeddings, cluster_centers, margin=0.01):
        """Run overlap filtering with precomputed embeddings."""
        from watermarking.primitives import compute_kmeans_margins
        from unittest.mock import MagicMock

        embedder = MagicMock()
        # Match a real encoder's CUDA output.
        embedder.encode.return_value = torch.tensor(embeddings, device="cuda", dtype=torch.float32)
        # Cluster centers must remain on CPU.
        centers = torch.tensor(cluster_centers, dtype=torch.float32)  # CPU

        texts = ["s"] * len(embeddings)
        margins, cluster_ids = compute_kmeans_margins(texts, embedder, centers)
        return margins > margin, cluster_ids

    def test_closest_two_clusters_very_different_accepted(self):
        # 2 cluster centers; 1 embedding clearly closest to center 0
        centers = [[1.0, 0.0], [0.0, 1.0]]
        embeds = [[1.0, 0.0]]  # identical to center 0

        not_rejected, cluster_ids = self._run(embeds, centers, margin=0.0)
        # When margin=0, everything passes (gap ≥ 0 trivially)
        self.assertTrue(not_rejected[0].item())
        self.assertEqual(cluster_ids[0].item(), 0)

    def test_equidistant_from_two_clusters_rejected(self):
        # Centers at (1,0) and (-1,0); embed at (0,1) — equidistant
        centers = [[1.0, 0.0], [-1.0, 0.0]]
        embeds = [[0.0, 1.0]]  # equidistant: cos to both = 0

        # Equal center distances fail any positive margin.
        not_rejected, _ = self._run(embeds, centers, margin=0.1)
        self.assertFalse(not_rejected[0].item())

    def test_batch_returns_correct_shape(self):
        centers = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        embeds = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]

        not_rejected, cluster_ids = self._run(embeds, centers, margin=0.01)
        self.assertEqual(len(not_rejected), 3)
        self.assertEqual(len(cluster_ids), 3)

    def test_assigned_to_nearest_cluster(self):
        centers = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        embeds = [[0.9, 0.1], [0.1, 0.9], [-0.9, 0.1]]  # near centers 0, 1, 2

        _, cluster_ids = self._run(embeds, centers, margin=0.0)
        self.assertEqual(cluster_ids[0].item(), 0)
        self.assertEqual(cluster_ids[1].item(), 1)
        self.assertEqual(cluster_ids[2].item(), 2)


if __name__ == "__main__":
    unittest.main()
