import unittest

from config.loader import load_config
from config.paths import (
    AttackSpec,
    attack_dir,
    generation_path,
    generation_subpath,
    attack_path,
    segmentation_cache_tag,
    target_dir,
    watermark_dir,
)


class TestGenerationPaths(unittest.TestCase):
    def test_context_fixed_and_none_paths(self):
        self.assertEqual(
            generation_path("data/c4", "lsh_fixed", "rejection", "sentence", "nltk"),
            "data/c4/lsh/fixed/rejection/sentence-nltk/candidates-64/watermarked",
        )
        self.assertEqual(
            generation_path("data/c4", "none", "rejection", "sentence", "spacy"),
            "data/c4/none/sentence-spacy",
        )

    def test_semcut_center_tag_includes_boundary_policy(self):
        self.assertEqual(
            segmentation_cache_tag("semspan", "spacy"),
            "semspan_spacy_max15_win5",
        )

    def test_every_candidate_budget_is_part_of_output_identity(self):
        self.assertEqual(
            generation_path(
                "data/c4", "lsh_fixed_diverse", "best-of-n", "semspan", "spacy",
                15, 5, 32,
            ),
            "data/c4/lsh/fixed_diverse/best-of-n/semspan-spacy-max15-win5/"
            "candidates-32/watermarked",
        )
        self.assertEqual(
            generation_path(
                "data/c4", "lsh_fixed_diverse", "best-of-n", "semspan", "spacy",
                15, 5, 128,
            ),
            "data/c4/lsh/fixed_diverse/best-of-n/semspan-spacy-max15-win5/"
            "candidates-128/watermarked",
        )


class TestAttackPaths(unittest.TestCase):
    def test_attack_path_is_sibling_of_watermarked_leaf(self):
        spec = AttackSpec("parrot-bigram", suffixes=("threshold=0.03",))
        self.assertEqual(
            attack_path(
                "data/c4/lsh/fixed/rejection/sentence-nltk/watermarked",
                spec,
            ),
            "data/c4/lsh/fixed/rejection/sentence-nltk/parrot-bigram-threshold=0.03",
        )

    def test_adaptive_suffixes_are_canonical(self):
        spec = AttackSpec(
            "adaptive",
            model="Qwen/Qwen2.5-3B-Instruct",
            prompt="standard",
            k=64,
            suffixes=("bag=min", "surr=semstamp_sbert"),
        )
        self.assertEqual(
            attack_path("runs/watermarked", spec),
            "runs/adaptive-Qwen2.5-3B-Instruct-standard-K64-min-bag=min-surr=semstamp_sbert",
        )

    def test_oracle_keeps_the_candidate_budget_without_the_adaptive_suffix(self):
        spec = AttackSpec(
            "oracle",
            model="Qwen/Qwen2.5-3B-Instruct",
            prompt="standard",
            k=64,
        )
        self.assertEqual(
            attack_path("runs/watermarked", spec),
            "runs/oracle-Qwen2.5-3B-Instruct-standard-K64",
        )

    def test_pegasus_temperature_is_part_of_the_leaf(self):
        spec = AttackSpec("pegasus", suffixes=("temp=2.0",))
        self.assertEqual(
            attack_path("runs/watermarked", spec),
            "runs/pegasus-temp=2.0",
        )


class TestConfigDerivedDirs(unittest.TestCase):
    def _cfg(self, *extra):
        return load_config(None, ["io.data_path=data/c4-val-def", "watermark.mode=lsh", *extra])

    def test_watermark_and_target_dir(self):
        cfg = self._cfg()
        self.assertEqual(
            watermark_dir(cfg),
            "data/c4-val-def/lsh/context/rejection/sentence-nltk/"
            "candidates-64/watermarked",
        )
        self.assertEqual(target_dir(cfg), watermark_dir(cfg))

    def test_attack_dir_and_attack_target(self):
        cfg = self._cfg(
            "io.target=attack", "attack.paraphraser=pegasus", "attack.temperature=2.0"
        )
        self.assertEqual(
            attack_dir(cfg),
            "data/c4-val-def/lsh/context/rejection/sentence-nltk/candidates-64/"
            "pegasus-temp=2.0",
        )
        self.assertEqual(target_dir(cfg), attack_dir(cfg))

    def test_controlled_reorder_strength_is_part_of_attack_path(self):
        cfg = self._cfg(
            "io.target=attack",
            "attack.paraphraser=controlled_reorder",
            "attack.word_edit_ratio=0.75",
        )
        self.assertEqual(
            attack_dir(cfg),
            "data/c4-val-def/lsh/context/rejection/sentence-nltk/candidates-64/"
            "controlled_reorder-ratio=0.75",
        )

    def test_boundary_exchange_strength_is_part_of_attack_path(self):
        cfg = self._cfg(
            "io.target=attack",
            "attack.paraphraser=boundary_exchange",
            "attack.word_edit_ratio=0.5",
        )
        self.assertEqual(
            attack_dir(cfg),
            "data/c4-val-def/lsh/context/rejection/sentence-nltk/candidates-64/"
            "boundary_exchange-ratio=0.5",
        )

    def test_adaptive_attack_dir(self):
        cfg = self._cfg(
            "io.target=attack",
            "attack.paraphraser=adaptive",
            "attack.custom_model=Qwen/Qwen2.5-3B-Instruct",
            "attack.prompt_style=standard",
            "attack.num_candidates=32",
            "attack.anchor=bag",
            "attack.surrogate_tag=bge",
        )
        self.assertEqual(
            attack_dir(cfg),
            "data/c4-val-def/lsh/context/rejection/sentence-nltk/candidates-64/"
            "adaptive-Qwen2.5-3B-Instruct-standard-K32-min-bag=min-surr=bge",
        )

    def test_adaptive_positional_has_no_bag_suffix(self):
        cfg = self._cfg(
            "io.target=attack",
            "attack.paraphraser=adaptive",
            "attack.custom_model=Qwen/Qwen2.5-3B-Instruct",
            "attack.prompt_style=standard",
            "attack.num_candidates=32",
        )
        self.assertEqual(
            attack_dir(cfg),
            "data/c4-val-def/lsh/context/rejection/sentence-nltk/candidates-64/"
            "adaptive-Qwen2.5-3B-Instruct-standard-K32-min",
        )

    def test_oracle_candidate_budget_separates_run_dirs(self):
        def oracle_dir(k):
            return attack_dir(self._cfg(
                "io.target=attack",
                "attack.paraphraser=oracle",
                "attack.custom_model=Qwen/Qwen2.5-3B-Instruct",
                "attack.prompt_style=standard",
                f"attack.num_candidates={k}",
            ))

        self.assertEqual(
            oracle_dir(32),
            "data/c4-val-def/lsh/context/rejection/sentence-nltk/candidates-64/"
            "oracle-Qwen2.5-3B-Instruct-standard-K32",
        )
        # The effort sweep must not overwrite itself.
        self.assertNotEqual(oracle_dir(8), oracle_dir(32))

    def test_adaptive_explicit_semspan_attacker_adds_aseg(self):
        cfg = self._cfg(
            "io.target=attack",
            "segmentation.type=semspan",
            "segmentation.attacker_type=semspan",
            "attack.paraphraser=adaptive",
            "attack.custom_model=Qwen/Qwen2.5-3B-Instruct",
            "attack.prompt_style=standard",
            "attack.num_candidates=32",
        )
        self.assertEqual(
            attack_dir(cfg),
            "data/c4-val-def/lsh/context/rejection/semspan-nltk-max15-win5/"
            "candidates-64/adaptive-Qwen2.5-3B-Instruct-standard-K32-min-"
            "aseg=semspan-nltk-max15-win5",
        )

    def test_output_dir_is_not_baked_into_derivation(self):
        # output_dir is an entry-point-level override, not part of pure derivation.
        cfg = self._cfg("io.output_dir=/tmp/custom")
        self.assertTrue(watermark_dir(cfg).endswith("watermarked"))

    def test_semcut_policy_is_part_of_watermark_path(self):
        default = self._cfg("segmentation.type=semspan")
        self.assertIn(
            "semspan-nltk-max15-win5", watermark_dir(default),
        )
        variants = (
            self._cfg("segmentation.type=semspan", "segmentation.semcut_max_words=14"),
            self._cfg("segmentation.type=semspan", "segmentation.semcut_window=4"),
        )
        paths = {watermark_dir(default), *(watermark_dir(cfg) for cfg in variants)}
        self.assertEqual(len(paths), 3)
        self.assertEqual(watermark_dir(default), watermark_dir(default))

    def test_non_semcut_paths_ignore_semcut_policy(self):
        default = self._cfg("segmentation.type=sentence")
        changed = self._cfg(
            "segmentation.type=sentence",
            "segmentation.semcut_max_words=12",
            "segmentation.semcut_window=6",
        )
        self.assertEqual(watermark_dir(default), watermark_dir(changed))

    def test_nondefault_generation_budget_separates_clean_and_attack_dirs(self):
        default = self._cfg("generation.sampling_method=best-of-n")
        n32 = self._cfg(
            "generation.sampling_method=best-of-n", "generation.num_candidates=32",
            "io.target=attack", "attack.paraphraser=pegasus",
        )
        self.assertIn("/candidates-64/watermarked", watermark_dir(default))
        self.assertIn("/candidates-32/watermarked", watermark_dir(n32))
        self.assertIn("/candidates-32/pegasus", attack_dir(n32))

    def test_semcut_attacker_policy_is_part_of_attack_path(self):
        cfg = self._cfg(
            "io.target=attack",
            "attack.paraphraser=adaptive",
            "attack.custom_model=example/model",
            "segmentation.attacker_type=semspan",
        )
        self.assertIn(
            "aseg=semspan-nltk-max15-win5", attack_dir(cfg),
        )
        changed = self._cfg(
            "io.target=attack",
            "attack.paraphraser=adaptive",
            "attack.custom_model=example/model",
            "segmentation.attacker_type=semspan",
            "segmentation.semcut_window=4",
        )
        self.assertNotEqual(attack_dir(cfg), attack_dir(changed))


if __name__ == "__main__":
    unittest.main()


class TestFixedDiversePaths(unittest.TestCase):
    def test_method_to_algo_mode(self):
        from config.paths import method_to_algo_mode
        self.assertEqual(method_to_algo_mode("lsh_fixed_diverse"), ("lsh", "fixed_diverse"))
        self.assertEqual(method_to_algo_mode("kmeans_fixed_diverse"), ("kmeans", "fixed_diverse"))
        # The plain modes are unchanged.
        self.assertEqual(method_to_algo_mode("lsh_fixed"), ("lsh", "fixed"))
        self.assertEqual(method_to_algo_mode("lsh"), ("lsh", "context"))
