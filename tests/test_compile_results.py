import json
import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from datasets import Dataset


_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _ROOT)
compile_results = importlib.import_module("visualization.compile_results")
encoder_transfer = importlib.import_module("visualization.encoder_transfer")
from visualization.extractors.common import read_per_sample  # noqa: E402


class AttackParsingTests(unittest.TestCase):
    def test_attack_allowlists_cover_only_the_twelve_paper_cells(self):
        cells = [
            (rung.family, rung.mask, rung.sampling, rung.segmentation)
            for family in ("lsh", "kmeans")
            for rung in compile_results.PAPER_LADDERS[family]
        ] + [
            (
                compile_results.PMARK.family,
                compile_results.PMARK.mask,
                compile_results.PMARK.sampling,
                compile_results.PMARK.segmentation,
            ),
            (
                compile_results.SAMARK.family,
                compile_results.SAMARK.mask,
                compile_results.SAMARK.sampling,
                compile_results.SAMARK.segmentation,
            ),
        ]
        expected_sizes = (33, 33, 36, 36, 36, 38, 33, 36, 36, 41, 17, 20)

        registries = [compile_results.paper_attack_leaves(*cell) for cell in cells]
        self.assertEqual(tuple(map(len, registries)), expected_sizes)
        for leaves in registries:
            for leaf in leaves - {"watermarked"}:
                with self.subTest(leaf=leaf):
                    compile_results.parse_attack(leaf)
        self.assertEqual(
            compile_results.paper_attack_leaves(
                "lsh", "fixed", "rejection", "sentence-nltk"
            ),
            frozenset(),
        )
        self.assertEqual(
            compile_results.paper_attack_leaves(
                "kmeans", "fixed_diverse", "best-of-n",
                "semspan-spacy-max12-win4",
            ),
            frozenset(),
        )

    def test_parses_representative_paper_attacks(self):
        cases = {
            "none": ("none", "none"),
            "pegasus": ("pegasus", "default"),
            "parrot-bigram-threshold=0.03": ("parrot", "bigram-0.03"),
            "dipper-lex20-order80": ("dipper", "lex20-order80"),
            "dipper-l80-o20": ("dipper", "lex80-order20"),
            "adp-K64": ("adaptive", "K64-positional-bge"),
            "adpbag-K32": ("adaptive", "K32-bag-bge"),
            "oracle-Qwen2.5-3B-Instruct-standard-K16": (
                "oracle", "K16-detector-aware"
            ),
            "adaptive-Qwen2.5-3B-Instruct-standard-K8-min-surr=bge": (
                "adaptive", "K8-min-positional-bge"
            ),
            (
                "adaptive-Qwen2.5-3B-Instruct-standard-K4-min-bag=min-"
                "surr=bge-aseg=semspan-nltk-max15-win5"
            ): ("adaptive", "K4-min-bag=min-bge"),
        }
        for leaf, (family, setting) in cases.items():
            with self.subTest(leaf=leaf):
                parsed = compile_results.parse_attack(leaf)
                self.assertEqual(parsed["attack_family"], family)
                self.assertEqual(parsed["attack_setting"], setting)

    def test_parses_controlled_probe_only_when_exactly_registered(self):
        parsed = compile_results.parse_attack("controlled_reorder-ratio=0.25")

        self.assertEqual(parsed["attack_family"], "other")
        self.assertEqual(parsed["attack_setting"], "controlled_reorder-ratio=0.25")

    def test_rejects_retired_and_unknown_attack_leaves(self):
        retired = (
            "openai",
            "openai-bigram",
            "openai-num_beams=10",
            "custom-Qwen2.5-3B-Instruct-standard",
            "baseqwen",
            "adpwb-K4",
            "adpbagwb-K4",
            "boundary_exchange-ratio=0.25",
            "controlled_reorder-ratio=0.33",
            "dipper-lex10-order10",
            "pegasus-bigram-threshold=0.10",
            "adaptive-Qwen2.5-3B-Instruct-standard-K4-min-surr=wbox",
        )
        for leaf in retired:
            with self.subTest(leaf=leaf):
                with self.assertRaisesRegex(
                    ValueError, "unsupported paper attack leaf"
                ):
                    compile_results.parse_attack(leaf)


class SegmentationParsingTests(unittest.TestCase):
    def test_parses_exact_paper_segmentations(self):
        sentence = compile_results.parse_segmentation("sentence-nltk")
        semspan = compile_results.parse_segmentation(
            "semspan-spacy-max15-win5"
        )

        self.assertEqual(sentence["segmentation_type"], "sentence")
        self.assertEqual(sentence["segmentation_backend"], "nltk")
        self.assertTrue(np.isnan(sentence["segmentation_max_words"]))
        self.assertTrue(np.isnan(sentence["segmentation_window"]))
        self.assertEqual(semspan, {
            "segmentation_type": "semspan",
            "segmentation_backend": "spacy",
            "segmentation_max_words": 15.0,
            "segmentation_window": 5.0,
        })

    def test_rejects_stale_or_unregistered_segmentations(self):
        for value in (
            "sentence-spacy",
            "semspan-nltk-max15-win5",
            "semspan-spacy-max12-win4",
            "semspan-spacy-max15-win4",
            "paragraph-custom",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "unsupported paper segmentation"
                ):
                    compile_results.parse_segmentation(value)


class JudgeRunExportTests(unittest.TestCase):
    def test_exports_three_runs_for_both_rubrics_and_reconstructs_headlines(self):
        quality_dims = {
            key: np.array([[0.2, 0.4, 0.6], [0.0, 0.0, 0.0]])
            for key in (
                "llm_quality_fluency_repeats",
                "llm_quality_coherence_repeats",
                "llm_quality_relevance_repeats",
                "llm_quality_informativeness_repeats",
            )
        }
        judge_dims = {
            "llm_judge_content_recall_repeats": np.array(
                [[1.0, 0.8, 0.6], [0.0, 0.0, 0.0]]
            ),
            "llm_judge_detail_precision_repeats": np.array(
                [[0.8, 0.6, 0.4], [0.0, 0.0, 0.0]]
            ),
            "llm_judge_information_injection_repeats": np.array(
                [[0.2, 0.4, 0.6], [0.0, 0.0, 0.0]]
            ),
            "llm_judge_contradiction_repeats": np.array(
                [[0.0, 0.2, 0.4], [0.0, 0.0, 0.0]]
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            np.savez(
                os.path.join(directory, "eval_quality_per_sample.npz"),
                llm_quality=np.array([0.4, 0.0]),
                llm_judge=np.array([0.85, 0.0]),
                **quality_dims,
                **judge_dims,
            )

            runs = compile_results.judge_run_arrays(directory, 3)

        self.assertEqual(set(runs), set(compile_results.JUDGE_RUN_KEYS))
        self.assertEqual(runs["llm_quality_fluency_run_2"].tolist()[:2], [0.4, 0.0])
        self.assertAlmostEqual(runs["llm_quality_run_1"][0], 0.2)
        self.assertAlmostEqual(runs["llm_quality_run_3"][0], 0.6)
        self.assertAlmostEqual(runs["llm_judge_run_1"][0], 0.9)
        self.assertAlmostEqual(runs["llm_judge_run_2"][0], 0.7)
        self.assertAlmostEqual(runs["llm_judge_run_3"][0], 0.5)
        self.assertEqual(runs["llm_judge_run_1"][1], 0.0)
        self.assertTrue(np.isnan(runs["llm_judge_run_1"][2]))

    def test_missing_quality_file_keeps_stable_nan_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = compile_results.judge_run_arrays(directory, 2)

        self.assertEqual(len(runs), 30)
        self.assertTrue(all(np.isnan(values).all() for values in runs.values()))

    def test_rebuilds_all_ten_aggregates_from_asymmetric_runs(self):
        runs = {}
        expected = {}
        for metrics, values in (
            (compile_results.LLM_QUALITY_METRICS, (0.2, 0.4, 0.6)),
            (compile_results.LLM_JUDGE_METRICS, (0.6, 0.7, 0.8)),
        ):
            for spec in metrics:
                for run, value in enumerate(values, 1):
                    runs[f"{spec.key}_run_{run}"] = np.array([value])
                expected[spec.key] = np.mean(values)

        rebuilt = compile_results.judge_mean_arrays(runs, 1)

        self.assertEqual(set(rebuilt), set(expected))
        for key, value in expected.items():
            self.assertAlmostEqual(rebuilt[key][0], value)


class ParquetOnlyResultTests(unittest.TestCase):
    def test_required_clean_cell_without_scores_fails_before_writing(self):
        discovered = [(
            "fixture", "lsh", "context", "rejection", "sentence-nltk",
            64.0, "", "", "watermark", "none", "/fixture/watermarked",
        )]
        with tempfile.TemporaryDirectory() as directory, patch.object(
            compile_results, "discover", return_value=discovered
        ), patch.object(
            compile_results, "load_z", return_value=(None, None, None, "direct")
        ):
            output = os.path.join(directory, "bundle")
            with self.assertRaisesRegex(RuntimeError, "no detector scores"):
                compile_results.compile_results(
                    ["--out", output], _datasets=["fixture"],
                    _require_complete=False, _skip_transfer=True,
                )
            self.assertFalse(os.path.exists(output))

    def test_required_clean_cell_without_quality_fails_before_writing(self):
        discovered = [(
            "fixture", "lsh", "context", "rejection", "sentence-nltk",
            64.0, "", "", "watermark", "none", "/fixture/watermarked",
        )]
        with tempfile.TemporaryDirectory() as directory, patch.object(
            compile_results, "discover", return_value=discovered
        ), patch.object(
            compile_results, "load_z",
            return_value=(np.array([1.0]), np.array([1.0]), None, "direct"),
        ), patch.object(
            compile_results, "per_sample_arrays", return_value={}
        ), patch.object(
            compile_results, "judge_run_arrays",
            return_value={
                key: np.array([np.nan]) for key in compile_results.JUDGE_RUN_KEYS
            },
        ):
            output = os.path.join(directory, "bundle")
            with self.assertRaisesRegex(RuntimeError, "lacks quality evidence"):
                compile_results.compile_results(
                    ["--out", output], _datasets=["fixture"],
                    _require_complete=False, _skip_transfer=True,
                )
            self.assertFalse(os.path.exists(output))

    def test_small_compile_writes_detailed_parquet_with_runs_and_mean_manifest(self):
        discovered = [(
            "fixture", "pmark", "online", "rejection", "sentence-nltk",
            64.0, "", "", "attack", "adp-K1", "/fixture/result",
        )]
        per_sample = {
            # Deliberately stale aggregates: compilation must replace them
            # before computing summaries and the quality gate.
            "llm_judge": np.array([0.95]),
            "llm_quality": np.array([0.95]),
        }
        run_values = {
            key: np.array([
                ((0.2, 0.4, 0.6) if key.startswith("llm_quality")
                 else (0.6, 0.7, 0.8))[run - 1]
            ])
            for run in range(1, 4)
            for key in compile_results.JUDGE_RUN_KEYS
            if key.endswith(f"_run_{run}")
        }

        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys, "argv", [
                "compile_results.py", "--out", directory,
            ]
        ), patch.object(
            compile_results, "discover", return_value=discovered
        ), patch.object(
            compile_results, "load_z",
            return_value=(np.array([0.0]), np.array([1.0]), None, "direct"),
        ), patch.object(
            compile_results, "per_sample_arrays", return_value=per_sample
        ), patch.object(
            compile_results, "judge_run_arrays", return_value=run_values
        ), patch.object(
            compile_results, "scalars", return_value={}
        ):
            compile_results.compile_results(
                _datasets=["fixture"], _require_complete=False,
                _skip_transfer=True,
            )

            parquet = os.path.join(directory, "per_sample.parquet")
            self.assertTrue(os.path.exists(parquet))
            self.assertFalse(os.path.exists(os.path.join(directory, "per_sample.csv")))
            compiled = pd.read_parquet(parquet)
            self.assertTrue(set(compile_results.JUDGE_RUN_KEYS) <= set(compiled.columns))
            for spec in compile_results.LLM_QUALITY_METRICS:
                self.assertAlmostEqual(compiled.loc[0, spec.key], 0.4)
            for spec in compile_results.LLM_JUDGE_METRICS:
                self.assertAlmostEqual(compiled.loc[0, spec.key], 0.7)
            self.assertEqual(compiled.loc[0, "quality_ok"], 0)
            self.assertEqual(compiled.loc[0, "success_at5"], 0)
            self.assertEqual(compiled.loc[0, "success_at1"], 0)
            with open(os.path.join(directory, "manifest.json")) as manifest_file:
                manifest = json.load(manifest_file)
            self.assertEqual(
                manifest["judge_runs"]["aggregation"],
                "mean across three judge runs",
            )
            self.assertNotIn("median", json.dumps(manifest["judge_runs"]).lower())

    def test_extractor_reads_only_parquet(self):
        frame = pd.DataFrame({
            "family": ["lsh"],
            "mask": ["context"],
            "sampling": ["rejection"],
            "segmentation": ["sentence-nltk"],
            "z": [1.25],
        })
        with tempfile.TemporaryDirectory() as directory:
            frame.to_parquet(os.path.join(directory, "per_sample.parquet"), index=False)
            rows = read_per_sample(Path(directory))

        self.assertEqual(rows[0]["z"], 1.25)

    def test_transfer_cache_reads_parquet_without_csv(self):
        frame = pd.DataFrame({"sentence_index": [0], "distance": [0.25]})
        with tempfile.TemporaryDirectory() as directory:
            frame.to_parquet(
                os.path.join(directory, "unit_encoder_transfer.parquet"), index=False,
            )
            manifest = {
                "source_fingerprint": encoder_transfer._source_fingerprint([]),
            }
            with open(
                os.path.join(directory, "unit_encoder_transfer.manifest.json"), "w"
            ) as manifest_file:
                json.dump(manifest, manifest_file)

            loaded, _ = encoder_transfer.build_transfer_table(
                directory, [], directory
            )

        self.assertFalse(os.path.exists(
            os.path.join(directory, "unit_encoder_transfer.csv")
        ))
        pd.testing.assert_frame_equal(loaded, frame)


class NoWatermarkCompileTests(unittest.TestCase):
    def test_discovers_canonical_none_baseline(self):
        with tempfile.TemporaryDirectory() as root, patch.object(
            compile_results, "ROOT", root
        ):
            directory = os.path.join(
                root, "data", "fixture", "none", "sentence-nltk"
            )
            os.makedirs(directory)
            open(os.path.join(directory, "dataset_info.json"), "w").close()

            discovered = list(compile_results.discover(["fixture"]))

        self.assertEqual(discovered, [(
            "fixture", "none", "none", "rejection", "sentence-nltk",
            1.0, "", "", "watermark", "none", directory,
        )])

    def test_load_z_preserves_quality_rows_without_detector_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            Dataset.from_dict({"text": ["one", "two"]}).save_to_disk(directory)
            para, wm, human, source = compile_results.load_z(
                directory, "none", "watermark"
            )

        self.assertEqual(len(para), 2)
        self.assertTrue(np.isnan(para).all())
        self.assertTrue(np.isnan(wm).all())
        self.assertIsNone(human)
        self.assertEqual(source, "unavailable")

    def test_compiles_none_quality_with_undefined_detection(self):
        discovered = [(
            "fixture", "none", "none", "rejection", "sentence-nltk",
            1.0, "", "", "watermark", "none", "/fixture/none",
        )]
        run_values = {
            key: (
                np.array([0.8, 0.9])
                if key.startswith("llm_quality")
                else np.full(2, np.nan)
            )
            for key in compile_results.JUDGE_RUN_KEYS
        }
        provider = {
            "pool": 1.0,
            "units_total": 2,
            "sentences_total": 2,
            "stats_path": "/fixture/none/generation_stats.json",
            "stats": {
                "n_units_tracked": 2,
                "mean_p_green_per_unit": 1.0,
                "std_p_green_per_unit": 0.0,
            },
            "prompt_evidence": {
                "source_dataset": "/fixture",
                "prompt_config": "/fixture/none/resolved_config.yaml",
                "prompt_rule": "fixture prompt",
                "prompts_verified": 2,
            },
        }

        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys, "argv", [
                "compile_results.py", "--out", directory,
            ]
        ), patch.object(
            compile_results, "discover", return_value=discovered
        ), patch.object(
            compile_results, "load_z",
            return_value=(np.full(2, np.nan), np.full(2, np.nan), None, "unavailable"),
        ), patch.object(
            compile_results, "per_sample_arrays", return_value={}
        ), patch.object(
            compile_results, "judge_run_arrays", return_value=run_values
        ), patch.object(
            compile_results, "scalars", return_value={}
        ), patch.object(
            compile_results, "load_provider_evidence", return_value=provider
        ):
            compile_results.compile_results(
                _datasets=["fixture"], _require_complete=False,
                _skip_transfer=True,
            )
            samples = pd.read_parquet(os.path.join(directory, "per_sample.parquet"))
            cells = pd.read_csv(os.path.join(directory, "cell_summary.csv"))
            datasets = pd.read_csv(os.path.join(directory, "dataset_summary.csv"))

        for frame in (samples, cells, datasets):
            self.assertEqual(frame.loc[0, "family"], "none")
            self.assertEqual(frame.loc[0, "scheme"], "No watermark")
            self.assertEqual(frame.loc[0, "stage"], "watermark")
        for frame in (cells, datasets):
            self.assertTrue(np.isnan(frame.loc[0, "TPR_at5"]))
        self.assertTrue(samples["detected_at5"].isna().all())
        self.assertAlmostEqual(cells.loc[0, "llm_quality"], 0.85)
        self.assertEqual(cells.loc[0, "provider_draws_per_sentence"], 1.0)


class ProviderDrawTests(unittest.TestCase):
    @staticmethod
    def _rejection_evidence(n, mean, std, units=10, sentences=12):
        return {
            "pool": 64.0,
            "units_total": units,
            "sentences_total": sentences,
            "stats": {
                "n_units_tracked": n,
                "mean_p_green_per_unit": mean,
                "std_p_green_per_unit": std,
                "mean_tries": 1.0 / mean,
                "tries_ci_lo_95": 1.0,
                "tries_ci_hi_95": 100.0,
            },
        }

    def test_rejection_pools_shard_moments_before_inversion(self):
        evidence = [
            self._rejection_evidence(10, 0.20, 0.10, units=10, sentences=11),
            self._rejection_evidence(30, 0.40, 0.20, units=30, sentences=35),
        ]

        result = compile_results.aggregate_provider_metrics(
            "lsh", "rejection", "sentence-nltk", evidence,
        )

        pooled_mean = (10 * 0.20 + 30 * 0.40) / 40
        pooled_ss = (
            9 * 0.10 ** 2 + 10 * (0.20 - pooled_mean) ** 2
            + 29 * 0.20 ** 2 + 30 * (0.40 - pooled_mean) ** 2
        )
        pooled_std = (pooled_ss / 39) ** 0.5
        half_width = compile_results.scipy_stats.t.ppf(0.975, 39) * pooled_std / 40 ** 0.5
        self.assertAlmostEqual(result["provider_draws_per_unit"], 1 / pooled_mean)
        self.assertAlmostEqual(
            result["provider_draws_per_unit_ci_lo_95"],
            1 / (pooled_mean + half_width),
        )
        self.assertAlmostEqual(
            result["provider_draws_per_unit_ci_hi_95"],
            1 / (pooled_mean - half_width),
        )
        self.assertEqual(result["provider_units_total"], 40)
        self.assertEqual(result["provider_sentences_total"], 46)
        self.assertEqual(result["provider_units_per_sentence"], 1.0)
        self.assertAlmostEqual(
            result["provider_draws_per_sentence"], 1 / pooled_mean,
        )

    def test_semspan_best_of_n_uses_ratio_of_pooled_totals(self):
        evidence = [
            {"pool": 64.0, "units_total": 100, "sentences_total": 40},
            {"pool": 64.0, "units_total": 20, "sentences_total": 20},
        ]

        result = compile_results.aggregate_provider_metrics(
            "kmeans", "best-of-n", "semspan-spacy-max15-win5", evidence,
        )

        self.assertEqual(result["provider_draws_per_unit"], 64.0)
        self.assertEqual(result["provider_units_total"], 120)
        self.assertEqual(result["provider_sentences_total"], 60)
        self.assertEqual(result["provider_units_per_sentence"], 2.0)
        self.assertEqual(result["provider_draws_per_sentence"], 128.0)
        self.assertEqual(result["provider_draws_per_sentence_ci_lo_95"], 128.0)
        self.assertEqual(result["provider_draws_per_sentence_ci_hi_95"], 128.0)

    def test_fixed_sentence_methods_report_exactly_64_draws(self):
        for family, sampling in (
            ("pmark", "rejection"),
            ("samark", "rejection"),
            ("lsh", "best-of-n"),
        ):
            with self.subTest(family=family, sampling=sampling):
                result = compile_results.aggregate_provider_metrics(
                    family, sampling, "sentence-nltk",
                    [{"pool": 64.0, "units_total": 30, "sentences_total": 34}],
                )
                self.assertEqual(result["provider_units_per_sentence"], 1.0)
                self.assertEqual(result["provider_draws_per_unit"], 64.0)
                self.assertEqual(result["provider_draws_per_sentence"], 64.0)
                self.assertEqual(result["provider_draws_per_sentence_ci_lo_95"], 64.0)
                self.assertEqual(result["provider_draws_per_sentence_ci_hi_95"], 64.0)

    def test_missing_rejection_field_names_source_file(self):
        stats = {
            "num_candidates": 64,
            "total_units": 3,
            "n_units_tracked": 3,
            "mean_p_green_per_unit": 0.25,
            "std_p_green_per_unit": 0.1,
            "mean_tries": 4.0,
            "tries_ci_lo_95": 2.0,
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(
            compile_results, "count_clean_output_sentences",
            return_value=(4, {}),
        ):
            with open(os.path.join(directory, "generation_stats.json"), "w") as output:
                json.dump(stats, output)
            with self.assertRaisesRegex(
                compile_results.ProviderEvidenceError,
                "generation_stats.json.*tries_ci_hi_95",
            ):
                compile_results.load_provider_evidence(
                    directory, "lsh", "rejection", "sentence-nltk", 64.0,
                    "/fixture/source",
                )

    def test_semstamp_sentence_count_strips_configured_exact_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            source_dir = os.path.join(directory, "source")
            marked_dir = os.path.join(directory, "marked")
            source_text = "A sufficiently long prompt sentence. Original remainder."
            prompt = compile_results.extract_prompt_from_text(source_text, 32)
            Dataset.from_dict({"text": [source_text]}).save_to_disk(source_dir)
            Dataset.from_dict({
                "text": [prompt + " Generated first sentence. Generated second sentence?"]
            }).save_to_disk(marked_dir)
            with open(os.path.join(marked_dir, "resolved_config.yaml"), "w") as output:
                output.write(
                    f"io:\n  data_path: {source_dir}\n"
                    "generation:\n  len_prompt: 32\n"
                )

            count, audit = compile_results.count_clean_output_sentences(
                marked_dir, "lsh", source_dir,
            )

        self.assertEqual(count, 2)
        self.assertEqual(audit["prompts_verified"], 1)
        self.assertIn("len_prompt=32", audit["prompt_rule"])

    def test_prompt_prefix_mismatch_names_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            source_dir = os.path.join(directory, "source")
            marked_dir = os.path.join(directory, "marked")
            source_text = "A sufficiently long prompt sentence. Original remainder."
            Dataset.from_dict({"text": [source_text]}).save_to_disk(source_dir)
            Dataset.from_dict({"text": ["Wrong prompt. Generated sentence."]}).save_to_disk(
                marked_dir
            )
            with open(os.path.join(marked_dir, "resolved_config.yaml"), "w") as output:
                output.write(
                    f"io:\n  data_path: {source_dir}\n"
                    "generation:\n  len_prompt: 32\n"
                )

            with self.assertRaisesRegex(
                compile_results.ProviderEvidenceError,
                "sample 0 stored text does not begin with the exact recovered prompt",
            ):
                compile_results.count_clean_output_sentences(
                    marked_dir, "lsh", source_dir,
                )


if __name__ == "__main__":
    unittest.main()
