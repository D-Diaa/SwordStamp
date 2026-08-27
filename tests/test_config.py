"""Unit tests for the config loader/schema (no GPU, no model weights)."""

import os
from pathlib import Path
import tempfile
import unittest

from config.loader import (
    deep_merge,
    dump_config,
    env_overlay,
    from_dict,
    load_config,
    parse_overrides,
    set_dotted,
)
from config.runtime import DEFAULT_VLLM_UTILIZATION, vllm_gpu_memory_utilization
from config.schema import AppConfig, GenConfig, RuntimeConfig, SegmentationConfig


class DeepMergeTest(unittest.TestCase):
    def test_nested_override_wins_and_siblings_preserved(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        deep_merge(base, {"a": {"y": 20}, "c": 4})
        self.assertEqual(base, {"a": {"x": 1, "y": 20}, "b": 3, "c": 4})


class SetDottedTest(unittest.TestCase):
    def test_creates_nested_path(self):
        data = {}
        set_dotted(data, "generation.model", "foo")
        self.assertEqual(data, {"generation": {"model": "foo"}})

    def test_rejects_empty_key(self):
        with self.assertRaises(ValueError):
            set_dotted({}, "", 1)


class ParseOverridesTest(unittest.TestCase):
    def test_scalar_coercion_via_yaml(self):
        data = parse_overrides(
            ["generation.delta=0.1", "generation.max_new_tokens=64", "quality.skip_per_pair=true"]
        )
        self.assertEqual(data["generation"]["delta"], 0.1)
        self.assertIsInstance(data["generation"]["delta"], float)
        self.assertEqual(data["generation"]["max_new_tokens"], 64)
        self.assertIsInstance(data["generation"]["max_new_tokens"], int)
        self.assertIs(data["quality"]["skip_per_pair"], True)

    def test_requires_equals(self):
        with self.assertRaises(ValueError):
            parse_overrides(["generation.delta"])


class FromDictTest(unittest.TestCase):
    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError):
            from_dict(GenConfig, {"nonexistent": 1})

    def test_bad_enum_raises(self):
        with self.assertRaises(ValueError):
            from_dict(GenConfig, {"backend": "bogus"})

    def test_partial_dict_keeps_defaults(self):
        cfg = from_dict(GenConfig, {"model": "x"})
        self.assertEqual(cfg.model, "x")
        self.assertEqual(cfg.max_new_tokens, GenConfig().max_new_tokens)

    def test_string_int_is_coerced(self):
        cfg = from_dict(GenConfig, {"max_new_tokens": "64"})
        self.assertEqual(cfg.max_new_tokens, 64)


class LoadConfigTest(unittest.TestCase):
    def test_defaults_roundtrip(self):
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "resolved.yaml")
            dump_config(cfg, path)
            resolved = Path(path).read_text()
            self.assertIn("semcut_max_words: 15", resolved)
            self.assertIn("semcut_window: 5", resolved)
            self.assertIn("semcut_batch_size: 512", resolved)
            reloaded = load_config([path])
        self.assertEqual(cfg, reloaded)
        self.assertEqual(cfg, AppConfig())

    def test_precedence_defaults_yaml_env_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.yaml")
            with open(path, "w") as fh:
                fh.write("generation:\n  model: from_yaml\n  max_new_tokens: 100\n")

            # yaml only
            self.assertEqual(load_config([path]).generation.model, "from_yaml")

            # env overlays yaml
            os.environ["SEMSTAMP__GENERATION__MODEL"] = "from_env"
            try:
                self.assertEqual(load_config([path]).generation.model, "from_env")
                # --set wins over both env and yaml
                cfg = load_config([path], ["generation.model=from_set"])
                self.assertEqual(cfg.generation.model, "from_set")
            finally:
                del os.environ["SEMSTAMP__GENERATION__MODEL"]

            # yaml value survives where not overridden
            self.assertEqual(load_config([path]).generation.max_new_tokens, 100)

    def test_semcut_defaults_and_precedence(self):
        defaults = AppConfig()
        self.assertEqual(
            (
                defaults.segmentation.semcut_max_words,
                defaults.segmentation.semcut_window,
            ),
            (15, 5),
        )
        self.assertEqual(defaults.runtime.semcut_batch_size, 512)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "semcut.yaml")
            with open(path, "w") as fh:
                fh.write(
                    "segmentation:\n"
                    "  semcut_max_words: 12\n"
                    "  semcut_window: 2\n"
                )
            os.environ["SEMSTAMP__SEGMENTATION__SEMCUT_MAX_WORDS"] = "14"
            os.environ["SEMSTAMP__SEGMENTATION__SEMCUT_WINDOW"] = "4"
            try:
                cfg = load_config(
                    [path], ["segmentation.semcut_max_words=16"],
                )
            finally:
                del os.environ["SEMSTAMP__SEGMENTATION__SEMCUT_MAX_WORDS"]
                del os.environ["SEMSTAMP__SEGMENTATION__SEMCUT_WINDOW"]

        self.assertEqual(cfg.segmentation.semcut_max_words, 16)
        self.assertEqual(cfg.segmentation.semcut_window, 4)

    def test_invalid_semcut_policy_is_rejected(self):
        invalid = (
            {"semcut_max_words": 0},
            {"semcut_window": 0},
            {"semcut_max_words": 9, "semcut_window": 5},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                SegmentationConfig(**values)

    def test_invalid_semcut_batch_size_is_rejected(self):
        with self.assertRaises(ValueError):
            RuntimeConfig(semcut_batch_size=0)

    def test_env_overlay_parsing(self):
        os.environ["SEMSTAMP__WATERMARK__SP_DIM"] = "16"
        try:
            overlay = env_overlay()
            self.assertEqual(overlay, {"watermark": {"sp_dim": 16}})
        finally:
            del os.environ["SEMSTAMP__WATERMARK__SP_DIM"]

    def test_top_level_unknown_section_raises(self):
        with self.assertRaises(ValueError):
            load_config(overrides=["bogus_section.key=1"])

    def test_invalid_segmentation_type_is_rejected(self):
        with self.assertRaises(ValueError):
            load_config(overrides=["segmentation.type=paragraph"])

    def test_adaptive_uses_one_surrogate_model(self):
        cfg = load_config(overrides=["attack.surrogate_model=example/encoder"])
        self.assertEqual(cfg.attack.surrogate_model, "example/encoder")

    def test_unknown_nested_field_is_rejected(self):
        with self.assertRaises(ValueError):
            load_config(overrides=["runtime.nonexistent=true"])

    def test_vllm_default_explicit_value_and_config_env(self):
        self.assertEqual(AppConfig().runtime.vllm_utilization, DEFAULT_VLLM_UTILIZATION)
        self.assertEqual(vllm_gpu_memory_utilization(0.61), 0.61)
        os.environ["SEMSTAMP__RUNTIME__VLLM_UTILIZATION"] = "0.73"
        try:
            cfg = load_config()
        finally:
            del os.environ["SEMSTAMP__RUNTIME__VLLM_UTILIZATION"]
        self.assertEqual(cfg.runtime.vllm_utilization, 0.73)

    def test_every_preset_loads(self):
        preset_dir = Path(__file__).resolve().parents[1] / "config" / "presets"
        presets = sorted(preset_dir.glob("*.yaml"))
        self.assertTrue(presets)
        for preset in presets:
            with self.subTest(preset=preset.name):
                load_config([str(preset)])


if __name__ == "__main__":
    unittest.main()


class FixedDiverseModeTest(unittest.TestCase):
    def test_watermark_mode_validates(self):
        from config.schema import WatermarkConfig
        for mode in ("lsh_fixed_diverse", "kmeans_fixed_diverse"):
            self.assertEqual(WatermarkConfig(mode=mode).mode, mode)
