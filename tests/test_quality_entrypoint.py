import dataclasses
import unittest
from unittest.mock import patch

from config.schema import AppConfig
from quality import __main__ as quality_main


class QualityEntrypointTests(unittest.TestCase):
    def test_single_directory_cli_delegates_to_phased_engine(self):
        cfg = dataclasses.replace(
            AppConfig(),
            io=dataclasses.replace(AppConfig().io, data_path="base"),
        )
        derived = ("target", "para_text", "reference", False)

        with patch.object(quality_main, "parse_args", return_value=cfg), \
                patch.object(quality_main, "derive_io", return_value=derived), \
                patch.object(quality_main, "run") as run:
            quality_main.main()

        args, kwargs = run.call_args
        self.assertEqual(args[0], ["target"])
        delegated_cfg = args[1]
        self.assertEqual(delegated_cfg.quality.column, "para_text")
        self.assertEqual(delegated_cfg.quality.reference, "reference")
        self.assertFalse(delegated_cfg.quality.skip_per_pair)
        self.assertEqual(kwargs, {"force": True})


if __name__ == "__main__":
    unittest.main()
