import tempfile
import unittest
from pathlib import Path
from unittest import mock

from config.paper import (
    DATASETS,
    LADDERS,
    ORACLE_KS,
    PMARK,
    PROVIDER_CANDIDATES,
    SAMARK,
)
from visualization.compile_results import (
    compile_results,
    require_complete_generation,
    required_generation_cells,
)
from visualization.__main__ import parser as visualization_parser
from visualization.extractors import EXTRACTORS
from visualization.extractors.common import BARS, FRONTIER_SCHEMES, write_table
from visualization.renderers.paper import (
    _direction_cue,
    render_all,
    render_teaser,
    render_transfer,
    render_whitebox,
)


class PaperVisualizationSpecTests(unittest.TestCase):
    def test_compile_cli_has_no_partial_or_legacy_surface(self):
        root = visualization_parser()
        subparsers = next(
            action for action in root._actions
            if action.__class__.__name__ == "_SubParsersAction"
        )
        self.assertEqual(
            set(subparsers.choices),
            {"compile", "extract", "render", "all"},
        )
        options = {
            option
            for action in subparsers.choices["compile"]._actions
            for option in action.option_strings
        }
        self.assertNotIn("--output", options)
        for retired in ("--datasets", "--families", "--num-candidates", "--oracle-ks"):
            self.assertNotIn(retired, options)

    def test_visualization_uses_the_canonical_paper_registry(self):
        self.assertEqual(PROVIDER_CANDIDATES, 64)
        self.assertEqual(
            DATASETS,
            ("c4-val-def-256", "c4-val-def-256b", "c4-val-def-512"),
        )
        self.assertEqual(ORACLE_KS, (4, 8, 16, 32, 64))
        self.assertEqual((PMARK.msig, SAMARK.msig), (4, 2))
        self.assertEqual((PMARK.num_candidates, SAMARK.num_candidates), (64, 64))
        self.assertEqual(len(EXTRACTORS), 10)
        self.assertNotIn("example_passages", EXTRACTORS)
        self.assertEqual(
            [rung.key for rung in LADDERS["lsh"]],
            ["base", "bestn", "fixed", "diverse", "span"],
        )
        self.assertEqual(
            [rung.key for rung in LADDERS["kmeans"]],
            ["kbase", "kbestn", "kfixed", "kdiverse", "kspan"],
        )
        self.assertEqual(BARS, ["q65", "q70", "q75", "q80", "q85", "q90", "q95"])
        self.assertEqual(
            [scheme.key for scheme in FRONTIER_SCHEMES],
            ["semstamp", "ksemstamp", "pmark-online", "samark",
             "swordstamp", "kswordstamp"],
        )
        self.assertTrue(callable(compile_results))

    def test_compiler_rejects_an_incomplete_generation_matrix(self):
        self.assertEqual(len(required_generation_cells()), 13)
        cell = next(iter(required_generation_cells()))
        row = (
            "fixture", *cell,
            64.0 if cell[0] in {"lsh", "kmeans", "pmark", "samark"} else
            1.0 if cell[0] == "none" else float("nan"),
            "", "", "watermark", "none", "/fixture",
        )
        with self.assertRaisesRegex(RuntimeError, "expected 13 clean cells"):
            require_complete_generation([row], ["fixture"])

    def test_missing_candidate_budget_is_accepted_only_for_human_null(self):
        rung = FRONTIER_SCHEMES[0]
        identity = {
            "scheme": rung.scheme,
            "mask": rung.mask,
            "sampling": rung.sampling,
            "segmentation": rung.segmentation,
            "generation_num_candidates": float("nan"),
        }
        self.assertTrue(rung.matches({**identity, "stage": "human"}))
        self.assertFalse(rung.matches({**identity, "stage": "attack"}))

        for comparison in FRONTIER_SCHEMES[2:4]:
            comparison_identity = {
                "scheme": comparison.scheme,
                "mask": comparison.mask,
                "sampling": comparison.sampling,
                "segmentation": comparison.segmentation,
                "generation_num_candidates": float("nan"),
            }
            self.assertTrue(comparison.matches({**comparison_identity, "stage": "human"}))
            self.assertFalse(comparison.matches({**comparison_identity, "stage": "attack"}))

class MatplotlibRendererTests(unittest.TestCase):
    @staticmethod
    def _curve(path: Path) -> None:
        write_table(
            path, ["K", "y", "em", "ep", "n"],
            [[4.0, .2, .01, .02, 8.0], [8.0, .35, .02, .02, 8.0]],
        )

    def test_whitebox_renderer_writes_png_and_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "tables" / "pgfplots"
            output = root / "figures"
            for scheme in ("ksemstamp", "kswordstamp"):
                for metric in ("evasion", "asr", "quality-pass-given-evasion"):
                    self._curve(data / "whitebox" / f"{metric}-{scheme}.dat")

            paths = render_whitebox(data, output)

            self.assertEqual({path.suffix for path in paths}, {".png", ".pdf"})
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))

    def test_transfer_renderer_reads_extracted_bands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "tables" / "pgfplots"
            output = root / "figures"
            for scheme in ("semstamp", "pmark"):
                write_table(
                    data / "transfer" / f"{scheme}-band.dat",
                    ["x", "y", "lo", "hi", "n"],
                    [[.1, .12, .08, .16, 40.0], [.2, .24, .18, .3, 45.0]],
                )

            paths = render_transfer(data, output)

            self.assertTrue(all(path.is_file() for path in paths))

    def test_teaser_renderer_keeps_the_paper_direction_cues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "tables" / "pgfplots"
            output = root / "figures"
            curve = [[.65, .2, .01, .02, 8.0], [.90, .1, .01, .01, 8.0]]
            for scheme in (
                "semstamp", "ksemstamp", "pmark-online", "samark",
                "swordstamp", "kswordstamp",
            ):
                write_table(
                    data / "teaser" / f"{scheme}.dat",
                    ["bar", "y", "em", "ep", "n"],
                    curve,
                )
            write_table(
                data / "fidelity_robustness" / "points.dat",
                ["id", "fidelity", "fem", "fep", "asr", "aem", "aep", "n"],
                [[float(index), 80 + index, 1, 1, 30 - index, 2, 2, 8]
                 for index in range(6)],
            )

            with mock.patch(
                "visualization.renderers.paper._direction_cue",
                wraps=_direction_cue,
            ) as cue:
                paths = render_teaser(data, output)

            calls = {call.args[1]: call.kwargs for call in cue.call_args_list}
            self.assertEqual(set(calls), {"more robust", "better"})
            self.assertEqual(calls["more robust"]["arrow_end"], (.88, .70))
            self.assertEqual(calls["better"]["arrow_end"], (.22, .72))
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))

    def test_render_all_covers_every_authoritative_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "tables" / "pgfplots"
            curve = [[.65, .2, .01, .02, 8.0], [.90, .1, .01, .01, 8.0]]
            schemes = ("semstamp", "ksemstamp", "pmark-online", "samark",
                       "swordstamp", "kswordstamp")
            for scheme in schemes:
                write_table(data / "teaser" / f"{scheme}.dat",
                            ["bar", "y", "em", "ep", "n"], curve)
                for series in ("sentence", "dipper", "adaptive"):
                    write_table(data / "cross_scheme" / f"{scheme}-{series}.dat",
                                ["bar", "y", "em", "ep", "n"], curve)
            write_table(
                data / "fidelity_robustness" / "points.dat",
                ["id", "fidelity", "fem", "fep", "asr", "aem", "aep", "n"],
                [[float(index), 80 + index, 1, 1, 30 - index, 2, 2, 8]
                 for index in range(6)],
            )
            for panel in ("reword", "reorder", "reseg"):
                for rung in ("base", "bestn", "fixed", "diverse", "span",
                             "kbase", "kbestn", "kfixed", "kdiverse", "kspan"):
                    write_table(
                        data / "channel_response" / f"{panel}-{rung}.dat",
                        ["x", "y", "em", "ep", "ratio", "n"],
                        [[0, 100, 0, 0, 0, 8], [.2, 75, 2, 3, .5, 8]],
                    )
            for scheme in ("ksemstamp", "kswordstamp"):
                for metric in ("evasion", "asr", "quality-pass-given-evasion"):
                    self._curve(data / "whitebox" / f"{metric}-{scheme}.dat")
            for scheme in ("semstamp", "pmark"):
                write_table(
                    data / "transfer" / f"{scheme}-band.dat",
                    ["x", "y", "lo", "hi", "n"],
                    [[.1, .12, .08, .16, 40], [.2, .24, .18, .3, 45]],
                )
            write_table(
                data / "attack_channels" / "ksemstamp.dat",
                ["series", "setting", "endpoint", "reword", "reorder", "reseg",
                 "zret", "evading_reword", "evading_reorder", "evading_reseg",
                 "evading_quality", "evading_n", "evading_quality_pass",
                 "evading_quality_pass_numer", "evading_quality_pass_denom"],
                [["pegasus", "Peg", "point", .2, .1, .1, 70, .25, .15, .1,
                  .8, 8, .75, 6, 8]],
            )

            paths = render_all(root, root)

            self.assertEqual(len(paths), 14)
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))


if __name__ == "__main__":
    unittest.main()
