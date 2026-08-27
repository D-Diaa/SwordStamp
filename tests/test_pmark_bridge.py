import importlib.util
import os
from pathlib import Path
import subprocess
import unittest

import numpy as np


_ROOT = os.path.dirname(os.path.dirname(__file__))
_CALIBRATION_PATH = os.path.join(_ROOT, "comparisons", "pmark", "calibration.py")
_SPEC = importlib.util.spec_from_file_location("pmark_calibration", _CALIBRATION_PATH)
pmark_calibration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pmark_calibration)


class PMarkBridgeTests(unittest.TestCase):
    def test_empirical_cutoff_respects_strict_tail_fpr(self):
        null = np.arange(1024, dtype=float)
        threshold = pmark_calibration.calibrated_threshold(null, 0.01)
        self.assertLessEqual(float(np.mean(null > threshold)), 0.01)

    def test_empirical_auroc_uses_tie_half_credit(self):
        self.assertEqual(pmark_calibration.empirical_auroc([1.0], [0.0, 1.0]), 0.75)

    def test_pmark_worker_uses_only_the_online_paper_method(self):
        with open(os.path.join(_ROOT, "scripts", "experiments", "_pmark_gen.sh")) as f:
            worker = f.read()
        self.assertIn('--num_samples "${N:-64}"', worker)
        self.assertNotIn("--start", worker)
        self.assertNotIn("--end", worker)
        self.assertIn("--median_method hd", worker)
        self.assertNotIn("--parallel", worker)
        self.assertNotIn("MEDIAN", worker)

    def test_sampler_and_detector_settings_are_wired(self):
        with open(os.path.join(_ROOT, "comparisons", "pmark", "utils", "detect.py")) as f:
            detector = f.read()
        with open(os.path.join(_ROOT, "scripts", "experiments", "pmark.sh")) as f:
            scheduler = f.read()
        self.assertIn("mode=online", scheduler)
        self.assertIn("pmark/online/rejection/sentence-nltk", scheduler)
        self.assertNotIn("MEDIANS", scheduler)
        self.assertNotIn("--median_method prior", scheduler)
        self.assertNotIn("pmark/prior/", scheduler)
        self.assertIn("K=150", detector)
        self.assertIn("TEMPERATURE=0.9", scheduler)
        self.assertIn("TOP_P=0.9", scheduler)
        self.assertIn("REPETITION_PENALTY=1.05", scheduler)
        self.assertIn("MAX_NEW_SENTENCES=12", scheduler)

    def test_samark_runner_uses_only_the_flags_run_paper_method(self):
        with open(os.path.join(_ROOT, "scripts", "experiments", "samark.sh")) as f:
            scheduler = f.read()
        self.assertIn("FLAG_SCOPE=run", scheduler)
        self.assertIn("samark/flags-$FLAG_SCOPE", scheduler)
        self.assertNotIn("FLAG_SCOPES", scheduler)

    def test_hf_exports_preserve_early_stopped_pmark_and_samark_rows(self):
        with open(os.path.join(_ROOT, "comparisons", "pmark", "pmark.py")) as f:
            pmark = f.read()
        self.assertNotIn("Skipping incomplete sample", pmark)
        self.assertNotIn('len(log.get("log", [])) != args.max_new_sentences', pmark)

        for name in ("samark_gen.py", "samark_gen_unwatermarked.py"):
            with self.subTest(generator=name):
                with open(os.path.join(_ROOT, "comparisons", "samark", name)) as f:
                    samark = f.read()
                export = samark[samark.index("def _export_hf_dataset"):]
                self.assertNotIn("min_new_sentences", export)
                self.assertNotIn("Skipping incomplete", export)

    def test_samark_uses_fixed_attack_policy_while_pmark_stays_context(self):
        with open(os.path.join(_ROOT, "scripts", "experiments", "samark.sh")) as f:
            samark = f.read()
        with open(os.path.join(_ROOT, "scripts", "experiments", "pmark.sh")) as f:
            pmark = f.read()
        self.assertIn("attacks_for_cell samark sentence fixed", samark)
        self.assertIn("attacks_for_cell pmark sentence context", pmark)

    def test_shared_scheduler_emits_only_the_paper_attack_cells(self):
        script = os.path.join(_ROOT, "scripts", "experiments", "attacks.sh")

        for name in ("attacks.sh", "swordstamp.sh", "_attack.sh", "pmark.sh", "samark.sh"):
            with self.subTest(script=name):
                text = Path(os.path.join(_ROOT, "scripts", "experiments", name)).read_text()
                for retired in (
                    "baseqwen", "openai", "adpwb", "adpbagwb",
                    "boundary_exchange",
                ):
                    self.assertNotIn(retired, text.lower())

        def scheduled(family, segmentation, mask, oracle=""):
            command = (
                'ORACLE_KS="$1"; source "$2"; '
                'attacks_for_cell "$3" "$4" "$5"'
            )
            result = subprocess.run(
                [
                    "bash", "-c", command, "paper-attacks", oracle, script,
                    family, segmentation, mask,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.splitlines()

        oracle = "4 8 16 32 64"
        cases = (
            (("lsh", "sentence", "context", ""), 32),
            (("kmeans", "sentence", "context", oracle), 37),
            (("kmeans", "semspan", "fixed_diverse", oracle), 40),
            (("pmark", "sentence", "context", ""), 16),
            (("samark", "sentence", "fixed", ""), 19),
        )
        for arguments, expected_count in cases:
            with self.subTest(cell=arguments[:3]):
                entries = scheduled(*arguments)
                self.assertEqual(len(entries), expected_count)
                serialized = "\n".join(entries).lower()
                for retired in (
                    "baseqwen", "openai", "adpwb", "adpbagwb",
                    "boundary_exchange",
                ):
                    self.assertNotIn(retired, serialized)

        self.assertFalse(any(
            entry.startswith("probe-")
            for entry in scheduled("pmark", "sentence", "context")
        ))
        self.assertFalse(any(
            entry.startswith("probe-")
            for entry in scheduled("samark", "sentence", "fixed")
        ))

    def test_experiment_runners_only_schedule_phased_quality_worker(self):
        for name in ("swordstamp.sh", "pmark.sh", "samark.sh"):
            with self.subTest(runner=name):
                with open(os.path.join(_ROOT, "scripts", "experiments", name)) as f:
                    runner = f.read()
                self.assertIn('_quality_batch.sh', runner)
                self.assertNotIn('_quality.sh', runner)

    def test_probe_quality_is_cpu_only_in_every_runner(self):
        with open(os.path.join(_ROOT, "scripts", "experiments", "_quality_batch.sh")) as f:
            worker = f.read()
        self.assertIn('${CPU_ONLY:+--cpu-only}', worker)
        for name in ("swordstamp.sh", "pmark.sh", "samark.sh"):
            with self.subTest(runner=name):
                with open(os.path.join(_ROOT, "scripts", "experiments", name)) as f:
                    runner = f.read()
                self.assertIn('enqueue_cpu "$J/_quality_batch.sh"', runner)
                self.assertIn('--env "CPU_ONLY=1"', runner)

    def test_default_probe_battery_matches_the_16_paper_conditions(self):
        script = os.path.join(_ROOT, "scripts", "experiments", "attacks.sh")
        result = subprocess.run(
            ["bash", "-c", f'source "{script}"; printf "%s\\n" "${{PROBE_ATTACKS[@]}}"'],
            check=True,
            capture_output=True,
            text=True,
        )
        entries = result.stdout.splitlines()
        self.assertEqual(len(entries), 16)
        counts = {
            stem: sum(entry.startswith(f"probe-{stem}-") for entry in entries)
            for stem in ("reorder", "split", "merge", "syn")
        }
        self.assertEqual(
            counts,
            {"reorder": 5, "split": 3, "merge": 3, "syn": 5},
        )
        self.assertFalse(any("boundary_exchange" in entry for entry in entries))


if __name__ == "__main__":
    unittest.main()
