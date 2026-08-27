import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "scripts/experiments/_gpu_policy.sh"


class WholeGpuPolicyTests(unittest.TestCase):
    def needs_whole_gpu(self, *args):
        command = [
            "bash",
            "-c",
            'source "$1"; shift; needs_whole_gpu "$@"',
            "bash",
            str(POLICY),
            *args,
        ]
        return subprocess.run(command, check=False).returncode == 0

    def test_vllm_attacks_are_exclusive(self):
        for paraphraser in ("adaptive", "custom", "oracle"):
            with self.subTest(paraphraser=paraphraser):
                self.assertTrue(self.needs_whole_gpu(
                    "scripts/experiments/_attack.sh",
                    "--set",
                    f"attack.paraphraser={paraphraser}",
                ))

    def test_non_vllm_attack_is_shareable(self):
        self.assertFalse(self.needs_whole_gpu(
            "scripts/experiments/_attack.sh",
            "--set",
            "attack.paraphraser=pegasus",
        ))


if __name__ == "__main__":
    unittest.main()
