"""Safety and CPU workflow contracts for the bundled scheduler installer."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shlex
import signal
import stat
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install_bundled_scheduler.sh"
VENDOR = ROOT / "tools" / "gpu_scheduler"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
COMMANDS = ("gpu-scheduler", "gpu-enqueue", "gpu-delete", "gpu-status")
LIBRARIES = ("common.py", "dispatcher.py", "warmup.py")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BundledSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="swordstamp-scheduler-")
        self.home = Path(self.temporary.name)
        self.env = os.environ.copy()
        self.env.update({
            "HOME": str(self.home),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        self.assertTrue(VENV_PYTHON.is_file(), "run scripts/setup.sh first")

    def tearDown(self):
        self.temporary.cleanup()

    def _installer(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(INSTALLER), *arguments],
            cwd=ROOT,
            env=self.env,
            stdin=subprocess.DEVNULL,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )

    def _cli(self, name: str, *arguments: str, timeout: int = 20):
        env = self.env.copy()
        env["PATH"] = f"{self.home / '.local' / 'bin'}:{env['PATH']}"
        return subprocess.run(
            [str(self.home / ".local" / "bin" / name), *arguments],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _install(self) -> None:
        result = self._installer("--yes")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("without starting a dispatcher", result.stdout)

    def _payload_paths(self) -> list[Path]:
        scheduler_home = self.home / ".gpu-scheduler"
        paths = [scheduler_home / "lib" / name for name in LIBRARIES]
        paths.extend(scheduler_home / "bin" / name for name in COMMANDS)
        paths.extend(self.home / ".local" / "bin" / name for name in COMMANDS)
        return paths

    def test_plan_and_noninteractive_refusal_do_not_write(self):
        plan = self._installer("--print-plan")
        self.assertEqual(plan.returncode, 0, plan.stderr)
        self.assertIn("Current state: fresh", plan.stdout)
        self.assertIn("performs no network access", plan.stdout)
        self.assertFalse((self.home / ".gpu-scheduler").exists())
        self.assertFalse((self.home / ".local").exists())

        refused = self._installer()
        self.assertEqual(refused.returncode, 2)
        self.assertIn("without --yes", refused.stderr)
        self.assertFalse((self.home / ".gpu-scheduler").exists())
        self.assertFalse((self.home / ".local").exists())

    def test_install_uses_frozen_python_and_all_clis_import(self):
        self._install()
        expected_python = str(VENV_PYTHON)
        runtime_bin = self.home / ".gpu-scheduler" / "bin"
        for name in COMMANDS:
            with self.subTest(command=name):
                launcher = self.home / ".local" / "bin" / name
                self.assertTrue(launcher.stat().st_mode & stat.S_IXUSR)
                tokens = shlex.split(launcher.read_text(encoding="utf-8").splitlines()[1])
                self.assertEqual(
                    tokens,
                    ["exec", expected_python, str(runtime_bin / name), "$@"],
                )
                result = self._cli(name, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)

        for name in LIBRARIES:
            self.assertEqual(
                _sha256(self.home / ".gpu-scheduler" / "lib" / name),
                _sha256(VENDOR / "src" / name),
            )
        for name in COMMANDS:
            self.assertEqual(
                _sha256(runtime_bin / name),
                _sha256(VENDOR / "bin" / name),
            )

    def test_exact_install_is_noop_even_if_dispatcher_pid_is_live(self):
        self._install()
        before = {
            path: (_sha256(path), path.stat().st_mtime_ns)
            for path in self._payload_paths()
        }
        sleeper = subprocess.Popen(["sleep", "30"])
        try:
            (self.home / ".gpu-scheduler" / "dispatcher.pid").write_text(
                f"{sleeper.pid}\n", encoding="utf-8"
            )
            result = self._installer()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("already installed exactly", result.stdout)
            after = {
                path: (_sha256(path), path.stat().st_mtime_ns)
                for path in self._payload_paths()
            }
            self.assertEqual(after, before)
        finally:
            sleeper.send_signal(signal.SIGTERM)
            sleeper.wait(timeout=5)

    def test_conflicting_install_is_not_overwritten(self):
        target = self.home / ".local" / "bin" / "gpu-enqueue"
        target.parent.mkdir(parents=True)
        sentinel = b"reviewer-owned command\n"
        target.write_bytes(sentinel)

        result = self._installer("--yes")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to overwrite", result.stderr)
        self.assertEqual(target.read_bytes(), sentinel)
        self.assertFalse((self.home / ".gpu-scheduler").exists())
        self.assertEqual(list(target.parent.iterdir()), [target])

    def test_nonexact_running_scheduler_is_refused(self):
        scheduler_home = self.home / ".gpu-scheduler"
        scheduler_home.mkdir()
        sleeper = subprocess.Popen(["sleep", "30"])
        try:
            (scheduler_home / "dispatcher.pid").write_text(
                f"{sleeper.pid}\n", encoding="utf-8"
            )
            result = self._installer("--yes")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("running scheduler", result.stderr)
            self.assertFalse((scheduler_home / "lib").exists())
            self.assertFalse((self.home / ".local").exists())
        finally:
            sleeper.send_signal(signal.SIGTERM)
            sleeper.wait(timeout=5)

    def test_installer_has_no_network_or_upstream_install_commands(self):
        source = INSTALLER.read_text(encoding="utf-8")
        for command in ("git", "curl", "wget", "ssh", "scp"):
            with self.subTest(command=command):
                self.assertNotRegex(source, rf"(?m)^\s*{command}\b")
        self.assertNotIn("tools/gpu_scheduler/install.sh", source)
        self.assertNotIn("http://", source)
        self.assertNotIn("https://", source)

    def test_cpu_dependency_dag_smoke(self):
        self._install()
        initialized = self._cli(
            "gpu-scheduler", "init", "--gpus", "0", "--no-warmup", "--poll", "0.05"
        )
        self.assertEqual(initialized.returncode, 0, initialized.stderr)
        started = self._cli("gpu-scheduler", "start")
        self.assertEqual(started.returncode, 0, started.stderr)

        work = self.home / "dag"
        work.mkdir()
        trace = work / "trace.txt"
        first = work / "first.sh"
        second = work / "second.sh"
        first.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "test \"${GPU_SCHEDULER_GPU_COUNT}\" = 0\n"
            "test -z \"${CUDA_VISIBLE_DEVICES}\"\n"
            "sleep 0.2\n"
            f"printf 'first\\n' > {shlex.quote(str(trace))}\n",
            encoding="utf-8",
        )
        second.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"test -f {shlex.quote(str(trace))}\n"
            f"printf 'second\\n' >> {shlex.quote(str(trace))}\n",
            encoding="utf-8",
        )

        try:
            first_result = self._cli(
                "gpu-enqueue", str(first), "--cpu", "--workdir", str(work)
            )
            self.assertEqual(first_result.returncode, 0, first_result.stderr)
            first_id = first_result.stdout.strip()
            self.assertRegex(first_id, r"^\d{8}_\d{6}_[a-z0-9]{6}$")

            second_result = self._cli(
                "gpu-enqueue", str(second), "--cpu", "--workdir", str(work),
                "--after", first_id,
            )
            self.assertEqual(second_result.returncode, 0, second_result.stderr)
            second_id = second_result.stdout.strip()
            self.assertRegex(second_id, r"^\d{8}_\d{6}_[a-z0-9]{6}$")

            waited = self._cli(
                "gpu-scheduler", "wait", first_id, second_id, "--timeout", "20s",
                timeout=25,
            )
            self.assertEqual(waited.returncode, 0, waited.stderr)
            self.assertEqual(trace.read_text(encoding="utf-8"), "first\nsecond\n")

            done = self.home / ".gpu-scheduler" / "done"
            self.assertTrue((done / f"{first_id}.json").is_file())
            self.assertTrue((done / f"{second_id}.json").is_file())
        finally:
            stopped = self._cli("gpu-scheduler", "stop")
            self.assertEqual(stopped.returncode, 0, stopped.stderr)
            deadline = time.monotonic() + 5
            pid_file = self.home / ".gpu-scheduler" / "dispatcher.pid"
            while pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertFalse(pid_file.exists(), "dispatcher did not stop")


if __name__ == "__main__":
    unittest.main()
