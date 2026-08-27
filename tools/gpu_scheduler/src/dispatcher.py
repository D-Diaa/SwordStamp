#!/usr/bin/env python3
"""
dispatcher.py — GPU Scheduler daemon (non-blocking event loop).

Responsibilities
----------------
1. Owns a fixed set of GPU indices (from config.json).
2. Keeps each fully idle GPU warm by running warmup.py in a subprocess.
3. Polls ~/.gpu-scheduler/queue/ for new task JSON files; CLI tools poke it
   with SIGHUP so enqueued work dispatches immediately instead of waiting a
   poll interval.
4. Dispatches tasks to GPU slots (up to tasks_per_gpu concurrent tasks per
   GPU), spreading load evenly across GPUs.
5. Enforces per-task timeouts and automatic retries.
6. When a task finishes and a GPU becomes fully idle, restarts warmup.
7. Writes its PID to dispatcher.pid so CLI tools can signal it.

Event-loop design
-----------------
The scheduling loop never blocks on anything slower than a file stat:

* Stopping a warmup process is a per-GPU state machine (SIGTERM → poll →
  SIGKILL after 8 s), advanced once per tick.
* The idle-GPU memory-readiness check runs nvidia-smi as a *polled*
  subprocess (AsyncSmiProbe) instead of a blocking wait. While a launch is
  pending, its slots are reserved in memory but the task file stays in
  queue/ (crash-safe). A GPU that never frees memory cancels the pending
  launch at the deadline and enters a short cooldown while the rest of the
  pool keeps dispatching.
* Killing a task tree (defer / kill / timeout) is a PendingKill state
  machine: SIGTERM each tick to catch new children, SIGKILL after 8 s, and
  the request result is written once the tree is actually dead.

While any launch or kill is in flight, the loop ticks at 50 ms; otherwise it
sleeps poll_interval (woken early by signals).

Signals
-------
SIGTERM / SIGINT – graceful shutdown: stop warmup processes, exit loop;
                   running tasks continue and are recovered on restart.
SIGHUP           – wake immediately; reload config only if config.json
                   changed on disk (gpu_indices add/remove live,
                   poll_interval, warmup_gb, warmup_enabled, tasks_per_gpu).

Only the dispatcher may move files out of queue/ (dequeue).  The enqueue
CLI only writes files into queue/.
"""

import logging
import os
import signal
import subprocess
import sys
import time
import fcntl
from datetime import datetime, timezone
from pathlib import Path

_LIB = Path(__file__).parent
sys.path.insert(0, str(_LIB))
from common import (
    SCHEDULER_HOME, QUEUE_DIR, RUNNING_DIR, DONE_DIR, FAILED_DIR, LOGS_DIR,
    EXIT_CODES_DIR, REQUESTS_DIR, REQUEST_RESULTS_DIR, CONFIG_FILE, PID_FILE,
    LOCK_FILE, DISPATCHER_LOG, DEFAULT_CONFIG,
    load_config, validate_config, write_task, read_task, list_tasks, sorted_queue,
    task_ids_in, task_ids_in_all_state_dirs, task_to_queue_record,
    set_queued_task_priority, write_dispatcher_request_result,
    kill_process_tree, is_cpu_task, is_paused, format_duration,
    process_tree_pids, signal_process_tree, alive_processes,
    TaskDirCache,
)

WARMUP_STOP_TIMEOUT = 8.0    # SIGTERM → SIGKILL escalation for warmup
KILL_TERM_TIMEOUT   = 8.0    # SIGTERM → SIGKILL escalation for task trees
KILL_HARD_TIMEOUT   = 4.0    # SIGKILL → give-up window
GATE_COOLDOWN       = 5.0    # per-GPU pause after a memory-readiness timeout
SMI_PROBE_TIMEOUT   = 5.0    # max lifetime of one nvidia-smi probe
SMI_PROBE_INTERVAL  = 0.5    # min spacing between gate probes on one GPU
RESULTS_TTL         = 300.0  # request-results/ garbage collection age
REQUEST_TTL         = 120.0  # requests older than this are expired, not executed
FAST_TICK           = 0.05   # loop tick while launches/kills are in flight


def _warmup_pid_file(gpu_index: int) -> Path:
    return SCHEDULER_HOME / f"warmup_{gpu_index}.pid"


def _pid_is_warmup(pid: int) -> bool:
    """True only if the PID's command line is actually a warmup.py process.

    Warmup pidfiles can outlive a crashed dispatcher; by the time we read one
    the PID may have been recycled by an unrelated process, which we must not
    kill.
    """
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ")
        return b"warmup.py" in cmd
    except OSError:
        return False


def _kill_orphaned_warmups(gpu_indices: list) -> None:
    """Kill warmup processes left over from a previous unclean shutdown."""
    for idx in gpu_indices:
        pid_file = _warmup_pid_file(idx)
        if not pid_file.exists():
            continue
        try:
            pid = int(pid_file.read_text().strip())
            if _pid_is_warmup(pid):
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
        except Exception:
            pass
        pid_file.unlink(missing_ok=True)


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    SCHEDULER_HOME.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers: list[logging.Handler] = [logging.FileHandler(DISPATCHER_LOG)]
    # When daemonized, stdout is already redirected into DISPATCHER_LOG —
    # a StreamHandler would write every line twice. Only mirror to stdout
    # when it is a real terminal (foreground mode).
    try:
        if sys.stdout.isatty():
            handlers.append(logging.StreamHandler(sys.stdout))
    except (ValueError, OSError):
        pass
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    return logging.getLogger("dispatcher")


# ── Async nvidia-smi probe ────────────────────────────────────────────────────

class AsyncSmiProbe:
    """One polled nvidia-smi free-memory query for a single GPU."""

    def __init__(self, index: int):
        self.index = index
        self.started = time.monotonic()
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=memory.free,memory.total",
                 "--format=csv,noheader,nounits", "--id=" + str(index)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except OSError:
            self.proc = None

    def cancel(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.kill()
                self.proc.wait()
            except Exception:
                pass
        self.proc = None

    def result(self) -> tuple[bool, float | None]:
        """(finished, free_fraction).  fraction None ⇒ unavailable: don't gate."""
        if self.proc is None:
            return True, None
        if self.proc.poll() is None:
            if time.monotonic() - self.started > SMI_PROBE_TIMEOUT:
                self.cancel()
                return True, None
            return False, None
        try:
            out, _ = self.proc.communicate(timeout=1)
        except Exception:
            return True, None
        if self.proc.returncode != 0 or not (out or "").strip():
            return True, None
        try:
            free_s, total_s = out.strip().splitlines()[0].split(",")
            free, total = float(free_s), float(total_s)
            return True, (free / total) if total > 0 else None
        except Exception:
            return True, None


class MetricsSampler:
    """Periodic whole-pool nvidia-smi sample attributed to running tasks.

    One polled subprocess at a time, started at most every metrics_interval
    seconds and only while at least one started GPU task exists. Parsed
    samples update each allocation's peak-memory / utilization counters; the
    aggregate is written into the task's done/failed JSON as "gpu_metrics".
    With tasks_per_gpu > 1 co-located tasks share GPU-level numbers.
    """

    def __init__(self, log: logging.Logger):
        self.log = log
        self.proc: subprocess.Popen | None = None
        self.started = 0.0
        self.next_at = 0.0
        self.unavailable = False   # nvidia-smi missing: stop trying

    def tick(self, now: float, interval: float, allocations) -> None:
        if self.unavailable or interval <= 0:
            return
        active = [
            a for a in allocations
            if not a.pending and not a.killing and a.gpu_indices
        ]
        if self.proc is not None:
            self._collect(now, active)
        elif active and now >= self.next_at:
            self._start(now, interval, active)

    def _start(self, now: float, interval: float, active: list) -> None:
        indices = sorted({idx for a in active for idx in a.gpu_indices})
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi",
                 "--query-gpu=index,utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits",
                 "--id=" + ",".join(str(i) for i in indices)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            self.started = now
            self.next_at = now + max(1.0, interval)
        except OSError:
            self.unavailable = True
            self.log.info("nvidia-smi unavailable; per-task GPU telemetry disabled.")

    def _collect(self, now: float, active: list) -> None:
        if self.proc.poll() is None:
            if now - self.started > SMI_PROBE_TIMEOUT:
                try:
                    self.proc.kill()
                    self.proc.wait()
                except Exception:
                    pass
                self.proc = None
            return
        try:
            out, _ = self.proc.communicate(timeout=1)
            rc = self.proc.returncode
        except Exception:
            out, rc = "", -1
        self.proc = None
        if rc != 0 or not (out or "").strip():
            return
        per_gpu = self._parse(out)
        if not per_gpu:
            return
        for allocation in active:
            utils = [per_gpu[i][0] for i in allocation.gpu_indices
                     if i in per_gpu and per_gpu[i][0] is not None]
            mems = [per_gpu[i][1] for i in allocation.gpu_indices
                    if i in per_gpu and per_gpu[i][1] is not None]
            if not utils and not mems:
                continue
            allocation.metric_samples += 1
            if utils:
                allocation.metric_util_sum += sum(utils) / len(utils)
                allocation.metric_util_samples += 1
            if mems:
                allocation.metric_peak_mem = max(allocation.metric_peak_mem, max(mems))

    @staticmethod
    def _parse(out: str) -> dict[int, tuple[int | None, int | None]]:
        def to_int(text: str) -> int | None:
            try:
                return int(float(text.strip()))
            except ValueError:
                return None

        per_gpu: dict[int, tuple[int | None, int | None]] = {}
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            idx = to_int(parts[0])
            if idx is None:
                continue
            per_gpu[idx] = (to_int(parts[1]), to_int(parts[2]))
        return per_gpu


# ── Slot / GPU state ──────────────────────────────────────────────────────────

class Allocation:
    """One task and any GPU slots it reserves.

    pending – slots reserved but the process has not been launched yet
              (the task file is still in queue/).
    killing – a PendingKill owns this allocation; the regular completion
              path must leave it alone.
    """

    def __init__(
        self,
        task: dict,
        gpu_indices: list[int],
        proc: subprocess.Popen | None = None,
        pid: int | None = None,
        log_fh=None,
    ):
        self.task = task
        self.gpu_indices = gpu_indices
        self.proc = proc
        self.pid = pid
        self.log_fh = log_fh
        self.pending = False
        self.killing = False
        self.timeout_deadline: float | None = None   # monotonic
        # GPU telemetry accumulated by MetricsSampler while the task runs.
        self.metric_samples = 0
        self.metric_util_samples = 0
        self.metric_util_sum = 0.0
        self.metric_peak_mem = 0


class Slot:
    """One execution slot within a GPU."""

    def __init__(self):
        self.allocation: Allocation | None = None

    @property
    def is_free(self) -> bool:
        return self.allocation is None

    @property
    def task(self) -> dict | None:
        return self.allocation.task if self.allocation is not None else None


class GPU:
    def __init__(self, index: int, tasks_per_gpu: int = 1):
        self.index = index
        self.tasks_per_gpu = tasks_per_gpu
        self.slots: list[Slot] = [Slot() for _ in range(tasks_per_gpu)]
        self.warmup_proc: subprocess.Popen | None = None
        # Readiness-gate state machine (see _advance_gpu_gate)
        self.warmup_stop_deadline: float | None = None
        self.gate_passed = False
        self.mem_probe: AsyncSmiProbe | None = None
        self.mem_probe_next = 0.0
        self.mem_wait_deadline: float | None = None
        self.mem_wait_announced = False
        self.cooldown_until = 0.0

    @property
    def running_count(self) -> int:
        return sum(1 for s in self.slots if not s.is_free)

    @property
    def started_count(self) -> int:
        """Slots whose task process has actually been launched."""
        return sum(
            1 for s in self.slots
            if s.allocation is not None and not s.allocation.pending
        )

    @property
    def free_slot_count(self) -> int:
        return self.tasks_per_gpu - self.running_count

    @property
    def has_free_slot(self) -> bool:
        return self.running_count < self.tasks_per_gpu

    @property
    def is_idle(self) -> bool:
        return self.running_count == 0

    @property
    def is_warming(self) -> bool:
        return self.is_idle and self.warmup_proc is not None

    def free_slot(self) -> "Slot | None":
        for s in self.slots:
            if s.is_free:
                return s
        return None

    def free_slots(self) -> list["Slot"]:
        return [s for s in self.slots if s.is_free]

    def reset_gate(self) -> None:
        self.gate_passed = False
        self.mem_wait_deadline = None
        self.mem_wait_announced = False
        self.mem_probe_next = 0.0
        if self.mem_probe is not None:
            self.mem_probe.cancel()
            self.mem_probe = None


# ── Pending state machines ────────────────────────────────────────────────────

class PendingLaunch:
    """A task whose slots are reserved while its GPUs become ready."""

    def __init__(self, task: dict, gpus: list[GPU], allocation: Allocation,
                 needs_gate: dict[int, bool]):
        self.task = task
        self.gpus = gpus
        self.allocation = allocation
        self.needs_gate = needs_gate


class PendingKill:
    """An in-flight task-tree termination (defer / kill / timeout)."""

    def __init__(
        self,
        allocation: Allocation,
        *,
        requeue: bool,
        fresh_enqueue_time: bool = False,
        note: str = "",
        request_id: str | None = None,
        request_action: str | None = None,
        retry_eligible: bool = False,
    ):
        self.allocation = allocation
        self.requeue = requeue
        self.fresh_enqueue_time = fresh_enqueue_time
        self.note = note
        self.request_id = request_id
        self.request_action = request_action
        self.retry_eligible = retry_eligible
        now = time.monotonic()
        self.term_deadline = now + KILL_TERM_TIMEOUT
        self.hard_deadline = now + KILL_TERM_TIMEOUT + KILL_HARD_TIMEOUT
        self.killed = False
        pid = allocation.pid or (
            allocation.proc.pid if allocation.proc is not None
            else allocation.task.get("pid")
        )
        self.root_pid = int(pid) if pid else None
        self.known_pids: set[int] = set()


# ── Dispatcher ────────────────────────────────────────────────────────────────

class Dispatcher:
    def __init__(self):
        self.log               = _setup_logging()
        self._lock_fh          = None
        self._acquire_singleton_lock()
        self.config            = validate_config(load_config())
        self._config_mtime     = self._read_config_mtime()
        self.gpus: dict[int, GPU] = {}
        self._running          = True
        self._reload_requested = False
        self._control_requested = False
        self._lib_dir          = Path(__file__).parent
        self.allocations: dict[str, Allocation] = {}
        self.pending_launches: list[PendingLaunch] = []
        self.pending_kills: list[PendingKill] = []
        self._dying_warmups: list[subprocess.Popen] = []
        self._queue_cache      = TaskDirCache()
        self._was_paused       = is_paused()
        self._next_results_gc  = 0.0
        self._metrics          = MetricsSampler(self.log)
        # Anti-starvation bookkeeping for blocked multi-GPU tasks:
        # task_id -> monotonic time it was first seen ready-but-unfittable.
        self._blocked_since: dict[str, float] = {}
        self._drain_announced: set[str] = set()

        self._init_gpus()
        self._recover_running_tasks()

    def _acquire_singleton_lock(self) -> None:
        SCHEDULER_HOME.mkdir(parents=True, exist_ok=True)
        # "a" (not "w"): a losing contender must not truncate the lock file
        # contents written by the live dispatcher before flock() rejects it.
        self._lock_fh = open(LOCK_FILE, "a")
        try:
            fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.log.error("Another dispatcher already holds %s.", LOCK_FILE)
            raise SystemExit(1)
        self._lock_fh.seek(0)
        self._lock_fh.truncate()
        self._lock_fh.write(str(os.getpid()) + "\n")
        self._lock_fh.flush()

    @staticmethod
    def _read_config_mtime() -> int | None:
        try:
            return CONFIG_FILE.stat().st_mtime_ns
        except OSError:
            return None

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_gpus(self) -> None:
        indices = self.config["gpu_indices"]
        tpg     = self.config.get("tasks_per_gpu", 1)
        _kill_orphaned_warmups(indices)
        for idx in indices:
            self.gpus[idx] = GPU(idx, tpg)
        self.log.info(
            "Managing GPUs: %s (tasks_per_gpu=%d).", list(self.gpus.keys()), tpg
        )

    # ── Signal handling ───────────────────────────────────────────────────────

    def _handle_shutdown(self, signum, frame) -> None:
        self.log.info("Signal %s received – initiating graceful shutdown.", signum)
        self._running = False

    def _handle_reload(self, signum, frame) -> None:
        # SIGHUP doubles as a cheap "wake up now" poke from the CLI tools;
        # _apply_reload only re-reads config when the file actually changed.
        self._reload_requested = True
        self._control_requested = True

    def _apply_reload(self) -> None:
        self._reload_requested = False
        mtime = self._read_config_mtime()
        if mtime == self._config_mtime:
            return   # just a wake-up poke, not a config change
        self._config_mtime = mtime
        try:
            new_cfg = validate_config(load_config())
        except Exception as e:
            self.log.error("Config reload failed: %s", e)
            return

        self.config["poll_interval"]       = new_cfg.get("poll_interval", 1.0)
        self.config["warmup_gb"]           = new_cfg.get("warmup_gb", 0.0)
        self.config["warmup_enabled"]      = new_cfg.get("warmup_enabled", True)
        self.config["gpu_ready_free_frac"] = new_cfg.get(
            "gpu_ready_free_frac", DEFAULT_CONFIG["gpu_ready_free_frac"])
        self.config["gpu_ready_timeout"]   = new_cfg.get("gpu_ready_timeout", 30.0)
        self.config["metrics_interval"]    = new_cfg.get(
            "metrics_interval", DEFAULT_CONFIG["metrics_interval"])
        self.config["multi_gpu_drain_sec"] = new_cfg.get(
            "multi_gpu_drain_sec", DEFAULT_CONFIG["multi_gpu_drain_sec"])

        # tasks_per_gpu: grow slots immediately; shrink is advisory (drain in place)
        new_tpg = new_cfg.get("tasks_per_gpu", 1)
        old_tpg = self.config.get("tasks_per_gpu", 1)
        if new_tpg != old_tpg:
            self.config["tasks_per_gpu"] = new_tpg
            for gpu in self.gpus.values():
                if new_tpg > len(gpu.slots):
                    gpu.slots.extend(Slot() for _ in range(new_tpg - len(gpu.slots)))
                    # A live capacity increase must not punch new holes in a
                    # whole-GPU allocation. This also preserves the existing
                    # guarantee for multi-GPU tasks.
                    exclusive_allocation = next(
                        (
                            slot.allocation for slot in gpu.slots
                            if slot.allocation is not None
                            and self._task_requires_exclusive_gpus(slot.allocation.task)
                        ),
                        None,
                    )
                    if exclusive_allocation is not None:
                        for slot in gpu.free_slots():
                            slot.allocation = exclusive_allocation
                if new_tpg < len(gpu.slots):
                    while len(gpu.slots) > new_tpg and gpu.slots[-1].is_free:
                        gpu.slots.pop()
                if new_tpg < gpu.running_count:
                    self.log.warning(
                        "GPU %d: running_count=%d exceeds new tasks_per_gpu=%d; "
                        "will drain before accepting new tasks.",
                        gpu.index, gpu.running_count, new_tpg,
                    )
                gpu.tasks_per_gpu = new_tpg
            self.log.info("tasks_per_gpu updated: %d → %d.", old_tpg, new_tpg)

        if not self.config["warmup_enabled"]:
            for gpu in self.gpus.values():
                if gpu.warmup_proc is not None:
                    self._retire_warmup(gpu)

        new_indices = set(new_cfg.get("gpu_indices", []))
        old_indices = set(self.gpus.keys())

        for idx in sorted(old_indices - new_indices):
            gpu = self.gpus[idx]
            self.log.info("GPU %d: removing from managed list.", idx)
            if gpu.warmup_proc is not None:
                self._retire_warmup(gpu)
            if gpu.running_count > 0:
                self._kill_and_requeue_gpu(gpu)
            del self.gpus[idx]

        for idx in sorted(new_indices - old_indices):
            self.log.info("GPU %d: adding to managed list.", idx)
            self.gpus[idx] = GPU(idx, self.config.get("tasks_per_gpu", 1))

        self.log.info(
            "Config reloaded – GPUs: %s, poll=%.1fs, warmup=%s, tasks_per_gpu=%d.",
            sorted(self.gpus.keys()),
            self.config["poll_interval"],
            ("off" if not self.config["warmup_enabled"]
             else (f"{self.config['warmup_gb']:.0f} GB" if self.config["warmup_gb"] else "auto")),
            self.config.get("tasks_per_gpu", 1),
        )

    # ── Durable process state ─────────────────────────────────────────────────

    def _exit_code_file(self, task_id: str) -> Path:
        return EXIT_CODES_DIR / f"{task_id}.rc"

    def _write_exit_code(self, task_id: str, rc: int) -> None:
        EXIT_CODES_DIR.mkdir(parents=True, exist_ok=True)
        self._exit_code_file(task_id).write_text(str(int(rc)) + "\n")

    def _read_exit_code(self, task_id: str) -> int | None:
        try:
            return int(self._exit_code_file(task_id).read_text().strip())
        except Exception:
            return None

    def _cleanup_exit_code(self, task_id: str) -> None:
        self._exit_code_file(task_id).unlink(missing_ok=True)

    def _process_alive(self, pid: int | None) -> bool:
        if not pid:
            return False
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, ValueError):
            return False
        except PermissionError:
            return True
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists():
            try:
                # /proc/<pid>/stat contains "... ) <state> ..."; Z means zombie.
                after_name = stat.read_text().rsplit(")", 1)[1].strip()
                if after_name.startswith("Z"):
                    return False
            except Exception:
                pass
        return True

    def _requeue_task(
        self,
        task: dict,
        source_path: Path | None,
        note: str = "",
        *,
        fresh_enqueue_time: bool = False,
    ) -> None:
        requeue_task = task_to_queue_record(task, fresh_enqueue_time=fresh_enqueue_time)
        write_task(QUEUE_DIR, requeue_task)
        if source_path:
            source_path.unlink(missing_ok=True)
        if note:
            self.log.warning("Task %s requeued: %s.", task.get("id", "?"), note)

    def _finalize_task(
        self,
        task: dict,
        task_path: Path | None,
        returncode: int | None,
        note: str = "",
    ) -> None:
        task_id = task["id"]
        result = {k: v for k, v in task.items() if not k.startswith("_")}
        result["end_time"] = datetime.now(tz=timezone.utc).isoformat()
        if returncode is None:
            result.pop("returncode", None)
        else:
            result["returncode"] = returncode
        if note:
            result["failure_note"] = note
        dest_dir = DONE_DIR if returncode == 0 else FAILED_DIR
        write_task(dest_dir, result)
        if task_path:
            task_path.unlink(missing_ok=True)
        self._cleanup_exit_code(task_id)

    # ── Retries ───────────────────────────────────────────────────────────────

    def _maybe_retry(self, task: dict, note: str) -> bool:
        """Requeue a failed task if it has retries left.  Returns True if requeued."""
        try:
            max_retries = int(task.get("max_retries", 0) or 0)
            attempt = int(task.get("attempt", 0) or 0)
        except (TypeError, ValueError):
            return False
        if max_retries <= 0 or attempt >= max_retries:
            return False
        queued = task_to_queue_record(task, fresh_enqueue_time=True)
        queued["attempt"] = attempt + 1
        queued["last_failure"] = note
        write_task(QUEUE_DIR, queued)
        self.log.warning(
            "Task %s failed (%s); auto-retry %d/%d queued.",
            task.get("id", "?"), note, attempt + 1, max_retries,
        )
        return True

    def _conclude_failure(
        self,
        task: dict,
        task_path: Path | None,
        *,
        returncode: int | None,
        note: str,
        retry_eligible: bool,
    ) -> None:
        """Fail a finished/killed task, or requeue it when retries remain."""
        retry_note = note or (f"rc={returncode}" if returncode is not None else "killed")
        if retry_eligible and self._maybe_retry(task, retry_note):
            if task_path:
                task_path.unlink(missing_ok=True)
            self._cleanup_exit_code(task["id"])
            return
        self._finalize_task(task, task_path, returncode=returncode, note=note)

    # ── Task bookkeeping helpers ──────────────────────────────────────────────

    def _task_recorded_gpu_indices(self, task: dict) -> list[int]:
        raw_indices = task.get("gpu_indices")
        if isinstance(raw_indices, list) and raw_indices:
            try:
                return [int(idx) for idx in raw_indices]
            except (TypeError, ValueError):
                return []
        gpu_idx = task.get("gpu_index")
        if gpu_idx is None:
            return []
        try:
            return [int(gpu_idx)]
        except (TypeError, ValueError):
            return []

    def _task_gpu_count(self, task: dict) -> int:
        if is_cpu_task(task):
            return 0
        raw_count = task.get("gpu_count")
        if raw_count is None:
            recorded = self._task_recorded_gpu_indices(task)
            return max(1, len(recorded))
        return int(raw_count)

    def _task_requires_exclusive_gpus(self, task: dict, gpu_count: int | None = None) -> bool:
        """Whether a GPU task must reserve every slot on its assigned GPU(s)."""
        if is_cpu_task(task):
            return False
        if task.get("exclusive") is True:
            return True
        if gpu_count is None:
            gpu_count = self._task_gpu_count(task)
        # Multi-GPU tasks have always been whole-GPU allocations. Keep that
        # behavior for existing task records that predate ``exclusive``.
        return gpu_count > 1

    def _task_timeout(self, task: dict) -> float | None:
        raw = task.get("timeout")
        if raw is None:
            return None
        try:
            timeout = float(raw)
        except (TypeError, ValueError):
            return None
        return timeout if timeout > 0 else None

    def _set_timeout_deadline(self, allocation: Allocation, *, elapsed: float = 0.0) -> None:
        timeout = self._task_timeout(allocation.task)
        if timeout is not None:
            allocation.timeout_deadline = time.monotonic() + max(0.0, timeout - elapsed)

    def _reserve_recovered_allocation(self, allocation: Allocation) -> None:
        exclusive = self._task_requires_exclusive_gpus(
            allocation.task, len(allocation.gpu_indices)
        )
        for idx in allocation.gpu_indices:
            gpu = self.gpus[idx]
            if exclusive:
                if not gpu.is_idle:
                    self.log.warning(
                        "GPU %d: recovered overlapping multi-GPU task %s; "
                        "using overflow slots until it drains.",
                        idx, allocation.task.get("id", "?"),
                    )
                while len(gpu.free_slots()) < gpu.tasks_per_gpu:
                    gpu.slots.append(Slot())
                slots = gpu.free_slots()[:gpu.tasks_per_gpu]
            else:
                while gpu.free_slot() is None:
                    gpu.slots.append(Slot())
                slots = [gpu.free_slot()]
            for slot in slots:
                slot.allocation = allocation

    def _release_allocation(self, allocation: Allocation) -> None:
        affected: list[GPU] = []
        for gpu in self.gpus.values():
            for slot in gpu.slots:
                if slot.allocation is allocation:
                    slot.allocation = None
                    affected.append(gpu)
        self.allocations.pop(allocation.task.get("id", ""), None)
        self._drop_pending_launch(allocation)
        for gpu in set(affected):
            if gpu.started_count == 0:
                # GPU is going back to (or through) idle: the next first task
                # on it must re-confirm memory readiness.
                gpu.reset_gate()

    def _drop_pending_launch(self, allocation: Allocation) -> None:
        self.pending_launches = [
            launch for launch in self.pending_launches
            if launch.allocation is not allocation
        ]

    def _attach_recovered_task(self, task: dict) -> bool:
        gpu_indices = self._task_recorded_gpu_indices(task)
        pid = task.get("pid")
        if is_cpu_task(task):
            task["gpu_count"] = 0
            task["gpu_indices"] = []
            task.pop("gpu_index", None)
            allocation = Allocation(task, [], proc=None, pid=pid, log_fh=None)
            self.allocations[task["id"]] = allocation
            self._recover_timeout_deadline(allocation)
            return True

        missing = [idx for idx in gpu_indices if idx not in self.gpus]
        if not gpu_indices or missing:
            self.log.warning(
                "Recovered live task %s on unmanaged GPU(s) %s; killing and requeueing.",
                task.get("id", "?"), missing or gpu_indices,
            )
            if pid:
                kill_process_tree(int(pid))
            self._requeue_task(task, task.get("_path"), note="assigned GPU is no longer managed")
            self._cleanup_exit_code(task["id"])
            return False

        task["gpu_indices"] = gpu_indices
        task["gpu_index"] = gpu_indices[0]
        try:
            task["gpu_count"] = self._task_gpu_count(task)
        except (TypeError, ValueError):
            task["gpu_count"] = max(1, len(gpu_indices))
        allocation = Allocation(task, gpu_indices, proc=None, pid=pid, log_fh=None)
        self.allocations[task["id"]] = allocation
        self._recover_timeout_deadline(allocation)
        self._reserve_recovered_allocation(allocation)
        for idx in gpu_indices:
            gpu = self.gpus[idx]
            if len(gpu.slots) > gpu.tasks_per_gpu:
                self.log.warning(
                    "GPU %d: recovered %d occupied slots with tasks_per_gpu=%d; draining overflow.",
                    gpu.index, gpu.running_count, gpu.tasks_per_gpu,
                )
        return True

    def _recover_timeout_deadline(self, allocation: Allocation) -> None:
        timeout = self._task_timeout(allocation.task)
        if timeout is None:
            return
        elapsed = 0.0
        try:
            start = datetime.fromisoformat(allocation.task.get("start_time", ""))
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            elapsed = max(0.0, (datetime.now(tz=timezone.utc) - start).total_seconds())
        except (TypeError, ValueError):
            pass
        self._set_timeout_deadline(allocation, elapsed=elapsed)

    def _recover_running_tasks(self) -> None:
        """Reattach live running/ tasks or finalize/requeue stale entries."""
        for task in list_tasks(RUNNING_DIR):
            task_path = task["_path"]
            task_id = task["id"]
            pid = task.get("pid")

            if not pid:
                self._requeue_task(task, task_path, note="running file had no pid")
                continue

            if self._process_alive(pid):
                if self._attach_recovered_task(task):
                    location = self._allocation_location(self.allocations[task_id])
                    self.log.info(
                        "Recovered running task %s (PID %s) on %s.",
                        task_id, pid, location,
                    )
                continue

            rc = self._read_exit_code(task_id)
            if rc is None:
                self._finalize_task(
                    task,
                    task_path,
                    returncode=-3,
                    note="process was gone during dispatcher recovery and no exit code was recorded",
                )
                self.log.warning("Recovered stale task %s as failed (exit code unavailable).", task_id)
            else:
                if rc == 0:
                    self._finalize_task(task, task_path, returncode=rc)
                else:
                    self._conclude_failure(
                        task, task_path, returncode=rc,
                        note=task.get("failure_note", ""),
                        retry_eligible=True,
                    )
                status = "done" if rc == 0 else f"FAILED (rc={rc})"
                self.log.info("Recovered completed task %s as %s.", task_id, status)

    # ── Task kill / requeue (async) ───────────────────────────────────────────

    def _kill_and_requeue_gpu(self, gpu: GPU) -> None:
        """Kill all running tasks on a GPU and re-enqueue them."""
        allocations: list[Allocation] = []
        for slot in gpu.slots:
            if slot.allocation is not None and slot.allocation not in allocations:
                allocations.append(slot.allocation)
        for allocation in allocations:
            self._kill_allocation(allocation, requeue=True)

    def _kill_allocation(
        self,
        allocation: Allocation,
        requeue: bool = True,
        *,
        fresh_enqueue_time: bool = False,
        fail_without_returncode: bool = False,
        note: str = "",
        request_id: str | None = None,
        request_action: str | None = None,
        retry_eligible: bool = False,
    ) -> None:
        """Start (or resolve) the termination of an allocation's task.

        Pending allocations have no process yet and resolve immediately;
        started tasks become a PendingKill advanced by the main loop.
        """
        task = allocation.task
        if task is None or allocation.killing:
            return

        if allocation.pending:
            self._resolve_pending_kill_now(
                allocation,
                requeue=requeue,
                fresh_enqueue_time=fresh_enqueue_time,
                fail_without_returncode=fail_without_returncode,
                note=note,
                request_id=request_id,
                request_action=request_action,
            )
            return

        allocation.killing = True
        kill = PendingKill(
            allocation,
            requeue=requeue,
            fresh_enqueue_time=fresh_enqueue_time,
            note=note,
            request_id=request_id,
            request_action=request_action,
            retry_eligible=retry_eligible,
        )
        self.pending_kills.append(kill)
        self._signal_kill(kill, signal.SIGTERM)

    def _resolve_pending_kill_now(
        self,
        allocation: Allocation,
        *,
        requeue: bool,
        fresh_enqueue_time: bool,
        fail_without_returncode: bool,
        note: str,
        request_id: str | None,
        request_action: str | None,
    ) -> None:
        """A reserved-but-not-launched task: no process to kill, just move files."""
        task = allocation.task
        task_id = task["id"]
        t_path = task.get("_path")   # still the queue/ file
        location = self._allocation_location(allocation)
        if requeue:
            # The task never started; just refresh its queue record in place.
            self._requeue_task(task, None, fresh_enqueue_time=fresh_enqueue_time)
            self.log.info("%s: pending task %s returned to queue.", location, task_id)
        elif fail_without_returncode:
            self._finalize_task(task, t_path, returncode=None, note=note)
            self.log.info("%s: pending task %s killed and moved to failed/.", location, task_id)
        self._release_allocation(allocation)
        if request_id:
            self._write_kill_request_result(
                request_id, request_action, task_id, location, allocation,
            )

    def _signal_kill(self, kill: PendingKill, sig: int) -> None:
        if kill.root_pid is None:
            return
        kill.known_pids = process_tree_pids(kill.root_pid, kill.known_pids)
        alive = alive_processes(kill.known_pids)
        if alive:
            signal_process_tree(kill.root_pid, alive, sig)

    def _advance_pending_kills(self) -> None:
        for kill in list(self.pending_kills):
            allocation = kill.allocation
            now = time.monotonic()
            alive: set[int] = set()
            if kill.root_pid is not None:
                kill.known_pids = process_tree_pids(kill.root_pid, kill.known_pids)
                alive = alive_processes(kill.known_pids)

            if alive:
                if now >= kill.hard_deadline:
                    self.log.warning(
                        "Task %s: process tree still alive after SIGKILL; finalizing anyway.",
                        allocation.task.get("id", "?"),
                    )
                elif now >= kill.term_deadline:
                    if not kill.killed:
                        kill.killed = True
                        self.log.warning(
                            "Task %s: SIGTERM timed out; escalating to SIGKILL.",
                            allocation.task.get("id", "?"),
                        )
                    signal_process_tree(kill.root_pid, alive, signal.SIGKILL)
                    continue
                else:
                    # Re-signal each tick to catch children spawned mid-shutdown.
                    signal_process_tree(kill.root_pid, alive, signal.SIGTERM)
                    continue

            # Tree is dead (or we gave up waiting): finalize.
            self.pending_kills.remove(kill)
            self._finalize_killed_allocation(kill)

    def _finalize_killed_allocation(self, kill: PendingKill) -> None:
        allocation = kill.allocation
        task = allocation.task
        task_id = task["id"]

        if allocation.proc is not None:
            try:
                allocation.proc.poll()   # reap if it exited
            except Exception:
                pass
        if allocation.log_fh:
            try:
                allocation.log_fh.close()
            except Exception:
                pass

        t_path = task.get("_path")
        location = self._allocation_location(allocation)
        if kill.requeue:
            self._requeue_task(task, None, fresh_enqueue_time=kill.fresh_enqueue_time)
            if t_path:
                t_path.unlink(missing_ok=True)
            self.log.info("%s: task %s killed and re-enqueued.", location, task_id)
        else:
            self._attach_gpu_metrics(allocation)
            self._conclude_failure(
                task, t_path,
                returncode=None,
                note=kill.note,
                retry_eligible=kill.retry_eligible,
            )
            self.log.info("%s: task %s killed (%s).", location, task_id,
                          kill.note or "moved to failed/")
        self._cleanup_exit_code(task_id)
        allocation.killing = False
        self._release_allocation(allocation)

        if kill.request_id:
            self._write_kill_request_result(
                kill.request_id, kill.request_action, task_id, location, allocation,
            )

    def _write_kill_request_result(
        self,
        request_id: str,
        request_action: str | None,
        task_id: str,
        location: str,
        allocation: Allocation,
    ) -> None:
        if request_action == "preempt":
            payload = {"task_id": task_id, "location": location}
            if allocation.gpu_indices:
                payload["gpu_index"] = allocation.gpu_indices[0]
            write_dispatcher_request_result(
                request_id,
                ok=True,
                message=f"deferred {task_id} from {location}",
                **payload,
            )
        else:
            write_dispatcher_request_result(
                request_id,
                ok=True,
                message=f"killed running task {task_id}; moved to failed",
                task_id=task_id,
                state="running",
            )

    def _allocation_location(self, allocation: Allocation) -> str:
        if is_cpu_task(allocation.task):
            return "CPU"
        return f"GPU(s) {allocation.gpu_indices}"

    def _attach_gpu_metrics(self, allocation: Allocation) -> None:
        """Fold sampled telemetry into the task dict before it is finalized."""
        if allocation.metric_samples <= 0:
            return
        metrics = {
            "samples": allocation.metric_samples,
            "peak_mem_mib": int(allocation.metric_peak_mem),
        }
        if allocation.metric_util_samples:
            metrics["avg_util"] = round(
                allocation.metric_util_sum / allocation.metric_util_samples, 1
            )
        allocation.task["gpu_metrics"] = metrics

    # ── Timeouts ──────────────────────────────────────────────────────────────

    def _check_timeouts(self) -> None:
        now = time.monotonic()
        for allocation in list(self.allocations.values()):
            if allocation.pending or allocation.killing:
                continue
            deadline = allocation.timeout_deadline
            if deadline is None or now < deadline:
                continue
            timeout = self._task_timeout(allocation.task) or 0.0
            note = f"timed out after {format_duration(timeout)}"
            self.log.warning(
                "Task %s exceeded its %s timeout; killing.",
                allocation.task.get("id", "?"), format_duration(timeout),
            )
            self._kill_allocation(
                allocation,
                requeue=False,
                fail_without_returncode=True,
                note=note,
                retry_eligible=True,
            )

    # ── Dispatcher request handling ──────────────────────────────────────────

    def _resolve_running_allocation(self, task_ref: str) -> Allocation:
        task_ref = str(task_ref).strip()
        if not task_ref:
            raise ValueError("empty task ID")

        matches: list[tuple[str, Allocation]] = []
        for allocation in self.allocations.values():
            task_id = allocation.task.get("id", "")
            if task_id.startswith(task_ref):
                matches.append((task_id, allocation))

        if not matches:
            raise LookupError(f"task not found in running: {task_ref}")

        unique_ids = sorted({task_id for task_id, _ in matches})
        if len(unique_ids) > 1:
            raise ValueError(f"ambiguous task prefix {task_ref!r}: {', '.join(unique_ids)}")

        return matches[0][1]

    def _request_preempt(self, request_id: str, task_ref: str) -> None:
        allocation = self._resolve_running_allocation(task_ref)
        if allocation.killing:
            raise ValueError(f"task {allocation.task.get('id', '?')} is already being killed")
        self._kill_allocation(
            allocation,
            requeue=True,
            fresh_enqueue_time=True,
            request_id=request_id,
            request_action="preempt",
        )

    def _request_kill_task(self, request_id: str, task_ref: str) -> None:
        task_ref = str(task_ref).strip()
        if not task_ref:
            raise ValueError("empty task ID")

        running_matches = [
            (allocation.task.get("id", ""), allocation)
            for allocation in self.allocations.values()
            if allocation.task.get("id", "").startswith(task_ref)
        ]
        queued_matches = [
            task
            for task in self._list_queue()
            if task.get("id", "").startswith(task_ref)
            and task.get("id", "") not in self.allocations
        ]
        unique_ids = sorted(
            {task_id for task_id, _ in running_matches}
            | {task.get("id", "") for task in queued_matches}
        )
        if not unique_ids:
            raise LookupError(f"task not found in running or queue: {task_ref}")
        if len(unique_ids) > 1:
            raise ValueError(f"ambiguous task prefix {task_ref!r}: {', '.join(unique_ids)}")

        if running_matches:
            allocation = running_matches[0][1]
            if allocation.killing:
                raise ValueError(f"task {allocation.task.get('id', '?')} is already being killed")
            self._kill_allocation(
                allocation,
                requeue=False,
                fail_without_returncode=True,
                note="killed via gpu-scheduler kill",
                request_id=request_id,
                request_action="kill_task",
            )
            return

        task = queued_matches[0]
        task_id = task["id"]
        self._finalize_task(
            task,
            task["_path"],
            returncode=None,
            note="killed before start via gpu-scheduler kill",
        )
        self.log.info("Queued task %s killed and moved to failed/.", task_id)
        write_dispatcher_request_result(
            request_id,
            ok=True,
            message=f"killed queue task {task_id}; moved to failed",
            task_id=task_id,
            state="queue",
        )

    def _request_expired(self, request: dict) -> bool:
        """A leftover request (e.g. the dispatcher died before processing it)
        must not fire long after the user issued it."""
        try:
            created = datetime.fromisoformat(request.get("created_at", ""))
        except (TypeError, ValueError):
            return False
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age = (datetime.now(tz=timezone.utc) - created).total_seconds()
        return age > REQUEST_TTL

    def _process_dispatcher_request(self, request: dict) -> None:
        request_id = request["id"]
        action = request.get("action")
        try:
            if self._request_expired(request):
                raise ValueError(
                    f"request expired (older than {REQUEST_TTL:.0f}s); not executing "
                    f"{action} — re-run the command if still wanted"
                )
            if action == "preempt":
                self._request_preempt(request_id, request.get("task_ref", ""))
            elif action == "kill_task":
                self._request_kill_task(request_id, request.get("task_ref", ""))
            elif action == "set_priority":
                task_id, old_priority, new_priority = set_queued_task_priority(
                    request.get("task_ref", ""),
                    int(request.get("priority")),
                )
                write_dispatcher_request_result(
                    request_id,
                    ok=True,
                    message=f"priority changed for {task_id}: {old_priority} -> {new_priority}",
                    task_id=task_id,
                    old_priority=old_priority,
                    new_priority=new_priority,
                )
            else:
                raise ValueError(f"unknown dispatcher request action: {action}")
        except Exception as exc:
            write_dispatcher_request_result(
                request_id,
                ok=False,
                message=str(exc),
            )
            self.log.warning("Dispatcher request %s failed: %s", request_id, exc)
        finally:
            request_path = request.get("_path")
            if request_path:
                request_path.unlink(missing_ok=True)

    def _process_dispatcher_requests(self) -> None:
        self._control_requested = False
        requests = sorted(
            list_tasks(REQUESTS_DIR),
            key=lambda request: (request.get("created_at", ""), request.get("id", "")),
        )
        for request in requests:
            self._process_dispatcher_request(request)

    def _gc_request_results(self) -> None:
        now = time.monotonic()
        if now < self._next_results_gc:
            return
        self._next_results_gc = now + 60.0
        cutoff = time.time() - RESULTS_TTL
        try:
            for p in REQUEST_RESULTS_DIR.glob("*.json"):
                try:
                    if p.stat().st_mtime < cutoff:
                        p.unlink()
                except OSError:
                    pass
        except OSError:
            pass

    # ── Warmup management ─────────────────────────────────────────────────────

    def _start_warmup(self, gpu: GPU) -> None:
        warmup_script = self._lib_dir / "warmup.py"
        warmup_gb     = self.config.get("warmup_gb", 0.0)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu.index)

        cmd = [sys.executable, str(warmup_script)]
        if warmup_gb > 0:
            cmd.append(str(warmup_gb))

        warmup_log = LOGS_DIR / f"warmup_gpu{gpu.index}.log"
        log_fh = None
        try:
            log_fh = open(warmup_log, "a")
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            log_fh.close()
            gpu.warmup_proc = proc
            gpu.warmup_stop_deadline = None
            gpu.reset_gate()   # warmup occupies VRAM again
            _warmup_pid_file(gpu.index).write_text(str(proc.pid) + "\n")
            label = f"{warmup_gb:.0f} GB" if warmup_gb > 0 else "auto (80% free VRAM)"
            self.log.info("GPU %d: warmup started (PID %d, target=%s).",
                          gpu.index, proc.pid, label)
        except Exception as exc:
            if log_fh is not None:
                try:
                    log_fh.close()
                except Exception:
                    pass
            self.log.error("GPU %d: failed to start warmup: %s", gpu.index, exc)

    def _request_warmup_stop(self, gpu: GPU) -> None:
        """Begin async warmup shutdown; _advance_gpu_gate watches it exit."""
        proc = gpu.warmup_proc
        if proc is None or gpu.warmup_stop_deadline is not None:
            return
        gpu.warmup_stop_deadline = time.monotonic() + WARMUP_STOP_TIMEOUT
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

    def _retire_warmup(self, gpu: GPU) -> None:
        """Detach a warmup process (GPU removed / warmup disabled); reap later."""
        proc = gpu.warmup_proc
        gpu.warmup_proc = None
        gpu.warmup_stop_deadline = None
        _warmup_pid_file(gpu.index).unlink(missing_ok=True)
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        self._dying_warmups.append(proc)

    def _reap_dying_warmups(self) -> None:
        still_dying = []
        for proc in self._dying_warmups:
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
                still_dying.append(proc)
        self._dying_warmups = still_dying

    def _stop_warmup_blocking(self, gpu: GPU) -> None:
        """Shutdown-path only: synchronously stop a warmup process."""
        proc = gpu.warmup_proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
            try:
                proc.wait(timeout=WARMUP_STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
                proc.wait()
        gpu.warmup_proc = None
        gpu.warmup_stop_deadline = None
        _warmup_pid_file(gpu.index).unlink(missing_ok=True)
        self.log.info("GPU %d: warmup stopped.", gpu.index)

    def _ensure_warmups_alive(self) -> None:
        """Reap warmup processes that crashed; main loop restarts them later."""
        for gpu in self.gpus.values():
            proc = gpu.warmup_proc
            if proc is None or proc.poll() is None:
                continue
            if gpu.warmup_stop_deadline is None:
                self.log.warning(
                    "GPU %d: warmup exited unexpectedly (rc=%d) – will restart.",
                    gpu.index, proc.returncode,
                )
            gpu.warmup_proc = None
            gpu.warmup_stop_deadline = None
            _warmup_pid_file(gpu.index).unlink(missing_ok=True)

    def _start_idle_warmups(self) -> None:
        if not self.config.get("warmup_enabled", True):
            return
        now = time.monotonic()
        for gpu in self.gpus.values():
            if gpu.is_idle and gpu.warmup_proc is None and now >= gpu.cooldown_until:
                self._start_warmup(gpu)

    # ── GPU readiness gate ────────────────────────────────────────────────────

    def _advance_gpu_gate(self, gpu: GPU) -> str:
        """Advance one GPU's idle→busy readiness gate.

        Returns "ready", "wait", or "timeout".  Never blocks: warmup shutdown
        and the nvidia-smi memory probe are both polled subprocesses.
        """
        now = time.monotonic()

        # 1. Warmup must be fully gone before its VRAM can be considered free.
        proc = gpu.warmup_proc
        if proc is not None:
            if proc.poll() is not None:
                gpu.warmup_proc = None
                gpu.warmup_stop_deadline = None
                _warmup_pid_file(gpu.index).unlink(missing_ok=True)
                self.log.info("GPU %d: warmup stopped.", gpu.index)
            else:
                if gpu.warmup_stop_deadline is None:
                    self._request_warmup_stop(gpu)
                elif now >= gpu.warmup_stop_deadline:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                return "wait"

        if gpu.gate_passed:
            return "ready"

        frac_required = self.config.get("gpu_ready_free_frac", 0.0)
        if not frac_required or frac_required <= 0:
            gpu.gate_passed = True
            return "ready"

        if gpu.mem_wait_deadline is None:
            gpu.mem_wait_deadline = now + self.config.get("gpu_ready_timeout", 30.0)

        if gpu.mem_probe is None:
            # Pace the probes: the loop ticks at FAST_TICK while a launch is in
            # flight, and respawning nvidia-smi 20×/s per GPU is pure waste.
            if now < gpu.mem_probe_next:
                return "wait"
            gpu.mem_probe = AsyncSmiProbe(gpu.index)
        finished, frac = gpu.mem_probe.result()
        if not finished:
            return "wait"
        gpu.mem_probe = None
        gpu.mem_probe_next = now + SMI_PROBE_INTERVAL

        if frac is None:
            # Cannot query (no nvidia-smi) → don't block dispatch.
            gpu.gate_passed = True
            return "ready"
        if frac >= frac_required:
            gpu.gate_passed = True
            return "ready"
        if now >= gpu.mem_wait_deadline:
            self.log.warning(
                "GPU %d still occupied (%.0f%% free < %.0f%% required) after %.0fs; "
                "deferring — free GPU memory or check for leaked/other processes.",
                gpu.index, frac * 100, frac_required * 100,
                self.config.get("gpu_ready_timeout", 30.0),
            )
            gpu.reset_gate()
            gpu.cooldown_until = now + GATE_COOLDOWN
            return "timeout"
        if not gpu.mem_wait_announced:
            gpu.mem_wait_announced = True
            self.log.info(
                "GPU %d: waiting for memory to release (%.0f%% free, need %.0f%%).",
                gpu.index, frac * 100, frac_required * 100,
            )
        return "wait"

    def _advance_pending_launches(self) -> None:
        if not self.pending_launches:
            return
        # Advance each gated GPU at most once per tick, even when several
        # pending launches share it (tasks_per_gpu > 1).
        gate_results: dict[int, str] = {}
        for launch in self.pending_launches:
            for gpu in launch.gpus:
                if launch.needs_gate.get(gpu.index) and gpu.index not in gate_results:
                    gate_results[gpu.index] = self._advance_gpu_gate(gpu)

        for launch in list(self.pending_launches):
            results = [
                gate_results.get(gpu.index, "ready") if launch.needs_gate.get(gpu.index) else "ready"
                for gpu in launch.gpus
            ]
            if any(result == "timeout" for result in results):
                # GPU memory never released: free the reservation; the task is
                # still in queue/ and other GPUs may pick it up immediately.
                self.pending_launches.remove(launch)
                self._release_allocation(launch.allocation)
                self.log.warning(
                    "Task %s: launch cancelled (GPU not memory-ready); requeued for "
                    "another attempt.", launch.task.get("id", "?"),
                )
            elif all(result == "ready" for result in results):
                self.pending_launches.remove(launch)
                self._start_task(launch)

    # ── Task selection ────────────────────────────────────────────────────────

    def _list_queue(self) -> list[dict]:
        return self._queue_cache.list_tasks(QUEUE_DIR)

    def _highest_ready_priority_tasks(self, done_ids: set) -> list[dict]:
        """Ready queued tasks in FIFO order for the best currently-ready priority."""
        ready: list[dict] = []
        target_priority = None
        for task in sorted_queue(self._list_queue()):
            if task.get("id") in self.allocations:
                continue   # reserved by a pending launch
            deps = task.get("depends_on") or []
            if not all(d in done_ids for d in deps):
                continue
            priority = task.get("priority", 0)
            if target_priority is None:
                target_priority = priority
            elif priority != target_priority:
                break
            ready.append(task)
        return ready

    def _gpu_count_failure(self, task: dict) -> tuple[int | None, str | None]:
        if is_cpu_task(task):
            return 0, None
        try:
            gpu_count = self._task_gpu_count(task)
        except (TypeError, ValueError):
            return None, f"invalid gpu_count: {task.get('gpu_count')!r}"
        if gpu_count < 1:
            return None, f"invalid gpu_count: {gpu_count}; must be >= 1"
        return gpu_count, None

    def _reap_dead_deps(self, failed_ids: set, known_ids: set) -> None:
        """Cascade-fail any queued task that depends on a task which has failed.

        Without this, an attack whose generation job failed would block forever.
        """
        for task in self._list_queue():
            if task.get("id") in self.allocations:
                continue
            deps = task.get("depends_on") or []
            dead = [d for d in deps if d in failed_ids]
            missing = [d for d in deps if d not in known_ids]
            if dead:
                note = "dependency failed: " + ", ".join(dead)
                self._fail_task(task, task["_path"], returncode=-2, note=note)
                self.log.warning("Task %s cascade-failed (%s).", task["id"], note)
            elif missing:
                note = "dependency missing: " + ", ".join(missing)
                self._fail_task(task, task["_path"], returncode=-2, note=note)
                self.log.warning("Task %s cascade-failed (%s).", task["id"], note)

    def _find_gpus_for_task(
        self, gpu_count: int, *, exclusive: bool, exclude: set[int] = frozenset()
    ) -> list[GPU] | None:
        """Return a deterministic GPU allocation for a task if one currently fits.

        `exclude` holds GPUs draining for a blocked whole-GPU task; backfill
        must not land on them.
        """
        if gpu_count == 0:
            return []
        now = time.monotonic()
        available = [
            g for g in self.gpus.values()
            if g.cooldown_until <= now and g.index not in exclude
        ]
        if gpu_count == 1 and not exclusive:
            candidates = [gpu for gpu in available if gpu.has_free_slot]
            if not candidates:
                return None
            return [max(candidates, key=lambda g: (g.free_slot_count, -g.index))]

        candidates = [
            gpu for gpu in sorted(available, key=lambda g: g.index)
            if gpu.is_idle
        ]
        if len(candidates) < gpu_count:
            return None
        return candidates[:gpu_count]

    def _maybe_drain_for_task(
        self, task: dict, gpu_count: int, *, exclusive: bool, draining: set[int]
    ) -> None:
        """Anti-starvation: reserve GPUs for a blocked whole-GPU task.

        Same-priority backfill normally wins over a whole-GPU task that cannot
        fit, which can starve it forever under a steady stream of shared-GPU
        work. Once the task has been ready-but-blocked for multi_gpu_drain_sec,
        the GPUs closest to idle are added to `draining` so this dispatch round
        stops backfilling them and they drain to fully idle.
        """
        task_id = task.get("id", "?")
        drain_after = self.config.get("multi_gpu_drain_sec", 0.0)
        if not exclusive or drain_after <= 0:
            return
        if gpu_count > len(self.gpus):
            return   # can never fit; draining would deadlock the whole pool
        now = time.monotonic()
        blocked_since = self._blocked_since.setdefault(task_id, now)
        if now - blocked_since < drain_after:
            return

        chosen = sorted(
            (g for g in self.gpus.values() if g.index not in draining),
            key=lambda g: (g.running_count, g.index),
        )[:gpu_count]
        draining.update(g.index for g in chosen)
        if task_id not in self._drain_announced:
            self._drain_announced.add(task_id)
            self.log.warning(
                "Exclusive task %s (%d GPU%s) blocked for %s; draining GPU(s) %s for it "
                "(backfill suspended there).",
                task_id, gpu_count, "" if gpu_count == 1 else "s",
                format_duration(now - blocked_since),
                sorted(g.index for g in chosen),
            )

    def _prune_blocked_tracking(self) -> None:
        """Drop anti-starvation state for tasks that left the queue."""
        if not self._blocked_since and not self._drain_announced:
            return
        queued_ids = {t.get("id") for t in self._list_queue()}
        for task_id in list(self._blocked_since):
            if task_id not in queued_ids:
                del self._blocked_since[task_id]
        self._drain_announced &= queued_ids

    def _select_ready_task_and_gpus(self, done_ids: set) -> tuple[dict, list[GPU]] | None:
        """Pick a ready task that fits, allowing backfill only within one priority.

        Backfill skips GPUs that are draining for a starving whole-GPU task
        (see _maybe_drain_for_task).
        """
        while True:
            ready_tasks = self._highest_ready_priority_tasks(done_ids)
            if not ready_tasks:
                return None

            failed_invalid = False
            draining: set[int] = set()
            for task in ready_tasks:
                gpu_count, failure = self._gpu_count_failure(task)
                if failure:
                    self._fail_task(task, task["_path"], returncode=-1, note=failure)
                    self.log.warning("Task %s failed before dispatch: %s.", task["id"], failure)
                    failed_invalid = True
                    continue

                exclusive = self._task_requires_exclusive_gpus(task, gpu_count)
                gpus = self._find_gpus_for_task(
                    gpu_count, exclusive=exclusive, exclude=draining
                )
                if gpus is not None:
                    self._blocked_since.pop(task["id"], None)
                    self._drain_announced.discard(task["id"])
                    return task, gpus

                self._maybe_drain_for_task(
                    task, gpu_count, exclusive=exclusive, draining=draining
                )

            if not failed_invalid:
                return None

    # ── Dispatch / launch ─────────────────────────────────────────────────────

    def _dispatch_queued(self) -> None:
        paused = is_paused()
        if paused != self._was_paused:
            self._was_paused = paused
            self.log.info("Dispatching %s.", "paused" if paused else "resumed")
        if paused:
            return

        done_ids   = task_ids_in(DONE_DIR)
        failed_ids = task_ids_in(FAILED_DIR)
        known_ids  = task_ids_in_all_state_dirs()
        self._reap_dead_deps(failed_ids, known_ids)   # cascade-fail tasks whose dep failed

        while self._running:
            selection = self._select_ready_task_and_gpus(done_ids)
            if selection is None:
                break
            task, gpus = selection
            self._begin_launch(task, gpus)
        self._prune_blocked_tracking()

    def _begin_launch(self, task: dict, gpus: list[GPU]) -> None:
        """Reserve slots for a task and queue it behind its GPUs' readiness gates."""
        gpu_indices = [gpu.index for gpu in gpus]
        allocation = Allocation(task, gpu_indices, proc=None, pid=None, log_fh=None)
        allocation.pending = True
        self.allocations[task["id"]] = allocation

        # Whether each GPU needs the idle→busy gate is decided *before*
        # reserving: a GPU with started tasks already proved itself.
        needs_gate = {gpu.index: gpu.started_count == 0 for gpu in gpus}
        self._reserve_new_allocation(allocation)
        launch = PendingLaunch(task, gpus, allocation, needs_gate)
        self.pending_launches.append(launch)
        for gpu in gpus:
            if needs_gate[gpu.index] and gpu.warmup_proc is not None:
                self._request_warmup_stop(gpu)

    def _reserve_new_allocation(self, allocation: Allocation) -> None:
        exclusive = self._task_requires_exclusive_gpus(
            allocation.task, len(allocation.gpu_indices)
        )
        for idx in allocation.gpu_indices:
            gpu = self.gpus[idx]
            if exclusive:
                slots = gpu.free_slots()
                assert len(slots) == len(gpu.slots)
            else:
                slot = gpu.free_slot()
                assert slot is not None
                slots = [slot]
            for slot in slots:
                slot.allocation = allocation

    def _start_task(self, launch: PendingLaunch) -> None:
        allocation = launch.allocation
        task      = launch.task
        task_id   = task["id"]
        script    = task["script"]
        workdir   = task.get("workdir", str(Path.home()))
        extra_env = task.get("env", {})
        task_path = task["_path"]
        gpu_indices = [gpu.index for gpu in launch.gpus]
        gpu_count = len(gpu_indices)

        log_file = LOGS_DIR / f"{task_id}.log"
        exit_code_file = self._exit_code_file(task_id)

        if not Path(script).is_file():
            self.log.error(
                "Task %s: script not found: %s – moving to failed/.", task_id, script
            )
            self._fail_task(task, task_path, returncode=-1,
                            note=f"script not found: {script}")
            self._release_allocation(allocation)
            return

        running_info = {k: v for k, v in task.items() if not k.startswith("_")}
        running_info["start_time"] = datetime.now(tz=timezone.utc).isoformat()
        running_info["gpu_count"] = gpu_count
        running_info["gpu_indices"] = gpu_indices
        if gpu_indices:
            running_info["gpu_index"] = gpu_indices[0]
        else:
            running_info["cpu"] = True
            running_info.pop("gpu_index", None)
        running_path = RUNNING_DIR / f"{task_id}.json"
        if running_path.exists():
            self.log.error("Task %s: running file already exists; leaving queued.", task_id)
            self._release_allocation(allocation)
            return
        try:
            task_path.rename(running_path)
        except FileNotFoundError:
            # Task was deleted/cleared while its launch was pending.
            self.log.info("Task %s disappeared before it could be claimed.", task_id)
            self._release_allocation(allocation)
            return
        except Exception as exc:
            self.log.error("Task %s: failed to claim queue file: %s", task_id, exc)
            self._release_allocation(allocation)
            return

        write_task(RUNNING_DIR, running_info)
        self._cleanup_exit_code(task_id)

        env = os.environ.copy()
        env.update(extra_env)
        visible_devices = ",".join(str(idx) for idx in gpu_indices)
        first_gpu = str(gpu_indices[0]) if gpu_indices else ""
        env["CUDA_VISIBLE_DEVICES"] = visible_devices
        env["GPU_SCHEDULER_TASK_ID"] = task_id
        env["GPU_SCHEDULER_GPU_COUNT"] = str(gpu_count)
        env["GPU_SCHEDULER_GPU_INDICES"] = visible_devices
        env["GPU_SCHEDULER_GPU_INDEX"] = first_gpu
        env["GPU_SCHEDULER_SCRIPT"] = script
        env["GPU_SCHEDULER_EXIT_CODE_FILE"] = str(exit_code_file)

        launcher = (
            'bash "$GPU_SCHEDULER_SCRIPT"; '
            'rc=$?; '
            'printf "%s\\n" "$rc" > "$GPU_SCHEDULER_EXIT_CODE_FILE"; '
            'exit "$rc"'
        )

        log_fh = None
        try:
            log_fh = open(log_file, "w")
            proc   = subprocess.Popen(
                ["bash", "-c", launcher],
                cwd=workdir,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as exc:
            location = "CPU" if is_cpu_task(running_info) else f"GPU(s) {gpu_indices}"
            self.log.error("%s: failed to start task %s: %s", location, task_id, exc)
            try:
                if log_fh:
                    log_fh.close()
            except Exception:
                pass
            failed_info = {k: v for k, v in running_info.items() if not k.startswith("_")}
            self._fail_task(failed_info, running_path, returncode=-1, note=str(exc))
            self._release_allocation(allocation)
            return

        running_info["pid"]        = proc.pid
        running_path = write_task(RUNNING_DIR, running_info)

        running_info["_path"] = running_path
        allocation.task = running_info
        allocation.proc = proc
        allocation.pid = proc.pid
        allocation.log_fh = log_fh
        allocation.pending = False
        self._set_timeout_deadline(allocation)

        if is_cpu_task(running_info):
            self.log.info("CPU: started task %s (PID %d, 0 GPUs requested).", task_id, proc.pid)
        else:
            self.log.info(
                "GPU(s) %s: started task %s (PID %d, %d GPU%s requested).",
                gpu_indices, task_id, proc.pid, gpu_count, "" if gpu_count == 1 else "s",
            )

    def _fail_task(self, task: dict, task_path: Path, returncode: int, note: str = "") -> None:
        self._finalize_task(task, task_path, returncode=returncode, note=note)

    def _check_running_tasks(self) -> None:
        """Collect finished tasks and free their slots."""
        for allocation in list(self.allocations.values()):
            if allocation.pending or allocation.killing:
                continue
            proc = allocation.proc
            task = allocation.task
            pid = allocation.pid or (proc.pid if proc is not None else task.get("pid"))
            if proc is not None:
                if proc.poll() is None:
                    continue
                rc = proc.returncode
                self._write_exit_code(task["id"], rc)
            else:
                if self._process_alive(pid):
                    continue
                recorded_rc = self._read_exit_code(task["id"])
                if recorded_rc is None:
                    rc = -3
                    task["failure_note"] = (
                        "process exited after dispatcher restart and no exit code was recorded"
                    )
                else:
                    rc = recorded_rc
            task_id = task["id"]
            t_path = task["_path"]

            if allocation.log_fh:
                try:
                    allocation.log_fh.close()
                except Exception:
                    pass

            self._attach_gpu_metrics(allocation)
            if rc == 0:
                self._finalize_task(task, t_path, returncode=rc)
                self.log.info("%s: task %s done.", self._allocation_location(allocation), task_id)
            else:
                self._conclude_failure(
                    task, t_path, returncode=rc,
                    note=task.get("failure_note", ""), retry_eligible=True,
                )
                self.log.info(
                    "%s: task %s FAILED (rc=%d).",
                    self._allocation_location(allocation), task_id, rc,
                )

            self._release_allocation(allocation)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _write_pid(self) -> None:
        PID_FILE.write_text(str(os.getpid()) + "\n")

    def _remove_pid(self) -> None:
        try:
            if PID_FILE.read_text().strip() == str(os.getpid()):
                PID_FILE.unlink(missing_ok=True)
        except FileNotFoundError:
            pass
        except Exception:
            self.log.warning("Could not remove pid file %s.", PID_FILE)

    def _sleep_interruptible(self, seconds: float) -> None:
        deadline = time.time() + max(0.0, seconds)
        while (
            self._running
            and not self._reload_requested
            and not self._control_requested
            and time.time() < deadline
        ):
            time.sleep(min(0.2, max(0.0, deadline - time.time())))

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGHUP,  self._handle_reload)

        self._write_pid()
        self.log.info("Dispatcher started (PID %d).", os.getpid())

        try:
            while self._running:
                if self._reload_requested:
                    self._apply_reload()

                # 1. Reap detached warmups and collect finished tasks.
                self._reap_dying_warmups()
                self._check_running_tasks()

                # 2. Enforce per-task timeouts and advance in-flight kills.
                self._check_timeouts()
                self._advance_pending_kills()

                # 3. Apply user requests before dispatching more queued work.
                if self._control_requested or any(REQUESTS_DIR.glob("*.json")):
                    self._process_dispatcher_requests()

                # 4. Reap crashed warmup processes.
                self._ensure_warmups_alive()

                # 5. Advance reserved launches through their readiness gates.
                self._advance_pending_launches()

                # 6. Dispatch queued tasks to available GPU slots, then push the
                #    new reservations through their gates (gate-free launches —
                #    CPU tasks, busy GPUs, disabled checks — start this tick).
                self._dispatch_queued()
                self._advance_pending_launches()

                # 7. Warmup goes on whatever is still idle *after* dispatch, so a
                #    busy queue no longer churns warmup start/stop between tasks.
                self._start_idle_warmups()

                self._metrics.tick(
                    time.monotonic(),
                    self.config.get("metrics_interval", 0.0),
                    self.allocations.values(),
                )
                self._gc_request_results()

                in_flight = bool(self.pending_launches or self.pending_kills)
                poll = FAST_TICK if in_flight else self.config.get("poll_interval", 1.0)
                self._sleep_interruptible(poll)

        finally:
            self.log.info("Shutting down – stopping warmup processes.")
            for gpu in self.gpus.values():
                if gpu.warmup_proc is not None:
                    self._stop_warmup_blocking(gpu)
            for proc in self._dying_warmups:
                if proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        pass
            # Running tasks are left alive; they will finish on their own.
            self._remove_pid()
            if self._lock_fh is not None:
                try:
                    fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_UN)
                    self._lock_fh.close()
                except Exception:
                    pass
            self.log.info("Dispatcher stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    Dispatcher().run()


if __name__ == "__main__":
    main()
