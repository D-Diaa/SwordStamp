"""
common.py — Shared constants, paths, and utilities for gpu-scheduler.

All components import from here so that paths are defined in one place.
"""

import fcntl
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
import random
import string

# ── Paths ─────────────────────────────────────────────────────────────────────

SCHEDULER_HOME = Path.home() / ".gpu-scheduler"
QUEUE_DIR      = SCHEDULER_HOME / "queue"
RUNNING_DIR    = SCHEDULER_HOME / "running"
DONE_DIR       = SCHEDULER_HOME / "done"
FAILED_DIR     = SCHEDULER_HOME / "failed"
LOGS_DIR       = SCHEDULER_HOME / "logs"
EXIT_CODES_DIR = SCHEDULER_HOME / "exit-codes"
REQUESTS_DIR   = SCHEDULER_HOME / "requests"
REQUEST_RESULTS_DIR = SCHEDULER_HOME / "request-results"
LIB_DIR        = SCHEDULER_HOME / "lib"
CONFIG_FILE    = SCHEDULER_HOME / "config.json"
PID_FILE       = SCHEDULER_HOME / "dispatcher.pid"
LOCK_FILE      = SCHEDULER_HOME / "dispatcher.lock"
DISPATCHER_LOG = SCHEDULER_HOME / "dispatcher.log"
PAUSED_FILE    = SCHEDULER_HOME / "paused"

ALL_DIRS = [
    QUEUE_DIR,
    RUNNING_DIR,
    DONE_DIR,
    FAILED_DIR,
    LOGS_DIR,
    EXIT_CODES_DIR,
    REQUESTS_DIR,
    REQUEST_RESULTS_DIR,
    LIB_DIR,
]

# ── Default config ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "gpu_indices":         [],    # List of GPU indices owned by the scheduler
    "warmup_gb":           0.0,   # Warmup target GB; 0 = auto (80 % of free GPU VRAM)
    "warmup_enabled":      True,  # Keep idle GPUs warm with a background matmul
    "poll_interval":       1.0,   # Seconds between dispatcher loop iterations
    "tasks_per_gpu":       1,     # Max concurrent tasks per GPU (slots per GPU)
    # Before dispatching a task, require the GPU's free VRAM to be at least this
    # fraction of total (after stopping warmup) — i.e. confirm memory is actually
    # released, so a vLLM task can't OOM on a still-occupied GPU. 0 disables the check.
    "gpu_ready_free_frac": 0.90,
    "gpu_ready_timeout":   30.0,  # Max seconds to wait for that memory to release.
    # Seconds between nvidia-smi samples used to record per-task GPU telemetry
    # (peak memory / average utilization) into done/failed task JSON. 0 disables.
    "metrics_interval":    10.0,
    # Anti-starvation: once a ready whole-GPU task has been blocked this many
    # seconds, the GPUs it needs stop accepting same-priority backfill and
    # drain to idle so the task can launch. 0 disables draining (a multi-GPU
    # or --exclusive task can then starve behind shared-GPU tasks).
    "multi_gpu_drain_sec": 600.0,
}

DISPATCHER_STOP_TIMEOUT = 60.0

# ── Task ID ────────────────────────────────────────────────────────────────────

def make_task_id() -> str:
    """Generate a unique, sortable task ID: YYYYMMDD_HHMMSS_XXXXXX."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{ts}_{suffix}"

def iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()

def task_id_from_arg(arg: str) -> str:
    """Accept either a full task ID or just the random suffix part."""
    return arg.strip()

# ── Durations ──────────────────────────────────────────────────────────────────

_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([dhms]?)", re.IGNORECASE)
_DURATION_UNITS = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}

def parse_duration(text: str) -> float:
    """Parse '90', '90s', '15m', '2h', '1h30m', '1.5h' → seconds."""
    text = str(text).strip()
    if not text:
        raise ValueError("empty duration")
    pos = 0
    total = 0.0
    for match in _DURATION_RE.finditer(text):
        if match.start() != pos:
            break
        total += float(match.group(1)) * _DURATION_UNITS[match.group(2).lower()]
        pos = match.end()
    if pos != len(text) or total <= 0:
        raise ValueError(f"invalid duration: {text!r} (try 90, 90s, 15m, 2h, 1h30m)")
    return total

def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m" + (f"{seconds % 60}s" if seconds % 60 else "")
    return f"{seconds // 3600}h" + (f"{(seconds % 3600) // 60}m" if seconds % 3600 else "")

# ── Pause / resume ─────────────────────────────────────────────────────────────

def is_paused() -> bool:
    return PAUSED_FILE.exists()

def set_paused(paused: bool) -> None:
    if paused:
        SCHEDULER_HOME.mkdir(parents=True, exist_ok=True)
        PAUSED_FILE.write_text(iso_now() + "\n")
    else:
        PAUSED_FILE.unlink(missing_ok=True)

# ── Config I/O ─────────────────────────────────────────────────────────────────

def load_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Config not found: {CONFIG_FILE}\n"
            "Run: gpu-scheduler init --gpus 0,1,2"
        )
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    # Merge defaults for any missing keys
    for k, v in DEFAULT_CONFIG.items():
        cfg.setdefault(k, v)
    return cfg

def validate_config(cfg: dict) -> dict:
    """Validate and normalize scheduler config values."""
    try:
        gpu_indices = [int(x) for x in cfg.get("gpu_indices", [])]
    except (TypeError, ValueError):
        raise ValueError("gpu_indices must be a list of integers")
    if len(gpu_indices) != len(set(gpu_indices)):
        raise ValueError("gpu_indices must not contain duplicates")
    if any(idx < 0 for idx in gpu_indices):
        raise ValueError("gpu_indices must be non-negative integers")

    try:
        poll_interval = float(cfg.get("poll_interval", DEFAULT_CONFIG["poll_interval"]))
        warmup_gb = float(cfg.get("warmup_gb", DEFAULT_CONFIG["warmup_gb"]))
        tasks_per_gpu = int(cfg.get("tasks_per_gpu", DEFAULT_CONFIG["tasks_per_gpu"]))
        ready_frac = float(cfg.get("gpu_ready_free_frac", DEFAULT_CONFIG["gpu_ready_free_frac"]))
        ready_timeout = float(cfg.get("gpu_ready_timeout", DEFAULT_CONFIG["gpu_ready_timeout"]))
        metrics_interval = float(cfg.get("metrics_interval", DEFAULT_CONFIG["metrics_interval"]))
        drain_sec = float(cfg.get("multi_gpu_drain_sec", DEFAULT_CONFIG["multi_gpu_drain_sec"]))
    except (TypeError, ValueError):
        raise ValueError("numeric config values must be valid numbers")

    if poll_interval <= 0:
        raise ValueError("poll_interval must be > 0")
    if warmup_gb < 0:
        raise ValueError("warmup_gb must be >= 0")
    if tasks_per_gpu < 1:
        raise ValueError("tasks_per_gpu must be >= 1")
    if not 0 <= ready_frac <= 1:
        raise ValueError("gpu_ready_free_frac must be between 0 and 1")
    if ready_timeout < 0:
        raise ValueError("gpu_ready_timeout must be >= 0")
    if metrics_interval < 0:
        raise ValueError("metrics_interval must be >= 0")
    if drain_sec < 0:
        raise ValueError("multi_gpu_drain_sec must be >= 0")

    normalized = dict(cfg)
    normalized["gpu_indices"] = gpu_indices
    normalized["poll_interval"] = poll_interval
    normalized["warmup_gb"] = warmup_gb
    normalized["warmup_enabled"] = bool(cfg.get("warmup_enabled", DEFAULT_CONFIG["warmup_enabled"]))
    normalized["tasks_per_gpu"] = tasks_per_gpu
    normalized["gpu_ready_free_frac"] = ready_frac
    normalized["gpu_ready_timeout"] = ready_timeout
    normalized["metrics_interval"] = metrics_interval
    normalized["multi_gpu_drain_sec"] = drain_sec
    return normalized

def save_config(cfg: dict) -> None:
    cfg = validate_config(cfg)
    SCHEDULER_HOME.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

# ── Dispatcher PID ─────────────────────────────────────────────────────────────

def read_dispatcher_pid() -> int | None:
    """Return the dispatcher's PID if the pidfile exists, else None."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        # Verify the process is actually alive
        os.kill(pid, 0)
        cmdline = Path(f"/proc/{pid}/cmdline")
        if cmdline.exists():
            try:
                cmd = cmdline.read_bytes().replace(b"\x00", b" ").decode(errors="ignore")
                if "dispatcher.py" not in cmd:
                    return None
            except Exception:
                return None
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None

def is_dispatcher_running() -> bool:
    return read_dispatcher_pid() is not None

# ── Task file I/O ──────────────────────────────────────────────────────────────

def write_task(directory: Path, task: dict) -> Path:
    """Write task dict to <directory>/<task_id>.json atomically."""
    directory.mkdir(parents=True, exist_ok=True)
    task_id = task["id"]
    path = directory / f"{task_id}.json"
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    tmp = directory / f".{task_id}.{os.getpid()}.{suffix}.tmp"
    tmp.write_text(json.dumps(task, indent=2) + "\n")
    tmp.replace(path)
    return path

def read_task(path: Path) -> dict | None:
    """Read and parse a task JSON file; return None on error."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def list_tasks(directory: Path) -> list[dict]:
    """Return all valid task dicts in a directory, unsorted."""
    tasks = []
    for p in directory.glob("*.json"):
        t = read_task(p)
        if t:
            t["_path"] = p
            tasks.append(t)
    return tasks


class TaskDirCache:
    """mtime+size-keyed cache of parsed task JSON files.

    State dirs like done/ grow without bound; re-parsing every file each poll
    is what made frequent status refreshes expensive. Unchanged files are
    served from memory; callers get shallow copies so cached entries stay
    pristine.
    """

    def __init__(self):
        self._cache: dict[str, tuple[int, int, dict]] = {}

    def list_tasks(self, directory: Path) -> list[dict]:
        tasks: list[dict] = []
        seen: set[str] = set()
        prefix = str(directory) + os.sep
        try:
            entries = list(os.scandir(directory))
        except OSError:
            return tasks
        for entry in entries:
            if not entry.name.endswith(".json") or entry.name.startswith("."):
                continue
            try:
                st = entry.stat()
            except OSError:
                continue
            seen.add(entry.path)
            cached = self._cache.get(entry.path)
            if cached and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
                task = cached[2]
            else:
                task = read_task(Path(entry.path))
                if not task or "id" not in task:
                    continue
                self._cache[entry.path] = (st.st_mtime_ns, st.st_size, task)
            t = dict(task)
            t["_path"] = Path(entry.path)
            tasks.append(t)
        # Evict entries for files deleted from this directory.
        for key in [k for k in self._cache if k.startswith(prefix) and k not in seen]:
            del self._cache[key]
        return tasks

def sorted_queue(tasks: list[dict]) -> list[dict]:
    """Sort tasks FIFO within each priority bucket (lower priority int = runs first).

    Task ID is the final tie-break so ordering stays deterministic when two
    tasks share an enqueue timestamp (e.g. array siblings).
    """
    return sorted(
        tasks,
        key=lambda t: (t.get("priority", 0), t.get("enqueue_time", ""), t.get("id", "")),
    )

def is_cpu_task(task: dict) -> bool:
    """Return True for tasks explicitly queued to run without a GPU allocation."""
    return task.get("cpu") is True

def task_ids_in(directory: Path) -> set:
    """Fast set of task IDs present in a state dir (filename stems, no JSON parse)."""
    return {p.stem for p in directory.glob("*.json")}

def task_ids_in_all_state_dirs() -> set:
    ids = set()
    for directory in (QUEUE_DIR, RUNNING_DIR, DONE_DIR, FAILED_DIR):
        ids.update(task_ids_in(directory))
    return ids

def resolve_task_id_prefix(task_id: str, directories=None) -> str:
    """Resolve a full task ID or unique prefix across scheduler state directories."""
    task_id = task_id.strip()
    if not task_id:
        raise ValueError("empty task ID")
    if directories is None:
        directories = (QUEUE_DIR, RUNNING_DIR, DONE_DIR, FAILED_DIR)
    matches = sorted(
        p.stem
        for directory in directories
        for p in directory.glob("*.json")
        if p.stem.startswith(task_id)
    )
    if not matches:
        raise LookupError(f"task not found: {task_id}")
    unique = sorted(set(matches))
    if len(unique) > 1:
        raise ValueError(f"ambiguous task prefix {task_id!r}: {', '.join(unique)}")
    return unique[0]

# ── Task state actions ────────────────────────────────────────────────────────

QUEUE_TASK_KEYS = (
    "id",
    "script",
    "workdir",
    "env",
    "priority",
    "enqueue_time",
    "depends_on",
    "gpu_count",
    "cpu",
    "exclusive",
    "timeout",
    "max_retries",
    "attempt",
    "last_failure",
)


def task_to_queue_record(task: dict, *, fresh_enqueue_time: bool = False) -> dict:
    """Return only the fields that belong in queue/ for a rerun/requeue."""
    queued = {k: task[k] for k in QUEUE_TASK_KEYS if k in task}
    if fresh_enqueue_time or "enqueue_time" not in queued:
        queued["enqueue_time"] = iso_now()
    return queued


def _cleanup_exit_code(task_id: str) -> None:
    (EXIT_CODES_DIR / f"{task_id}.rc").unlink(missing_ok=True)


def retry_failed_task(task_ref: str) -> tuple[str, dict]:
    """Move a failed task back to queue/ with a fresh enqueue timestamp."""
    task_id = resolve_task_id_prefix(task_ref, directories=(FAILED_DIR,))
    failed_path = FAILED_DIR / f"{task_id}.json"
    task = read_task(failed_path)
    if not task:
        raise LookupError(f"failed task metadata unreadable: {task_id}")

    queue_path = QUEUE_DIR / f"{task_id}.json"
    if queue_path.exists():
        raise FileExistsError(f"task already exists in queue: {task_id}")

    queued = task_to_queue_record(task, fresh_enqueue_time=True)
    write_task(QUEUE_DIR, queued)
    failed_path.unlink(missing_ok=True)
    _cleanup_exit_code(task_id)
    return task_id, queued


def set_queued_task_priority(task_ref: str, priority: int) -> tuple[str, int, int]:
    """Update a queued task's priority, preserving its FIFO enqueue timestamp."""
    task_id = resolve_task_id_prefix(task_ref, directories=(QUEUE_DIR,))
    queue_path = QUEUE_DIR / f"{task_id}.json"
    task = read_task(queue_path)
    if not task:
        raise LookupError(f"queued task metadata unreadable: {task_id}")

    old_priority = int(task.get("priority", 0))
    task = {k: v for k, v in task.items() if not k.startswith("_")}
    task["priority"] = int(priority)
    write_task(QUEUE_DIR, task)
    return task_id, old_priority, int(priority)


def _process_alive(pid: int | None) -> bool:
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
            after_name = stat.read_text().rsplit(")", 1)[1].strip()
            if after_name.startswith("Z"):
                return False
        except Exception:
            pass
    return True


def _proc_ppid_map() -> dict[int, int]:
    """Return a best-effort pid -> parent-pid map from /proc."""
    by_pid: dict[int, int] = {}
    proc_dir = Path("/proc")
    if not proc_dir.exists():
        return by_pid
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            after_name = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            by_pid[int(entry.name)] = int(after_name[1])
        except Exception:
            continue
    return by_pid


def _process_tree_pids(root_pid: int, known_pids: set[int] | None = None) -> set[int]:
    """Return root_pid and all known descendants visible in /proc."""
    roots = {int(root_pid)}
    if known_pids:
        roots.update(int(pid) for pid in known_pids if pid)

    by_parent: dict[int, list[int]] = {}
    for pid, ppid in _proc_ppid_map().items():
        by_parent.setdefault(ppid, []).append(pid)

    tree = set(roots)
    frontier = list(roots)
    while frontier:
        parent = frontier.pop()
        for child in by_parent.get(parent, []):
            if child not in tree:
                tree.add(child)
                frontier.append(child)
    return tree


def _alive_processes(pids: set[int]) -> set[int]:
    return {pid for pid in pids if _process_alive(pid)}


def _signal_process_tree(root_pid: int, pids: set[int], sig: int) -> None:
    """Signal a process tree, using process groups created inside that tree too."""
    current_pid = os.getpid()
    current_pgid = os.getpgrp()

    pgids: set[int] = set()
    for pid in pids:
        if pid == current_pid:
            continue
        try:
            pgid = os.getpgid(pid)
        except (ProcessLookupError, OSError):
            continue
        # The scheduler starts task roots in their own session, so the root
        # process group is the normal way to catch subprocesses. If a child
        # created another group where the group leader is still in this tree,
        # signal that group as well.
        if pid == root_pid or pgid in pids:
            pgids.add(pgid)

    for pgid in sorted(pgids):
        if pgid == current_pgid:
            continue
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, OSError):
            pass

    for pid in sorted(pids, reverse=True):
        if pid == current_pid:
            continue
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, OSError):
            pass


def kill_process_tree(pid: int, *, term_timeout: float = 8.0) -> None:
    """Terminate a process and descendants, escalating to SIGKILL after timeout."""
    import signal

    if not pid:
        return

    known = _process_tree_pids(int(pid))
    if not _alive_processes(known):
        return

    _signal_process_tree(int(pid), known, signal.SIGTERM)
    deadline = time.time() + max(0.0, term_timeout)
    while time.time() < deadline:
        known = _process_tree_pids(int(pid), known)
        alive = _alive_processes(known)
        if not alive:
            return
        _signal_process_tree(int(pid), alive, signal.SIGTERM)
        time.sleep(0.2)

    known = _process_tree_pids(int(pid), known)
    alive = _alive_processes(known)
    if alive:
        _signal_process_tree(int(pid), alive, signal.SIGKILL)
        kill_deadline = time.time() + 2.0
        while time.time() < kill_deadline:
            known = _process_tree_pids(int(pid), known)
            if not _alive_processes(known):
                return
            time.sleep(0.1)


def kill_proc_group(pid: int, *, term_timeout: float = 8.0) -> None:
    """Backward-compatible wrapper for process-tree termination."""
    kill_process_tree(pid, term_timeout=term_timeout)


# Public aliases so the dispatcher can drive tree termination incrementally
# (one signalling pass per loop tick) instead of blocking in kill_process_tree.
process_tree_pids = _process_tree_pids
signal_process_tree = _signal_process_tree
alive_processes = _alive_processes


def _failed_without_returncode_record(task: dict, note: str = "") -> dict:
    result = {k: v for k, v in task.items() if not k.startswith("_")}
    result["end_time"] = iso_now()
    result.pop("returncode", None)
    if note:
        result["failure_note"] = note
    return result


def _resolve_task_file_prefix(task_ref: str, directories) -> tuple[str, Path]:
    task_ref = task_ref.strip()
    if not task_ref:
        raise ValueError("empty task ID")

    matches: list[Path] = []
    for directory in directories:
        matches.extend(p for p in directory.glob("*.json") if p.stem.startswith(task_ref))

    if not matches:
        raise LookupError(f"task not found: {task_ref}")

    ids = sorted({path.stem for path in matches})
    if len(ids) > 1:
        raise ValueError(f"ambiguous task prefix {task_ref!r}: {', '.join(ids)}")

    for path in matches:
        if path.stem == ids[0]:
            return path.parent.name, path

    raise LookupError(f"task not found: {task_ref}")


def kill_task_direct(task_ref: str, *, note: str | None = None) -> tuple[str, str, dict]:
    """Move a queued/running task to failed/ without recording a return code.

    This is intended for use only while the dispatcher is stopped. When it is
    running, submit a dispatcher request so the action is serialized with the
    scheduling loop.
    """
    state, path = _resolve_task_file_prefix(task_ref, directories=(RUNNING_DIR, QUEUE_DIR))
    task = read_task(path)
    if not task:
        raise LookupError(f"{state} task metadata unreadable: {path.stem}")

    if state == RUNNING_DIR.name:
        pid = task.get("pid")
        if pid:
            kill_process_tree(int(pid))

    if note is None:
        if state == QUEUE_DIR.name:
            note = "killed before start via gpu-scheduler kill"
        else:
            note = "killed via gpu-scheduler kill"

    result = _failed_without_returncode_record(task, note=note)
    write_task(FAILED_DIR, result)
    path.unlink(missing_ok=True)
    _cleanup_exit_code(task["id"])
    return task["id"], state, result


def preempt_running_task_direct(task_ref: str, *, fresh_enqueue_time: bool = True) -> tuple[str, dict]:
    """Kill a recorded running task and move it back to queue/.

    This is intended for use only while the dispatcher is stopped. When it is
    running, submit a dispatcher request so the action is serialized with the
    scheduling loop.
    """
    task_id = resolve_task_id_prefix(task_ref, directories=(RUNNING_DIR,))
    running_path = RUNNING_DIR / f"{task_id}.json"
    task = read_task(running_path)
    if not task:
        raise LookupError(f"running task metadata unreadable: {task_id}")

    queue_path = QUEUE_DIR / f"{task_id}.json"
    if queue_path.exists():
        raise FileExistsError(f"task already exists in queue: {task_id}")

    pid = task.get("pid")
    if pid:
        kill_proc_group(int(pid))

    queued = task_to_queue_record(task, fresh_enqueue_time=fresh_enqueue_time)
    write_task(QUEUE_DIR, queued)
    running_path.unlink(missing_ok=True)
    _cleanup_exit_code(task_id)
    return task_id, queued


def submit_dispatcher_request(action: str, **payload) -> str:
    """Write a request for the live dispatcher and return the request ID."""
    request_id = make_task_id()
    request = {
        "id": request_id,
        "action": action,
        "created_at": iso_now(),
        **payload,
    }
    write_task(REQUESTS_DIR, request)
    return request_id


def write_dispatcher_request_result(
    request_id: str,
    *,
    ok: bool,
    message: str,
    **payload,
) -> None:
    result = {
        "id": request_id,
        "ok": bool(ok),
        "message": message,
        "created_at": iso_now(),
        **payload,
    }
    write_task(REQUEST_RESULTS_DIR, result)


def wait_dispatcher_request_result(request_id: str, timeout: float = 10.0) -> dict | None:
    deadline = time.time() + max(0.0, timeout)
    result_path = REQUEST_RESULTS_DIR / f"{request_id}.json"
    while time.time() < deadline:
        result = read_task(result_path)
        if result:
            return result
        time.sleep(0.05)
    return read_task(result_path)


def cleanup_dispatcher_request_result(request_id: str) -> None:
    (REQUEST_RESULTS_DIR / f"{request_id}.json").unlink(missing_ok=True)


def signal_dispatcher() -> bool:
    """Poke the dispatcher after writing a request or queue action."""
    pid = read_dispatcher_pid()
    if pid is None:
        return False
    import signal

    try:
        os.kill(pid, signal.SIGHUP)
        return True
    except ProcessLookupError:
        return False

# ── ANSI colours for terminal output ──────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
GREY   = "\033[90m"

def colour(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET

def die(msg: str) -> None:
    import sys
    print(colour(f"error: {msg}", RED), file=sys.stderr)
    sys.exit(1)
