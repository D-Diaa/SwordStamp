#!/usr/bin/env python3
"""
warmup.py — GPU warm-up worker.

Keeps a single GPU busy with a repeated matmul so it stays at operating
temperature and peak clock speed.  The dispatcher launches one instance of
this script per idle GPU and kills it (SIGTERM) before assigning a task.

Memory target
-------------
By default the warmup occupies **80 % of currently free VRAM** so it does not
fight other processes already using the GPU.
Three square float16 matrices (A, B, C = A @ B) are allocated; their combined
size is ~80 % of free VRAM at warmup startup.

An explicit target can be passed as the first argument (in GB) to override the
80 % of free VRAM default — useful for multi-tenant machines.

Environment
-----------
CUDA_VISIBLE_DEVICES must be set by the caller so this process only sees
the one GPU it should warm.  Internally it always addresses cuda:0.

Usage
-----
    python warmup.py [target_gb]

    target_gb  – override the 80 %-of-VRAM default with a fixed GB target.
                 Pass 0 or omit to use the 80 % auto-detect behaviour.
"""

import math
import sys
import time


def _sleep_forever() -> None:
    """Fallback: GPU/torch not available – just park the process."""
    while True:
        time.sleep(60)


def _try_alloc(d: int, device) -> tuple | None:
    """Try to allocate two d×d float16 matrices.  Return (A, B) or None."""
    import torch
    try:
        A = torch.randn(d, d, dtype=torch.float16, device=device)
        B = torch.randn(d, d, dtype=torch.float16, device=device)
        return A, B
    except RuntimeError:
        return None


def main() -> None:
    # ── Parse optional override argument ─────────────────────────────────────
    override_gb: float = 0.0
    if len(sys.argv) > 1:
        try:
            override_gb = float(sys.argv[1])
        except ValueError:
            pass

    # ── Import torch ─────────────────────────────────────────────────────────
    try:
        import torch
    except ImportError:
        print("[warmup] PyTorch not available – warmup disabled", flush=True)
        _sleep_forever()
        return

    if not torch.cuda.is_available():
        print("[warmup] CUDA not available – warmup disabled", flush=True)
        _sleep_forever()
        return

    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES handles remapping

    # ── Determine target memory ───────────────────────────────────────────────
    if override_gb > 0:
        target_bytes = override_gb * (1024 ** 3)
        source_label = f"{override_gb:.1f} GB (override)"
    else:
        # 80 % of currently FREE VRAM (so we don't fight other processes)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        target_bytes = free_bytes * 0.80
        source_label = (f"{target_bytes / (1024**3):.1f} GB "
                        f"(80 % of {free_bytes / (1024**3):.1f} GB free / "
                        f"{total_bytes / (1024**3):.1f} GB total)")

    # Three d×d float16 matrices: 3 * d² * 2 bytes = target_bytes
    d = int(math.sqrt(target_bytes / 6))

    gb_alloc = 3 * d * d * 2 / (1024 ** 3)
    print(f"[warmup] Target: {source_label}", flush=True)
    print(f"[warmup] Allocating {d}×{d} float16 matrices (~{gb_alloc:.1f} GB for A+B+C)", flush=True)

    # ── Allocate (fall back to smaller sizes on OOM) ──────────────────────────
    result = _try_alloc(d, device)
    if result is None:
        for scale in (0.70, 0.60, 0.50, 0.40, 0.30):
            d2 = int(d * math.sqrt(scale))   # maintain area ratio
            gb2 = 3 * d2 * d2 * 2 / (1024 ** 3)
            print(f"[warmup] OOM – retrying with d={d2} (~{gb2:.1f} GB)", flush=True)
            result = _try_alloc(d2, device)
            if result is not None:
                d = d2
                break

    if result is None:
        print("[warmup] Could not allocate warmup tensors – warmup disabled", flush=True)
        _sleep_forever()
        return

    A, B = result
    gb_used = 3 * d * d * 2 / (1024 ** 3)
    print(f"[warmup] Running (d={d}, {gb_used:.1f} GB allocated). SIGTERM to stop.", flush=True)

    # ── Warm-up loop ──────────────────────────────────────────────────────────
    # Never exit except via SIGTERM (which Python converts to SystemExit).
    # Duty cycle: 75 % compute, 25 % idle.
    # After each matmul (duration T), sleep T/3 so compute/(compute+idle) = 75 %.
    # On OOM during the matmul, shrink matrices by 10 % and keep going.
    iteration = 0
    C = None
    while True:
        try:
            t0 = time.perf_counter()
            C = torch.mm(A, B)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            iteration += 1
            if iteration % 50 == 0:
                print(f"[warmup] iteration {iteration}", flush=True)

            time.sleep(elapsed / 3)   # 25 % idle: T / (T + T/3) = 75 %

        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                # Non-OOM runtime error – log and keep going
                print(f"[warmup] RuntimeError (non-OOM): {exc} – continuing", flush=True)
                time.sleep(1)
                continue

            # OOM during matmul: free current tensors and retry at 90 % size.
            # C may be unbound if the very first matmul OOMed.
            print(f"[warmup] OOM during matmul – shrinking matrices", flush=True)
            del A, B
            C = None
            torch.cuda.empty_cache()
            time.sleep(1)

            d = int(d * 0.90)
            if d < 1024:
                # Below this the matmul is so cheap the loop degenerates into a
                # busy-spin that warms nothing; park until the dispatcher kills us.
                print(f"[warmup] Matrices too small to be useful (d={d}) – "
                      "sleeping until task arrives", flush=True)
                _sleep_forever()
                return
            result = _try_alloc(d, device)
            if result is None:
                print(f"[warmup] Cannot reallocate at d={d} – sleeping until task arrives",
                      flush=True)
                _sleep_forever()
                return

            A, B = result
            gb_used = 3 * d * d * 2 / (1024 ** 3)
            print(f"[warmup] Resized to d={d} ({gb_used:.1f} GB)", flush=True)

        except Exception as exc:
            # Any other unexpected exception – log and keep going
            print(f"[warmup] Unexpected error: {exc} – sleeping 5 s", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
