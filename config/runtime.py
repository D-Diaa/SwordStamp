DEFAULT_VLLM_UTILIZATION = 0.9


def vllm_gpu_memory_utilization(configured: float | None = None) -> float:
    """Resolve the configured vLLM GPU-memory fraction."""
    return DEFAULT_VLLM_UTILIZATION if configured is None else float(configured)


def _main():
    """Run a configuration-resolution smoke test."""
    v = vllm_gpu_memory_utilization()
    assert v == DEFAULT_VLLM_UTILIZATION, v
    print(f"default: {v}")

    v2 = vllm_gpu_memory_utilization(0.7)
    assert v2 == 0.7, v2
    print(f"configured=0.7 → {v2}")

    print("runtime smoke ok")


if __name__ == "__main__":
    _main()
