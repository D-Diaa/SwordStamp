"""Data model shared by all segmentation backends."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:
    """A segment with normalized scoring text and lossless display text."""

    type: str = "sentence"
    normalized: str = ""
    display: str = ""


def _main():
    """Run unit-model smoke tests."""
    u = Unit()
    assert u.type == "sentence" and u.normalized == "" and u.display == ""
    print(f"default Unit: {u!r}")

    sent = Unit(
        "sentence",
        "the cat sat on the mat.",
        "The cat sat on the mat.",
    )
    assert sent.normalized.islower()
    print(f"sentence Unit: normalized={sent.normalized!r}")

    span = Unit("semspan", "the cat sat.", "The cat sat.")
    assert span.type == "semspan"
    print(f"semantic-span Unit: {span.normalized!r}")

    try:
        sent.normalized = "mutate"  # type: ignore[misc]
        raise AssertionError("expected FrozenInstanceError")
    except Exception as exc:
        print(f"frozen guard: {type(exc).__name__}")

    print("unit smoke ok")


if __name__ == "__main__":
    _main()
