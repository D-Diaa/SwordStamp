"""Load layered application configuration."""

from __future__ import annotations

import dataclasses
import os
import types
import typing
from typing import Any

import yaml

from config.schema import AppConfig

ENV_PREFIX = "SEMSTAMP"
_ENV_SEP = "__"


def deep_merge(base: dict, override: dict | None) -> dict:
    """Merge ``override`` into ``base`` in place."""
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def set_dotted(data: dict, dotted_key: str, value: Any) -> None:
    """Set a value using a dotted key."""
    keys = [k for k in dotted_key.split(".") if k]
    if not keys:
        raise ValueError(f"Empty config key in override: {dotted_key!r}")
    cursor = data
    for key in keys[:-1]:
        nxt = cursor.setdefault(key, {})
        if not isinstance(nxt, dict):
            raise ValueError(f"Cannot set {dotted_key!r}: {key!r} is not a mapping")
        cursor = nxt
    cursor[keys[-1]] = value


def parse_overrides(pairs: list[str] | None) -> dict:
    """Parse dotted command-line overrides."""
    data: dict = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        set_dotted(data, key.strip(), yaml.safe_load(raw))
    return data


def env_overlay(prefix: str = ENV_PREFIX) -> dict:
    """Read prefixed environment variables into a nested mapping."""
    data: dict = {}
    head = prefix + _ENV_SEP
    for name, value in os.environ.items():
        if not name.startswith(head):
            continue
        dotted = name[len(head):].lower().replace(_ENV_SEP, ".")
        set_dotted(data, dotted, yaml.safe_load(value))
    return data


def _strip_optional(hint):
    """Remove ``None`` from an optional type hint."""
    origin = typing.get_origin(hint)
    if origin is typing.Union or origin is getattr(types, "UnionType", ()):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return hint


def _coerce(hint, value):
    """Coerce a value to its declared scalar type."""
    if value is None:
        return None
    origin = typing.get_origin(hint)
    if origin is list:
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return list(value)
    if hint is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if hint is int:
        return int(value)
    if hint is float:
        return float(value)
    if hint is str:
        return str(value)
    return value


def from_dict(cls, data: dict | None):
    """Build nested dataclasses and reject unknown keys."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a mapping for {cls.__name__}, got {type(data).__name__}"
        )
    hints = typing.get_type_hints(cls)
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - field_names)
    if unknown:
        raise ValueError(f"Unknown config key(s) for {cls.__name__}: {unknown}")

    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        hint = _strip_optional(hints[f.name])
        if dataclasses.is_dataclass(hint):
            kwargs[f.name] = from_dict(hint, data[f.name])
        else:
            kwargs[f.name] = _coerce(hint, data[f.name])
    return cls(**kwargs)


def load_config(
    config_paths: list[str] | None = None,
    overrides: list[str] | None = None,
    *,
    env_prefix: str | None = ENV_PREFIX,
) -> AppConfig:
    """Load config with precedence YAML, environment, then overrides."""
    data: dict = {}
    for path in config_paths or []:
        with open(path) as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file {path!r} must contain a mapping at top level")
        deep_merge(data, loaded)
    if env_prefix:
        deep_merge(data, env_overlay(env_prefix))
    deep_merge(data, parse_overrides(overrides))
    return from_dict(AppConfig, data)


def to_dict(cfg: AppConfig) -> dict:
    return dataclasses.asdict(cfg)


def dump_config(cfg: AppConfig, path: str) -> None:
    """Write a resolved config as YAML."""
    with open(path, "w") as handle:
        yaml.safe_dump(to_dict(cfg), handle, sort_keys=False, default_flow_style=False)


def _main():
    """Run a config round-trip smoke test."""
    import os
    import tempfile

    preset_yaml = (
        "watermark:\n"
        "  sp_dim: 16\n"
        "  lmbd: 0.5\n"
        "generation:\n"
        "  backend: hf\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(preset_yaml)
        yaml_path = f.name

    dump_path = yaml_path + ".out.yaml"
    try:
        cfg = load_config([yaml_path], overrides=["watermark.lmbd=0.3"])
        assert cfg.watermark.sp_dim == 16
        assert cfg.watermark.lmbd == 0.3
        assert cfg.generation.backend == "hf"
        print(f"loaded: sp_dim={cfg.watermark.sp_dim}, lmbd={cfg.watermark.lmbd}, "
              f"backend={cfg.generation.backend!r}")

        dump_config(cfg, dump_path)
        cfg2 = load_config([dump_path])
        assert cfg2.watermark.sp_dim == cfg.watermark.sp_dim
        assert cfg2.watermark.lmbd == cfg.watermark.lmbd
        print(f"round-trip: sp_dim={cfg2.watermark.sp_dim}, lmbd={cfg2.watermark.lmbd}")
    finally:
        os.unlink(yaml_path)
        if os.path.exists(dump_path):
            os.unlink(dump_path)

    print("loader smoke ok")


if __name__ == "__main__":
    _main()
