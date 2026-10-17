from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

SettingsType = TypeVar("SettingsType")


class ConfigError(ValueError):
    pass


def _unwrap_optional(*, annotation: Any) -> tuple[Any, bool]:
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        arguments = [argument for argument in typing.get_args(annotation) if argument is not type(None)]
        if len(arguments) == 1:
            return arguments[0], True
    return annotation, False


def _convert_value(
    *,
    annotation: Any,
    value: Any,
    location: str,
) -> Any:
    inner, optional = _unwrap_optional(annotation=annotation)
    if value is None:
        if optional:
            return None
        raise ConfigError(f"{location} must not be null")
    if dataclasses.is_dataclass(inner):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{location} must be a mapping")
        return build_dataclass(
            settings_type=inner,
            values=value,
            location=location,
        )
    origin = typing.get_origin(inner)
    if origin is tuple:
        item_types = typing.get_args(inner)
        item_type = item_types[0] if item_types else Any
        return tuple(
            _convert_value(
                annotation=item_type,
                value=item,
                location=f"{location}[{index}]",
            )
            for index, item in enumerate(value)
        )
    if origin in (dict, Mapping) or inner in (dict, Mapping):
        return dict(value)
    if inner is float and isinstance(value, int):
        return float(value)
    return value


def build_dataclass(
    *,
    settings_type: type[SettingsType],
    values: Mapping[str, Any],
    location: str = "config",
) -> SettingsType:
    known = {field.name: field for field in dataclasses.fields(settings_type)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ConfigError(f"Unknown keys in {location}: {unknown}")
    hints = typing.get_type_hints(settings_type)
    arguments = {
        name: _convert_value(
            annotation=hints[name],
            value=value,
            location=f"{location}.{name}",
        )
        for name, value in values.items()
    }
    missing = sorted(
        name
        for name, field in known.items()
        if name not in arguments
        and field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
    )
    if missing:
        raise ConfigError(f"Missing required keys in {location}: {missing}")
    return settings_type(**arguments)


def merge_overrides(
    *,
    values: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(values)
    for dotted_key, value in overrides.items():
        cursor = merged
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor[part] = dict(cursor.get(part) or {})
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return merged


def load_settings(
    *,
    settings_type: type[SettingsType],
    config_path: Path,
    overrides: Mapping[str, Any] | None = None,
) -> SettingsType:
    import yaml

    values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    merged = merge_overrides(
        values=values,
        overrides=overrides or {},
    )
    return build_dataclass(
        settings_type=settings_type,
        values=merged,
        location=str(config_path),
    )


def settings_to_dict(*, settings: Any) -> dict[str, Any]:
    def convert(*, value: Any) -> Any:
        if dataclasses.is_dataclass(value):
            return {field.name: convert(value=getattr(value, field.name)) for field in dataclasses.fields(value)}
        if isinstance(value, tuple):
            return [convert(value=item) for item in value]
        if isinstance(value, Mapping):
            return {key: convert(value=item) for key, item in value.items()}
        return value

    return convert(value=settings)


def parse_override_pairs(*, pairs: typing.Sequence[str]) -> dict[str, Any]:
    import yaml

    overrides: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(f"Override must look like key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        overrides[key.strip()] = yaml.safe_load(raw)
    return overrides
