"""
CLI with TOML/YAML/JSON config file support.

Drop-in replacement for tyro.cli with config file support:
    # Instead of:
    from tyro import cli
    # Use:
    from pydantic_config import cli

Usage:
    from pydantic_config import cli, BaseConfig

    class Config(BaseConfig):
        lr: float = 1e-4
        batch_size: int = 32

    config = cli(Config)

Supports loading config files with @ syntax:
    python train.py @ config.toml --lr 1e-3
    python train.py --model @ model.toml --data @ data.toml
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import os
import shutil
import sys
import types
from typing import Any, Literal, Optional, TypeVar, Union, get_args, get_origin, overload

import tyro
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

T = TypeVar("T")


def _coerce_str_value(v: str) -> bool | int | float | str:
    """Coerce a single string to bool, int, float, or leave as str."""
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    try:
        int_val = int(v)
        if str(int_val) == v:
            return int_val
    except ValueError:
        pass
    try:
        float_val = float(v)
        if str(float_val) == v:
            return float_val
    except ValueError:
        pass
    return v


def _coerce_dict_values(d: dict) -> dict:
    """Coerce all-string dict values to proper Python types.

    Only runs when every value is a string (i.e. from CLI parsing).
    TOML/programmatic dicts already have proper types and pass through unchanged.
    """
    if not d or not all(isinstance(v, str) for v in d.values()):
        return d
    return {k: _coerce_str_value(v) for k, v in d.items()}


def _is_dict_annotation(annotation: type) -> bool:
    """Check if an annotation is a dict type (bare ``dict`` or ``dict[K, V]``)."""
    if hasattr(annotation, "__metadata__"):
        annotation = get_args(annotation)[0]
    return annotation is dict or get_origin(annotation) is dict


class BaseConfig(BaseModel):
    """Base configuration class with strict validation (extra fields forbidden)."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _none_str_to_none(cls, data: dict) -> dict:
        """Convert ``"None"`` string values to ``None`` so TOML files can express null."""
        if not isinstance(data, dict):
            return data
        for key, value in data.items():
            if value == "None":
                data[key] = None
        return data

    @model_validator(mode="before")
    @classmethod
    def _coerce_dict_str_values(cls, data: dict) -> dict:
        """Coerce string values in dict-typed fields back to proper Python types.

        tyro parses untyped dict values as strings. This detects dict fields
        whose values are all strings (i.e. from CLI parsing) and converts
        them back to int/float/bool.
        """
        if not isinstance(data, dict):
            return data
        for field_name, field_info in cls.model_fields.items():
            if _is_dict_annotation(field_info.annotation) and field_name in data:
                val = data[field_name]
                if isinstance(val, dict):
                    data[field_name] = _coerce_dict_values(val)
        return data

    @model_validator(mode="before")
    @classmethod
    def _default_discriminator_types(cls, data: dict) -> dict:
        """For discriminated-union fields whose default carries a ``type`` tag, inject it when missing."""
        if not isinstance(data, dict):
            return data
        for field_name, field_info in cls.model_fields.items():
            val = data.get(field_name)
            if isinstance(val, dict) and "type" not in val:
                default = field_info.default
                if isinstance(default, BaseModel) and hasattr(default, "type"):
                    val["type"] = default.type
        return data


CONFIG_FILE_SIGN = "@"


# ANSI color codes
_RESET = "\033[0m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_BRIGHT_RED = "\033[91m"


def _supports_color() -> bool:
    """Check if the terminal supports ANSI colors."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not hasattr(sys.stderr, "isatty"):
        return False
    if not sys.stderr.isatty():
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return True


def _colorize(text: str, *codes: str) -> str:
    """Apply ANSI color codes to text if colors are supported."""
    if not _supports_color():
        return text
    return "".join(codes) + text + _RESET


class ConfigFileError(Exception):
    """Error loading or parsing a config file."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _print_config_error_and_exit(error: ConfigFileError) -> None:
    """Print a config file error in a nice box format and exit."""
    width = min(80, max(40, shutil.get_terminal_size().columns))
    inner_width = width - 4  # Account for "│ " and " │"

    # Box drawing characters
    top_left, top_right = "╭", "╮"
    bot_left, bot_right = "╰", "╯"
    horiz, vert = "─", "│"

    def wrap_text(text: str, max_width: int) -> list[str]:
        """Wrap text to fit within max_width."""
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if not current_line:
                current_line = word
            elif len(current_line) + 1 + len(word) <= max_width:
                current_line += " " + word
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines or [""]

    def box_line(content: str) -> str:
        """Create a line inside the box with proper padding."""
        padding = inner_width - len(content)
        return f"{_colorize(vert, _RED)} {content}{' ' * padding} {_colorize(vert, _RED)}"

    # Build the error message content
    lines = []

    # Title line
    title = "Config file error"
    title_plain_len = 2 + len(title) + 1
    lines.append(
        _colorize(top_left, _RED)
        + f"{horiz} {_colorize(title, _RED, _BOLD)} "
        + _colorize(horiz * (width - title_plain_len - 2) + top_right, _RED)
    )

    # Content
    message = error.message
    if "Failed to validate config" in message:
        parts = message.split(": ", 1)
        if len(parts) == 2:
            # Source info line
            for line in wrap_text(parts[0] + ":", inner_width):
                lines.append(box_line(line))

            # Horizontal rule
            lines.append(box_line(_colorize(horiz * inner_width, _RED)))

            # Pydantic error details
            pydantic_lines = parts[1].split("\n")
            for pydantic_line in pydantic_lines:
                if not pydantic_line:
                    continue
                # First line (validation error count)
                if "validation error" in pydantic_line:
                    for wrapped in wrap_text(pydantic_line, inner_width):
                        lines.append(box_line(_colorize(wrapped, _BRIGHT_RED)))
                # Field name (not indented)
                elif pydantic_line and not pydantic_line.startswith(" "):
                    for wrapped in wrap_text(f"  {pydantic_line}", inner_width):
                        lines.append(box_line(_colorize(wrapped, _BOLD)))
                # Error details (indented)
                elif pydantic_line.startswith("  "):
                    for wrapped in wrap_text(f"    {pydantic_line.strip()}", inner_width):
                        lines.append(box_line(_colorize(wrapped, _DIM)))
        else:
            for line in wrap_text(message, inner_width):
                lines.append(box_line(line))
    else:
        for line in wrap_text(message, inner_width):
            lines.append(box_line(line))

    # Bottom border
    lines.append(_colorize(f"{bot_left}{horiz * (width - 2)}{bot_right}", _RED))

    # Print to stderr
    for line in lines:
        print(line, file=sys.stderr)

    sys.exit(1)


def _load_config_file(path: str) -> dict:
    """Load a config file (JSON, YAML, or TOML) and return its contents as a dict."""
    try:
        with open(path, "rb") as f:
            if path.endswith(".json"):
                try:
                    return json.load(f)
                except json.JSONDecodeError as e:
                    raise ConfigFileError(f"Invalid JSON in {path}: {e}")

            elif path.endswith(".yaml") or path.endswith(".yml"):
                if importlib.util.find_spec("yaml") is None:
                    raise ConfigFileError(f"Cannot load {path}: pyyaml not installed. Install with: pip install pyyaml")
                import yaml

                try:
                    return yaml.load(f, Loader=yaml.FullLoader)
                except yaml.YAMLError as e:
                    raise ConfigFileError(f"Invalid YAML in {path}: {e}")

            elif path.endswith(".toml"):
                if importlib.util.find_spec("tomli") is None:
                    raise ConfigFileError(f"Cannot load {path}: tomli not installed. Install with: pip install tomli")
                import tomli

                try:
                    return tomli.load(f)
                except tomli.TOMLDecodeError as e:
                    raise ConfigFileError(f"Invalid TOML in {path}: {e}")

            else:
                raise ConfigFileError(f"Unsupported file type: {path}. Supported: .json, .yaml, .yml, .toml")
    except FileNotFoundError:
        raise ConfigFileError(f"Config file not found: {path}")


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts. Values from override take precedence."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _strip_annotated(annotation):
    """Unwrap ``Annotated[T, ...]`` → ``T``."""
    if hasattr(annotation, "__metadata__"):
        return get_args(annotation)[0]
    return annotation


def _is_union(annotation) -> bool:
    origin = get_origin(annotation)
    return origin is Union or origin is getattr(types, "UnionType", None)


# Cache mirror classes per (root cls) so repeated cli(cls) calls reuse them.
_MIRROR_CACHE: "dict[type, type]" = {}


def _mirror_annotation(annotation, cache: dict):
    """Compute the mirror form of a field annotation for the all-Optional mirror.

    The mirror class is used to capture CLI overrides as a sparse dict (see
    ``cli()``). Annotations are rewritten so every value is plausible for tyro
    to parse while staying compatible with ``model_dump(exclude_none=True)``
    to recover only what the user actually wrote:

      - Plain ``BaseModel`` → recurse into a mirror of that sub-model.
      - ``Optional[BaseModel]`` / discriminated union: not mirrored (CLI args
        for these paths are pre-extracted by ``_expand_bare_optional_flags``
        before tyro runs). The mirror field becomes ``Any`` with default
        ``None`` so tyro ignores any leftover overrides.
      - ``list[BaseModel]`` → list of the mirror sub-model.
      - ``dict[K, BaseModel]`` → dict mapping K to mirror sub-model.
      - Annotated[T, ...] → Annotated[mirror(T), ...] (preserves metadata
        that tyro / pydantic care about).
      - Everything else → keep as-is.
    """
    metadata = ()
    inner = annotation
    if hasattr(inner, "__metadata__"):
        metadata = inner.__metadata__
        inner = get_args(inner)[0]

    origin = get_origin(inner)

    if _is_union(inner):
        # Union of BaseModels or Optional[BaseModel] — pre-extracted before tyro,
        # so the mirror just needs to accept any value here without parsing it.
        non_none = [a for a in get_args(inner) if a is not type(None)]
        if non_none and all(isinstance(a, type) and issubclass(a, BaseModel) for a in non_none):
            return Any
        # Other unions (e.g. int | str): mirror each arm individually.
        new_args = tuple(_mirror_annotation(a, cache) for a in get_args(inner))
        return Union[new_args]

    if isinstance(inner, type) and issubclass(inner, BaseModel):
        mirrored = _build_mirror_cls(inner, cache)
        return _reapply_annotated(mirrored, metadata)

    if origin is list:
        args = get_args(inner)
        if args:
            item = _mirror_annotation(args[0], cache)
            return _reapply_annotated(list[item], metadata)
        return _reapply_annotated(inner, metadata)

    if origin is dict:
        args = get_args(inner)
        if len(args) == 2:
            key_ann, val_ann = args
            return _reapply_annotated(dict[key_ann, _mirror_annotation(val_ann, cache)], metadata)
        return _reapply_annotated(inner, metadata)

    return _reapply_annotated(inner, metadata)


def _reapply_annotated(inner, metadata):
    if not metadata:
        return inner
    from typing import Annotated  # local import to avoid widening top-of-file imports

    return Annotated[(inner, *metadata)]


def _build_mirror_cls(cls: type, cache: dict | None = None) -> type:
    """Build a "sparse mirror" of ``cls`` for CLI-override capture.

    The mirror has the same field names as ``cls`` but every primitive field
    becomes ``Optional[T] = None`` and every sub-BaseModel becomes
    ``MirrorOfSub = MirrorOfSub()`` (non-Optional with an empty-default
    instance so tyro will accept ``--sub.field value`` overrides).

    Dumping the resulting instance with ``model_dump(exclude_none=True)``
    yields a sparse dict containing only the values the user actually set on
    the CLI, which can be deep-merged into the TOML dict and validated against
    the real ``cls`` exactly once.
    """
    if cache is None:
        cache = _MIRROR_CACHE
    if cls in cache:
        return cache[cls]

    field_defs: dict[str, tuple] = {}
    # Insert a sentinel before recursing to break cycles (self-referential models).
    cache[cls] = cls  # placeholder

    for name, finfo in cls.model_fields.items():
        mirrored_ann = _mirror_annotation(finfo.annotation, cache)
        inner = _strip_annotated(mirrored_ann)
        # Sub-BaseModel: non-Optional with empty default so tyro accepts dotted overrides.
        if isinstance(inner, type) and issubclass(inner, BaseModel) and inner not in (BaseModel,):
            field_defs[name] = (mirrored_ann, Field(default_factory=inner))
        else:
            # Wrap leaf in Optional so it can default to None and be dropped by exclude_none.
            field_defs[name] = (Optional[mirrored_ann], None)

    mirror = create_model(
        f"_{cls.__name__}__Mirror",
        __base__=BaseModel,
        **field_defs,
    )
    mirror.model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
    cache[cls] = mirror
    return mirror


def _enumerate_cli_paths(cls: type, prefix: str = "") -> set[str]:
    """Walk ``cls.model_fields`` to enumerate every dotted kebab-case path that
    can appear as a ``--flag`` on the CLI.

    Used together with ``_scan_cli_set_paths`` to detect *which* paths the user
    actually typed in argv — separate from *what value* tyro parsed. This is
    what lets us tell ``--seq-len 0`` apart from "didn't set seq_len" even
    though both leave ``seq_len`` equal to the field default.
    """
    paths: set[str] = set()
    if not hasattr(cls, "model_fields"):
        return paths
    for name, finfo in cls.model_fields.items():
        kebab = name.replace("_", "-")
        full = f"{prefix}.{kebab}" if prefix else kebab
        ann = _strip_annotated(finfo.annotation)
        if _is_union(ann):
            non_none = [a for a in get_args(ann) if a is not type(None)]
            if len(non_none) == 1:
                ann = non_none[0]
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            paths.update(_enumerate_cli_paths(ann, prefix=full))
        else:
            paths.add(full)
    return paths


def _scan_cli_set_paths(args: list[str], valid_paths: set[str]) -> set[str]:
    """Identify which dotted snake-case paths from ``valid_paths`` were
    explicitly mentioned in ``args`` as ``--flag`` or ``--flag=value``.

    Recognises boolean negation: ``--no-thing`` counts as setting ``thing``.
    Returns snake-case keys (so they match Pydantic attribute names).
    """
    seen: set[str] = set()
    for arg in args:
        if not arg.startswith("--"):
            continue
        flag = arg[2:]
        if "=" in flag:
            flag = flag.split("=", 1)[0]
        if flag in valid_paths:
            seen.add(flag.replace("-", "_"))
            continue
        if flag.startswith("no-"):
            unnegated = flag[3:]
            if unnegated in valid_paths:
                seen.add(unnegated.replace("-", "_"))
    return seen


def _extract_set_paths(instance: BaseModel, set_paths: set[str]) -> dict:
    """Build a sparse dict by reading only the listed dotted snake-case paths
    off ``instance``. Used to recover user-supplied CLI values after tyro has
    flattened everything onto a fully-populated mirror instance.
    """
    result: dict = {}
    for path in set_paths:
        parts = path.split(".")
        value: Any = instance
        ok = True
        for part in parts:
            if value is None or not hasattr(value, part):
                ok = False
                break
            value = getattr(value, part)
        if not ok:
            continue
        if isinstance(value, BaseModel):
            value = value.model_dump()
        # Build nested dict at this dotted path.
        leaf: dict = result
        for part in parts[:-1]:
            leaf = leaf.setdefault(part, {})
        leaf[parts[-1]] = value
    return result


def _process_args(args: list[str]) -> tuple[list[str], dict, dict[str, dict]]:
    """
    Process command line args to extract config file references.

    Returns:
        - remaining_args: args with config file refs removed (for tyro)
        - root_config: merged config from root-level @ files
        - nested_configs: dict mapping arg names to their loaded configs

    Supports:
        - `@ config.toml` (with space, root level)
        - `--model @ model.toml` (with space, nested)
        - `--model @model.toml` (without space, nested)
    """
    remaining_args = []
    root_config: dict = {}
    nested_configs: dict[str, dict] = {}

    i = 0
    while i < len(args):
        arg = args[i]

        # Root level config: `@ config.toml`
        if arg == CONFIG_FILE_SIGN:
            if i + 1 >= len(args):
                raise ConfigFileError("@ must be followed by a config file path")
            config_path = args[i + 1]
            loaded = _load_config_file(config_path)
            root_config = _deep_merge(root_config, loaded)
            i += 2
            continue

        # Handle --arg @ file.toml or --arg @file.toml
        if arg.startswith("--"):
            arg_name = arg[2:]  # Remove --

            # Check if next arg is @ (with space)
            if i + 1 < len(args) and args[i + 1] == CONFIG_FILE_SIGN:
                if i + 2 >= len(args):
                    raise ConfigFileError(f"@ after {arg} must be followed by a config file path")
                config_path = args[i + 2]
                loaded = _load_config_file(config_path)
                nested_configs[arg_name] = loaded
                i += 3
                continue

            # Check if next arg starts with @ (without space): --arg @file.toml
            if i + 1 < len(args) and args[i + 1].startswith(CONFIG_FILE_SIGN) and len(args[i + 1]) > 1:
                config_path = args[i + 1][1:]  # Remove @
                loaded = _load_config_file(config_path)
                nested_configs[arg_name] = loaded
                i += 2
                continue

        # Regular arg, keep it
        remaining_args.append(arg)
        i += 1

    return remaining_args, root_config, nested_configs


def _nest_config(key_path: str, config: dict) -> dict:
    """
    Nest a config dict under a dotted key path.

    Example:
        _nest_config("model.encoder", {"layers": 6})
        -> {"model": {"encoder": {"layers": 6}}}
    """
    parts = key_path.split(".")
    result = config
    for part in reversed(parts):
        result = {part: result}
    return result


def _is_optional_model(annotation: type) -> bool:
    """Check if annotation is Optional[SomeBaseModel] (i.e. SomeBaseModel | None)."""
    if hasattr(annotation, "__metadata__"):
        annotation = get_args(annotation)[0]
    origin = get_origin(annotation)
    if origin is not Union and origin is not getattr(types, "UnionType", None):
        return False
    args = get_args(annotation)
    non_none = [a for a in args if a is not type(None)]
    return len(non_none) == 1 and isinstance(non_none[0], type) and issubclass(non_none[0], BaseModel)


def _is_multi_model_union(annotation: type) -> bool:
    """Check if annotation is a discriminated union of multiple BaseModel subclasses."""
    if hasattr(annotation, "__metadata__"):
        annotation = get_args(annotation)[0]
    origin = get_origin(annotation)
    if origin is not Union and origin is not getattr(types, "UnionType", None):
        return False
    args = get_args(annotation)
    non_none = [a for a in args if a is not type(None)]
    return len(non_none) > 1 and all(isinstance(a, type) and issubclass(a, BaseModel) for a in non_none)


def _find_optional_model_paths(cls: type, prefix: str = "") -> set[str]:
    """Recursively find all CLI arg paths (kebab-case) that map to Optional[BaseModel] or discriminated union fields."""
    paths: set[str] = set()
    if not hasattr(cls, "model_fields"):
        return paths
    for field_name, field_info in cls.model_fields.items():
        field_kebab = field_name.replace("_", "-")
        full_path = f"{prefix}.{field_kebab}" if prefix else field_kebab
        annotation = field_info.annotation
        if _is_optional_model(annotation) or _is_multi_model_union(annotation):
            paths.add(full_path)
        inner = annotation
        if hasattr(inner, "__metadata__"):
            inner = get_args(inner)[0]
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            paths.update(_find_optional_model_paths(inner, prefix=full_path))
    return paths


def _find_optional_model_inner_classes(cls: type, prefix: str = "") -> dict[str, type]:
    """Map each Optional[BaseModel] CLI path (kebab-case) to its inner BaseModel class.

    Used to fabricate a default instance so tyro renders sub-fields in ``--help``
    for fields that would otherwise default to None.
    """
    result: dict[str, type] = {}
    if not hasattr(cls, "model_fields"):
        return result
    for field_name, field_info in cls.model_fields.items():
        field_kebab = field_name.replace("_", "-")
        full_path = f"{prefix}.{field_kebab}" if prefix else field_kebab
        annotation = field_info.annotation
        if _is_optional_model(annotation):
            inner = annotation
            if hasattr(inner, "__metadata__"):
                inner = get_args(inner)[0]
            non_none = [a for a in get_args(inner) if a is not type(None)]
            result[full_path] = non_none[0]
        inner = annotation
        if hasattr(inner, "__metadata__"):
            inner = get_args(inner)[0]
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            result.update(_find_optional_model_inner_classes(inner, prefix=full_path))
    return result


def _find_multi_union_variants(cls: type, prefix: str = "") -> dict[str, tuple[list[type], type | None]]:
    """Map each multi-model-union CLI path (kebab-case) to (all variants, default variant class).

    Used to render help panels for variants that aren't the field's default,
    so users discover the sub-fields of every variant in ``--help``.
    """
    result: dict[str, tuple[list[type], type | None]] = {}
    if not hasattr(cls, "model_fields"):
        return result
    for field_name, field_info in cls.model_fields.items():
        field_kebab = field_name.replace("_", "-")
        full_path = f"{prefix}.{field_kebab}" if prefix else field_kebab
        annotation = field_info.annotation
        if _is_multi_model_union(annotation):
            inner = annotation
            if hasattr(inner, "__metadata__"):
                inner = get_args(inner)[0]
            variants = [a for a in get_args(inner) if a is not type(None)]
            default_cls = type(field_info.default) if isinstance(field_info.default, BaseModel) else None
            result[full_path] = (variants, default_cls)
        inner_cls = annotation
        if hasattr(inner_cls, "__metadata__"):
            inner_cls = get_args(inner_cls)[0]
        if isinstance(inner_cls, type) and issubclass(inner_cls, BaseModel):
            result.update(_find_multi_union_variants(inner_cls, prefix=full_path))
    return result


def _path_is_set_in_config(config: dict, path: str) -> bool:
    """Check whether a dotted kebab-case path resolves to a value in the merged config dict."""
    parts = [p.replace("-", "_") for p in path.split(".")]
    cur = config
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return cur is not None


def _path_value_on_model(obj, path: str):
    """Walk a dotted kebab-case path on a Pydantic model, returning the value or None if missing."""
    parts = [p.replace("-", "_") for p in path.split(".")]
    cur = obj
    for part in parts:
        if cur is None or not isinstance(cur, BaseModel):
            return None
        cur = getattr(cur, part, None)
    return cur


def _set_path_to_none(obj: BaseModel, path: str) -> None:
    """Set the field at a dotted kebab-case path to None on a Pydantic model."""
    parts = [p.replace("-", "_") for p in path.split(".")]
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part, None)
        if target is None:
            return
    setattr(target, parts[-1], None)


def _annotate_optional_panel_titles(text: str, optional_paths: list[str]) -> str:
    """Inject "(optional, default: None)" into tyro panel titles for the given paths.

    Tyro renders ``╭─ wandb options ───╮`` for each top-level group; this rewrites
    those titles so the help itself communicates which fields default to None.
    The line width is preserved by trimming the trailing dashes.
    """
    if not optional_paths:
        return text
    marker = "(optional, default: None)"
    lines = text.split("\n")
    for path in optional_paths:
        prefix = f"╭─ {path} options "
        for i, line in enumerate(lines):
            if not line.startswith(prefix) or not line.endswith("╮"):
                continue
            original_width = len(line)
            new_title = f"{prefix}{marker} "
            dashes = original_width - len(new_title) - 1
            if dashes < 1:
                lines[i] = f"{new_title}─╮"
            else:
                lines[i] = f"{new_title}{'─' * dashes}╮"
            break
    return "\n".join(lines)


def _format_type_for_help(annotation) -> str:
    """Render an annotation as a tyro-style metavar (e.g. INT, STR, {a,b})."""
    if annotation is None:
        return ""
    if hasattr(annotation, "__metadata__"):
        annotation = get_args(annotation)[0]
    origin = get_origin(annotation)
    if origin is Literal:
        return "{" + ",".join(str(c) for c in get_args(annotation)) + "}"
    primitive_names = {int: "INT", float: "FLOAT", str: "STR", bool: "{True,False}"}
    if annotation in primitive_names:
        return primitive_names[annotation]
    if origin is list:
        args = get_args(annotation)
        inner = args[0] if args else None
        return f"[{_format_type_for_help(inner)} [...]]"
    if origin is Union or origin is getattr(types, "UnionType", None):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return f"{_format_type_for_help(non_none[0])}|None"
        return "|".join(_format_type_for_help(a) for a in non_none)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return ""
    return getattr(annotation, "__name__", str(annotation)).upper()


def _render_variant_panel(path: str, variant_cls: type, term_width: int) -> list[str]:
    """Render a tyro-style help panel listing a union variant's fields."""
    rows: list[tuple[str, str]] = []
    for fname, finfo in variant_cls.model_fields.items():
        type_str = _format_type_for_help(finfo.annotation)
        flag = f"--{path}.{fname.replace('_', '-')}"
        if type_str:
            flag = f"{flag} {type_str}"
        default = finfo.default
        if isinstance(default, BaseModel):
            note = ""
        elif default is None:
            note = "(default: None)"
        elif repr(default) == "PydanticUndefined":
            note = ""
        else:
            note = f"(default: {default})"
        rows.append((flag, note))

    if not rows:
        return []

    flag_w = max(len(f) for f, _ in rows)
    body_lines = [f"{f:<{flag_w}}  {n}".rstrip() for f, n in rows]
    title = f"{path} variant: {variant_cls.__name__}"
    inner = max(max(len(line) for line in body_lines), len(title) + 2)
    box_total = min(max(inner + 4, len(title) + 6), max(40, term_width))
    inner = box_total - 4

    horiz = "─" * max(1, box_total - len(title) - 5)
    lines = [f"╭─ {title} {horiz}╮"]
    for line in body_lines:
        truncated = line if len(line) <= inner else line[: inner - 1] + "…"
        lines.append(f"│ {truncated:<{inner}} │")
    lines.append(f"╰{'─' * (box_total - 2)}╯")
    return lines


def _print_union_variant_panels(cls: type) -> None:
    """Print help panels for every non-default variant of multi-model union fields."""
    variants_map = _find_multi_union_variants(cls)
    if not variants_map:
        return
    term_width = min(80, max(40, shutil.get_terminal_size().columns))
    for path, (variants, default_cls) in variants_map.items():
        for variant_cls in variants:
            if variant_cls is default_cls:
                continue
            for line in _render_variant_panel(path, variant_cls, term_width):
                print(line)


_JSON_VALUE_TYPES = (dict, list)


def _is_json_value_field(annotation: type) -> bool:
    """Check if annotation is a type that should accept JSON-encoded CLI values.

    Covers ``dict``, ``list``, and their ``Optional`` / ``Annotated`` wrappers.
    """
    if hasattr(annotation, "__metadata__"):
        annotation = get_args(annotation)[0]
    origin = get_origin(annotation)
    if annotation in _JSON_VALUE_TYPES or origin in _JSON_VALUE_TYPES:
        return True
    if origin is Union or origin is getattr(types, "UnionType", None):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            inner = non_none[0]
            return inner in _JSON_VALUE_TYPES or get_origin(inner) in _JSON_VALUE_TYPES
    return False


def _find_json_value_field_paths(cls: type, prefix: str = "") -> set[str]:
    """Recursively find all CLI arg paths (kebab-case) that map to JSON-value fields."""
    paths: set[str] = set()
    if not hasattr(cls, "model_fields"):
        return paths
    for field_name, field_info in cls.model_fields.items():
        field_kebab = field_name.replace("_", "-")
        full_path = f"{prefix}.{field_kebab}" if prefix else field_kebab
        annotation = field_info.annotation
        if _is_json_value_field(annotation):
            paths.add(full_path)
        inner = annotation
        if hasattr(inner, "__metadata__"):
            inner = get_args(inner)[0]
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            paths.update(_find_json_value_field_paths(inner, prefix=full_path))
    return paths


def _extract_json_value_args(args: list[str], json_paths: set[str]) -> tuple[list[str], dict]:
    """Intercept CLI args for dict/list fields and parse their JSON values.

    Handles any CLI arg whose path maps to a dict or list field and whose
    value looks like JSON (starts with ``{`` or ``[``).  For example::

        --extra-kwargs '{"key": 123}'     → dict field
        --buffer.env-ratios '[0.7, 0.3]'  → list field

    Returns (remaining_args, config_overrides_as_nested_dict).
    """
    remaining: list[str] = []
    overrides: dict = {}

    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            path = arg[2:]
            if path in json_paths and i + 1 < len(args) and args[i + 1][:1] in ("{", "["):
                parsed = json.loads(args[i + 1])
                snake_path = path.replace("-", "_")
                nested = _nest_config(snake_path, parsed)
                overrides = _deep_merge(overrides, nested)
                i += 2
                continue
        remaining.append(args[i])
        i += 1

    return remaining, overrides


def _match_optional_prefix(path: str, optional_paths: set[str]) -> str | None:
    """If ``path`` starts with an optional model path followed by '.', return it."""
    for opt_path in optional_paths:
        if path.startswith(opt_path + "."):
            return opt_path
    return None


def _expand_bare_optional_flags(
    args: list[str], optional_paths: set[str]
) -> tuple[list[str], dict]:
    """Handle CLI args for Optional[BaseModel] fields that tyro cannot parse.

    Handles two patterns:
    1. Bare flags: ``--compile`` enables CompileConfig with defaults.
    2. Sub-field overrides: ``--wandb.project foo`` sets a sub-field on an
       Optional model that defaults to None.  The arg and value are removed
       from the CLI args and injected into the config dict so pydantic
       handles type coercion.

    Returns (remaining_args, config_overrides_as_nested_dict).
    """
    remaining: list[str] = []
    overrides: dict = {}

    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            path = arg[2:]

            # Pattern 1: bare flag (e.g. --compile)
            if path in optional_paths:
                next_is_value = i + 1 < len(args) and not args[i + 1].startswith("-") and not args[i + 1].startswith("@")
                if not next_is_value:
                    snake_path = path.replace("-", "_")
                    nested = _nest_config(snake_path, {})
                    overrides = _deep_merge(overrides, nested)
                    i += 1
                    continue

            # Pattern 2: sub-field override (e.g. --wandb.project foo)
            matched = _match_optional_prefix(path, optional_paths)
            if matched is not None:
                snake_path = path.replace("-", "_")
                if i + 1 < len(args) and not args[i + 1].startswith("--") and not args[i + 1].startswith("@"):
                    value: str | dict | list = args[i + 1]
                    if isinstance(value, str) and value.startswith(("{", "[")):
                        value = json.loads(value)
                    nested = _nest_config(snake_path, value)
                    overrides = _deep_merge(overrides, nested)
                    i += 2
                    continue
                else:
                    # Bare sub-flag (e.g. --wandb.enabled with no value → True)
                    nested = _nest_config(snake_path, True)
                    overrides = _deep_merge(overrides, nested)
                    i += 1
                    continue

        remaining.append(args[i])
        i += 1

    return remaining, overrides


def _build_default_from_config(cls: type[T], config: dict, config_path: str | None = None) -> T | None:
    """Build a default instance from config dict for tyro.

    Raises ConfigFileError if the config cannot be validated against the model.
    """
    if not config:
        return None
    try:
        return _dict_to_instance(cls, config)
    except Exception as e:
        source = f" from '{config_path}'" if config_path else ""
        raise ConfigFileError(f"Failed to validate config{source}: {e}") from e


@overload
def cli(cls: type[T]) -> T: ...


@overload
def cli(cls: type[T], *, args: list[str]) -> T: ...


@overload
def cli(cls: type[T], *, default: T) -> T: ...


@overload
def cli(cls: type[T], *, args: list[str], default: T) -> T: ...


def cli(
    cls: type[T],
    *,
    args: list[str] | None = None,
    default: T | None = None,
    prog: str | None = None,
    description: str | None = None,
) -> T:
    """
    Parse CLI arguments into a typed config object, with support for config files.

    Drop-in replacement for tyro.cli() with additional support for loading
    config files using the @ syntax:
        - `@ config.toml` - Load root-level config
        - `--model @ model.toml` - Load config nested under 'model'
        - `--model @model.toml` - Same as above (no space)

    Args:
        cls: The type to parse into (Pydantic BaseConfig or BaseModel)
        args: Command line args to parse (defaults to sys.argv[1:])
        default: Default instance to use for missing values
        prog: Program name for help text
        description: Description for help text

    Returns:
        Parsed and validated config object

    Example:
        class TrainConfig(BaseConfig):
            lr: float = 1e-4
            batch_size: int = 32

        class Config(BaseConfig):
            train: TrainConfig
            seed: int = 42

        # Can be called as:
        # python train.py @ config.toml --train.lr 1e-3
        # python train.py --train @ train.toml --seed 123

        config = cli(Config)
    """
    use_sys_argv = args is None
    if args is None:
        args = sys.argv[1:]

    try:
        # 1. Parse TOML / nested config files referenced via ``@`` into a raw dict.
        remaining_args, root_config, nested_configs = _process_args(args)
        toml_dict = root_config
        for key_path, config in nested_configs.items():
            nested = _nest_config(key_path, config)
            toml_dict = _deep_merge(toml_dict, nested)

        # 2. Pre-extract CLI overrides for types tyro can't parse directly:
        #    - Optional[BaseModel] / discriminated-union sub-field flags
        #      (e.g. ``--wandb.project foo``, ``--data.type b``)
        #    - JSON-encoded dict/list values (e.g. ``--env-ratios '[0.7, 0.3]'``)
        # These args are stripped from ``remaining_args`` and accumulated into
        # ``cli_overrides`` as raw strings / parsed JSON.
        cli_overrides: dict = {}
        optional_paths = _find_optional_model_paths(cls)
        if optional_paths:
            remaining_args, bare_overrides = _expand_bare_optional_flags(remaining_args, optional_paths)
            cli_overrides = _deep_merge(cli_overrides, bare_overrides)

        json_paths = _find_json_value_field_paths(cls)
        if json_paths:
            remaining_args, json_overrides = _extract_json_value_args(remaining_args, json_paths)
            cli_overrides = _deep_merge(cli_overrides, json_overrides)

        # 3. Handle ``--help`` against the real ``cls`` so users see its schema.
        #    Build a stitched default that includes any caller-provided default
        #    layered with TOML so optional sub-fields render in the help panels.
        inactive_optional_paths: list[str] = []
        wants_help = "--help" in remaining_args or "-h" in remaining_args
        if wants_help:
            help_default = _build_help_default(cls, default, toml_dict, cli_overrides, inactive_optional_paths)
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    tyro.cli(
                        tyro.conf.AvoidSubcommands[cls],
                        args=remaining_args,
                        default=help_default,
                        prog=prog,
                        description=description,
                    )
            except SystemExit:
                sys.stdout.write(_annotate_optional_panel_titles(buf.getvalue(), inactive_optional_paths))
                _print_union_variant_panels(cls)
                raise

        # 4. Capture remaining CLI overrides via a "sparse mirror" of ``cls``:
        #    every field becomes Optional with a None / empty default so tyro can
        #    parse any subset of CLI args without complaining about missing
        #    required fields. We then scan argv to identify *which* paths the
        #    user actually typed (so ``--seq-len 0`` is distinct from "unset",
        #    and ``--name None`` survives even though it ties the mirror's
        #    default), and read those paths off the mirror result.
        mirror_cls = _build_mirror_cls(cls)
        mirror_result = tyro.cli(
            tyro.conf.AvoidSubcommands[mirror_cls],
            args=remaining_args,
            default=mirror_cls(),
            prog=prog,
            description=description,
        )
        valid_paths = _enumerate_cli_paths(mirror_cls)
        set_paths = _scan_cli_set_paths(remaining_args, valid_paths)
        if set_paths:
            cli_overrides = _deep_merge(cli_overrides, _extract_set_paths(mirror_result, set_paths))

        # 5. Build the final merged dict in precedence order:
        #    caller default ⊂ TOML ⊂ CLI overrides, then validate once. This
        #    means ``cls``'s validators fire exactly once on the real merged
        #    data, and ``model_fields_set`` on sub-configs faithfully reflects
        #    "did the user / TOML actually write this key?" — no leakage from
        #    earlier default-construction passes.
        default_dict: dict = {}
        if default is not None:
            default_dict = default.model_dump(exclude_unset=True) if isinstance(default, BaseModel) else {}
        merged = _deep_merge(_deep_merge(default_dict, toml_dict), cli_overrides)
        return cls.model_validate(merged)
    except ConfigFileError as e:
        # Only print formatted error when running from CLI (sys.argv)
        # When args are explicitly passed, re-raise for programmatic handling
        if use_sys_argv:
            _print_config_error_and_exit(e)
        raise


def _build_help_default(cls, caller_default, toml_dict: dict, cli_overrides: dict, inactive_optional_paths: list[str]):
    """Construct the ``default=`` instance passed to tyro for the ``--help`` pass.

    Help text is rendered against the real ``cls``, so we need a constructible
    instance. We approximate it by validating the merged TOML + CLI dict
    against ``cls`` and patching in placeholder instances for Optional[BaseModel]
    sub-fields the user hasn't activated (so their sub-flags still render in
    the help panels). Validation failures here are non-fatal — we fall back
    to ``cls()`` when possible and let tyro surface the real error after.
    """
    try:
        merged = _deep_merge(toml_dict, cli_overrides)
        if merged:
            base = cls.model_validate(merged)
        elif caller_default is not None:
            base = caller_default
        else:
            base = cls.model_validate({})
    except Exception:
        try:
            base = cls.model_validate({})
        except Exception:
            return caller_default

    optional_inner_classes = _find_optional_model_inner_classes(cls)
    for path, inner_cls in optional_inner_classes.items():
        if _path_is_set_in_config(toml_dict, path) or _path_is_set_in_config(cli_overrides, path):
            continue
        if _path_value_on_model(base, path) is not None:
            continue
        try:
            placeholder = inner_cls()
        except Exception:
            continue
        _set_path_on_model(base, path, placeholder)
        inactive_optional_paths.append(path)
    return base


def _set_path_on_model(obj, path: str, value):
    """Assign ``value`` to a dotted kebab-case path on a Pydantic model."""
    parts = [p.replace("-", "_") for p in path.split(".")]
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part, None)
        if target is None:
            return
    setattr(target, parts[-1], value)
