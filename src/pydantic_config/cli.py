"""
Pydantic-driven CLI with TOML / YAML / JSON config file support.

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

import copy
import importlib.util
import json
import os
import re
import shutil
import sys
import types
import typing
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, overload

from pydantic import AliasChoices, BaseModel, ConfigDict, ValidationError, model_validator

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


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visible_len(text: str) -> int:
    """Length of ``text`` excluding ANSI colour escape sequences.

    The box-drawing renderer pads rows to ``inner_width`` using this so
    coloured content (e.g. ``_colorize("foo", _BOLD)``) still produces a
    right-aligned border.
    """
    return len(_ANSI_RE.sub("", text))


class ConfigFileError(Exception):
    """Error loading or parsing a config file."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _term_width() -> int:
    """Width to render boxes at — the actual terminal width, with a floor of 40.

    No upper cap: by default the error / help panels render full-screen so the
    box draws look like a coherent UI element rather than an 80-column island
    in a wide terminal.
    """
    return max(40, shutil.get_terminal_size().columns)


def _print_config_error_and_exit(error: ConfigFileError) -> None:
    """Print a config file error in a nice box format and exit."""
    width = _term_width()
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
        """Create a line inside the box with proper padding.

        Uses ``_visible_len`` so ANSI escape codes inside ``content`` don't get
        counted as printable width — otherwise coloured rows make the right
        border drift left.
        """
        padding = inner_width - _visible_len(content)
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


def _loc_to_cli_flag(loc: tuple) -> str:
    """Convert a Pydantic error ``loc`` tuple to the matching CLI flag form.

    ``("trainer", "model", "seq_len")`` → ``--trainer.model.seq-len``.
    List indices and any non-str segments are appended in ``[i]`` notation
    so users can still locate them, e.g. ``("items", 0)`` → ``--items[0]``.
    """
    if not loc:
        return "<root>"
    parts: list[str] = []
    for segment in loc:
        if isinstance(segment, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{segment}]"
            else:
                parts.append(f"[{segment}]")
        else:
            parts.append(str(segment).replace("_", "-"))
    return "--" + ".".join(parts)


def _format_validation_error_for_cli(error: ValidationError) -> str:
    """Render a ``pydantic.ValidationError`` as a CLI-flag-flavoured multi-line
    message suitable for ``_print_config_error_and_exit``.

    Each Pydantic error becomes one row of the form
    ``<cli-flag>: <msg> (got <input>)``. The input is omitted when it's a
    container (dict / list), since the per-leaf errors that follow show the
    real culprit.
    """
    errors = error.errors()
    count = len(errors)
    header = f"Failed to validate config: {count} validation error{'s' if count != 1 else ''} for {error.title}"
    lines: list[str] = [header]
    for err in errors:
        loc = err.get("loc", ())
        flag = _loc_to_cli_flag(tuple(loc))
        msg = err.get("msg") or err.get("type", "validation error")
        input_value = err.get("input")
        suffix = ""
        if input_value is not None and not isinstance(input_value, (dict, list)):
            suffix = f" (got {input_value!r})"
        lines.append(flag)
        lines.append(f"  {msg}{suffix}")
    return "\n".join(lines)


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


def _format_default_for_help(default: Any) -> str:
    """Render a field default as the trailing ``(default: X)`` annotation for help.

    Rules in order:
      - ``BaseModel`` instance → no annotation (its fields appear in a sub-panel)
      - ``None``                → ``(default: None)``
      - ``PydanticUndefined``   → no annotation (required field)
      - callable (``default_factory``) → call it and recurse so e.g. ``list``
        renders as ``(default: [])`` rather than ``<function list>``
      - everything else → ``(default: {value})``
    """
    if isinstance(default, BaseModel):
        return ""
    if default is None:
        return "(default: None)"
    if repr(default) == "PydanticUndefined":
        return ""
    if callable(default):
        try:
            return _format_default_for_help(default())
        except Exception:
            return ""
    return f"(default: {default})"


def _format_field_note(finfo) -> str:
    """Combine ``Field(description=...)`` with the default annotation into the
    trailing column shown next to a flag in ``--help``.

    Layout: ``<description>  (default: X)`` with a two-space gap when both are
    present; if either is empty it's just the other. Returns an empty string
    for required fields with no description (the flag itself is enough).
    """
    description = (finfo.description or "").strip()
    default = _format_default_for_help(finfo.default)
    if description and default:
        return f"{description}  {default}"
    return description or default


def _render_panel(title: str, rows: list[tuple[str, str]], term_width: int) -> list[str]:
    """Render a box-drawn help panel.

    ``rows`` is a list of ``(flag_with_metavar, default_annotation)`` pairs.
    Empty ``rows`` returns an empty list (caller should suppress the panel).

    The panel spans the full ``term_width`` so successive panels line up
    vertically and the help output reads as a coherent UI. If a body line is
    wider than the terminal, the panel grows to fit it (and the long row is
    truncated with an ellipsis if it still overflows the safety margin).
    """
    if not rows:
        return []

    flag_w = max(len(f) for f, _ in rows)
    body_lines = [f"{f:<{flag_w}}  {n}".rstrip() for f, n in rows]
    content_width = max(max(len(line) for line in body_lines), len(title) + 2)
    box_total = max(term_width, content_width + 4, len(title) + 6)
    inner = box_total - 4

    horiz = "─" * max(1, box_total - len(title) - 5)
    lines = [f"╭─ {title} {horiz}╮"]
    for line in body_lines:
        truncated = line if len(line) <= inner else line[: inner - 1] + "…"
        lines.append(f"│ {truncated:<{inner}} │")
    lines.append(f"╰{'─' * (box_total - 2)}╯")
    return lines


def _render_variant_panel(path: str, variant_cls: type, term_width: int) -> list[str]:
    """Render a help panel listing a union variant's fields."""
    rows: list[tuple[str, str]] = []
    for fname, finfo in variant_cls.model_fields.items():
        type_str = _format_type_for_help(finfo.annotation)
        flag = f"--{path}.{fname.replace('_', '-')}"
        if type_str:
            flag = f"{flag} {type_str}"
        rows.append((flag, _format_field_note(finfo)))
    return _render_panel(f"{path} variant: {variant_cls.__name__}", rows, term_width)


def _strip_annotated(annotation):
    """Unwrap ``Annotated[T, ...]`` → ``T``."""
    if hasattr(annotation, "__metadata__"):
        return get_args(annotation)[0]
    return annotation


def _is_union(annotation) -> bool:
    """``True`` if ``annotation`` is ``Union[...]`` or PEP-604 ``X | Y``."""
    origin = get_origin(annotation)
    return origin is Union or origin is getattr(types, "UnionType", None)


def _collect_help_panels(
    cls: type, term_width: int, prefix: str = ""
) -> tuple[list[tuple[str, str]], list[str]]:
    """Walk ``cls.model_fields`` to collect help rows and sub-panel lines.

    Returns ``(rows, panel_lines)`` where ``rows`` is the list of leaf
    ``(flag, default_annotation)`` pairs to emit in the *current* panel and
    ``panel_lines`` is a flat list of lines (panels separated by blank lines)
    for every sub-config / Optional[BaseModel] / multi-model union field
    discovered below this point.
    """
    rows: list[tuple[str, str]] = []
    panel_lines: list[str] = []

    for fname, finfo in cls.model_fields.items():
        kebab = fname.replace("_", "-")
        full_path = f"{prefix}.{kebab}" if prefix else kebab
        annotation = finfo.annotation

        if _is_multi_model_union(annotation):
            inner = _strip_annotated(annotation)
            variants = [a for a in get_args(inner) if a is not type(None)]
            for variant_cls in variants:
                panel_lines.extend(_render_variant_panel(full_path, variant_cls, term_width))
                panel_lines.append("")
            continue

        if _is_optional_model(annotation):
            inner = _strip_annotated(annotation)
            non_none = [a for a in get_args(inner) if a is not type(None)]
            inner_cls = non_none[0]
            sub_rows, sub_panels = _collect_help_panels(inner_cls, term_width, prefix=full_path)
            panel_lines.extend(
                _render_panel(f"{full_path} options (optional, default: None)", sub_rows, term_width)
            )
            panel_lines.append("")
            panel_lines.extend(sub_panels)
            continue

        inner = _strip_annotated(annotation)
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            sub_rows, sub_panels = _collect_help_panels(inner, term_width, prefix=full_path)
            panel_lines.extend(_render_panel(f"{full_path} options", sub_rows, term_width))
            panel_lines.append("")
            panel_lines.extend(sub_panels)
            continue

        # Leaf field.
        type_str = _format_type_for_help(annotation)
        flag = f"--{full_path}"
        if type_str:
            flag = f"{flag} {type_str}"
        rows.append((flag, _format_field_note(finfo)))

    return rows, panel_lines


def _render_help(cls: type, prog: str | None = None, description: str | None = None) -> str:
    """Render the full ``--help`` text for ``cls`` as a single string.

    Layout: a usage line, optional description, an "options" panel listing
    leaf fields directly on ``cls``, then a separate panel per sub-config
    (recursively flattened with dotted paths), per Optional[BaseModel] field
    (annotated "(optional, default: None)"), and per multi-model union
    variant. Reuses ``_render_panel`` so all panels share the same
    box-drawing style.
    """
    prog = prog or os.path.basename(sys.argv[0])
    term_width = _term_width()

    root_rows, sub_panels = _collect_help_panels(cls, term_width)

    lines: list[str] = [f"usage: {prog} [-h] [@ FILE] [OPTIONS]"]
    if description:
        lines.append("")
        for paragraph in description.splitlines():
            lines.append(paragraph)
    lines.append("")

    header_rows = [("-h, --help", "show this help message and exit"), *root_rows]
    lines.extend(_render_panel("options", header_rows, term_width))
    lines.append("")
    lines.extend(sub_panels)

    return "\n".join(lines).rstrip() + "\n"


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


def _annotation_is_bool(annotation) -> bool:
    """``True`` if ``annotation`` is ``bool``, ``Optional[bool]``, etc."""
    inner = _strip_annotated(annotation)
    if inner is bool:
        return True
    if _is_union(inner):
        non_none = [a for a in get_args(inner) if a is not type(None)]
        return len(non_none) == 1 and non_none[0] is bool
    return False


def _annotation_is_list(annotation) -> bool:
    """``True`` if ``annotation`` is ``list[T]`` or ``Optional[list[T]]``."""
    inner = _strip_annotated(annotation)
    if get_origin(inner) is list:
        return True
    if _is_union(inner):
        non_none = [a for a in get_args(inner) if a is not type(None)]
        return len(non_none) == 1 and get_origin(non_none[0]) is list
    return False


class _FieldMeta(typing.NamedTuple):
    snake_path: str  # dotted snake_case path for nesting into the override dict
    annotation: object  # field annotation
    is_bool: bool
    is_list: bool


def _build_field_meta_map(cls: type, prefix: str = "") -> tuple[dict[str, _FieldMeta], set[str]]:
    """Walk ``cls.model_fields`` to build a kebab-case-path → ``_FieldMeta`` map
    plus a set of "interior" kebab-case paths (paths whose target is itself a
    ``BaseModel``). Both maps are needed by ``_parse_cli_to_dict`` so it can
    look up leaves in O(1) and give a clean ``--wandb foo``-style error when
    the user lands on an interior node.

    Field-level ``validation_alias=AliasChoices(...)`` entries are added at the
    same depth — every alias name produces an additional entry so users can
    write the CLI flag under any accepted spelling.
    """
    leaves: dict[str, _FieldMeta] = {}
    interior: set[str] = set()
    if not hasattr(cls, "model_fields"):
        return leaves, interior

    for field_name, field_info in cls.model_fields.items():
        annotation = field_info.annotation

        # Field names + every alias accepted at validation time.
        names = [field_name]
        alias = field_info.validation_alias
        if alias is not None:
            from pydantic import AliasChoices

            if isinstance(alias, AliasChoices):
                for choice in alias.choices:
                    if isinstance(choice, str) and choice != field_name:
                        names.append(choice)
            elif isinstance(alias, str) and alias != field_name:
                names.append(alias)

        for name in names:
            kebab = name.replace("_", "-")
            full_path = f"{prefix}.{kebab}" if prefix else kebab
            snake_path = (prefix.replace("-", "_") + "." if prefix else "") + name

            inner = _strip_annotated(annotation)
            is_plain_basemodel = isinstance(inner, type) and issubclass(inner, BaseModel)
            if is_plain_basemodel:
                interior.add(full_path)
                child_leaves, child_interior = _build_field_meta_map(inner, prefix=full_path)
                leaves.update(child_leaves)
                interior.update(child_interior)
            else:
                leaves[full_path] = _FieldMeta(
                    snake_path=snake_path,
                    annotation=annotation,
                    is_bool=_annotation_is_bool(annotation),
                    is_list=_annotation_is_list(annotation),
                )

    return leaves, interior


def _coerce_bool_literal(value: str) -> bool | None:
    """Recognise the boolean string literals tyro previously coerced for us."""
    if value in ("true", "True", "1"):
        return True
    if value in ("false", "False", "0"):
        return False
    return None


def _set_nested(out: dict, snake_path: str, value: Any) -> None:
    """Set ``value`` at ``snake_path`` inside ``out``, creating intermediate dicts."""
    parts = snake_path.split(".")
    node = out
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            # An earlier write put a non-dict here; bail rather than clobber.
            return
    node[parts[-1]] = value


def _parse_cli_to_dict(
    args: list[str], cls: type, optional_paths: set[str]
) -> tuple[list[str], dict]:
    """Parse remaining CLI tokens into a sparse override dict.

    Runs *after* ``_process_args``, ``_expand_bare_optional_flags`` and
    ``_extract_json_value_args`` have stripped ``@``-file references,
    Optional[BaseModel] / discriminated-union sub-flags, and JSON-encoded
    dict/list values. What's left is leaf-scalar / bool / typed-list overrides
    against ``cls.model_fields``.

    Returns ``(remaining_args, overrides)`` — anything not recognised as a
    ``--flag`` token stays in ``remaining_args`` so the caller can decide
    whether to error.

    Behaviour:
      - ``--name value`` and ``--name=value`` are equivalent.
      - ``--flag`` on a bool field sets it to True; ``--no-flag`` sets it to False.
        A bool flag may also take an explicit ``true|false|1|0`` value.
      - ``--items 0.7 0.3`` consumes consecutive non-``--`` tokens as list members.
        A leading ``-`` (e.g. ``-1e-3``) is a value, not a flag — only ``--`` boundaries break list consumption.
      - Unknown leaves and tokens that land on an interior (BaseModel) path raise
        ``ConfigFileError`` with a "did you mean" suggestion built from the leaf paths.
    """
    leaves, interior = _build_field_meta_map(cls)
    remaining: list[str] = []
    overrides: dict = {}
    unknown: list[tuple[str, str | None]] = []

    i = 0
    n = len(args)
    while i < n:
        token = args[i]
        if not token.startswith("--"):
            remaining.append(token)
            i += 1
            continue

        flag_part, eq_value = (token[2:].split("=", 1) + [None])[:2] if "=" in token[2:] else (token[2:], None)

        # 1. Defensive skip for paths handled upstream by _expand_bare_optional_flags.
        #    Those should already be gone, but if a caller passes a custom args list
        #    directly to cli(), be lenient.
        if flag_part in optional_paths or _match_optional_prefix(flag_part, optional_paths):
            remaining.append(token)
            i += 1
            continue

        # 2. --no-flag negation for booleans.
        if flag_part.startswith("no-") and flag_part[3:] in leaves and leaves[flag_part[3:]].is_bool:
            _set_nested(overrides, leaves[flag_part[3:]].snake_path, False)
            i += 1
            continue

        # 3. Interior (BaseModel) path → guide the user toward dotted form.
        if flag_part in interior:
            sub_flags = sorted(p for p in leaves if p.startswith(flag_part + "."))
            hint = f" Try one of: {', '.join('--' + p for p in sub_flags[:5])}" if sub_flags else ""
            raise ConfigFileError(
                f"--{flag_part} is a config group, not a leaf field.{hint}"
            )

        # 4. Leaf field lookup.
        meta = leaves.get(flag_part)
        if meta is None:
            unknown.append((flag_part, eq_value))
            # Consume value-shaped follower so it doesn't pollute ``remaining``.
            if eq_value is None and i + 1 < n and not args[i + 1].startswith("--"):
                i += 2
            else:
                i += 1
            continue

        if meta.is_bool:
            if eq_value is not None:
                coerced = _coerce_bool_literal(eq_value)
                _set_nested(overrides, meta.snake_path, coerced if coerced is not None else eq_value)
                i += 1
            elif i + 1 < n and (lit := _coerce_bool_literal(args[i + 1])) is not None:
                _set_nested(overrides, meta.snake_path, lit)
                i += 2
            else:
                _set_nested(overrides, meta.snake_path, True)
                i += 1
            continue

        if meta.is_list and eq_value is None:
            # Greedy consume contiguous non-flag tokens. ``--`` (any length ≥2) is a boundary;
            # a single ``-`` prefix (e.g. ``-1e-3``) is a value.
            j = i + 1
            values: list[str] = []
            while j < n and not args[j].startswith("--"):
                values.append(args[j])
                j += 1
            if values:
                _set_nested(overrides, meta.snake_path, values)
                i = j
            else:
                # Bare ``--items`` with no follower — treat as empty list override.
                _set_nested(overrides, meta.snake_path, [])
                i += 1
            continue

        # 5. Scalar field: consume the next token (or use the ``=value``).
        if eq_value is not None:
            _set_nested(overrides, meta.snake_path, eq_value)
            i += 1
            continue
        if i + 1 >= n:
            raise ConfigFileError(f"--{flag_part} requires a value")
        _set_nested(overrides, meta.snake_path, args[i + 1])
        i += 2

    if unknown:
        import difflib

        lines = ["Unrecognized command-line option(s):"]
        valid_kebab = sorted(leaves)
        for flag, _ in unknown:
            suggestions = difflib.get_close_matches(flag, valid_kebab, n=3, cutoff=0.6)
            if suggestions:
                lines.append(f"  --{flag}   (did you mean {', '.join('--' + s for s in suggestions)}?)")
            else:
                lines.append(f"  --{flag}")
        raise ConfigFileError("\n".join(lines))

    return remaining, overrides


def _canonical_key_map(cls: type) -> dict[str, str]:
    """For each field on ``cls`` with a ``validation_alias``, return a map from
    every accepted spelling to a single canonical key (the first
    ``AliasChoices`` entry, or the alias string when a bare string is used).

    Fields without an alias are absent from the map, so plain keys pass through
    unchanged. The canonical name is the one pydantic prefers when multiple
    matches are present — using it consistently lets multi-source merges
    (TOML + CLI) collapse to a single key before validation.
    """
    mapping: dict[str, str] = {}
    if not hasattr(cls, "model_fields"):
        return mapping
    for field_name, finfo in cls.model_fields.items():
        alias = finfo.validation_alias
        if alias is None:
            continue
        if isinstance(alias, AliasChoices):
            choices = [c for c in alias.choices if isinstance(c, str)]
            if not choices:
                continue
            primary = choices[0]
            for c in choices:
                mapping[c] = primary
        elif isinstance(alias, str):
            mapping[alias] = alias
    return mapping


def _normalize_alias_keys(cls: type, data: Any) -> Any:
    """Recursively rewrite alias keys in ``data`` to their canonical form.

    Without this pass, a field reachable under two names (e.g. ``seed`` and
    ``random_seed`` via ``AliasChoices("seed", "random_seed")``) can survive
    the deep-merge step with both keys present — TOML supplying one, CLI the
    other — and ``extra="forbid"`` then rejects the duplicate. Iteration order
    is preserved so the higher-precedence layer in ``_deep_merge`` (CLI > file
    > default) wins when both names collide on the same field.
    """
    if not isinstance(data, dict) or not hasattr(cls, "model_fields"):
        return data

    key_map = _canonical_key_map(cls)

    # Map every accepted key (canonical + aliases) to its inner BaseModel,
    # used to recurse into nested groups while normalizing.
    inner_map: dict[str, type] = {}
    for field_name, finfo in cls.model_fields.items():
        inner = _strip_annotated(finfo.annotation)
        if not (isinstance(inner, type) and issubclass(inner, BaseModel)):
            continue
        inner_map[field_name] = inner
        alias = finfo.validation_alias
        if isinstance(alias, AliasChoices):
            for c in alias.choices:
                if isinstance(c, str):
                    inner_map[c] = inner
        elif isinstance(alias, str):
            inner_map[alias] = inner

    result: dict = {}
    for key, value in data.items():
        canonical = key_map.get(key, key)
        inner_cls = inner_map.get(key)
        if inner_cls is not None and isinstance(value, dict):
            value = _normalize_alias_keys(inner_cls, value)
        result[canonical] = value
    return result


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
        # 1. Parse ``@ file.toml`` references into a raw TOML/JSON/YAML dict.
        remaining_args, root_config, nested_configs = _process_args(args)
        toml_dict = root_config
        for key_path, config in nested_configs.items():
            nested = _nest_config(key_path, config)
            toml_dict = _deep_merge(toml_dict, nested)

        # 2. Pre-extract CLI args that can't be matched by leaf path lookup —
        #    Optional[BaseModel] / discriminated-union sub-flags, and JSON-encoded
        #    list/dict values. These end up in ``cli_overrides`` as raw strings
        #    (or already-parsed JSON), to be deep-merged with TOML at the end.
        cli_overrides: dict = {}
        optional_paths = _find_optional_model_paths(cls)
        if optional_paths:
            remaining_args, bare_overrides = _expand_bare_optional_flags(remaining_args, optional_paths)
            cli_overrides = _deep_merge(cli_overrides, bare_overrides)

        json_paths = _find_json_value_field_paths(cls)
        if json_paths:
            remaining_args, json_overrides = _extract_json_value_args(remaining_args, json_paths)
            cli_overrides = _deep_merge(cli_overrides, json_overrides)

        # 3. --help is rendered from ``cls.model_fields``, no tyro round-trip.
        if "--help" in remaining_args or "-h" in remaining_args:
            sys.stdout.write(_render_help(cls, prog=prog, description=description))
            sys.exit(0)

        # 4. Parse remaining ``--flag value`` / ``--flag=value`` tokens against the
        #    leaf paths of ``cls``. The parser handles bools (``--no-flag``), typed
        #    lists, aliases, and emits a ``ConfigFileError`` with suggestions on
        #    unknown flags.
        remaining_args, flag_overrides = _parse_cli_to_dict(remaining_args, cls, optional_paths)
        cli_overrides = _deep_merge(cli_overrides, flag_overrides)

        if remaining_args:
            raise ConfigFileError(
                f"Unrecognized arguments: {' '.join(remaining_args)}"
            )

        # 5. Compose the precedence layers and validate once. Order:
        #    caller default  ⊂  TOML/@-file  ⊂  CLI overrides.
        #    Validators on ``cls`` fire exactly once on the merged dict, so
        #    ``model_fields_set`` faithfully records what the user wrote.
        default_dict: dict = {}
        if default is not None and isinstance(default, BaseModel):
            default_dict = default.model_dump(exclude_unset=True)
        merged = _deep_merge(_deep_merge(default_dict, toml_dict), cli_overrides)
        # Collapse ``validation_alias`` duplicates so e.g. ``seed`` (TOML) +
        # ``random_seed`` (CLI alias) merge into a single canonical key.
        merged = _normalize_alias_keys(cls, merged)
        try:
            return cls.model_validate(merged)
        except ValidationError as e:
            # Surface pydantic errors with CLI-flag-flavoured wording so users
            # see ``--foo: Input should be a valid integer (got 'dskfj')`` rather
            # than a raw pydantic_core traceback.
            raise ConfigFileError(_format_validation_error_for_cli(e)) from e
    except ConfigFileError as e:
        # Only print formatted error when running from CLI (sys.argv);
        # when args are explicitly passed, re-raise for programmatic handling.
        if use_sys_argv:
            _print_config_error_and_exit(e)
        raise
