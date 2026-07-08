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

import ast
import copy
import difflib
import importlib.util
import inspect
import json
import os
import re
import shutil
import sys
import textwrap
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
        if not (float_val != float_val) and v not in ("inf", "-inf", "+inf"):  # reject NaN/inf
            # Guard against coercing strings with leading zeros (e.g. "007")
            # that aren't valid float literals in the conventional sense.
            if v[:1].isdigit() and v != str(float_val) and not any(c in v for c in "eE."):
                pass
            else:
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


def _dict_value_type_is_str(annotation: type) -> bool:
    """``True`` if ``annotation`` is ``dict[K, str]`` (annotated value type is ``str``)."""
    if hasattr(annotation, "__metadata__"):
        annotation = get_args(annotation)[0]
    if get_origin(annotation) is not dict:
        return False
    args = get_args(annotation)
    return len(args) == 2 and args[1] is str


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

        CLI-parsed dict values arrive as strings. This detects dict fields
        whose values are all strings and converts them back to int/float/bool.
        Fields annotated ``dict[K, str]`` are left alone — the values are
        already the correct type and coercion would clobber e.g. ``"0"`` → ``0``.
        """
        if not isinstance(data, dict):
            return data
        for field_name, field_info in cls.model_fields.items():
            if not _is_dict_annotation(field_info.annotation) or field_name not in data:
                continue
            if _dict_value_type_is_str(field_info.annotation):
                continue
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


def _resolve_bool_option(explicit: bool | None, env_var: str, default: bool) -> bool:
    """Resolve a boolean option: explicit arg > env var > default."""
    if explicit is not None:
        return explicit
    env_val = os.environ.get(env_var, "").lower()
    if env_val in ("1", "true", "yes"):
        return True
    if env_val in ("0", "false", "no"):
        return False
    return default


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


def _term_width(wide: bool = True) -> int:
    """Width to render boxes at.

    ``wide=True`` (default): full terminal width (floor 40).
    ``wide=False``: capped at 80 columns.
    """
    cols = shutil.get_terminal_size().columns
    if wide:
        return max(40, cols)
    return min(80, max(40, cols))


def _print_config_error_and_exit(
    error: ConfigFileError, *, plain: bool = False, wide: bool = True
) -> None:
    """Print a config file error in a nice box format and exit."""
    use_color = not plain and _supports_color()

    def colorize(text: str, *codes: str) -> str:
        if not use_color:
            return text
        return "".join(codes) + text + _RESET

    width = _term_width(wide)
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
        return f"{colorize(vert, _RED)} {content}{' ' * padding} {colorize(vert, _RED)}"

    # Build the error message content
    lines = []

    # Title line
    title = "Config file error"
    title_plain_len = 2 + len(title) + 1
    lines.append(
        colorize(top_left, _RED)
        + f"{horiz} {colorize(title, _RED, _BOLD)} "
        + colorize(horiz * (width - title_plain_len - 2) + top_right, _RED)
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
            lines.append(box_line(colorize(horiz * inner_width, _RED)))

            # Pydantic error details
            pydantic_lines = parts[1].split("\n")
            for pydantic_line in pydantic_lines:
                if not pydantic_line:
                    continue
                # First line (validation error count)
                if "validation error" in pydantic_line:
                    for wrapped in wrap_text(pydantic_line, inner_width):
                        lines.append(box_line(colorize(wrapped, _BRIGHT_RED)))
                # Field name (not indented)
                elif pydantic_line and not pydantic_line.startswith(" "):
                    for wrapped in wrap_text(f"  {pydantic_line}", inner_width):
                        lines.append(box_line(colorize(wrapped, _BOLD)))
                # Error details (indented)
                elif pydantic_line.startswith("  "):
                    for wrapped in wrap_text(f"    {pydantic_line.strip()}", inner_width):
                        lines.append(box_line(colorize(wrapped, _DIM)))
        else:
            for line in wrap_text(message, inner_width):
                lines.append(box_line(line))
    else:
        for line in wrap_text(message, inner_width):
            lines.append(box_line(line))

    # Bottom border
    lines.append(colorize(f"{bot_left}{horiz * (width - 2)}{bot_right}", _RED))

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


def _suggest_flag(unknown: str, known: list[str], n: int = 1) -> list[str]:
    """Return up to ``n`` close matches for ``unknown`` from ``known`` flags."""
    return difflib.get_close_matches(unknown, known, n=n, cutoff=0.6)


def _env_var_for_loc(loc: tuple, env_locs: dict[tuple[str, ...], str]) -> str | None:
    """Find the env var that supplied the value at ``loc``, if any.

    Exact match on the str segments first; then retry with one interior
    segment dropped, because a discriminated-union error loc carries the
    variant tag (``("optimizer", "adamw", "momentum")``) that the injected
    env path (``("optimizer", "momentum")``) doesn't have.
    """
    parts = tuple(s for s in loc if isinstance(s, str))
    if parts in env_locs:
        return env_locs[parts]
    for i in range(1, len(parts) - 1):
        candidate = parts[:i] + parts[i + 1 :]
        if candidate in env_locs:
            return env_locs[candidate]
    return None


def _format_validation_error_for_cli(
    error: ValidationError,
    known_flags: list[str] | None = None,
    env_locs: dict[tuple[str, ...], str] | None = None,
) -> str:
    """Render a ``pydantic.ValidationError`` as a CLI-flag-flavoured multi-line
    message suitable for ``_print_config_error_and_exit``.

    Each Pydantic error becomes one row of the form
    ``<cli-flag>: <msg> (got <input>)``. The input is omitted when it's a
    container (dict / list), since the per-leaf errors that follow show the
    real culprit.

    When ``known_flags`` is provided, "extra fields not permitted" errors
    get a "did you mean --X?" suggestion via ``difflib``. When ``env_locs``
    marks a loc as env-var-supplied, the row is headed by the env var name
    (``PRL_FOO__BAR (from env)``) instead of a CLI flag.
    """
    errors = error.errors()
    count = len(errors)
    header = f"Failed to validate config: {count} validation error{'s' if count != 1 else ''} for {error.title}"
    lines: list[str] = [header]
    for err in errors:
        loc = err.get("loc", ())
        flag = _loc_to_cli_flag(tuple(loc))
        if env_locs:
            env_var = _env_var_for_loc(tuple(loc), env_locs)
            if env_var is not None:
                flag = f"{env_var} (from env)"
        msg = err.get("msg") or err.get("type", "validation error")
        input_value = err.get("input")
        suffix = ""
        if input_value is not None and not isinstance(input_value, (dict, list)):
            suffix = f" (got {input_value!r})"
        if known_flags and err.get("type") == "extra_forbidden" and loc:
            unknown_kebab = str(loc[-1]).replace("_", "-")
            suggestions = _suggest_flag(unknown_kebab, known_flags)
            if suggestions:
                suffix += f"  — did you mean --{suggestions[0]}?"
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
        - remaining_args: args with config file refs removed
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


def _nest_config(key_path: str, config: Any) -> dict:
    """
    Nest a value under a dotted key path.

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


def _model_fields_shorthand(cls: type) -> str:
    """Render a ``{field1, field2, ...}`` shorthand for a BaseModel's fields."""
    if not hasattr(cls, "model_fields"):
        return cls.__name__
    names = list(cls.model_fields.keys())
    if len(names) <= 4:
        return "{" + ", ".join(names) + "}"
    return "{" + ", ".join(names[:3]) + ", ...}"


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
        if inner is not None and isinstance(inner, type) and issubclass(inner, BaseModel):
            return f"list[{_model_fields_shorthand(inner)}]"
        return f"[{_format_type_for_help(inner)} [...]]"
    if origin is dict:
        args = get_args(annotation)
        if len(args) == 2 and isinstance(args[1], type) and issubclass(args[1], BaseModel):
            key_str = _format_type_for_help(args[0])
            return f"dict[{key_str}, {_model_fields_shorthand(args[1])}]"
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
    if isinstance(default, list) and default and isinstance(default[0], BaseModel):
        n = len(default)
        cls_name = type(default[0]).__name__
        return f"default: [{n} {cls_name}]" if n > 1 else f"default: [1 {cls_name}]"
    if default is None:
        return "default: None"
    if repr(default) == "PydanticUndefined":
        return "required"
    if callable(default):
        try:
            return _format_default_for_help(default())
        except Exception:
            return ""
    return f"default: {default}"


def _extract_field_docstrings(cls: type) -> dict[str, str]:
    """Extract PEP 224-style attribute docstrings from ``cls``.

    A string literal immediately following an annotated assignment is treated
    as that field's description::

        class Config(BaseConfig):
            seed: int = 42
            \"\"\"Random seed for reproducibility.\"\"\"

    Walks ``cls.__mro__`` so docstrings on inherited fields are also picked up;
    subclass docstrings shadow base-class docstrings.

    Returns a ``{field_name: docstring}`` dict. Fields without a trailing
    string literal are absent from the result.
    """
    result: dict[str, str] = {}
    for base in reversed(cls.__mro__):
        if base is object:
            continue
        try:
            source = textwrap.dedent(inspect.getsource(base))
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == base.__name__:
                class_node = node
                break
        else:
            continue

        body = class_node.body
        for i, stmt in enumerate(body):
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
                continue
            if i + 1 >= len(body):
                continue
            nxt = body[i + 1]
            if isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant) and isinstance(nxt.value.value, str):
                result[stmt.target.id] = nxt.value.value.strip()
    return result


def _field_description(finfo, docstring: str = "") -> str:
    """Return the description text for a field (``Field(description=...)``
    with PEP 224 docstring as fallback)."""
    return (finfo.description or docstring or "").strip()


def _field_short_flag(finfo) -> str | None:
    """Return the single-character ``validation_alias`` for a field, if any.

    This is the alias that ``_expand_short_flags`` exposes as a ``-x`` short
    flag, surfaced in ``--help`` as ``-x, --long`` (mirroring ``-h, --help``).
    """
    alias = finfo.validation_alias
    names: list[str] = []
    if isinstance(alias, AliasChoices):
        names = [c for c in alias.choices if isinstance(c, str)]
    elif isinstance(alias, str):
        names = [alias]
    for name in names:
        if len(name) == 1:
            return name
    return None


# A help row is (flag_with_metavar, description, annotation) where annotation
# is the right-aligned ``(default: X)`` / ``(required)`` tag.
_HelpRow = tuple[str, str, str]


# A single outlier flag (deep dotted path, or a Literal with a giant inline
# enum metavar) shouldn't widen the entire panel's flag column. Beyond this
# cap, oversize flags are rendered on their own line and the description
# wraps below them.
_MAX_FLAG_COL = 60


def _render_panel(
    title: str, rows: list[_HelpRow], term_width: int, min_flag_w: int = 0
) -> list[str]:
    """Render a box-drawn help panel.

    ``rows`` is a list of ``(flag, description, annotation)`` triples. The
    annotation (``default: X`` / ``required``) is appended inline to the
    description in parentheses. Long descriptions wrap onto continuation
    lines indented to the description column. Flags longer than
    ``_MAX_FLAG_COL`` are emitted on their own line so they can't starve
    the description column.
    """
    if not rows:
        return []

    # Merge annotation into description so we render a single text column.
    merged: list[tuple[str, str]] = [
        (flag, f"{desc} ({anno})" if desc and anno else (anno or desc))
        for flag, desc, anno in rows
    ]

    short_flag_widths = [len(f) for f, _ in merged if len(f) <= _MAX_FLAG_COL]
    flag_w = max(min_flag_w, max(short_flag_widths, default=0))
    desc_col = flag_w + 2  # where descriptions start

    # Box width is driven by the fixed flag column, never by description
    # or title length — both wrap to fit.
    box_total = max(term_width, desc_col + 4)
    inner = box_total - 4
    max_desc_w = inner - desc_col

    def _wrap(text: str, width: int) -> list[str]:
        # Hard-break tokens longer than ``width`` (e.g. a giant
        # ``Literal[...]`` metavar) so they can't overflow the box.
        words: list[str] = []
        for raw in text.split():
            while len(raw) > width:
                words.append(raw[:width])
                raw = raw[width:]
            words.append(raw)
        out: list[str] = []
        cur = ""
        for word in words:
            if not cur:
                cur = word
            elif len(cur) + 1 + len(word) <= width:
                cur += " " + word
            else:
                out.append(cur)
                cur = word
        if cur:
            out.append(cur)
        return out

    # Title goes on the top border up to ``box_total - 6`` chars (room for
    # the ``╭─ `` prefix and ``  ─╮`` suffix). Overflow wraps into ``│ …  │``
    # continuation rows below the border.
    title_chunks = _wrap(title, box_total - 6) if title else [""]
    horiz = "─" * max(1, box_total - len(title_chunks[0]) - 5)
    out: list[str] = [f"╭─ {title_chunks[0]} {horiz}╮"]
    for chunk in title_chunks[1:]:
        out.append(f"│ {chunk:<{inner}} │")

    def _box_line(text: str) -> str:
        return f"│ {text:<{inner}} │"

    for flag, text in merged:
        # Oversize flag: print on its own line, then wrap text below.
        # If the flag itself is longer than ``inner``, wrap it too — long
        # ``Literal`` metavars (e.g. ``{a,b,c,d,e}``) can blow past the box.
        if len(flag) > flag_w:
            flag_chunks = [flag] if len(flag) <= inner else _wrap(flag, inner)
            for chunk in flag_chunks:
                out.append(_box_line(chunk))
            wrapped = _wrap(text, max_desc_w) if text and max_desc_w > 0 else ([text] if text else [""])
            for chunk in wrapped:
                out.append(_box_line(f"{' ' * desc_col}{chunk}"))
            continue

        if text and max_desc_w > 0 and len(text) > max_desc_w:
            wrapped = _wrap(text, max_desc_w)
            first = wrapped[0] if wrapped else ""
            out.append(_box_line(f"{flag:<{flag_w}}  {first}"))
            for chunk in wrapped[1:]:
                out.append(_box_line(f"{' ' * desc_col}{chunk}"))
        else:
            out.append(_box_line(f"{flag:<{flag_w}}  {text}".rstrip()))

    out.append(f"╰{'─' * (box_total - 2)}╯")
    return out


def _strip_annotated(annotation):
    """Unwrap ``Annotated[T, ...]`` → ``T``."""
    if hasattr(annotation, "__metadata__"):
        return get_args(annotation)[0]
    return annotation


def _is_union(annotation) -> bool:
    """``True`` if ``annotation`` is ``Union[...]`` or PEP-604 ``X | Y``."""
    origin = get_origin(annotation)
    return origin is Union or origin is getattr(types, "UnionType", None)


_HelpPanel = tuple[str, list[_HelpRow]]


def _list_inner_model(annotation) -> type | None:
    """If ``annotation`` is ``list[SomeBaseModel]``, return ``SomeBaseModel``."""
    inner = _strip_annotated(annotation)
    if get_origin(inner) is list:
        args = get_args(inner)
        if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            return args[0]
    if _is_union(inner):
        non_none = [a for a in get_args(inner) if a is not type(None)]
        if len(non_none) == 1:
            return _list_inner_model(non_none[0])
    return None


def _dict_inner_model(annotation) -> type | None:
    """If ``annotation`` is ``dict[K, SomeBaseModel]``, return ``SomeBaseModel``."""
    inner = _strip_annotated(annotation)
    if get_origin(inner) is dict:
        args = get_args(inner)
        if len(args) == 2 and isinstance(args[1], type) and issubclass(args[1], BaseModel):
            return args[1]
    if _is_union(inner):
        non_none = [a for a in get_args(inner) if a is not type(None)]
        if len(non_none) == 1:
            return _dict_inner_model(non_none[0])
    return None


def _panel_description(inner_cls: type, finfo, field_docstring: str = "") -> str:
    """Derive a one-line description for a sub-config panel title.

    Precedence: ``Field(description=...)`` > PEP 224 docstring below the field
    > the inner class's ``__doc__`` (first line only).
    """
    desc = (finfo.description or field_docstring or "").strip()
    if desc:
        return desc
    cls_doc = getattr(inner_cls, "__doc__", None)
    if cls_doc:
        return cls_doc.strip().split("\n")[0].strip()
    return ""


def _collect_help_panels(
    cls: type, prefix: str = ""
) -> tuple[list[_HelpRow], list[_HelpPanel]]:
    """Walk ``cls.model_fields`` to collect help rows and sub-panel specs.

    Returns ``(rows, sub_panels)`` where ``rows`` is the list of leaf
    ``(flag, annotation)`` pairs to emit in the *current* panel and
    ``sub_panels`` is a list of ``(title, rows)`` tuples for every
    sub-config / Optional[BaseModel] / multi-model union variant.
    """
    rows: list[_HelpRow] = []
    sub_panels: list[_HelpPanel] = []
    docstrings = _extract_field_docstrings(cls)

    def _make_row(flag: str, finfo, ds: str = "") -> _HelpRow:
        default = finfo.default
        if repr(default) == "PydanticUndefined" and finfo.default_factory is not None:
            default = finfo.default_factory
        return (flag, _field_description(finfo, ds), _format_default_for_help(default))

    for fname, finfo in cls.model_fields.items():
        kebab = fname.replace("_", "-")
        full_path = f"{prefix}.{kebab}" if prefix else kebab
        annotation = finfo.annotation

        if _is_multi_model_union(annotation):
            inner = _strip_annotated(annotation)
            variants = [a for a in get_args(inner) if a is not type(None)]
            for variant_cls in variants:
                vdocs = _extract_field_docstrings(variant_cls)
                vrows: list[_HelpRow] = []
                for vfname, vfinfo in variant_cls.model_fields.items():
                    type_str = _format_type_for_help(vfinfo.annotation)
                    flag = f"--{full_path}.{vfname.replace('_', '-')}"
                    if type_str:
                        flag = f"{flag} {type_str}"
                    vrows.append(_make_row(flag, vfinfo, vdocs.get(vfname, "")))
                sub_panels.append((f"{full_path} variant: {variant_cls.__name__}", vrows))
            continue

        if _is_optional_model(annotation):
            inner = _strip_annotated(annotation)
            non_none = [a for a in get_args(inner) if a is not type(None)]
            inner_cls = non_none[0]
            child_rows, child_panels = _collect_help_panels(inner_cls, prefix=full_path)
            if isinstance(finfo.default, BaseModel):
                tag = "optional, enabled by default"
            else:
                tag = "optional, default: None"
            desc = _panel_description(inner_cls, finfo, docstrings.get(fname, ""))
            title = f"{full_path} options ({tag})"
            if desc:
                title = f"{full_path}: {desc} ({tag})"
            sub_panels.append((title, child_rows))
            sub_panels.extend(child_panels)
            continue

        inner = _strip_annotated(annotation)
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            child_rows, child_panels = _collect_help_panels(inner, prefix=full_path)
            desc = _panel_description(inner, finfo, docstrings.get(fname, ""))
            title = f"{full_path}: {desc}" if desc else f"{full_path} options"
            sub_panels.append((title, child_rows))
            sub_panels.extend(child_panels)
            continue

        # list[BaseModel] or dict[str, BaseModel] — show as a leaf with a
        # reference sub-panel describing the inner model's fields.
        list_model = _list_inner_model(annotation)
        dict_model = _dict_inner_model(annotation)
        item_cls = list_model or dict_model
        if item_cls is not None:
            type_str = _format_type_for_help(annotation)
            short = _field_short_flag(finfo)
            name = f"-{short}, --{full_path}" if short else f"--{full_path}"
            flag = f"{name} {type_str}" if type_str else name
            rows.append(_make_row(flag, finfo, docstrings.get(fname, "")))
            child_rows, child_panels = _collect_help_panels(item_cls, prefix=f"{full_path}[*]")
            tag = "list item" if list_model else "dict value"
            desc = _panel_description(item_cls, finfo, docstrings.get(fname, ""))
            title = f"{full_path}[*]: {desc} ({tag}, via @ file or JSON)" if desc else f"{full_path}[*] fields ({tag}, via @ file or JSON)"
            sub_panels.append((title, child_rows))
            sub_panels.extend(child_panels)
            continue

        # Leaf field.
        type_str = _format_type_for_help(annotation)
        short = _field_short_flag(finfo)
        name = f"-{short}, --{full_path}" if short else f"--{full_path}"
        flag = f"{name} {type_str}" if type_str else name
        rows.append(_make_row(flag, finfo, docstrings.get(fname, "")))

    return rows, sub_panels


def _render_help(
    cls: type, prog: str | None = None, description: str | None = None, wide: bool = True
) -> str:
    """Render the full ``--help`` text for ``cls`` as a single string.

    Layout: a usage line, optional description, an "options" panel listing
    leaf fields directly on ``cls``, then a separate panel per sub-config
    (recursively flattened with dotted paths), per Optional[BaseModel] field
    (annotated "(optional, default: None)"), and per multi-model union
    variant. Reuses ``_render_panel`` so all panels share the same
    box-drawing style.
    """
    prog = prog or os.path.basename(sys.argv[0])
    term_width = _term_width(wide)

    root_rows, sub_panels = _collect_help_panels(cls)

    header_rows: list[_HelpRow] = [("-h, --help", "show this help message and exit", ""), *root_rows]
    all_panels: list[_HelpPanel] = [("options", header_rows), *sub_panels]

    lines: list[str] = [f"usage: {prog} [-h] [@ FILE] [OPTIONS]"]
    if description:
        lines.append("")
        for paragraph in description.splitlines():
            lines.append(paragraph)
    lines.append("")

    # Each panel sizes its flag column to its own widest flag. A single
    # global width would let one deeply-nested flag (e.g.
    # ``--orchestrator.eval.env[*].sampling.reasoning-effort``) starve
    # every other panel's description column.
    for title, rows in all_panels:
        lines.extend(_render_panel(title, rows, term_width))
        lines.append("")

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


def _find_short_flag_paths(cls: type, prefix: str = "") -> dict[str, str]:
    """Map single-dash short flags to the long CLI path they stand in for.

    A *short flag* is declared as a single-character entry in a field's
    ``validation_alias`` (``AliasChoices``). For example::

        n_samples: Annotated[int, Field(validation_alias=AliasChoices("n_samples", "n"))] = 1

    registers ``-n`` → ``--n-samples`` (the field's canonical long flag). Because
    expansion happens before every other arg pass, ``-n`` becomes a true synonym
    for ``--n-samples`` across all config forms — scalars, bools, lists, JSON
    dict/list values, optional sub-configs, and TOML overrides — with no special
    casing in the parser.

    Returns ``{char: long_kebab_path}``. Nested fields produce dotted paths
    (e.g. a short flag on ``model.lr`` maps to ``model.lr``). On collision the
    last-registered field wins; short flags are expected to be unique like any
    short option.
    """
    shorts: dict[str, str] = {}
    if not hasattr(cls, "model_fields"):
        return shorts
    for field_name, finfo in cls.model_fields.items():
        kebab = field_name.replace("_", "-")
        full_path = f"{prefix}.{kebab}" if prefix else kebab

        short = _field_short_flag(finfo)
        if short is not None:
            shorts[short] = full_path

        inner = _strip_annotated(finfo.annotation)
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            shorts.update(_find_short_flag_paths(inner, prefix=full_path))
    return shorts


def _expand_short_flags(args: list[str], shorts: dict[str, str]) -> list[str]:
    """Rewrite single-dash short flags to their long ``--`` equivalents.

    Handles ``-n value`` and ``-n=value`` (the value, if attached, is preserved
    on the rewritten flag). Tokens that aren't a registered short flag — bare
    ``-``, ``--`` long flags, and negative-number values like ``-1e-3`` — pass
    through untouched, so this is safe to run before the other arg passes.
    """
    if not shorts:
        return args
    out: list[str] = []
    for token in args:
        if token.startswith("-") and not token.startswith("--") and len(token) > 1:
            key, eq, value = token[1:].partition("=")
            if key in shorts:
                long = f"--{shorts[key]}"
                out.append(f"{long}={value}" if eq else long)
                continue
        out.append(token)
    return out


def _match_optional_prefix(path: str, optional_paths: set[str]) -> str | None:
    """If ``path`` starts with an optional model path followed by '.', return it."""
    for opt_path in optional_paths:
        if path.startswith(opt_path + "."):
            return opt_path
    return None


def _expand_bare_optional_flags(
    args: list[str], optional_paths: set[str]
) -> tuple[list[str], dict]:
    """Handle CLI args for Optional[BaseModel] fields.

    Patterns:
    1. ``--wandb`` (bare flag) — enable with defaults.
    2. ``--wandb None`` — disable (set to None).
    3. ``--no-wandb`` — disable (set to None).
    4. ``--wandb.project foo`` — enable + sub-field override.

    Returns (remaining_args, config_overrides_as_nested_dict).
    """
    remaining: list[str] = []
    overrides: dict = {}

    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            path = arg[2:]

            # Pattern 3: --no-<optional> negation (e.g. --no-wandb)
            if path.startswith("no-") and path[3:] in optional_paths:
                snake_path = path[3:].replace("-", "_")
                nested = _nest_config(snake_path, "None")
                overrides = _deep_merge(overrides, nested)
                i += 1
                continue

            # Pattern 3b: --no-<optional>.<field> (e.g. --no-compile.fullgraph)
            if path.startswith("no-"):
                no_stripped = path[3:]
                matched = _match_optional_prefix(no_stripped, optional_paths)
                if matched is not None:
                    snake_path = no_stripped.replace("-", "_")
                    nested = _nest_config(snake_path, False)
                    overrides = _deep_merge(overrides, nested)
                    i += 1
                    continue

            # Pattern 1 / 2: bare flag or explicit "None"
            if path in optional_paths:
                next_is_value = i + 1 < len(args) and not args[i + 1].startswith("-") and not args[i + 1].startswith("@")
                if next_is_value and args[i + 1] == "None":
                    # Pattern 2: --wandb None → disable
                    snake_path = path.replace("-", "_")
                    nested = _nest_config(snake_path, "None")
                    overrides = _deep_merge(overrides, nested)
                    i += 2
                    continue
                if not next_is_value:
                    # Pattern 1: bare flag → enable with defaults
                    snake_path = path.replace("-", "_")
                    nested = _nest_config(snake_path, {})
                    overrides = _deep_merge(overrides, nested)
                    i += 1
                    continue

            # Pattern 4: sub-field override (e.g. --wandb.project foo)
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


class _EnvVarMeta(typing.NamedTuple):
    snake_path: str  # dotted snake_case path using canonical (post-alias) keys
    annotation: object
    is_bool: bool
    is_json: bool  # dict/list-typed field — value must be a JSON literal
    is_model: bool = False  # nullable model field addressed as a leaf (enable/disable)


def _field_env_names(field_name: str, field_info) -> tuple[list[str], str]:
    """Accepted spellings for a field in env-var names, plus its canonical key.

    Multi-character ``validation_alias`` entries each get their own env name;
    single-character aliases are CLI short flags and don't make sense as env
    vars. All spellings inject under the canonical key (the first
    ``AliasChoices`` entry, matching ``_canonical_key_map``) so the alias
    normalization pass keeps env below TOML/CLI in precedence.
    """
    names = [field_name]
    canonical = field_name
    alias = field_info.validation_alias
    if isinstance(alias, AliasChoices):
        choices = [c for c in alias.choices if isinstance(c, str)]
        if choices:
            canonical = choices[0]
            names = list(dict.fromkeys([field_name, *choices]))
    elif isinstance(alias, str):
        canonical = alias
        names = list(dict.fromkeys([field_name, alias]))
    return [n for n in names if len(n) > 1 or n == field_name], canonical


def _build_env_var_map(cls: type, env_prefix: str = "", snake_prefix: str = "") -> dict[str, _EnvVarMeta]:
    """Walk ``cls.model_fields`` to build an env-var-name → ``_EnvVarMeta`` map.

    Names are UPPER_SNAKE path segments joined with a double underscore
    (``trainer.model.seq_len`` → ``TRAINER__MODEL__SEQ_LEN``); field names keep
    their own single underscores, so ``__`` is unambiguous as the level
    delimiter (a field name itself containing ``__`` would collide). The map is
    what makes lookup model-driven: only variables matching a real field are
    ever read, so unrelated ``<PREFIX>_*`` variables in the environment are
    ignored.

    Unlike ``_build_field_meta_map`` this recurses into ``Optional[BaseModel]``
    fields (the field also stays addressable as a leaf so ``PREFIX_WANDB=None``
    can disable it) and into every variant of a discriminated union (the
    discriminator picks the variant at validation; setting a field of the
    non-selected variant fails validation the same way a TOML key would).
    """
    entries: dict[str, _EnvVarMeta] = {}
    if not hasattr(cls, "model_fields"):
        return entries

    for field_name, field_info in cls.model_fields.items():
        annotation = field_info.annotation
        names, canonical = _field_env_names(field_name, field_info)
        snake_path = f"{snake_prefix}.{canonical}" if snake_prefix else canonical
        env_names = [f"{env_prefix}__{n.upper()}" if env_prefix else n.upper() for n in names]

        inner = _strip_annotated(annotation)
        if _is_json_value_field(annotation):
            for env_name in env_names:
                entries[env_name] = _EnvVarMeta(snake_path, annotation, is_bool=False, is_json=True)
        elif isinstance(inner, type) and issubclass(inner, BaseModel):
            for env_name in env_names:
                entries.update(_build_env_var_map(inner, env_name, snake_path))
        elif _is_optional_model(annotation) or _is_multi_model_union(annotation):
            models = [a for a in get_args(inner) if a is not type(None)]
            for env_name in env_names:
                if len(models) < len(get_args(inner)):  # None allowed → settable as a leaf
                    entries[env_name] = _EnvVarMeta(snake_path, annotation, is_bool=False, is_json=False, is_model=True)
                for model in models:
                    entries.update(_build_env_var_map(model, env_name, snake_path))
        else:
            for env_name in env_names:
                entries[env_name] = _EnvVarMeta(
                    snake_path, annotation, is_bool=_annotation_is_bool(annotation), is_json=False
                )

    return entries


def _collect_env_overrides(cls: type, env_prefix: str) -> tuple[dict, dict[tuple[str, ...], str]]:
    """Read ``<PREFIX>_<PATH>`` environment variables into a nested override dict.

    Returns the dict plus an attribution map (snake-path tuple → env var name)
    used to point validation errors at the env var instead of a CLI flag.
    """
    prefix = env_prefix.rstrip("_")
    overrides: dict = {}
    locs: dict[tuple[str, ...], str] = {}
    for name, meta in _build_env_var_map(cls).items():
        full_name = f"{prefix}_{name}"
        raw = os.environ.get(full_name)
        if raw is None:
            continue
        value: Any = raw
        if meta.is_bool:
            coerced = _coerce_bool_literal(raw)
            if coerced is not None:
                value = coerced
        elif meta.is_model:
            # Mirror the CLI's semantics for nullable model fields: true is the
            # bare --flag (enable with defaults), false is --no-flag (disable).
            coerced = _coerce_bool_literal(raw)
            if coerced is True:
                value = {}
            elif coerced is False:
                value = "None"
        elif meta.is_json:
            stripped = raw.strip()
            if not stripped.startswith(("{", "[")):
                raise ConfigFileError(
                    f"Invalid value for environment variable {full_name}: "
                    f"list/dict fields take a JSON literal, e.g. '[1,2]' (got {raw!r})"
                )
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ConfigFileError(f"Invalid JSON in environment variable {full_name}: {e}")
        _set_nested(overrides, meta.snake_path, value)
        locs[tuple(meta.snake_path.split("."))] = full_name
    return overrides, locs


def _path_present(data: dict, parts: tuple[str, ...]) -> bool:
    """``True`` if ``parts`` names an existing path inside nested dict ``data``."""
    node: Any = data
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


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
      - Unknown leaves are stored as overrides under their raw path so
        ``model_validator(mode="before")`` can remap legacy keys. If no
        validator handles them, pydantic's ``extra="forbid"`` rejects them.
    """
    leaves, interior = _build_field_meta_map(cls)
    remaining: list[str] = []
    overrides: dict = {}

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
            # Unknown flag — store it as an override under its raw path.
            # A ``model_validator(mode="before")`` on the config class can
            # remap legacy keys (e.g. ``model.*`` → ``student.model.*``).
            # If no validator handles it, pydantic's ``extra="forbid"``
            # rejects it at validation time.
            snake_path = flag_part.replace("-", "_")
            if eq_value is not None:
                _set_nested(overrides, snake_path, eq_value)
                i += 1
            elif i + 1 < n and not args[i + 1].startswith("--"):
                _set_nested(overrides, snake_path, args[i + 1])
                i += 2
            else:
                _set_nested(overrides, snake_path, True)
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
def cli(cls: type[T], *, env_prefix: str | None = ...) -> T: ...


@overload
def cli(cls: type[T], *, args: list[str], env_prefix: str | None = ...) -> T: ...


@overload
def cli(cls: type[T], *, default: T, env_prefix: str | None = ...) -> T: ...


@overload
def cli(cls: type[T], *, args: list[str], default: T, env_prefix: str | None = ...) -> T: ...


def cli(
    cls: type[T],
    *,
    args: list[str] | None = None,
    default: T | None = None,
    env_prefix: str | None = None,
    prog: str | None = None,
    description: str | None = None,
    plain: bool | None = None,
    wide: bool | None = None,
) -> T:
    """
    Parse CLI arguments into a typed config object, with support for config files.

    Supports loading config files using the @ syntax:
        - `@ config.toml` - Load root-level config
        - `--model @ model.toml` - Load config nested under 'model'
        - `--model @model.toml` - Same as above (no space)

    Args:
        cls: The type to parse into (Pydantic BaseConfig or BaseModel)
        args: Command line args to parse (defaults to sys.argv[1:])
        default: Default instance to use for missing values
        env_prefix: Enable environment-variable overrides under this prefix.
            Every field becomes settable via ``<PREFIX>_<PATH>``, where nesting
            levels are joined with a double underscore, e.g. ``env_prefix="PRL"``
            makes ``PRL_TRAINER__MODEL__SEQ_LEN`` set ``trainer.model.seq_len``.
            Precedence: CLI > config file > env var > ``default`` > class default.
        prog: Program name for help text
        description: Description for help text
        plain: Disable colored output. Falls back to env var
            ``PYDANTIC_CONFIG_PLAIN`` (default: False).
        wide: Use full terminal width for help and error panels. Falls back
            to env var ``PYDANTIC_CONFIG_WIDE`` (default: True).

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

    # Strip reserved CLI-level flags (--plain / --no-wide) before parsing the
    # user's config model. These override the env var / default but are
    # themselves overridden by an explicit ``cli(..., plain=True)`` call.
    if plain is None and "--plain" in args:
        args = [a for a in args if a != "--plain"]
        plain = True
    if wide is None and "--no-wide" in args:
        args = [a for a in args if a != "--no-wide"]
        wide = False

    plain_resolved = _resolve_bool_option(plain, "PYDANTIC_CONFIG_PLAIN", False)
    wide_resolved = _resolve_bool_option(wide, "PYDANTIC_CONFIG_WIDE", True)

    try:
        # 1. Parse ``@ file.toml`` references into a raw TOML/JSON/YAML dict.
        remaining_args, root_config, nested_configs = _process_args(args)
        toml_dict = root_config
        for key_path, config in nested_configs.items():
            nested = _nest_config(key_path, config)
            toml_dict = _deep_merge(toml_dict, nested)

        # 1b. Expand single-dash short flags (e.g. ``-n`` → ``--n-samples``)
        #     declared as single-character ``validation_alias`` entries. Runs
        #     first so every downstream pass sees the canonical long form.
        remaining_args = _expand_short_flags(remaining_args, _find_short_flag_paths(cls))

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

        # 3. --help is rendered directly from ``cls.model_fields``.
        if "--help" in remaining_args or "-h" in remaining_args:
            sys.stdout.write(_render_help(cls, prog=prog, description=description, wide=wide_resolved))
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

        # Build the set of known kebab-case leaf flags for "did you mean" hints.
        known_leaves, _ = _build_field_meta_map(cls)
        known_flags = sorted(known_leaves.keys())

        # 5. Compose the precedence layers and validate once. Order:
        #    caller default  ⊂  env vars  ⊂  TOML/@-file  ⊂  CLI overrides.
        #    Validators on ``cls`` fire exactly once on the merged dict, so
        #    ``model_fields_set`` faithfully records what the user wrote.
        default_dict: dict = {}
        if default is not None and isinstance(default, BaseModel):
            default_dict = default.model_dump(exclude_unset=True)
        env_dict: dict = {}
        env_locs: dict[tuple[str, ...], str] = {}
        if env_prefix:
            env_dict, env_locs = _collect_env_overrides(cls, env_prefix)
        if env_locs:
            # Attribute errors to an env var only where the env value is what
            # survived the merge; compare on canonical keys since TOML/CLI may
            # spell the field via an alias.
            norm_toml = _normalize_alias_keys(cls, toml_dict)
            norm_cli = _normalize_alias_keys(cls, cli_overrides)
            env_locs = {
                path: name
                for path, name in env_locs.items()
                if not (_path_present(norm_toml, path) or _path_present(norm_cli, path))
            }
        merged = _deep_merge(_deep_merge(_deep_merge(default_dict, env_dict), toml_dict), cli_overrides)
        # Collapse ``validation_alias`` duplicates so e.g. ``seed`` (TOML) +
        # ``random_seed`` (CLI alias) merge into a single canonical key.
        merged = _normalize_alias_keys(cls, merged)
        try:
            return cls.model_validate(merged)
        except ValidationError as e:
            # Surface pydantic errors with CLI-flag-flavoured wording so users
            # see ``--foo: Input should be a valid integer (got 'dskfj')`` rather
            # than a raw pydantic_core traceback.
            raise ConfigFileError(
                _format_validation_error_for_cli(e, known_flags=known_flags, env_locs=env_locs)
            ) from e
    except ConfigFileError as e:
        # Only print formatted error when running from CLI (sys.argv);
        # when args are explicitly passed, re-raise for programmatic handling.
        if use_sys_argv:
            _print_config_error_and_exit(e, plain=plain_resolved, wide=wide_resolved)
        raise
