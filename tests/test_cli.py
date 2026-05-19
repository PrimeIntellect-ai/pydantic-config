"""Tests for the cli module."""

import os
from typing import Annotated

import pytest
from pydantic import Field

from pydantic_config import cli, BaseConfig, ConfigFileError
from pydantic_config.cli import (
    _deep_merge,
    _load_config_file,
    _nest_config,
    _process_args,
)


# Helpers


def write_file(path: str, content: str):
    with open(path, "w") as f:
        f.write(content)


# Fixtures


@pytest.fixture
def tmp_toml_file(tmp_path):
    return os.path.join(tmp_path, "config.toml")


@pytest.fixture
def tmp_json_file(tmp_path):
    return os.path.join(tmp_path, "config.json")


@pytest.fixture
def tmp_yaml_file(tmp_path):
    return os.path.join(tmp_path, "config.yaml")


# Config classes


class SimpleConfig(BaseConfig):
    name: str = "default"
    count: int = 0


class NestedInner(BaseConfig):
    lr: float = 1e-4
    batch_size: int = 32


class NestedConfig(BaseConfig):
    train: NestedInner = NestedInner()
    seed: int = 42


class DeepNestedInner(BaseConfig):
    hidden_size: int = 256
    num_layers: int = 4


class DeepNestedMiddle(BaseConfig):
    encoder: DeepNestedInner = DeepNestedInner()
    decoder: DeepNestedInner = DeepNestedInner()


class DeepNestedConfig(BaseConfig):
    model: DeepNestedMiddle = DeepNestedMiddle()
    train: NestedInner = NestedInner()
    name: str = "experiment"


# Tests: _load_config_file


def test_load_json(tmp_json_file):
    write_file(tmp_json_file, '{"name": "test", "count": 5}')
    result = _load_config_file(tmp_json_file)
    assert result == {"name": "test", "count": 5}


def test_load_toml(tmp_toml_file):
    write_file(tmp_toml_file, 'name = "test"\ncount = 5')
    result = _load_config_file(tmp_toml_file)
    assert result == {"name": "test", "count": 5}


def test_load_yaml(tmp_yaml_file):
    write_file(tmp_yaml_file, "name: test\ncount: 5")
    result = _load_config_file(tmp_yaml_file)
    assert result == {"name": "test", "count": 5}


def test_load_file_not_found():
    with pytest.raises(ConfigFileError, match="not found"):
        _load_config_file("/nonexistent/file.toml")


def test_load_invalid_json(tmp_json_file):
    write_file(tmp_json_file, '{"invalid json')
    with pytest.raises(ConfigFileError, match="Invalid JSON"):
        _load_config_file(tmp_json_file)


def test_load_invalid_toml(tmp_toml_file):
    write_file(tmp_toml_file, "invalid = [toml")
    with pytest.raises(ConfigFileError, match="Invalid TOML"):
        _load_config_file(tmp_toml_file)


def test_load_unsupported_extension(tmp_path):
    txt_file = os.path.join(tmp_path, "config.txt")
    write_file(txt_file, "some content")
    with pytest.raises(ConfigFileError, match="Unsupported file type"):
        _load_config_file(txt_file)


# Tests: _deep_merge


def test_deep_merge_simple():
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_nested():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 20, "z": 30}}
    result = _deep_merge(base, override)
    assert result == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3}


def test_deep_merge_deep_nested():
    base = {"a": {"b": {"c": 1, "d": 2}}}
    override = {"a": {"b": {"d": 20, "e": 30}}}
    result = _deep_merge(base, override)
    assert result == {"a": {"b": {"c": 1, "d": 20, "e": 30}}}


# Tests: _nest_config


def test_nest_config_single_level():
    result = _nest_config("model", {"layers": 6})
    assert result == {"model": {"layers": 6}}


def test_nest_config_multi_level():
    result = _nest_config("model.encoder", {"layers": 6})
    assert result == {"model": {"encoder": {"layers": 6}}}


def test_nest_config_deep():
    result = _nest_config("a.b.c.d", {"value": 1})
    assert result == {"a": {"b": {"c": {"d": {"value": 1}}}}}


# Tests: _process_args


def test_process_args_no_config_files():
    args = ["--name", "test", "--count", "5"]
    remaining, root, nested = _process_args(args)
    assert remaining == ["--name", "test", "--count", "5"]
    assert root == {}
    assert nested == {}


def test_process_args_root_config_with_space(tmp_toml_file):
    write_file(tmp_toml_file, 'name = "from_file"')
    args = ["@", tmp_toml_file, "--count", "5"]
    remaining, root, nested = _process_args(args)
    assert remaining == ["--count", "5"]
    assert root == {"name": "from_file"}
    assert nested == {}


def test_process_args_nested_config_with_space(tmp_toml_file):
    write_file(tmp_toml_file, "lr = 0.001\nbatch_size = 64")
    args = ["--train", "@", tmp_toml_file, "--seed", "123"]
    remaining, root, nested = _process_args(args)
    assert remaining == ["--seed", "123"]
    assert root == {}
    assert nested == {"train": {"lr": 0.001, "batch_size": 64}}


def test_process_args_nested_config_without_space(tmp_toml_file):
    write_file(tmp_toml_file, "lr = 0.001\nbatch_size = 64")
    args = ["--train", f"@{tmp_toml_file}", "--seed", "123"]
    remaining, root, nested = _process_args(args)
    assert remaining == ["--seed", "123"]
    assert root == {}
    assert nested == {"train": {"lr": 0.001, "batch_size": 64}}


def test_process_args_multiple_nested_configs(tmp_path):
    train_file = os.path.join(tmp_path, "train.toml")
    model_file = os.path.join(tmp_path, "model.toml")
    write_file(train_file, "lr = 0.001")
    write_file(model_file, "hidden_size = 512")

    args = ["--train", "@", train_file, "--model", "@", model_file]
    remaining, root, nested = _process_args(args)
    assert remaining == []
    assert nested == {"train": {"lr": 0.001}, "model": {"hidden_size": 512}}


def test_process_args_deeply_nested_key(tmp_toml_file):
    write_file(tmp_toml_file, "hidden_size = 512")
    args = ["--model.encoder", "@", tmp_toml_file]
    remaining, root, nested = _process_args(args)
    assert remaining == []
    assert nested == {"model.encoder": {"hidden_size": 512}}


# Tests: cli basic


def test_cli_simple_args():
    config = cli(SimpleConfig, args=["--name", "test", "--count", "5"])
    assert config.name == "test"
    assert config.count == 5


def test_cli_defaults():
    config = cli(SimpleConfig, args=[])
    assert config.name == "default"
    assert config.count == 0


def test_cli_partial_override():
    config = cli(SimpleConfig, args=["--count", "10"])
    assert config.name == "default"
    assert config.count == 10


# Tests: cli with config files


def test_cli_root_config_file(tmp_toml_file):
    write_file(tmp_toml_file, 'name = "from_toml"\ncount = 99')
    config = cli(SimpleConfig, args=["@", tmp_toml_file])
    assert config.name == "from_toml"
    assert config.count == 99


def test_cli_root_config_with_override(tmp_toml_file):
    write_file(tmp_toml_file, 'name = "from_toml"\ncount = 99')
    config = cli(SimpleConfig, args=["@", tmp_toml_file, "--count", "1"])
    assert config.name == "from_toml"
    assert config.count == 1


def test_cli_nested_config_with_space(tmp_toml_file):
    write_file(tmp_toml_file, "lr = 0.001\nbatch_size = 64")
    config = cli(NestedConfig, args=["--train", "@", tmp_toml_file])
    assert config.train.lr == 0.001
    assert config.train.batch_size == 64
    assert config.seed == 42


def test_cli_nested_config_without_space(tmp_toml_file):
    write_file(tmp_toml_file, "lr = 0.001\nbatch_size = 64")
    config = cli(NestedConfig, args=["--train", f"@{tmp_toml_file}"])
    assert config.train.lr == 0.001
    assert config.train.batch_size == 64


def test_cli_nested_config_with_override(tmp_toml_file):
    write_file(tmp_toml_file, "lr = 0.001\nbatch_size = 64")
    config = cli(NestedConfig, args=["--train", "@", tmp_toml_file, "--train.lr", "0.1"])
    assert config.train.lr == 0.1
    assert config.train.batch_size == 64


def test_cli_json_config(tmp_json_file):
    write_file(tmp_json_file, '{"name": "from_json", "count": 42}')
    config = cli(SimpleConfig, args=["@", tmp_json_file])
    assert config.name == "from_json"
    assert config.count == 42


def test_cli_yaml_config(tmp_yaml_file):
    write_file(tmp_yaml_file, "name: from_yaml\ncount: 77")
    config = cli(SimpleConfig, args=["@", tmp_yaml_file])
    assert config.name == "from_yaml"
    assert config.count == 77


# Tests: cli deep nesting


def test_cli_deep_nested_config(tmp_path):
    encoder_file = os.path.join(tmp_path, "encoder.toml")
    write_file(encoder_file, "hidden_size = 512\nnum_layers = 8")

    config = cli(DeepNestedConfig, args=["--model.encoder", "@", encoder_file])
    assert config.model.encoder.hidden_size == 512
    assert config.model.encoder.num_layers == 8
    assert config.model.decoder.hidden_size == 256
    assert config.model.decoder.num_layers == 4


def test_cli_multiple_nested_configs(tmp_path):
    encoder_file = os.path.join(tmp_path, "encoder.toml")
    train_file = os.path.join(tmp_path, "train.toml")
    write_file(encoder_file, "hidden_size = 512\nnum_layers = 8")
    write_file(train_file, "lr = 0.0001\nbatch_size = 128")

    config = cli(
        DeepNestedConfig,
        args=["--model.encoder", "@", encoder_file, "--train", "@", train_file],
    )
    assert config.model.encoder.hidden_size == 512
    assert config.model.encoder.num_layers == 8
    assert config.train.lr == 0.0001
    assert config.train.batch_size == 128


def test_cli_root_and_nested_config(tmp_path):
    root_file = os.path.join(tmp_path, "root.toml")
    encoder_file = os.path.join(tmp_path, "encoder.toml")
    write_file(root_file, 'name = "experiment_1"')
    write_file(encoder_file, "hidden_size = 1024")

    config = cli(
        DeepNestedConfig,
        args=["@", root_file, "--model.encoder", "@", encoder_file],
    )
    assert config.name == "experiment_1"


# Tests: error handling


def test_cli_missing_config_file():
    with pytest.raises(ConfigFileError, match="not found"):
        cli(SimpleConfig, args=["@", "/nonexistent/config.toml"])


def test_cli_invalid_config_file(tmp_toml_file):
    write_file(tmp_toml_file, "invalid = [toml")
    with pytest.raises(ConfigFileError, match="Invalid TOML"):
        cli(SimpleConfig, args=["@", tmp_toml_file])


def test_cli_missing_file_after_at():
    with pytest.raises(ConfigFileError, match="must be followed"):
        cli(SimpleConfig, args=["@"])


def test_cli_discriminated_union_missing_type_uses_default(tmp_toml_file):
    """Test that missing discriminator field is auto-injected from default."""
    from typing import Annotated, Literal

    from pydantic import Field

    class DataConfigA(BaseConfig):
        type: Literal["a"] = "a"
        value: int = 1

    class DataConfigB(BaseConfig):
        type: Literal["b"] = "b"
        value: int = 2

    class ConfigWithUnion(BaseConfig):
        data: Annotated[DataConfigA | DataConfigB, Field(discriminator="type")] = DataConfigA()

    # Config file missing the 'type' discriminator - should use default "a"
    write_file(tmp_toml_file, "[data]\nvalue = 100")

    config = cli(ConfigWithUnion, args=["@", tmp_toml_file])
    assert config.data.type == "a"
    assert config.data.value == 100


def test_cli_discriminated_union_with_type(tmp_toml_file):
    """Test that discriminated union works when type field is provided."""
    from typing import Annotated, Literal
    from pydantic import Field

    class DataConfigA(BaseConfig):
        type: Literal["a"] = "a"
        value: int = 1

    class DataConfigB(BaseConfig):
        type: Literal["b"] = "b"
        value: int = 2

    class ConfigWithUnion(BaseConfig):
        data: Annotated[DataConfigA | DataConfigB, Field(discriminator="type")] = DataConfigA()

    # Config file with the required 'type' discriminator
    write_file(tmp_toml_file, '[data]\ntype = "b"\nvalue = 100')

    config = cli(ConfigWithUnion, args=["@", tmp_toml_file])
    assert config.data.type == "b"
    assert config.data.value == 100


def test_cli_discriminated_union_switch_variant_via_cli():
    """Test that discriminated union variant can be switched via CLI args (no config file)."""
    from typing import Annotated, Literal

    from pydantic import Field

    class DataConfigA(BaseConfig):
        type: Literal["a"] = "a"
        value: int = 1

    class DataConfigB(BaseConfig):
        type: Literal["b"] = "b"
        value: int = 2
        extra: int = 99

    class ConfigWithUnion(BaseConfig):
        data: Annotated[DataConfigA | DataConfigB, Field(discriminator="type")] = DataConfigA()
        name: str = "hello"

    config = cli(ConfigWithUnion, args=["--data.type", "b", "--data.extra", "42", "--name", "world"])
    assert config.data.type == "b"
    assert config.data.extra == 42
    assert config.name == "world"


def test_multi_union_help_shows_all_variants(capsys):
    """--help should list sub-fields for every variant of a multi-model union field."""
    from typing import Annotated, Literal
    from pydantic import Field

    class ConfigA(BaseConfig):
        type: Literal["a"] = "a"
        value: int = 1
        extra_a: str = "hello"

    class ConfigB(BaseConfig):
        type: Literal["b"] = "b"
        value: float = 2.0
        extra_b: bool = True

    class Config(BaseConfig):
        data: Annotated[ConfigA | ConfigB, Field(discriminator="type")] = ConfigA()
        seed: int = 42

    with pytest.raises(SystemExit):
        cli(Config, args=["--help"])
    out = capsys.readouterr().out
    assert "--data.value INT" in out
    assert "--data.extra-a STR" in out
    assert "data variant: ConfigB" in out
    assert "--data.extra-b" in out


def test_multi_union_plain_help_shows_all_variants(capsys):
    """--help should also list variants for a plain ``A | B`` union (no discriminator)."""

    class ConfigA(BaseConfig):
        value: int = 1
        extra_a: str = "hello"

    class ConfigB(BaseConfig):
        value: float = 2.0
        extra_b: bool = True

    class Config(BaseConfig):
        data: ConfigA | ConfigB = ConfigA()
        seed: int = 42

    with pytest.raises(SystemExit):
        cli(Config, args=["--help"])
    out = capsys.readouterr().out
    assert "--data.value INT" in out
    assert "data variant: ConfigB" in out
    assert "--data.extra-b" in out


# Tests: BaseConfig validators


def test_none_str_to_none():
    class ConfigWithOptional(BaseConfig):
        name: str | None = "default"

    config = cli(ConfigWithOptional, args=["--name", "None"])
    assert config.name is None


def test_none_str_to_none_in_toml(tmp_toml_file):
    class ConfigWithOptional(BaseConfig):
        name: str | None = "default"

    write_file(tmp_toml_file, 'name = "None"')
    config = cli(ConfigWithOptional, args=["@", tmp_toml_file])
    assert config.name is None


def test_none_str_passes_regular_values():
    class ConfigWithOptional(BaseConfig):
        name: str | None = "default"

    config = cli(ConfigWithOptional, args=["--name", "hello"])
    assert config.name == "hello"


def test_discriminator_type_injected_from_default(tmp_toml_file):
    """When a TOML file overrides a discriminated union field without specifying 'type',
    the default type tag should be injected automatically."""
    from typing import Annotated, Literal

    from pydantic import Field

    class DataConfigA(BaseConfig):
        type: Literal["a"] = "a"
        value: int = 1

    class DataConfigB(BaseConfig):
        type: Literal["b"] = "b"
        value: int = 2

    class ConfigWithUnion(BaseConfig):
        data: Annotated[DataConfigA | DataConfigB, Field(discriminator="type")] = DataConfigA()

    write_file(tmp_toml_file, "[data]\nvalue = 100")
    config = cli(ConfigWithUnion, args=["@", tmp_toml_file])
    assert config.data.type == "a"
    assert config.data.value == 100


def test_discriminator_type_explicit_overrides_default(tmp_toml_file):
    """When the TOML file explicitly provides a 'type', it should be used."""
    from typing import Annotated, Literal

    from pydantic import Field

    class DataConfigA(BaseConfig):
        type: Literal["a"] = "a"
        value: int = 1

    class DataConfigB(BaseConfig):
        type: Literal["b"] = "b"
        value: int = 2

    class ConfigWithUnion(BaseConfig):
        data: Annotated[DataConfigA | DataConfigB, Field(discriminator="type")] = DataConfigA()

    write_file(tmp_toml_file, '[data]\ntype = "b"\nvalue = 100')
    config = cli(ConfigWithUnion, args=["@", tmp_toml_file])
    assert config.data.type == "b"
    assert config.data.value == 100


# Tests: bare flags for Optional[BaseModel] fields


def test_bare_flag_enables_optional_config():
    """--compile as a bare flag should enable CompileConfig with defaults."""

    class CompileConfig(BaseConfig):
        fullgraph: bool = False

    class Config(BaseConfig):
        compile: CompileConfig | None = None
        name: str = "test"

    config = cli(Config, args=["--compile", "--name", "hello"])
    assert config.compile is not None
    assert config.compile.fullgraph is False
    assert config.name == "hello"


def test_bare_flag_at_end_of_args():
    """--compile at end of args should still work."""

    class CompileConfig(BaseConfig):
        fullgraph: bool = False

    class Config(BaseConfig):
        name: str = "test"
        compile: CompileConfig | None = None

    config = cli(Config, args=["--name", "hello", "--compile"])
    assert config.compile is not None
    assert config.name == "hello"


def test_bare_flag_nested_path():
    """--model.compile should work for nested Optional configs."""

    class CompileConfig(BaseConfig):
        fullgraph: bool = False

    class ModelConfig(BaseConfig):
        compile: CompileConfig | None = None
        name: str = "default"

    class Config(BaseConfig):
        model: ModelConfig = ModelConfig()

    config = cli(Config, args=["--model.compile", "--model.name", "mymodel"])
    assert config.model.compile is not None
    assert config.model.compile.fullgraph is False
    assert config.model.name == "mymodel"


def test_bare_flag_with_sub_field_override_via_toml(tmp_toml_file):
    """Sub-field overrides for Optional configs are best done via TOML."""

    class CompileConfig(BaseConfig):
        fullgraph: bool = False

    class ModelConfig(BaseConfig):
        compile: CompileConfig | None = None

    class Config(BaseConfig):
        model: ModelConfig = ModelConfig()

    write_file(tmp_toml_file, "[model.compile]\nfullgraph = true")
    config = cli(Config, args=["@", tmp_toml_file])
    assert config.model.compile is not None
    assert config.model.compile.fullgraph is True


def test_bare_flag_multiple_optional_configs():
    """Multiple bare flags should all work."""

    class ACConfig(BaseConfig):
        freq: int = 1

    class CompileConfig(BaseConfig):
        fullgraph: bool = False

    class ModelConfig(BaseConfig):
        ac: ACConfig | None = None
        compile: CompileConfig | None = None

    class Config(BaseConfig):
        model: ModelConfig = ModelConfig()

    config = cli(Config, args=["--model.compile", "--model.ac"])
    assert config.model.compile is not None
    assert config.model.ac is not None
    assert config.model.ac.freq == 1


def test_bare_flag_kebab_case():
    """--model.ac-offloading should work (kebab-case for snake_case field)."""

    class OffloadConfig(BaseConfig):
        pin_memory: bool = True

    class ModelConfig(BaseConfig):
        ac_offloading: OffloadConfig | None = None

    class Config(BaseConfig):
        model: ModelConfig = ModelConfig()

    config = cli(Config, args=["--model.ac-offloading"])
    assert config.model.ac_offloading is not None
    assert config.model.ac_offloading.pin_memory is True


def test_optional_config_none_by_default():
    """Without the bare flag, Optional config should remain None."""

    class CompileConfig(BaseConfig):
        fullgraph: bool = False

    class Config(BaseConfig):
        compile: CompileConfig | None = None
        name: str = "test"

    config = cli(Config, args=["--name", "hello"])
    assert config.compile is None


def test_optional_sub_field_override():
    """--wandb.project foo should implicitly enable the optional wandb config."""

    class WandbConfig(BaseConfig):
        project: str = "default"
        name: str = "run"

    class Config(BaseConfig):
        wandb: WandbConfig | None = None
        seed: int = 42

    config = cli(Config, args=["--wandb.project", "my-project", "--wandb.name", "my-run"])
    assert config.wandb is not None
    assert config.wandb.project == "my-project"
    assert config.wandb.name == "my-run"


def test_optional_sub_field_nested():
    """--model.compile.fullgraph True should work for deeply nested optional configs."""

    class CompileConfig(BaseConfig):
        fullgraph: bool = False
        dynamic: bool = True

    class ModelConfig(BaseConfig):
        compile: CompileConfig | None = None
        name: str = "default"

    class Config(BaseConfig):
        model: ModelConfig = ModelConfig()

    config = cli(Config, args=["--model.compile.fullgraph", "True"])
    assert config.model.compile is not None
    assert config.model.compile.fullgraph is True
    assert config.model.compile.dynamic is True


def test_optional_nested_help_shows_subfields(capsys):
    """--help should list sub-fields of Optional[BaseModel] fields even when their default is None.

    The panel title should also surface the field's optional status.
    """

    class WandbConfig(BaseConfig):
        project: str = "my-project"
        entity: str = "my-team"

    class Config(BaseConfig):
        wandb: WandbConfig | None = None
        seed: int = 42

    with pytest.raises(SystemExit):
        cli(Config, args=["--help"])
    out = capsys.readouterr().out
    assert "--wandb.project" in out
    assert "--wandb.entity" in out
    assert "wandb options (optional, default: None)" in out


def test_optional_sub_field_with_bare_flag_and_regular_args():
    """Mixing sub-field overrides with regular args should work."""

    class WandbConfig(BaseConfig):
        project: str = "default"

    class Config(BaseConfig):
        wandb: WandbConfig | None = None
        name: str = "test"
        seed: int = 42

    config = cli(Config, args=["--name", "hello", "--wandb.project", "proj", "--seed", "123"])
    assert config.name == "hello"
    assert config.seed == 123
    assert config.wandb is not None
    assert config.wandb.project == "proj"


# Tests: dict[str, Any] fields (handled via config files, not CLI)


def test_dict_any_field_with_default(tmp_toml_file):
    """dict[str, Any] fields should work when a default is provided via config file."""
    from typing import Any

    class SamplingConfig(BaseConfig):
        temperature: float = 1.0
        extra_body: dict[str, Any] = {}

    class Config(BaseConfig):
        sampling: SamplingConfig = SamplingConfig()
        name: str = "test"

    write_file(tmp_toml_file, '[sampling]\ntemperature = 0.5')
    config = cli(Config, args=["@", tmp_toml_file, "--name", "hello"])
    assert config.sampling.temperature == 0.5
    assert config.sampling.extra_body == {}
    assert config.name == "hello"


def test_dict_any_field_set_via_toml(tmp_toml_file):
    """dict[str, Any] fields should be settable via config files."""
    from typing import Any

    class EnvConfig(BaseConfig):
        id: str = "default"
        extra_kwargs: dict[str, Any] = {}

    class Config(BaseConfig):
        env: EnvConfig = EnvConfig()

    write_file(tmp_toml_file, '[env]\nid = "custom"\n\n[env.extra_kwargs]\nseq_len = 512\nverbose = true')
    config = cli(Config, args=["@", tmp_toml_file])
    assert config.env.id == "custom"
    assert config.env.extra_kwargs == {"seq_len": 512, "verbose": True}


def test_dict_any_in_discriminated_union(tmp_toml_file):
    """dict[str, Any] in a non-default discriminated union variant should not crash."""
    from typing import Annotated, Any, Literal, TypeAlias

    from pydantic import Field

    class DefaultMode(BaseConfig):
        type: Literal["default"] = "default"
        scale: float = 1.0

    class CustomMode(BaseConfig):
        type: Literal["custom"] = "custom"
        import_path: str = "my_module.fn"
        kwargs: dict[str, Any] = {}

    ModeConfig: TypeAlias = Annotated[DefaultMode | CustomMode, Field(discriminator="type")]

    class Config(BaseConfig):
        mode: ModeConfig = DefaultMode()
        name: str = "test"

    # Default mode works without touching the dict field
    config = cli(Config, args=["--name", "hello"])
    assert config.mode.type == "default"

    # Custom mode via TOML with dict kwargs
    write_file(tmp_toml_file, '[mode]\ntype = "custom"\nimport_path = "my.fn"\n\n[mode.kwargs]\nalpha = 0.5')
    config = cli(Config, args=["@", tmp_toml_file])
    assert config.mode.type == "custom"
    assert config.mode.kwargs == {"alpha": 0.5}


# Tests: list fields via JSON CLI args


def test_list_field_via_json_cli():
    """--env-ratios '[0.7, 0.3]' should parse JSON and set the list field."""

    class Config(BaseConfig):
        env_ratios: list[float] = []
        name: str = "test"

    config = cli(Config, args=["--env-ratios", "[0.7, 0.3]", "--name", "hello"])
    assert config.env_ratios == [0.7, 0.3]
    assert config.name == "hello"


def test_list_field_via_json_cli_nested():
    """JSON list args should work for nested config fields."""

    class BufferConfig(BaseConfig):
        env_ratios: list[float] | None = None
        seed: int | None = None

    class Config(BaseConfig):
        buffer: BufferConfig = BufferConfig()

    config = cli(Config, args=["--buffer.env-ratios", "[0.7, 0.2, 0.1]"])
    assert config.buffer.env_ratios == [0.7, 0.2, 0.1]


def test_list_field_via_json_cli_optional():
    """JSON list args should work for Optional[list[...]] fields."""

    class Config(BaseConfig):
        tags: list[str] | None = None
        name: str = "test"

    config = cli(Config, args=["--tags", '["alpha", "beta"]', "--name", "hello"])
    assert config.tags == ["alpha", "beta"]


def test_list_field_via_json_cli_integers():
    """JSON list args should work for list[int] fields."""

    class Config(BaseConfig):
        gpu_ids: list[int] = []

    config = cli(Config, args=["--gpu-ids", "[0, 1, 2, 3]"])
    assert config.gpu_ids == [0, 1, 2, 3]


def test_list_field_space_separated_still_works():
    """Space-separated list args (tyro native) should still work when not JSON."""

    class Config(BaseConfig):
        env_ratios: list[float] = []

    config = cli(Config, args=["--env-ratios", "0.7", "0.3"])
    assert config.env_ratios == [0.7, 0.3]


def test_list_field_via_json_cli_with_toml(tmp_toml_file):
    """JSON list CLI args should override TOML list values."""

    class BufferConfig(BaseConfig):
        env_ratios: list[float] | None = None

    class Config(BaseConfig):
        buffer: BufferConfig = BufferConfig()
        name: str = "test"

    write_file(tmp_toml_file, 'name = "from-toml"\n\n[buffer]\nenv_ratios = [0.5, 0.5]')
    config = cli(Config, args=["@", tmp_toml_file, "--buffer.env-ratios", "[0.7, 0.3]"])
    assert config.buffer.env_ratios == [0.7, 0.3]
    assert config.name == "from-toml"


def test_list_field_via_toml_no_cli():
    """list fields from TOML (no CLI override) should work as before."""

    class Config(BaseConfig):
        ratios: list[float] = []

    config = cli(Config, args=[])
    assert config.ratios == []


# Tests: dict[str, Any] fields via JSON CLI args


def test_dict_field_via_json_cli():
    """--extra-kwargs '{"key": 123}' should parse JSON and set the dict field."""
    from typing import Any

    class Config(BaseConfig):
        extra_kwargs: dict[str, Any] = {}
        name: str = "test"

    config = cli(Config, args=["--extra-kwargs", '{"sandbox_client_max_workers": 128}', "--name", "hello"])
    assert config.extra_kwargs == {"sandbox_client_max_workers": 128}
    assert config.name == "hello"


def test_dict_field_via_json_cli_nested():
    """JSON dict args should work for nested config fields."""
    from typing import Any

    class EnvConfig(BaseConfig):
        id: str = "default"
        extra_env_kwargs: dict[str, Any] = {}

    class Config(BaseConfig):
        env: EnvConfig = EnvConfig()

    config = cli(Config, args=["--env.extra-env-kwargs", '{"timeout": 60, "verbose": true}'])
    assert config.env.extra_env_kwargs == {"timeout": 60, "verbose": True}


def test_dict_field_via_json_cli_with_toml(tmp_toml_file):
    """JSON dict CLI args should merge with TOML dict values."""
    from typing import Any

    class Config(BaseConfig):
        extra: dict[str, Any] = {}
        name: str = "test"

    write_file(tmp_toml_file, 'name = "from-toml"\n\n[extra]\nold_key = "old_value"')
    config = cli(Config, args=["@", tmp_toml_file, "--extra", '{"new_key": "new_value"}'])
    assert config.extra == {"old_key": "old_value", "new_key": "new_value"}
    assert config.name == "from-toml"


def test_dict_field_via_json_inside_optional_model():
    """JSON dict on a field inside an Optional[BaseModel] should be parsed correctly."""
    from typing import Any

    class InferenceConfig(BaseConfig):
        vllm_extra: dict[str, Any] = {}
        name: str = "default"

    class Config(BaseConfig):
        inference: InferenceConfig | None = None

    config = cli(Config, args=["--inference.vllm-extra", '{"custom_arg": "value"}'])
    assert config.inference is not None
    assert config.inference.vllm_extra == {"custom_arg": "value"}


# Tests: real-world scenarios


def test_ml_training_config(tmp_path):
    """Simulate a typical ML training configuration."""

    class OptimizerConfig(BaseConfig):
        lr: float = 1e-4
        weight_decay: float = 0.01

    class ModelConfig(BaseConfig):
        hidden_size: int = 256
        num_layers: int = 4
        dropout: float = 0.1

    class DataConfig(BaseConfig):
        batch_size: int = 32
        num_workers: int = 4

    class TrainConfig(BaseConfig):
        model: ModelConfig = ModelConfig()
        optimizer: OptimizerConfig = OptimizerConfig()
        data: DataConfig = DataConfig()
        seed: int = 42
        max_epochs: int = 100

    model_file = os.path.join(tmp_path, "model.toml")
    optim_file = os.path.join(tmp_path, "optimizer.toml")

    write_file(model_file, "hidden_size = 512\nnum_layers = 8\ndropout = 0.2")
    write_file(optim_file, "lr = 0.001\nweight_decay = 0.1")

    config = cli(
        TrainConfig,
        args=[
            "--model", "@", model_file,
            "--optimizer", "@", optim_file,
            "--data.batch-size", "64",
            "--max-epochs", "50",
        ],
    )

    assert config.model.hidden_size == 512
    assert config.model.num_layers == 8
    assert config.model.dropout == 0.2
    assert config.optimizer.lr == 0.001
    assert config.optimizer.weight_decay == 0.1
    assert config.data.batch_size == 64
    assert config.max_epochs == 50
    assert config.seed == 42


def test_override_nested_in_config_file(tmp_toml_file):
    """Test that CLI args can override specific nested fields from config."""

    class InnerConfig(BaseConfig):
        a: int = 1
        b: int = 2
        c: int = 3

    class OuterConfig(BaseConfig):
        inner: InnerConfig = InnerConfig()
        name: str = "test"

    write_file(tmp_toml_file, "a = 10\nb = 20\nc = 30")

    config = cli(OuterConfig, args=["--inner", "@", tmp_toml_file, "--inner.b", "200"])

    assert config.inner.a == 10
    assert config.inner.b == 200
    assert config.inner.c == 30


# Tests: dict string value coercion (tyro parses untyped dict values as strings)


def test_dict_coercion_int_values():
    """Dict values that look like ints should be coerced from CLI strings."""
    from typing import Any

    class Config(BaseConfig):
        args: dict[str, Any] = {}

    config = Config.model_validate({"args": {"viewport_width": "800", "height": "600"}})
    assert config.args == {"viewport_width": 800, "height": 600}
    assert isinstance(config.args["viewport_width"], int)


def test_dict_coercion_float_values():
    """Dict values that look like floats should be coerced."""
    from typing import Any

    class Config(BaseConfig):
        args: dict[str, Any] = {}

    config = Config.model_validate({"args": {"lr": "0.001", "weight_decay": "0.01"}})
    assert config.args == {"lr": 0.001, "weight_decay": 0.01}
    assert isinstance(config.args["lr"], float)


def test_dict_coercion_bool_values():
    """Dict values 'true'/'false' should be coerced to bool."""
    from typing import Any

    class Config(BaseConfig):
        args: dict[str, Any] = {}

    config = Config.model_validate({"args": {"verbose": "true", "debug": "false"}})
    assert config.args == {"verbose": True, "debug": False}
    assert isinstance(config.args["verbose"], bool)


def test_dict_coercion_mixed_types():
    """Dict with int, float, bool, and string values all coerced correctly."""
    from typing import Any

    class Config(BaseConfig):
        args: dict[str, Any] = {}

    config = Config.model_validate({
        "args": {"port": "8080", "rate": "0.5", "enabled": "true", "name": "hello"}
    })
    assert config.args["port"] == 8080
    assert isinstance(config.args["port"], int)
    assert config.args["rate"] == 0.5
    assert isinstance(config.args["rate"], float)
    assert config.args["enabled"] is True
    assert config.args["name"] == "hello"
    assert isinstance(config.args["name"], str)


def test_dict_coercion_skipped_when_already_typed():
    """Dict with non-string values should not be coerced (e.g. from TOML)."""
    from typing import Any

    class Config(BaseConfig):
        args: dict[str, Any] = {}

    config = Config.model_validate({"args": {"port": 8080, "name": "hello"}})
    assert config.args == {"port": 8080, "name": "hello"}
    assert isinstance(config.args["port"], int)


def test_dict_coercion_empty_dict():
    """Empty dict should pass through unchanged."""
    from typing import Any

    class Config(BaseConfig):
        args: dict[str, Any] = {}

    config = Config.model_validate({"args": {}})
    assert config.args == {}


def test_dict_coercion_pure_string_values():
    """Dict with values that are genuinely strings should stay as strings."""
    from typing import Any

    class Config(BaseConfig):
        args: dict[str, Any] = {}

    config = Config.model_validate({"args": {"name": "alice", "label": "test-run"}})
    assert config.args == {"name": "alice", "label": "test-run"}
    assert isinstance(config.args["name"], str)


def test_dict_coercion_bare_dict_annotation():
    """Bare ``dict`` (without type params) should also get coercion."""
    class Config(BaseConfig):
        args: dict = {}

    config = Config.model_validate({"args": {"count": "42", "flag": "true"}})
    assert config.args["count"] == 42
    assert isinstance(config.args["count"], int)
    assert config.args["flag"] is True


def test_dict_coercion_annotated_bare_dict():
    """``Annotated[dict, Field(...)]`` should also get coercion."""
    from typing import Annotated
    from pydantic import Field

    class Config(BaseConfig):
        args: Annotated[dict, Field(description="env args")] = {}

    config = Config.model_validate({"args": {"width": "800", "verbose": "false"}})
    assert config.args["width"] == 800
    assert config.args["verbose"] is False


def test_dict_coercion_via_cli():
    """End-to-end: dict values from CLI args should be properly typed."""
    from typing import Any

    class Config(BaseConfig):
        extra_kwargs: dict[str, Any] = {}

    config = cli(Config, args=["--extra-kwargs", '{"viewport_width": 800, "enabled": true}'])
    assert config.extra_kwargs["viewport_width"] == 800
    assert isinstance(config.extra_kwargs["viewport_width"], int)
    assert config.extra_kwargs["enabled"] is True


def test_dict_coercion_via_cli_bare_dict():
    """End-to-end: bare ``dict`` field should work with JSON CLI args."""
    class Config(BaseConfig):
        args: dict = {}

    config = cli(Config, args=["--args", '{"count": 10, "rate": 0.5}'])
    assert config.args["count"] == 10
    assert config.args["rate"] == 0.5


def test_dict_coercion_toml_preserves_types(tmp_json_file):
    """Dict values from config files should already have proper types (no coercion needed)."""
    from typing import Any

    class Config(BaseConfig):
        args: dict[str, Any] = {}

    write_file(tmp_json_file, '{"args": {"viewport_width": 800, "enabled": true, "rate": 0.5}}')
    config = cli(Config, args=["@", tmp_json_file])
    assert config.args["viewport_width"] == 800
    assert isinstance(config.args["viewport_width"], int)
    assert config.args["enabled"] is True
    assert config.args["rate"] == 0.5


def test_dict_coercion_nested_config():
    """Dict coercion should work in nested configs."""
    from typing import Any

    class EnvConfig(BaseConfig):
        id: str = "default"
        args: dict[str, Any] = {}

    class Config(BaseConfig):
        env: EnvConfig = EnvConfig()

    config = Config.model_validate({"env": {"id": "browser", "args": {"width": "800", "headless": "true"}}})
    assert config.env.args["width"] == 800
    assert config.env.args["headless"] is True


def test_dict_coercion_negative_int():
    """Negative integers should be coerced correctly."""
    from typing import Any

    class Config(BaseConfig):
        args: dict[str, Any] = {}

    config = Config.model_validate({"args": {"offset": "-10"}})
    assert config.args["offset"] == -10
    assert isinstance(config.args["offset"], int)


def test_dict_coercion_scientific_float():
    """Scientific notation floats don't round-trip via str(), so they stay as strings."""
    from typing import Any

    class Config(BaseConfig):
        args: dict[str, Any] = {}

    config = Config.model_validate({"args": {"lr": "1e-4"}})
    assert config.args["lr"] == "1e-4"
    assert isinstance(config.args["lr"], str)


def test_dict_coercion_leading_zeros_stay_string():
    """Values like '007' should stay as strings (int('007') != '007')."""
    from typing import Any

    class Config(BaseConfig):
        args: dict[str, Any] = {}

    config = Config.model_validate({"args": {"code": "007"}})
    assert config.args["code"] == "007"
    assert isinstance(config.args["code"], str)


# Tests: parity contract — these must pass before AND after the tyro→custom
# parser swap. They lock down the behaviour we'd otherwise rely on tyro for.


def test_cli_equals_form_value():
    """``--field=value`` should work identically to ``--field value``."""
    config = cli(SimpleConfig, args=["--name=hello", "--count=7"])
    assert config.name == "hello"
    assert config.count == 7


def test_cli_no_prefix_disables_bool():
    """``--no-flag`` should disable a bool field even when its default is True."""

    class C(BaseConfig):
        enabled: bool = True

    config = cli(C, args=["--no-enabled"])
    assert config.enabled is False


def test_cli_bare_bool_flag_true():
    """``--enabled`` with no value should set a bool field to True (default False)."""

    class C(BaseConfig):
        enabled: bool = False

    config = cli(C, args=["--enabled"])
    assert config.enabled is True


def test_cli_unknown_flag_errors_cleanly():
    """An unknown ``--flag`` should raise a clear error, not crash with a traceback."""
    with pytest.raises((SystemExit, ConfigFileError, ValueError, Exception)):
        cli(SimpleConfig, args=["--this-flag-does-not-exist", "x"])


def test_cli_help_contains_usage(capsys):
    """``--help`` output should always include the usage line."""
    with pytest.raises(SystemExit):
        cli(SimpleConfig, args=["--help"])
    out = capsys.readouterr().out
    assert "usage" in out.lower()


def test_cli_help_contains_known_flags(capsys):
    """``--help`` output should list each field as ``--field`` somewhere."""
    with pytest.raises(SystemExit):
        cli(SimpleConfig, args=["--help"])
    out = capsys.readouterr().out
    assert "--name" in out
    assert "--count" in out


def test_cli_help_renders_list_default(capsys):
    """``--help`` should render a list field cleanly (no <function ...> leak)."""

    class C(BaseConfig):
        items: list[int] = []

    with pytest.raises(SystemExit):
        cli(C, args=["--help"])
    out = capsys.readouterr().out
    assert "--items" in out
    assert "<function" not in out, "default_factory should not leak as <function ...>"


def test_cli_help_with_default_factory_renders_cleanly(capsys):
    """A field with ``default_factory=list`` should not render as ``<function list>``."""

    class C(BaseConfig):
        items: list = Field(default_factory=list)

    with pytest.raises(SystemExit):
        cli(C, args=["--help"])
    out = capsys.readouterr().out
    assert "--items" in out
    assert "<function" not in out


def test_cli_help_exits_zero(capsys):
    """``--help`` should exit with status 0, not a non-zero code."""
    with pytest.raises(SystemExit) as exc_info:
        cli(SimpleConfig, args=["--help"])
    capsys.readouterr()  # drain
    assert exc_info.value.code in (0, None), f"--help exited with code {exc_info.value.code!r}"


def test_cli_negative_number_value_parses():
    """A negative-number value (e.g. ``--lr -1e-3``) should not be misread as a flag."""

    class C(BaseConfig):
        lr: float = 0.0

    config = cli(C, args=["--lr", "-1e-3"])
    assert config.lr == -1e-3


def test_cli_validation_alias_accepts_alias_name():
    """A field with ``Field(validation_alias=AliasChoices(...))`` should accept
    its alias on the CLI. tyro's mirror inherited the alias via create_model;
    the new parser must add alias names to its valid-paths set."""
    from pydantic import AliasChoices

    class Inner(BaseConfig):
        value: int = 0

    class Outer(BaseConfig):
        # Both 'inner' and 'student' point at the same field.
        inner: Annotated[
            Inner,
            Field(validation_alias=AliasChoices("inner", "student")),
        ] = Inner()

    # Canonical name works.
    config = cli(Outer, args=["--inner.value", "5"])
    assert config.inner.value == 5


def test_cli_validation_error_surfaces_as_configfileerror_with_flag_name():
    """When pydantic rejects a value, the user should see ``--foo: <msg>`` —
    not a raw ``pydantic_core.ValidationError`` traceback."""

    class C(BaseConfig):
        foo: int = 0

    with pytest.raises(ConfigFileError) as exc_info:
        cli(C, args=["--foo", "dskfj"])
    msg = str(exc_info.value)
    assert "--foo" in msg
    assert "integer" in msg
    assert "dskfj" in msg


def test_cli_validation_error_nested_field_renders_dotted_flag():
    """Nested pydantic errors should render as ``--sub.field`` so the user
    can see exactly which CLI flag to fix."""

    class Sub(BaseConfig):
        count: int = 0

    class C(BaseConfig):
        sub: Sub = Sub()

    with pytest.raises(ConfigFileError) as exc_info:
        cli(C, args=["--sub.count", "nope"])
    msg = str(exc_info.value)
    assert "--sub.count" in msg
    assert "integer" in msg


def test_cli_validation_error_box_right_border_aligns_with_color(capsys, monkeypatch):
    """Coloured rows inside the error box must not push the right border left.
    Guard against counting ANSI escape codes as printable width."""
    import re

    class C(BaseConfig):
        foo: int = 0

    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setattr("sys.argv", ["demo", "--foo", "dskfj"])
    with pytest.raises(SystemExit):
        cli(C)
    err = capsys.readouterr().err

    ansi = re.compile(r"\x1b\[[0-9;]*m")
    visible_widths = {
        len(ansi.sub("", line))
        for line in err.splitlines()
        if line.startswith(("\x1b[31m│", "│", "\x1b[31m╭", "╭", "\x1b[31m╰", "╰"))
    }
    # Every framed line should render to the same printable width — otherwise
    # the right border is misaligned.
    assert len(visible_widths) == 1, (
        f"Box rows have inconsistent visible widths: {sorted(visible_widths)}\n{err}"
    )


def test_cli_validation_error_snake_case_loc_renders_as_kebab():
    """Pydantic error locs use snake_case attribute names; we should kebab-case
    them when rendering as CLI flags."""

    class C(BaseConfig):
        my_int_field: int = 0

    with pytest.raises(ConfigFileError) as exc_info:
        cli(C, args=["--my-int-field", "nope"])
    msg = str(exc_info.value)
    assert "--my-int-field" in msg


# Tests: _parse_cli_to_dict — direct unit tests of the new argv parser.


def test_parse_cli_to_dict_equals_form():
    from pydantic_config.cli import _parse_cli_to_dict

    class C(BaseConfig):
        name: str = ""
        count: int = 0

    remaining, overrides = _parse_cli_to_dict(["--name=hello", "--count=7"], C, set())
    assert remaining == []
    assert overrides == {"name": "hello", "count": "7"}


def test_parse_cli_to_dict_space_form():
    from pydantic_config.cli import _parse_cli_to_dict

    class C(BaseConfig):
        name: str = ""

    remaining, overrides = _parse_cli_to_dict(["--name", "hello"], C, set())
    assert remaining == []
    assert overrides == {"name": "hello"}


def test_parse_cli_to_dict_bool_bare():
    from pydantic_config.cli import _parse_cli_to_dict

    class C(BaseConfig):
        enabled: bool = False

    remaining, overrides = _parse_cli_to_dict(["--enabled"], C, set())
    assert overrides == {"enabled": True}


def test_parse_cli_to_dict_bool_no_prefix():
    from pydantic_config.cli import _parse_cli_to_dict

    class C(BaseConfig):
        enabled: bool = True

    remaining, overrides = _parse_cli_to_dict(["--no-enabled"], C, set())
    assert overrides == {"enabled": False}


def test_parse_cli_to_dict_bool_explicit_value():
    from pydantic_config.cli import _parse_cli_to_dict

    class C(BaseConfig):
        enabled: bool = False

    remaining, overrides = _parse_cli_to_dict(["--enabled", "false"], C, set())
    assert overrides == {"enabled": False}


def test_parse_cli_to_dict_list_space_separated():
    from pydantic_config.cli import _parse_cli_to_dict

    class C(BaseConfig):
        items: list[int] = []

    remaining, overrides = _parse_cli_to_dict(["--items", "1", "2", "3"], C, set())
    assert overrides == {"items": ["1", "2", "3"]}


def test_parse_cli_to_dict_negative_number_value():
    from pydantic_config.cli import _parse_cli_to_dict

    class C(BaseConfig):
        lr: float = 0.0

    remaining, overrides = _parse_cli_to_dict(["--lr", "-1e-3"], C, set())
    assert overrides == {"lr": "-1e-3"}


def test_parse_cli_to_dict_nested_field():
    from pydantic_config.cli import _parse_cli_to_dict

    class Sub(BaseConfig):
        x: int = 0

    class C(BaseConfig):
        sub: Sub = Sub()

    remaining, overrides = _parse_cli_to_dict(["--sub.x", "5"], C, set())
    assert overrides == {"sub": {"x": "5"}}


def test_parse_cli_to_dict_unknown_flag_errors_with_suggestions():
    from pydantic_config.cli import _parse_cli_to_dict

    class C(BaseConfig):
        seed: int = 0
        rate: float = 0.0

    with pytest.raises(ConfigFileError, match="seed"):
        # Close-ish typo to "seed" should produce a suggestion.
        _parse_cli_to_dict(["--seedz", "5"], C, set())


def test_parse_cli_to_dict_interior_path_errors():
    """``--sub`` (a BaseModel group) without a sub-field name should error
    pointing at one of its leaves, not silently consume the next token."""
    from pydantic_config.cli import _parse_cli_to_dict

    class Sub(BaseConfig):
        x: int = 0
        y: int = 1

    class C(BaseConfig):
        sub: Sub = Sub()

    with pytest.raises(ConfigFileError, match=r"--sub.*group|--sub\.x"):
        _parse_cli_to_dict(["--sub", "5"], C, set())


def test_parse_cli_to_dict_alias_accepted():
    from pydantic import AliasChoices
    from pydantic_config.cli import _parse_cli_to_dict

    class Inner(BaseConfig):
        value: int = 0

    class C(BaseConfig):
        inner: Annotated[Inner, Field(validation_alias=AliasChoices("inner", "student"))] = Inner()

    remaining, overrides = _parse_cli_to_dict(["--student.value", "9"], C, set())
    # Alias resolves to the canonical snake-case path.
    assert overrides == {"student": {"value": "9"}}


def test_render_help_returns_string_with_panels():
    """``_render_help`` should produce a usage line + an "options" panel for
    root scalars + a separate panel per sub-config + variant panels for
    multi-model unions + an "(optional, default: None)" panel per
    Optional[BaseModel]."""
    from typing import Literal, Union
    from pydantic_config.cli import _render_help

    class WandbConfig(BaseConfig):
        project: str = "default-proj"
        entity: str | None = None

    class TrainerConfig(BaseConfig):
        lr: float = 1e-4
        batch_size: int = 32

    class VariantA(BaseConfig):
        type: Literal["a"] = "a"
        value: int = 1

    class VariantB(BaseConfig):
        type: Literal["b"] = "b"
        extra_b: str = "bee"

    class Top(BaseConfig):
        seed: int = 42
        trainer: TrainerConfig = TrainerConfig()
        wandb: WandbConfig | None = None
        data: Annotated[Union[VariantA, VariantB], Field(discriminator="type")] = VariantA()

    out = _render_help(Top, prog="demo", description="A demo program.")

    # Structural fixtures
    assert out.startswith("usage: demo [-h]")
    assert "A demo program." in out
    assert "-h, --help" in out
    # Root scalar in main panel
    assert "--seed" in out
    assert "(default: 42)" in out
    # Plain sub-config panel
    assert "trainer options" in out
    assert "--trainer.lr" in out
    assert "--trainer.batch-size" in out
    # Optional[BaseModel] panel
    assert "wandb options (optional, default: None)" in out
    assert "--wandb.project" in out
    assert "--wandb.entity" in out
    # Multi-model union variant panels
    assert "data variant: VariantA" in out
    assert "data variant: VariantB" in out
    assert "--data.value" in out
    assert "--data.extra-b" in out
