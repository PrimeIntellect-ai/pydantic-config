# Pydantic Config

A Pydantic-driven CLI with TOML / YAML / JSON config file support.

```python
from pydantic_config import cli, BaseConfig

class Config(BaseConfig):
    lr: float = 1e-4
    batch_size: int = 32

config = cli(Config)
```

## Install

```bash
pip install git+https://github.com/PrimeIntellect-ai/pydantic-config
```

For TOML support:
```bash
pip install "prime-pydantic-config[toml] @ git+https://github.com/PrimeIntellect-ai/pydantic-config"
```

For all formats (TOML + YAML):
```bash
pip install "prime-pydantic-config[all] @ git+https://github.com/PrimeIntellect-ai/pydantic-config"
```

## Features

Every example below uses [`examples/train.py`](examples/train.py), a dummy
training config that exercises the patterns common in `prime-rl`-style
training scripts.

### Help output

`--help` is auto-generated from the model. Each `BaseModel` field becomes its
own panel; discriminated-union variants get a panel each; `Optional[BaseModel]`
fields are annotated `(optional, default: None)`.

```bash
python examples/train.py --help
```

<p align="center">
  <img src="assets/help.svg" alt="Help output" width="700">
</p>

### Config files via `@`

Load a whole config from a TOML, YAML, or JSON file. CLI args layered on top
always win — same precedence as `default` ⊂ file ⊂ CLI.

```bash
python examples/train.py @ examples/train.toml
python examples/train.py @ examples/train.yaml
python examples/train.py @ examples/train.toml --seed 0 --no-model.compile
```

### Required fields

A field without a default must be passed. The error is rendered as a boxed
message naming the missing CLI flag, not a raw pydantic traceback.

```bash
python examples/train.py   # errors: --run-name is required
```

<p align="center">
  <img src="assets/required_error.svg" alt="Missing required argument" width="500">
</p>

### Nested config groups

Sub-configs are addressed via dotted paths. Field names are kebab-cased on the
CLI; pydantic still validates against the snake_case attribute.

```bash
python examples/train.py --run-name r1 --model.hidden-size 4096 --data.num-workers 16
```

### Bool flags and `--no-` negation

Bare `--flag` sets a bool to `True`; `--no-flag` sets it to `False`. Works on
nested fields too.

```bash
python examples/train.py --run-name r1 --no-model.compile --no-data.shuffle
```

### Lists

Lists accept either space-separated values or a JSON literal. Negative numbers
(e.g. `-1e-3`) are values, not flags.

```bash
python examples/train.py --run-name r1 --checkpoint-steps 100 200 500
python examples/train.py --run-name r1 --checkpoint-steps '[100, 200, 500]'
```

### Dicts

Dict fields take a JSON literal on the CLI. A TOML/YAML dict and a CLI dict
deep-merge — CLI keys win on conflict but don't wipe the file's keys.

```bash
python examples/train.py --run-name r1 --extra-kwargs '{"seq_len": 4096}'
```

### Optional sub-configs

A field typed `WandbConfig | None = None` is off by default. The bare flag
turns it on with defaults; a sub-field flag both activates the sub-config and
overrides the field.

```bash
python examples/train.py --run-name r1 --wandb                                 # enable with defaults
python examples/train.py --run-name r1 --wandb.project demo --wandb.entity me  # enable + override
python examples/train.py --run-name r1 --wandb @ examples/wandb.toml           # enable from a file
```

### Discriminated unions

Multi-variant fields (e.g. `optimizer: AdamWConfig | MuonConfig`) are switched
by the `type` tag. Each variant renders its own help panel. The default
variant's `type` is auto-injected, so partial overrides keep the same variant.

```bash
python examples/train.py --run-name r1 --optimizer.weight-decay 0.05               # stay on default (adamw)
python examples/train.py --run-name r1 --optimizer.type muon --optimizer.lr 2e-3   # switch to muon
python examples/train.py --run-name r1 --optimizer @ examples/optimizer.toml       # load a variant from a file
```

### Validation aliases

`Field(validation_alias=AliasChoices("seed", "random_seed"))` makes both names
accepted on the CLI and in config files. The library normalizes either form to
the canonical key before validation, so mixing TOML + CLI under different
names is safe (CLI still wins on conflict).

```bash
python examples/train.py --run-name r1 --random-seed 7      # CLI alias
python examples/train.py @ examples/train.toml              # TOML uses random_seed
python examples/train.py @ examples/train.toml --seed 99    # TOML alias + CLI canonical override
```

### Validation errors point at the CLI flag

Pydantic's `ValidationError` is wrapped so the user sees the offending flag
inline, not a raw `pydantic_core` traceback.

```bash
python examples/train.py --run-name r1 --seed nope
```

<p align="center">
  <img src="assets/config_error.svg" alt="Config validation error" width="700">
</p>

### Unknown flags get a suggestion

Typos are caught with a `difflib`-powered "did you mean" hint.

```bash
python examples/train.py --run-name r1 --seedz 5   # → did you mean --seed?
```

### Config file not found

```bash
python examples/train.py @ nonexistent.toml
```

<p align="center">
  <img src="assets/file_not_found.svg" alt="Config file not found" width="700">
</p>

## Development

```bash
uv sync --extra all
uv run pytest
```
