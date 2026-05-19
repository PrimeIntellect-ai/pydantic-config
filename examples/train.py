"""End-to-end demo of pydantic-config covering the patterns used in prime-rl
training scripts: nested configs, required fields, bool toggles with --no-,
lists, dicts, Optional sub-configs, discriminated-union optimizer choice,
field descriptions (via ``Field(description=...)`` or PEP 224-style attribute
docstrings), and validation aliases.

Try it:
    python examples/train.py --help
    python examples/train.py @ examples/train.toml
    python examples/train.py @ examples/train.toml --seed 0 --no-model.compile
"""

from pathlib import Path
from pprint import pprint
from typing import Annotated, Literal

from pydantic import AliasChoices, Field

from pydantic_config import cli, BaseConfig


class WandbConfig(BaseConfig):
    project: str = "prime-rl"
    """W&B project name"""

    entity: str | None = None
    """W&B team or user; defaults to the logged-in user"""

    tags: list[str] = []
    """Tags attached to the run"""


class AdamWConfig(BaseConfig):
    type: Literal["adamw"] = "adamw"
    lr: float = 3e-4
    """Peak learning rate"""
    weight_decay: float = 0.01
    """L2 weight-decay coefficient"""
    betas: list[float] = [0.9, 0.95]
    """Adam (beta1, beta2) moments"""


class MuonConfig(BaseConfig):
    type: Literal["muon"] = "muon"
    lr: float = 2e-3
    """Peak learning rate"""
    momentum: float = 0.95
    """Newton-Schulz momentum"""


OptimizerConfig = Annotated[AdamWConfig | MuonConfig, Field(discriminator="type")]


class DataConfig(BaseConfig):
    path: Path = Path("./data")
    """Path to the dataset directory"""
    num_workers: int = 4
    """DataLoader worker processes"""
    shuffle: bool = True
    """Shuffle the training set each epoch"""


class CompileConfig(BaseConfig):
    backend: str = "inductor"
    """torch.compile backend"""
    mode: str = "default"
    """Compilation mode (default, reduce-overhead, max-autotune)"""
    fullgraph: bool = False
    """Require the entire model to be capturable in a single graph"""


class ModelConfig(BaseConfig):
    name: str = "qwen-1b"
    """Checkpoint name or HuggingFace ID"""
    hidden_size: int = 2048
    """Transformer hidden dimension"""
    num_layers: int = 24
    """Number of transformer blocks"""


class Config(BaseConfig):
    run_name: str = Field(description="Unique identifier for this training run")

    seed: int = Field(
        42,
        description="Random seed for reproducibility (alias: --random-seed)",
        validation_alias=AliasChoices("seed", "random_seed"),
    )
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    """Mixed-precision dtype"""
    output_dir: Path = Path("./output")
    """Where checkpoints and logs are written"""

    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()

    optimizer: OptimizerConfig = AdamWConfig()

    compile: CompileConfig | None = CompileConfig()
    """torch.compile settings (pass --no-compile to disable)"""

    wandb: WandbConfig | None = None

    checkpoint_steps: list[int] = []
    """Steps at which to save a checkpoint"""
    extra_kwargs: dict = {}
    """Arbitrary extra config passed to the trainer"""


def main(config: Config):
    pprint(config.model_dump())


if __name__ == "__main__":
    main(cli(Config))
