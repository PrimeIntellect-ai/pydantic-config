"""End-to-end demo of pydantic-config covering the patterns used in prime-rl
training scripts: nested configs, required fields, bool toggles with --no-,
lists, dicts, Optional sub-configs, discriminated-union optimizer choice,
field descriptions surfaced in --help, and validation aliases.

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
    project: str = Field("prime-rl", description="W&B project name")
    entity: str | None = Field(None, description="W&B team or user; defaults to the logged-in user")
    tags: list[str] = Field([], description="Tags attached to the run")


class AdamWConfig(BaseConfig):
    type: Literal["adamw"] = "adamw"
    lr: float = Field(3e-4, description="Peak learning rate")
    weight_decay: float = Field(0.01, description="L2 weight-decay coefficient")
    betas: list[float] = Field([0.9, 0.95], description="Adam (beta1, beta2) moments")


class MuonConfig(BaseConfig):
    type: Literal["muon"] = "muon"
    lr: float = Field(2e-3, description="Peak learning rate")
    momentum: float = Field(0.95, description="Newton-Schulz momentum")


OptimizerConfig = Annotated[AdamWConfig | MuonConfig, Field(discriminator="type")]


class DataConfig(BaseConfig):
    path: Path = Field(Path("./data"), description="Path to the dataset directory")
    num_workers: int = Field(4, description="DataLoader worker processes")
    shuffle: bool = Field(True, description="Shuffle the training set each epoch")


class ModelConfig(BaseConfig):
    name: str = Field("qwen-1b", description="Checkpoint name or HuggingFace ID")
    hidden_size: int = Field(2048, description="Transformer hidden dimension")
    num_layers: int = Field(24, description="Number of transformer blocks")
    compile: bool = Field(True, description="Wrap the model in torch.compile")


class Config(BaseConfig):
    run_name: str = Field(description="Unique identifier for this training run")

    # ``validation_alias=AliasChoices(...)`` lets the user also pass --random-seed
    # on the CLI or write ``random_seed = ...`` in TOML/YAML. Listing the canonical
    # name keeps it accepted as well.
    seed: int = Field(
        42,
        description="Random seed for reproducibility (alias: --random-seed)",
        validation_alias=AliasChoices("seed", "random_seed"),
    )
    precision: Literal["bf16", "fp16", "fp32"] = Field("bf16", description="Mixed-precision dtype")
    output_dir: Path = Field(Path("./output"), description="Where checkpoints and logs are written")

    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()

    optimizer: OptimizerConfig = AdamWConfig()

    wandb: WandbConfig | None = None

    checkpoint_steps: list[int] = Field([], description="Steps at which to save a checkpoint")
    extra_kwargs: dict = Field({}, description="Arbitrary extra config passed to the trainer")


def main(config: Config):
    pprint(config.model_dump())


if __name__ == "__main__":
    main(cli(Config))
