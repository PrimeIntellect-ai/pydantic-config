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

from pydantic import AliasChoices, Field, field_validator, model_validator

from pydantic_config import cli, BaseConfig


class WandbConfig(BaseConfig):
    """Weights & Biases logging."""

    project: str = "prime-rl"
    """W&B project name"""

    entity: str | None = None
    """W&B team or user; defaults to the logged-in user"""

    tags: list[str] = []
    """Tags attached to the run"""


class AdamWConfig(BaseConfig):
    type: Literal["adamw"] = "adamw"
    lr: float = Field(3e-4, gt=0, description="Peak learning rate")
    weight_decay: float = Field(0.01, ge=0, description="L2 weight-decay coefficient")
    betas: list[float] = [0.9, 0.95]
    """Adam (beta1, beta2) moments"""


class MuonConfig(BaseConfig):
    type: Literal["muon"] = "muon"
    lr: float = Field(2e-3, gt=0, description="Peak learning rate")
    momentum: float = 0.95
    """Newton-Schulz momentum"""


OptimizerConfig = Annotated[AdamWConfig | MuonConfig, Field(discriminator="type")]


class EnvConfig(BaseConfig):
    """RL environment settings."""

    name: str = "math"
    """Environment identifier"""
    weight: float = Field(1.0, ge=0, description="Sampling weight for this env")
    num_workers: int = Field(4, ge=0, description="Parallel rollout workers")


class DataConfig(BaseConfig):
    """Dataset and dataloader settings."""

    path: Path = Path("./data")
    """Path to the dataset directory"""
    num_workers: int = Field(4, ge=0, description="DataLoader worker processes")
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
    hidden_size: int = Field(2048, gt=0, description="Transformer hidden dimension")
    num_layers: int = Field(32, gt=0, description="Number of transformer blocks")

    @model_validator(mode="after")
    def _check_hidden_divisible_by_layers(self):
        if self.hidden_size % self.num_layers != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_layers ({self.num_layers})"
            )
        return self


class StudentConfig(BaseConfig):
    model: ModelConfig = ModelConfig()


class Config(BaseConfig):
    run_name: str = Field(description="Unique identifier for this training run")

    seed: int = Field(
        42,
        description="Random seed for reproducibility (aliases: --random-seed, -s)",
        validation_alias=AliasChoices("seed", "random_seed", "s"),
    )
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    """Mixed-precision dtype"""
    output_dir: Path = Path("./output")
    """Where checkpoints and logs are written"""

    student: StudentConfig = StudentConfig()
    data: DataConfig = DataConfig()

    @model_validator(mode="before")
    @classmethod
    def _migrate_model_to_student(cls, data: dict) -> dict:
        """Legacy support: remap ``model.*`` → ``student.model.*``."""
        if isinstance(data, dict) and "model" in data and "student" not in data:
            data["student"] = {"model": data.pop("model")}
        return data

    optimizer: OptimizerConfig = AdamWConfig()

    compile: CompileConfig | None = CompileConfig()
    """torch.compile settings"""

    wandb: WandbConfig | None = None

    envs: list[EnvConfig] = [EnvConfig()]
    """RL environments to train on"""

    checkpoint_steps: list[int] = []
    """Steps at which to save a checkpoint"""
    extra_kwargs: dict = {}
    """Arbitrary extra config passed to the trainer"""

    @field_validator("checkpoint_steps")
    @classmethod
    def _steps_must_be_sorted(cls, v: list[int]) -> list[int]:
        if v != sorted(v):
            raise ValueError(f"checkpoint_steps must be in ascending order, got {v}")
        return v


def main(config: Config):
    pprint(config.model_dump())


if __name__ == "__main__":
    main(cli(Config))
