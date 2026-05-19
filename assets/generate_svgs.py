"""Generate SVG terminal screenshots for README."""

import subprocess
import os

from rich.console import Console
from rich.text import Text


TERM_COLUMNS = 110
WIDTH = TERM_COLUMNS
TRAIN = ["uv", "run", "python", "examples/train.py"]


def capture(cmd: list[str]) -> str:
    env = os.environ.copy()
    env["TERM"] = "xterm-256color"
    env["FORCE_COLOR"] = "1"
    env["COLUMNS"] = str(TERM_COLUMNS)
    env.pop("NO_COLOR", None)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result.stdout + result.stderr


def to_svg(ansi_text: str, title: str, filename: str):
    console = Console(record=True, width=WIDTH, force_terminal=True)
    console.print(Text.from_ansi(ansi_text.strip()))
    console.save_svg(filename, title=title)
    print(f"  {filename}")


def extract_box(output: str) -> str:
    lines = output.split("\n")
    result = []
    in_box = False
    for line in lines:
        if "╭" in line:
            in_box = True
        if in_box:
            result.append(line)
        if "╯" in line and in_box:
            break
    return "\n".join(result)


def extract_help(output: str) -> str:
    lines = output.split("\n")
    result = []
    started = False
    for line in lines:
        if "usage:" in line.lower():
            started = True
        if started:
            result.append(line)
    return "\n".join(result)


def session(cmd_display: str, cmd: list[str]) -> str:
    output = capture(cmd)
    return f"$ {cmd_display}\n{output.rstrip()}"


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Generating SVGs...")

    # 1. Help
    out = capture(TRAIN + ["--help"])
    to_svg(extract_help(out), "uv run python examples/train.py --help", "assets/help.svg")

    # 2. Config file
    to_svg(
        session("uv run python examples/train.py @ examples/train.toml", TRAIN + ["@", "examples/train.toml"]),
        "Config file via @",
        "assets/config_file.svg",
    )

    # 3. Required error
    to_svg(extract_box(capture(TRAIN)), "Missing required argument", "assets/required_error.svg")

    # 4. Nested override
    to_svg(
        session(
            "uv run python examples/train.py --run-name r1 --model.hidden-size 4096 --data.num-workers 16",
            TRAIN + ["--run-name", "r1", "--model.hidden-size", "4096", "--data.num-workers", "16"],
        ),
        "Nested config groups",
        "assets/nested.svg",
    )

    # 5. Bool --no-
    to_svg(
        session(
            "uv run python examples/train.py --run-name r1 --no-model.compile --no-data.shuffle",
            TRAIN + ["--run-name", "r1", "--no-model.compile", "--no-data.shuffle"],
        ),
        "Bool --no- negation",
        "assets/bool_negation.svg",
    )

    # 6. Lists
    to_svg(
        session(
            "uv run python examples/train.py --run-name r1 --checkpoint-steps 100 200 500",
            TRAIN + ["--run-name", "r1", "--checkpoint-steps", "100", "200", "500"],
        ),
        "List values",
        "assets/lists.svg",
    )

    # 7. Dicts
    to_svg(
        session(
            'python examples/train.py --run-name r1 --extra-kwargs \'{"seq_len": 4096}\'',
            TRAIN + ["--run-name", "r1", "--extra-kwargs", '{"seq_len": 4096}'],
        ),
        "Dict values",
        "assets/dicts.svg",
    )

    # 8. Optional sub-config
    to_svg(
        session(
            "uv run python examples/train.py --run-name r1 --wandb.project demo --wandb.entity me",
            TRAIN + ["--run-name", "r1", "--wandb.project", "demo", "--wandb.entity", "me"],
        ),
        "Optional sub-config",
        "assets/optional.svg",
    )

    # 9. Discriminated union switch
    to_svg(
        session(
            "uv run python examples/train.py --run-name r1 --optimizer.type muon --optimizer.lr 2e-3",
            TRAIN + ["--run-name", "r1", "--optimizer.type", "muon", "--optimizer.lr", "2e-3"],
        ),
        "Discriminated union",
        "assets/union_switch.svg",
    )

    # 10. Disable optional (enabled-by-default compile → off)
    to_svg(
        session(
            "uv run python examples/train.py --run-name r1 --no-compile",
            TRAIN + ["--run-name", "r1", "--no-compile"],
        ),
        "Disable optional sub-config",
        "assets/disable_optional.svg",
    )

    # 11. Validation alias
    to_svg(
        session(
            "uv run python examples/train.py --run-name r1 --random-seed 7",
            TRAIN + ["--run-name", "r1", "--random-seed", "7"],
        ),
        "Validation alias",
        "assets/alias.svg",
    )

    # 12. Validation error
    to_svg(
        extract_box(capture(TRAIN + ["--run-name", "r1", "--seed", "nope"])),
        "Validation error",
        "assets/config_error.svg",
    )

    # 13. Unknown flag
    to_svg(
        extract_box(capture(TRAIN + ["--run-name", "r1", "--seedz", "5"])),
        "Unknown flag suggestion",
        "assets/unknown_flag.svg",
    )

    # 14. File not found
    to_svg(
        extract_box(capture(TRAIN + ["@", "nonexistent.toml"])),
        "Config file not found",
        "assets/file_not_found.svg",
    )

    print("Done.")
