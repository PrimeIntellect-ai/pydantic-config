"""Generate SVG terminal screenshots for README."""

import subprocess
import os

from rich.console import Console
from rich.text import Text


def capture_with_colors(cmd: list[str]) -> str:
    """Capture command output with ANSI colors.

    ``FORCE_COLOR=1`` makes ``pydantic_config``'s ``_supports_color()``
    return True even without a TTY, so ANSI codes are emitted into the
    captured pipe.
    """
    env = os.environ.copy()
    env["TERM"] = "xterm-256color"
    env["FORCE_COLOR"] = "1"
    env.pop("NO_COLOR", None)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result.stdout + result.stderr


def ansi_to_svg(ansi_text: str, title: str, filename: str, width: int = 85):
    """Convert ANSI text to SVG using rich."""
    console = Console(record=True, width=width, force_terminal=True)

    # Parse ANSI and print to console
    text = Text.from_ansi(ansi_text.strip())
    console.print(text)

    # Export to SVG
    console.save_svg(filename, title=title)
    print(f"Saved {filename}")


def extract_box(output: str) -> str:
    """Extract just the box portion from output."""
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
    """Extract help output (all boxes)."""
    lines = output.split("\n")
    result = []
    started = False
    box_count = 0
    for line in lines:
        if "usage:" in line.lower():
            started = True
        if started:
            result.append(line)
            if "╯" in line:
                box_count += 1
                # Stop after every panel in examples/train.py:
                # options + model + data + 2 optimizer variants + wandb.
                if box_count >= 6:
                    break
    return "\n".join(result)


if __name__ == "__main__":
    # Change to repo root (parent of assets directory)
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Help output
    output = capture_with_colors(["uv", "run", "python", "examples/train.py", "--help"])
    help_text = extract_help(output)
    ansi_to_svg(help_text, "pydantic-config --help", "assets/help.svg", width=82)

    # Required error
    output = capture_with_colors(["uv", "run", "python", "examples/train.py"])
    box_text = extract_box(output)
    ansi_to_svg(box_text, "Missing Required Argument", "assets/required_error.svg", width=60)

    # Config validation error
    output = capture_with_colors(
        ["uv", "run", "python", "examples/train.py", "--run-name", "r1", "--seed", "nope"]
    )
    box_text = extract_box(output)
    ansi_to_svg(box_text, "Config Validation Error", "assets/config_error.svg", width=82)

    # File not found
    output = capture_with_colors(["uv", "run", "python", "examples/train.py", "@", "nonexistent.toml"])
    box_text = extract_box(output)
    ansi_to_svg(box_text, "Config File Not Found", "assets/file_not_found.svg", width=82)
