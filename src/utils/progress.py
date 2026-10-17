from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from tqdm import tqdm

COUNT_BAR = (
    "{desc} ▕{bar}▏ {percentage:3.0f}% • {n_fmt}/{total_fmt} • {rate_fmt}" " • ⏳ {elapsed}<{remaining}{postfix}"
)
TIME_BAR = "{desc} ▕{bar}▏ {percentage:3.0f}% • ⏳ {elapsed}<{remaining}{postfix}"
NO_COLOR_VARIABLE = "NO_COLOR"


@dataclass(frozen=True, slots=True)
class Stage:
    emoji: str
    title: str
    colour: str


DOWNLOAD = Stage(emoji="📥", title="Download", colour="cyan")
PRETRAIN = Stage(emoji="🧩", title="Pretrain", colour="green")
TRAIN = Stage(emoji="🧠", title="Train", colour="blue")
VALIDATE = Stage(emoji="🔎", title="Validate", colour="magenta")
EVALUATE = Stage(emoji="📊", title="Evaluate", colour="yellow")
RESIDUALS = Stage(emoji="🔗", title="Residuals", colour="cyan")
ROUTER = Stage(emoji="🧭", title="Router", colour="blue")
SELECT = Stage(emoji="🔍", title="Select", colour="magenta")
STREAM = Stage(emoji="🌐", title="Stream", colour="green")
CALIBRATE = Stage(emoji="🚀", title="Calibrate", colour="yellow")
EXPERIMENT = Stage(emoji="🧬", title="Experiment", colour="cyan")


def progress(
    *,
    stage: Stage,
    total: float | None,
    unit: str = "it",
    show_counts: bool = True,
    leave: bool = True,
    detail: str = "",
) -> tqdm:
    layout = COUNT_BAR if show_counts else TIME_BAR
    use_colour = NO_COLOR_VARIABLE not in os.environ and sys.stderr.isatty()
    description = f"{stage.emoji} {stage.title}" if not detail else f"{stage.emoji} {stage.title} {detail}"
    return tqdm(
        total=total,
        unit=unit,
        desc=description,
        colour=stage.colour if use_colour else None,
        bar_format=layout if total else None,
        dynamic_ncols=True,
        leave=leave,
        file=sys.stderr,
    )


def finish(*, bar: tqdm) -> None:
    if bar.total is not None and bar.n < bar.total:
        bar.update(bar.total - bar.n)
    bar.close()
