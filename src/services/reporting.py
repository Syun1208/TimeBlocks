from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from enum import Enum
from pathlib import Path

import pandas as pd

from src.data.classes.settings import ReportSettings
from src.services.results import METRICS_FILE
from src.utils.plotting import apply_academic_style, plot_critical_difference, plot_metric_curves, save_figure
from src.utils.ranking import rank_models

logger = logging.getLogger(__name__)

TABLE_FILE = "table.csv"
MARKDOWN_FILE = "table.md"
RANKING_FILE = "ranking.json"
FIGURES_DIRECTORY = "figures"
AVERAGE_COLUMN = "Average"
FLOAT_FORMAT = "{:.3f}"


class ReportKind(str, Enum):
    TABLE = "table"
    CRITICAL_DIFFERENCE = "critical_difference"
    BLOCKS_CURVE = "blocks_curve"


class MetricsLoader:
    def load(
        self,
        *,
        inputs: Sequence[Path],
        settings: ReportSettings,
    ) -> pd.DataFrame:
        rows = []
        for directory in inputs:
            path = directory / METRICS_FILE
            if not path.exists():
                raise FileNotFoundError(f"No {METRICS_FILE} in {directory}")
            rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
        frame = pd.DataFrame(rows)
        frame = frame[(frame["task"] == settings.task) & (frame["setting"] == settings.setting)]
        if frame.empty or settings.metric not in frame:
            raise ValueError(f"No {settings.metric} rows for task {settings.task} and setting {settings.setting}")
        return frame


class ReportBuilder:
    def __init__(
        self,
        *,
        settings: ReportSettings,
        loader: MetricsLoader,
    ) -> None:
        self._settings = settings
        self._loader = loader

    def _table(
        self,
        *,
        frame: pd.DataFrame,
        output_dir: Path,
    ) -> pd.DataFrame:
        table = frame.pivot_table(
            index="model",
            columns="dataset",
            values=self._settings.metric,
            aggfunc="mean",
        )
        table[AVERAGE_COLUMN] = table.mean(axis=1)
        table = table.sort_values(
            by=AVERAGE_COLUMN,
            ascending=self._settings.lower_is_better,
        )
        table.to_csv(output_dir / TABLE_FILE)
        (output_dir / MARKDOWN_FILE).write_text(
            table.to_markdown(floatfmt=".3f"),
            encoding="utf-8",
        )
        return table

    def _critical_difference(
        self,
        *,
        table: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        complete = table.drop(columns=AVERAGE_COLUMN).dropna(axis=1)
        scores = complete.to_numpy().T
        if self._settings.lower_is_better:
            scores = -scores
        result = rank_models(
            models=list(complete.index),
            scores=scores,
        )
        (output_dir / RANKING_FILE).write_text(
            json.dumps(
                {
                    "models": result.models,
                    "average_ranks": result.average_ranks,
                    "friedman_p_value": result.friedman_p_value,
                    "cliques": result.cliques,
                    "datasets": list(complete.columns),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        figure = plot_critical_difference(
            models=result.models,
            average_ranks=result.average_ranks,
            cliques=result.cliques,
        )
        save_figure(
            figure=figure,
            path=output_dir / FIGURES_DIRECTORY / (self._settings.figure_name or "critical_difference"),
        )

    def _blocks_curve(
        self,
        *,
        frame: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        grouped = frame.groupby(["dataset", "num_blocks"])[self._settings.metric].mean().reset_index()
        curves = {
            dataset: (
                group["num_blocks"].to_numpy(dtype=float),
                group[self._settings.metric].to_numpy(dtype=float),
            )
            for dataset, group in grouped.groupby("dataset")
        }
        figure = plot_metric_curves(
            curves=curves,
            x_label="Number of blocks (J)",
            y_label=self._settings.metric.upper(),
        )
        save_figure(
            figure=figure,
            path=output_dir / FIGURES_DIRECTORY / (self._settings.figure_name or "blocks_curve"),
        )
        grouped.to_csv(output_dir / TABLE_FILE, index=False)

    def build(
        self,
        *,
        inputs: Sequence[Path],
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        apply_academic_style()
        frame = self._loader.load(
            inputs=inputs,
            settings=self._settings,
        )
        kind = ReportKind(self._settings.kind)
        if kind is ReportKind.BLOCKS_CURVE:
            self._blocks_curve(
                frame=frame,
                output_dir=output_dir,
            )
            return output_dir
        table = self._table(
            frame=frame,
            output_dir=output_dir,
        )
        logger.info("Report table\n%s", table.to_string(float_format=FLOAT_FORMAT.format))
        if kind is ReportKind.CRITICAL_DIFFERENCE:
            self._critical_difference(
                table=table,
                output_dir=output_dir,
            )
        return output_dir
