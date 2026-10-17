from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_FILE = "config.yaml"
METRICS_FILE = "metrics.jsonl"
SUMMARY_FILE = "summary.csv"


def run_name_from_config(
    *,
    config_path: Path,
    configs_root: Path,
) -> str:
    relative = config_path.resolve().relative_to(configs_root.resolve()).with_suffix("")
    return ".".join(relative.parts)


class ResultStore:
    def __init__(self, *, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_run(
        cls,
        *,
        outputs_root: Path,
        run_name: str,
        run_date: date | None = None,
    ) -> "ResultStore":
        stamp = (run_date or date.today()).isoformat()
        return cls(directory=outputs_root / f"{stamp}_{run_name}")

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def metrics_path(self) -> Path:
        return self._directory / METRICS_FILE

    def save_config(self, *, values: Mapping[str, Any]) -> Path:
        import yaml

        path = self._directory / CONFIG_FILE
        path.write_text(yaml.safe_dump(dict(values), sort_keys=False), encoding="utf-8")
        return path

    def append(self, *, record: Mapping[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record)) + "\n")
        logger.info("Result %s", {key: value for key, value in record.items() if not isinstance(value, (list, dict))})

    def records(self) -> list[dict[str, Any]]:
        if not self.metrics_path.exists():
            return []
        return [json.loads(line) for line in self.metrics_path.read_text(encoding="utf-8").splitlines() if line]

    def write_summary(self) -> Path | None:
        import pandas as pd

        records = self.records()
        if not records:
            return None
        frame = pd.DataFrame(
            [{key: value for key, value in record.items() if not isinstance(value, (list, dict))} for record in records]
        )
        path = self._directory / SUMMARY_FILE
        frame.to_csv(path, index=False)
        return path
