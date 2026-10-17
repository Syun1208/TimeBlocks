from __future__ import annotations

import importlib
import sys
import threading
from pathlib import Path
from types import ModuleType

TIME_SERIES_LIBRARY = "Time-Series-Library"
ANOMALY_TRANSFORMER = "Anomaly-Transformer"
DCDETECTOR = "KDD2023-DCdetector"
LAG_LLAMA = "lag-llama"
SYMTIME = "SymTime"
SYMTIME_CHECKPOINT = Path("symtime_drive") / "finetuning.pth"
ISOLATED_PACKAGES = ("models", "model", "layers", "utils", "data_provider", "exp")

_IMPORT_LOCK = threading.Lock()


class ThirdPartyError(ImportError):
    pass


def _belongs_to(
    *,
    name: str,
    packages: tuple[str, ...],
) -> bool:
    return any(name == package or name.startswith(f"{package}.") for package in packages)


class IsolatedImporter:
    def __init__(self, *, third_party_root: Path) -> None:
        self._root = third_party_root

    def repository(self, *, name: str) -> Path:
        path = self._root / name
        if not path.is_dir():
            raise ThirdPartyError(f"Missing third-party repository {path}. Run: bash scripts/setup_third_party.sh")
        return path

    def load(
        self,
        *,
        repository: str,
        module: str,
    ) -> ModuleType:
        root = self.repository(name=repository)
        packages = tuple(sorted({*ISOLATED_PACKAGES, module.split(".")[0]}))
        with _IMPORT_LOCK:
            saved = {
                name: value
                for name, value in sys.modules.items()
                if _belongs_to(
                    name=name,
                    packages=packages,
                )
            }
            for name in saved:
                del sys.modules[name]
            sys.path.insert(0, str(root))
            try:
                return importlib.import_module(module)
            finally:
                sys.path.remove(str(root))
                loaded = [
                    name
                    for name in sys.modules
                    if _belongs_to(
                        name=name,
                        packages=packages,
                    )
                ]
                for name in loaded:
                    sys.modules[f"_third_party.{repository}.{name}"] = sys.modules.pop(name)
                sys.modules.update(saved)
