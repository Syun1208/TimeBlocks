from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIGS_ROOT = ROOT / "configs"
EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_INTERRUPTED = 130

logger = logging.getLogger("timeblocks")


def existing_path(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{value} does not exist")
    return path


def _prepare(
    *,
    arguments: argparse.Namespace,
    app_name: str,
):
    from src.utils.env import resolve_storage
    from src.utils.logging_setup import setup_logging

    storage = resolve_storage(start=ROOT)
    log_file = setup_logging(
        app_name=app_name,
        log_dir=storage.logs,
        level=logging.DEBUG if arguments.verbose else logging.INFO,
    )
    logger.info("Log file %s", log_file)
    logger.info("Storage root %s", storage.root)
    return storage


def _settings(
    *,
    arguments: argparse.Namespace,
    settings_type: type,
):
    from src.utils.config_loader import load_settings, parse_override_pairs, settings_to_dict

    settings = load_settings(
        settings_type=settings_type,
        config_path=arguments.config,
        overrides=parse_override_pairs(pairs=arguments.set),
    )
    logger.debug("Settings %s", settings_to_dict(settings=settings))
    return settings


def _shard_targets(
    *,
    arguments: argparse.Namespace,
    settings,
):
    from dataclasses import replace

    if arguments.num_shards == 1 or not settings.target_datasets:
        return settings if arguments.shard_index == 0 else None
    targets = settings.target_datasets[arguments.shard_index :: arguments.num_shards]
    if not targets:
        logger.info("Shard %d has no target datasets", arguments.shard_index)
        return None
    logger.info(
        "Shard %d/%d handles targets %s",
        arguments.shard_index,
        arguments.num_shards,
        list(targets),
    )
    return replace(
        settings,
        target_datasets=targets,
    )


def _result_store(
    *,
    arguments: argparse.Namespace,
    storage,
    settings,
):
    from src.services.factories import build_result_store
    from src.utils.config_loader import settings_to_dict

    output_dir = arguments.output_dir
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = storage.outputs / output_dir
    return build_result_store(
        storage=storage,
        config_path=arguments.config,
        configs_root=CONFIGS_ROOT,
        resolved=settings_to_dict(settings=settings),
        output_dir=output_dir,
    )


def run_download(arguments: argparse.Namespace) -> int:
    from src.data.classes.settings import DownloadSettings
    from src.services.factories import build_download_service

    storage = _prepare(
        arguments=arguments,
        app_name="download",
    )
    settings = _settings(
        arguments=arguments,
        settings_type=DownloadSettings,
    )
    build_download_service(
        settings=settings,
        storage=storage,
    ).run()
    return EXIT_OK


def run_pretrain(arguments: argparse.Namespace) -> int:
    from src.data.classes.settings import PretrainSettings
    from src.services.factories import build_pretraining

    storage = _prepare(
        arguments=arguments,
        app_name="pretrain",
    )
    settings = _settings(
        arguments=arguments,
        settings_type=PretrainSettings,
    )
    service, grid = build_pretraining(
        settings=settings,
        storage=storage,
    )
    jobs = grid.jobs()
    logger.info("Pretraining grid has %d end-to-end models", len(jobs))
    if arguments.list_jobs:
        for job in jobs:
            print(job.model_id)
        return EXIT_OK
    service.run(
        jobs=jobs,
        shard_index=arguments.shard_index,
        num_shards=arguments.num_shards,
    )
    return EXIT_OK


def run_blockbase(arguments: argparse.Namespace) -> int:
    from src.data.classes.settings import BlockbaseSettings
    from src.services.factories import build_blockbase_index

    storage = _prepare(
        arguments=arguments,
        app_name="blockbase",
    )
    full_settings = _settings(
        arguments=arguments,
        settings_type=BlockbaseSettings,
    )
    settings = _shard_targets(
        arguments=arguments,
        settings=full_settings,
    )
    if settings is None:
        return EXIT_OK
    build_blockbase_index(
        settings=settings,
        storage=storage,
    ).run()
    return EXIT_OK


def run_timeblocks_evaluate(arguments: argparse.Namespace) -> int:
    from src.data.classes.settings import ZeroShotSettings
    from src.services.factories import build_timeblocks_experiment

    storage = _prepare(
        arguments=arguments,
        app_name="timeblocks_evaluate",
    )
    full_settings = _settings(
        arguments=arguments,
        settings_type=ZeroShotSettings,
    )
    settings = _shard_targets(
        arguments=arguments,
        settings=full_settings,
    )
    if settings is None:
        return EXIT_OK
    build_timeblocks_experiment(
        settings=settings,
        storage=storage,
        results=_result_store(
            arguments=arguments,
            storage=storage,
            settings=full_settings,
        ),
    ).run()
    return EXIT_OK


def run_baseline(arguments: argparse.Namespace) -> int:
    from src.data.classes.settings import BaselineSettings
    from src.services.factories import build_baseline_experiment

    storage = _prepare(
        arguments=arguments,
        app_name="baseline",
    )
    full_settings = _settings(
        arguments=arguments,
        settings_type=BaselineSettings,
    )
    settings = _shard_targets(
        arguments=arguments,
        settings=full_settings,
    )
    if settings is None:
        return EXIT_OK
    build_baseline_experiment(
        settings=settings,
        storage=storage,
        results=_result_store(
            arguments=arguments,
            storage=storage,
            settings=full_settings,
        ),
    ).run()
    return EXIT_OK


def run_report(arguments: argparse.Namespace) -> int:
    from src.data.classes.settings import ReportSettings
    from src.services.reporting import MetricsLoader, ReportBuilder

    storage = _prepare(
        arguments=arguments,
        app_name="report",
    )
    settings = _settings(
        arguments=arguments,
        settings_type=ReportSettings,
    )
    output_dir = arguments.output_dir or Path(f"report_{arguments.config.stem}")
    if not output_dir.is_absolute():
        output_dir = storage.outputs / output_dir
    ReportBuilder(
        settings=settings,
        loader=MetricsLoader(),
    ).build(
        inputs=[storage.outputs / path if not path.is_absolute() else path for path in arguments.inputs],
        output_dir=output_dir,
    )
    logger.info("Report written to %s", output_dir)
    return EXIT_OK


def _common(*, parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=existing_path, required=True, help="YAML config under configs/")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a config value")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory, relative paths go under the storage outputs/"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug logs on the console")
    parser.add_argument("--shard-index", type=int, default=0, help="Index of this worker when splitting the work")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of workers splitting the work")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TimeBlocks reproduction: blockbase, router, StreamCore, baselines")
    groups = parser.add_subparsers(dest="group", required=True)

    data = groups.add_parser("data", help="Dataset utilities").add_subparsers(dest="command", required=True)
    download = data.add_parser("download", help="Download forecasting, anomaly and UCR datasets")
    _common(parser=download)
    download.set_defaults(handler=run_download)

    timeblocks = groups.add_parser("timeblocks", help="TimeBlocks pipeline").add_subparsers(
        dest="command",
        required=True,
    )
    pretrain = timeblocks.add_parser("pretrain", help="Algorithm A1: pretrain end-to-end models into the Blockbase")
    _common(parser=pretrain)
    pretrain.add_argument("--list-jobs", action="store_true", help="Print the grid and exit")
    pretrain.set_defaults(handler=run_pretrain)

    blockbase = timeblocks.add_parser("blockbase", help="Algorithm A2 and Sec 3.2.3: fingerprints, K-Means, router")
    _common(parser=blockbase)
    blockbase.set_defaults(handler=run_blockbase)

    evaluate = timeblocks.add_parser("evaluate", help="Algorithm A3: build inference-time models and evaluate")
    _common(parser=evaluate)
    evaluate.set_defaults(handler=run_timeblocks_evaluate)

    baseline = groups.add_parser("baseline", help="Baselines").add_subparsers(dest="command", required=True)
    baseline_run = baseline.add_parser("run", help="Train on source datasets and evaluate on target datasets")
    _common(parser=baseline_run)
    baseline_run.set_defaults(handler=run_baseline)

    report = groups.add_parser("report", help="Aggregate metrics into tables and figures")
    _common(parser=report)
    report.add_argument("--inputs", type=Path, nargs="+", required=True, help="Run directories under outputs/")
    report.set_defaults(handler=run_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if not 0 <= arguments.shard_index < arguments.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    try:
        return arguments.handler(arguments)
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return EXIT_INTERRUPTED
    except (FileNotFoundError, ValueError, KeyError) as error:
        logger.error("%s: %s", type(error).__name__, error)
        logger.debug("Traceback", exc_info=True)
        return EXIT_USAGE_ERROR
    except Exception:
        logger.exception("Run failed")
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())
