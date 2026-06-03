import argparse
import json
import re
import time
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch
from sqlalchemy.exc import OperationalError
from optuna.trial import TrialState

import settings as config
from callbacks import EpochProgressPrinter, RelativeImprovementEarlyStopping
from data import GravCollapseDataModule
from model import MLP


TRAINING_PARAM_KEYS = (
    "num_hidden_layers",
    "num_neurons_per_hidden_layer",
    "learning_rate",
    "batch_size",
)


def parse_devices(devices):
    if isinstance(devices, int):
        return devices
    if isinstance(devices, str) and devices.isdigit():
        return int(devices)
    return devices


def objective(trial, args, split_dir):
    search = config.OPTUNA_SEARCH_SPACE
    params = {
        "num_hidden_layers": trial.suggest_int(
            "num_hidden_layers", search["num_layers"]["low"], search["num_layers"]["high"]
        ),
        "num_neurons_per_hidden_layer": trial.suggest_int(
            "num_neurons_per_hidden_layer",
            search["hidden_units"]["low"],
            search["hidden_units"]["high"],
            step=search["hidden_units"]["step"],
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            search["learning_rate"]["low"],
            search["learning_rate"]["high"],
            log=search["learning_rate"]["log"],
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", search["batch_size"]["choices"]
        ),
    }

    data = GravCollapseDataModule(
        data_dir=str(split_dir),
        batch_size=params["batch_size"],
        num_workers=args.num_workers,
    )
    data.setup("fit")
    model_config = {
        "num_inputs": data.num_features,
        "output_size": data.num_targets,
        "num_hidden_layers": params["num_hidden_layers"],
        "num_neurons_per_hidden_layer": params["num_neurons_per_hidden_layer"],
        "learning_rate": params["learning_rate"],
    }
    model = MLP(model_config)
    train_batches = len(data.train_dataloader())
    val_batches = len(data.val_dataloader())
    num_parameters = sum(param.numel() for param in model.parameters())

    print(
        "[Optuna trial "
        f"{trial.number}] architecture: "
        f"input={model_config['num_inputs']}, "
        f"hidden_layers={model_config['num_hidden_layers']}, "
        f"hidden_units={model_config['num_neurons_per_hidden_layer']}, "
        f"output={model_config['output_size']}, "
        f"parameters={num_parameters:,}; "
        f"training: batch_size={params['batch_size']}, "
        f"learning_rate={params['learning_rate']:.6g}, "
        f"train_samples={len(data.train_ds):,}, "
        f"val_samples={len(data.val_ds):,}, "
        f"train_batches={train_batches:,}, "
        f"val_batches={val_batches:,}",
        flush=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.tune_epochs,
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        precision=args.precision,
        callbacks=[
            EpochProgressPrinter(
                prefix=f"[Optuna trial {trial.number}]",
                metric_names=("train_loss", "val_loss"),
            ),
            RelativeImprovementEarlyStopping(
                monitor="val_loss",
                min_relative_improvement=args.min_relative_improvement,
                patience=args.patience,
                mode="min",
                verbose=False,
            )
        ],
        logger=pl.loggers.TensorBoardLogger(
            save_dir=str(args.results_dir / "lightning_logs"),
            name="optuna",
            version=f"trial_{trial.number}",
        ),
        enable_checkpointing=False,
        enable_model_summary=False,
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=data)
    val_loss = trainer.callback_metrics.get("val_loss")
    if val_loss is None:
        raise RuntimeError("Trial completed without val_loss")

    trial.set_user_attr("params_for_training", params)
    value = float(val_loss.detach().cpu())
    print(f"[Optuna trial {trial.number}] complete val_loss={value:.6g}", flush=True)
    del trainer, model, data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return value


def save_best_params(best_params, results_dir, best_value):
    payload = {key: best_params[key] for key in TRAINING_PARAM_KEYS}
    payload["best_value"] = float(best_value)
    json_path = results_dir / "best_params.json"
    txt_path = results_dir / "best_params.txt"
    json_path.write_text(json.dumps(payload, indent=2))
    with txt_path.open("w") as f:
        f.write(f"num_layers={best_params['num_hidden_layers']}\n")
        f.write(f"hidden_units={best_params['num_neurons_per_hidden_layer']}\n")
        f.write(f"learning_rate={best_params['learning_rate']}\n")
        f.write(f"batch_size={best_params['batch_size']}\n")
    return json_path, txt_path


def sqlite_path_from_storage(storage):
    match = re.fullmatch(r"sqlite:///(.+)", storage)
    if match is None:
        return None
    return Path(match.group(1)).expanduser().resolve()


def reset_sqlite_storage(storage):
    sqlite_path = sqlite_path_from_storage(storage)
    if sqlite_path is None:
        raise ValueError("--journal-mode fresh is only supported for sqlite:/// storage URLs.")
    for path in (
        sqlite_path,
        Path(f"{sqlite_path}-journal"),
        Path(f"{sqlite_path}-wal"),
        Path(f"{sqlite_path}-shm"),
    ):
        if path.exists():
            path.unlink()
            print(f"Removed {path}", flush=True)


def finished_trial_count(study):
    finished_states = {TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL}
    return sum(
        1
        for trial in study.get_trials(deepcopy=False)
        if trial.state in finished_states
    )


def create_study_with_retry(args, storage, retries=5, delay_seconds=2.0):
    """Create or load an Optuna study, tolerating concurrent SQLite initialization."""
    for attempt in range(retries + 1):
        try:
            return optuna.create_study(
                study_name=args.study_name,
                storage=storage,
                direction="minimize",
                load_if_exists=True,
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            )
        except OperationalError as exc:
            if "already exists" not in str(exc).lower() or attempt == retries:
                raise
            sleep_time = delay_seconds * (attempt + 1)
            print(
                "Optuna storage initialization raced with another process; "
                f"retrying in {sleep_time:.1f}s.",
                flush=True,
            )
            time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(
        description="Optimize MLP hyperparameters for a split dataset directory."
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default=None,
        help="Split directory with train/val/test.csv (default: best-sampler split).",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Alias for dataset_path.")
    parser.add_argument("--results-dir", type=Path, default=config.DEFAULT_OPTUNA_RESULTS_DIR)
    parser.add_argument("--num-trials", type=int, default=config.OPTUNA_N_TRIALS)
    parser.add_argument("--tune-epochs", type=int, default=config.OPTUNA_TUNE_EPOCHS)
    parser.add_argument("--study-name", type=str, default=config.OPTUNA_STUDY_NAME)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument(
        "--journal-mode",
        choices=("resume", "fresh"),
        default=config.OPTUNA_JOURNAL_MODE,
        help=(
            "resume: reuse the existing Optuna journal and run only enough trials "
            "to reach --num-trials finished trials. fresh: remove the SQLite "
            "journal and start a new study."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--accelerator", type=str, default=config.ACCELERATOR)
    parser.add_argument("--devices", default=config.NUM_DEVICES)
    parser.add_argument("--precision", default=config.PRECISION)
    parser.add_argument("--patience", type=int, default=config.OPTUNA_PRUNER_PATIENCE)
    parser.add_argument(
        "--min-relative-improvement",
        type=float,
        default=config.OPTUNA_MIN_RELATIVE_IMPROVEMENT,
    )
    args = parser.parse_args()

    dataset_path = args.data_dir or args.dataset_path
    split_dir = config.resolve_split_dir(dataset_path)
    args.results_dir = Path(args.results_dir).expanduser().resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    configured_storage = None if config.OPTUNA_STORAGE == "auto" else config.OPTUNA_STORAGE
    storage = args.storage or configured_storage or f"sqlite:///{args.results_dir / 'optuna.sqlite3'}"
    if args.journal_mode == "fresh":
        reset_sqlite_storage(storage)
    study = create_study_with_retry(args, storage)
    existing_finished_trials = 0 if args.journal_mode == "fresh" else finished_trial_count(study)
    trials_to_run = max(0, args.num_trials - existing_finished_trials)
    print(
        f"Starting Optuna study '{args.study_name}' with journal_mode={args.journal_mode}, "
        f"finished_trials={existing_finished_trials}, target_trials={args.num_trials}, "
        f"trials_to_run={trials_to_run}, and up to {args.tune_epochs} epochs per trial.",
        flush=True,
    )
    if trials_to_run > 0:
        study.optimize(lambda trial: objective(trial, args, split_dir), n_trials=trials_to_run)
    else:
        print("No new trials needed; journal already meets the requested target.", flush=True)

    raw_best_params = study.best_trial.user_attrs.get("params_for_training", dict(study.best_params))
    best_params = {key: raw_best_params[key] for key in TRAINING_PARAM_KEYS}
    json_path, txt_path = save_best_params(best_params, args.results_dir, study.best_value)

    summary = {
        "split_dir": str(split_dir),
        "study_name": args.study_name,
        "storage": storage,
        "journal_mode": args.journal_mode,
        "target_trials": args.num_trials,
        "finished_trials": finished_trial_count(study),
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": best_params,
    }
    (args.results_dir / "optimization_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Best value: {study.best_value:.6g}")
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
