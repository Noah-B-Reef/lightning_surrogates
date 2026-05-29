import argparse
import json
import os

import optuna
import pytorch_lightning as pl
import torch

import config
from callbacks import RelativeImprovementEarlyStopping
from data import GravCollapseDataModule
from model import MLP


def objective(trial, args):
    params = {
        "num_hidden_layers": trial.suggest_int(
            "num_hidden_layers",
            config.OPTUNA_SEARCH_SPACE["num_layers"]["low"],
            config.OPTUNA_SEARCH_SPACE["num_layers"]["high"],
        ),
        "num_neurons_per_hidden_layer": trial.suggest_int(
            "num_neurons_per_hidden_layer",
            config.OPTUNA_SEARCH_SPACE["hidden_units"]["low"],
            config.OPTUNA_SEARCH_SPACE["hidden_units"]["high"],
            step=config.OPTUNA_SEARCH_SPACE["hidden_units"]["step"],
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            config.OPTUNA_SEARCH_SPACE["learning_rate"]["low"],
            config.OPTUNA_SEARCH_SPACE["learning_rate"]["high"],
            log=config.OPTUNA_SEARCH_SPACE["learning_rate"]["log"],
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", config.OPTUNA_SEARCH_SPACE["batch_size"]["choices"]
        ),
        "forecast_horizon": args.forecast_horizon,
    }

    data = GravCollapseDataModule(
        data_dir=args.data_dir,
        batch_size=params["batch_size"],
        num_workers=args.num_workers,
        forecast_horizon=params["forecast_horizon"],
    )
    data.setup("fit")
    model = MLP(
        {
            "num_inputs": data.num_features,
            "output_size": data.num_targets,
            "num_hidden_layers": params["num_hidden_layers"],
            "num_neurons_per_hidden_layer": params["num_neurons_per_hidden_layer"],
            "learning_rate": params["learning_rate"],
            "forecast_horizon": params["forecast_horizon"],
        }
    )
    trainer = pl.Trainer(
        max_epochs=args.tune_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=[
            RelativeImprovementEarlyStopping(
                monitor="val_loss", patience=args.patience, mode="min", verbose=False
            )
        ],
        logger=pl.loggers.TensorBoardLogger(
            save_dir=os.path.join(args.results_dir, "lightning_logs"),
            name="optuna",
            version=f"trial_{trial.number}",
        ),
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=data)
    val_loss = trainer.callback_metrics.get("val_loss")
    if val_loss is None:
        raise RuntimeError("Trial completed without val_loss")
    trial.set_user_attr("params_for_training", params)
    value = float(val_loss.detach().cpu())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(config.DEFAULT_SPLIT_DIR))
    parser.add_argument("--results-dir", type=str, default="results/optimization_results")
    parser.add_argument("--num-trials", type=int, default=config.OPTUNA_N_TRIALS)
    parser.add_argument("--tune-epochs", type=int, default=config.OPTUNA_TUNE_EPOCHS)
    parser.add_argument("--study-name", type=str, default=config.OPTUNA_STUDY_NAME)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--forecast-horizon", type=int, default=config.FORECAST_HORIZON)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--accelerator", type=str, default=config.ACCELERATOR)
    parser.add_argument("--devices", default=config.NUM_DEVICES)
    parser.add_argument("--precision", default=config.PRECISION)
    parser.add_argument("--patience", type=int, default=config.OPTUNA_PRUNER_PATIENCE)
    args = parser.parse_args()

    if isinstance(args.devices, str) and args.devices.isdigit():
        args.devices = int(args.devices)
    os.makedirs(args.results_dir, exist_ok=True)
    storage = args.storage or f"sqlite:///{os.path.join(args.results_dir, 'optuna.sqlite3')}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(lambda trial: objective(trial, args), n_trials=args.num_trials)

    best_params = study.best_trial.user_attrs.get("params_for_training", study.best_params)
    with open(os.path.join(args.results_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    with open(os.path.join(args.results_dir, "best_params.txt"), "w") as f:
        f.write(f"num_layers={best_params['num_hidden_layers']}\n")
        f.write(f"hidden_units={best_params['num_neurons_per_hidden_layer']}\n")
        f.write(f"learning_rate={best_params['learning_rate']}\n")
        f.write(f"batch_size={best_params['batch_size']}\n")
        f.write(f"forecast_horizon={best_params.get('forecast_horizon', args.forecast_horizon)}\n")
    print(f"Best value: {study.best_value:.6g}")


if __name__ == "__main__":
    main()
