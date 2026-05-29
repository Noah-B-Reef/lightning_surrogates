import argparse
import json
import os
import shutil

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import config
from callbacks import RelativeImprovementEarlyStopping
from data import GravCollapseDataModule
from model import MLP


class MetricsHistoryLogger(pl.Callback):
    def __init__(self):
        super().__init__()
        self.history = {"train_loss": [], "val_loss": [], "train_mse": [], "val_mse": []}

    def on_train_epoch_end(self, trainer, pl_module):
        for key in ("train_loss", "train_mse"):
            value = trainer.callback_metrics.get(key)
            if value is not None:
                self.history[key].append(float(value.detach().cpu()))

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        for key in ("val_loss", "val_mse"):
            value = trainer.callback_metrics.get(key)
            if value is not None:
                self.history[key].append(float(value.detach().cpu()))


def load_best_config(config_file="best_params.txt"):
    key_mapping = {
        "num_layers": "num_hidden_layers",
        "hidden_units": "num_neurons_per_hidden_layer",
        "num_hidden_layers": "num_hidden_layers",
        "num_neurons_per_hidden_layer": "num_neurons_per_hidden_layer",
        "learning_rate": "learning_rate",
        "batch_size": "batch_size",
        "forecast_horizon": "forecast_horizon",
    }
    search_paths = [
        config_file,
        os.path.join("results", "optimization_results", config_file),
        os.path.join("results", "optuna", config_file),
    ]
    for path in search_paths:
        if not os.path.exists(path):
            continue
        params = {}
        with open(path) as f:
            for line in f:
                if "=" not in line:
                    continue
                key, value = line.strip().split("=", 1)
                key = key_mapping.get(key, key)
                if key in {
                    "num_hidden_layers",
                    "num_neurons_per_hidden_layer",
                    "batch_size",
                    "forecast_horizon",
                }:
                    params[key] = int(float(value))
                elif key == "learning_rate":
                    params[key] = float(value)
        if params:
            return params
    return {
        "num_hidden_layers": config.NUM_LAYERS,
        "num_neurons_per_hidden_layer": config.HIDDEN_UNITS,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
        "forecast_horizon": config.FORECAST_HORIZON,
    }


def plot_history(history, output_path):
    plt.figure(figsize=(10, 6))
    if history["train_loss"]:
        plt.plot(history["train_loss"], label="Train loss")
    if history["val_loss"]:
        plt.plot(history["val_loss"], label="Validation loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Smooth L1 loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_final_model(
    best_config,
    num_epochs=None,
    checkpoint_path="mlp_grav_collapse.ckpt",
    results_dir="results",
    data_dir=None,
    save_epoch_checkpoints=True,
    accelerator=config.ACCELERATOR,
    devices=config.NUM_DEVICES,
    precision=config.PRECISION,
):
    num_epochs = int(num_epochs or config.EPOCHS)
    data_dir = data_dir or str(config.DEFAULT_SPLIT_DIR)
    os.makedirs(results_dir, exist_ok=True)

    log_dir = os.path.join(results_dir, "lightning_logs", "mlp_grav_collapse")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    data = GravCollapseDataModule(
        data_dir=data_dir,
        batch_size=best_config["batch_size"],
        num_workers=config.NUM_WORKERS,
        forecast_horizon=best_config.get("forecast_horizon", config.FORECAST_HORIZON),
    )
    data.setup("fit")
    model_config = {
        "num_inputs": data.num_features,
        "output_size": data.num_targets,
        "forecast_horizon": best_config.get("forecast_horizon", config.FORECAST_HORIZON),
        "num_hidden_layers": best_config["num_hidden_layers"],
        "num_neurons_per_hidden_layer": best_config["num_neurons_per_hidden_layer"],
        "learning_rate": best_config["learning_rate"],
    }
    model = MLP(model_config)

    metrics = MetricsHistoryLogger()
    callbacks = [
        RelativeImprovementEarlyStopping(monitor="val_loss", patience=8, mode="min"),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join(results_dir, "checkpoints"),
            filename="mlp_best-{epoch:04d}-{val_loss:.5f}",
            save_top_k=1,
            mode="min",
        ),
        metrics,
    ]
    if save_epoch_checkpoints:
        callbacks.append(
            ModelCheckpoint(
                dirpath=os.path.join(results_dir, "epoch_checkpoints"),
                filename="epoch-{epoch:04d}",
                save_top_k=-1,
                every_n_epochs=1,
            )
        )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        logger=TensorBoardLogger(
            save_dir=os.path.join(results_dir, "lightning_logs"),
            name="mlp_grav_collapse",
        ),
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=data)

    final_checkpoint = os.path.join(results_dir, checkpoint_path)
    best_checkpoint = callbacks[1].best_model_path
    if best_checkpoint:
        shutil.copy2(best_checkpoint, final_checkpoint)
    else:
        trainer.save_checkpoint(final_checkpoint)

    with open(os.path.join(results_dir, "trained_model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)
    plot_history(metrics.history, os.path.join(results_dir, "loss_curves.png"))
    return model, trainer


def main(
    num_epochs=None,
    config_file="best_params.txt",
    checkpoint_path="mlp_grav_collapse.ckpt",
    use_defaults=False,
    results_dir="results",
    data_dir=None,
):
    if use_defaults:
        best_config = {
            "num_hidden_layers": config.NUM_LAYERS,
            "num_neurons_per_hidden_layer": config.HIDDEN_UNITS,
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "forecast_horizon": config.FORECAST_HORIZON,
        }
    else:
        best_config = load_best_config(config_file)
    return train_final_model(
        best_config,
        num_epochs=num_epochs,
        checkpoint_path=checkpoint_path,
        results_dir=results_dir,
        data_dir=data_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--config-file", type=str, default="best_params.txt")
    parser.add_argument("--checkpoint", type=str, default="mlp_grav_collapse.ckpt")
    parser.add_argument("--use-defaults", action="store_true")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()
    main(
        num_epochs=args.epochs,
        config_file=args.config_file,
        checkpoint_path=args.checkpoint,
        use_defaults=args.use_defaults,
        results_dir=args.results_dir,
        data_dir=args.data_dir,
    )
