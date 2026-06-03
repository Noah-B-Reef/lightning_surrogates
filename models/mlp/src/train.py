import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import settings as config
from callbacks import EpochProgressPrinter, RelativeImprovementEarlyStopping
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


def parse_devices(devices):
    if isinstance(devices, int):
        return devices
    if isinstance(devices, str) and devices.isdigit():
        return int(devices)
    return devices


def load_best_config(config_file):
    """Load model/training hyperparameters from JSON or key=value text."""
    path = Path(config_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        return {
            "num_hidden_layers": int(payload["num_hidden_layers"]),
            "num_neurons_per_hidden_layer": int(payload["num_neurons_per_hidden_layer"]),
            "learning_rate": float(payload["learning_rate"]),
            "batch_size": int(payload["batch_size"]),
        }

    key_mapping = {
        "num_layers": "num_hidden_layers",
        "hidden_units": "num_neurons_per_hidden_layer",
        "num_hidden_layers": "num_hidden_layers",
        "num_neurons_per_hidden_layer": "num_neurons_per_hidden_layer",
        "learning_rate": "learning_rate",
        "batch_size": "batch_size",
    }
    params = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.strip().split("=", 1)
        key = key_mapping.get(key, key)
        if key in {
            "num_hidden_layers",
            "num_neurons_per_hidden_layer",
            "batch_size",
        }:
            params[key] = int(float(value))
        elif key in {"learning_rate"}:
            params[key] = float(value)
    if not params:
        raise ValueError(f"No hyperparameters found in {path}")
    return params


def default_config():
    return {
        "num_hidden_layers": config.NUM_LAYERS,
        "num_neurons_per_hidden_layer": config.HIDDEN_UNITS,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
    }


def plot_history(history, output_path):
    output_path = Path(output_path)
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
    split_dir,
    results_dir,
    *,
    num_epochs=None,
    checkpoint_name=config.CHECKPOINT_NAME,
    save_epoch_checkpoints=True,
    accelerator=config.ACCELERATOR,
    devices=config.NUM_DEVICES,
    precision=config.PRECISION,
    num_workers=config.NUM_WORKERS,
):
    split_dir = config.resolve_split_dir(split_dir)
    results_dir = Path(results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    num_epochs = int(num_epochs or config.EPOCHS)

    log_dir = results_dir / "lightning_logs" / "mlp_grav_collapse"
    if log_dir.exists():
        shutil.rmtree(log_dir)

    data = GravCollapseDataModule(
        data_dir=str(split_dir),
        batch_size=int(best_config["batch_size"]),
        num_workers=int(num_workers),
    )
    data.setup("fit")
    model_config = {
        "num_inputs": data.num_features,
        "output_size": data.num_targets,
        "num_hidden_layers": int(best_config["num_hidden_layers"]),
        "num_neurons_per_hidden_layer": int(best_config["num_neurons_per_hidden_layer"]),
        "learning_rate": float(best_config["learning_rate"]),
    }
    model = MLP(model_config)
    train_batches = len(data.train_dataloader())
    val_batches = len(data.val_dataloader())
    num_parameters = sum(param.numel() for param in model.parameters())

    print(
        "Starting final training: "
        f"epochs={num_epochs}, "
        f"batch_size={best_config['batch_size']}, "
        f"learning_rate={best_config['learning_rate']:.6g}, "
        f"train_samples={len(data.train_ds):,}, "
        f"val_samples={len(data.val_ds):,}, "
        f"train_batches={train_batches:,}, "
        f"val_batches={val_batches:,}, "
        f"parameters={num_parameters:,}",
        flush=True,
    )

    metrics = MetricsHistoryLogger()
    callbacks = [
        RelativeImprovementEarlyStopping(monitor="val_loss", patience=8, mode="min"),
        EpochProgressPrinter(
            prefix="[Final training]",
            metric_names=("train_loss", "val_loss", "train_mse", "val_mse"),
        ),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=str(results_dir / "checkpoints"),
            filename="mlp_best-{epoch:04d}-{val_loss:.5f}",
            save_top_k=1,
            mode="min",
        ),
        metrics,
    ]
    if save_epoch_checkpoints:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(results_dir / "epoch_checkpoints"),
                filename="epoch-{epoch:04d}",
                save_top_k=-1,
                every_n_epochs=1,
            )
        )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        devices=parse_devices(devices),
        precision=precision,
        callbacks=callbacks,
        logger=TensorBoardLogger(
            save_dir=str(results_dir / "lightning_logs"),
            name="mlp_grav_collapse",
        ),
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=data)

    final_checkpoint = results_dir / checkpoint_name
    best_checkpoint = callbacks[2].best_model_path
    if best_checkpoint:
        shutil.copy2(best_checkpoint, final_checkpoint)
    else:
        trainer.save_checkpoint(final_checkpoint)

    (results_dir / "trained_model_config.json").write_text(json.dumps(model_config, indent=2))
    plot_history(metrics.history, results_dir / "loss_curves.png")
    print(
        f"Final training complete: checkpoint={final_checkpoint}, "
        f"best_checkpoint={best_checkpoint or 'none'}",
        flush=True,
    )
    return {
        "checkpoint": str(final_checkpoint),
        "best_checkpoint": best_checkpoint,
        "model_config": model_config,
        "split_dir": str(split_dir),
        "results_dir": str(results_dir),
    }


def main(
    dataset_path=None,
    num_epochs=None,
    config_file=None,
    checkpoint_name="mlp_grav_collapse.ckpt",
    use_defaults=False,
    results_dir=config.DEFAULT_RESULTS_DIR,
    data_dir=None,
    accelerator=config.ACCELERATOR,
    devices=config.NUM_DEVICES,
    precision=config.PRECISION,
    num_workers=config.NUM_WORKERS,
    save_epoch_checkpoints=True,
):
    split_dir = data_dir or dataset_path
    if use_defaults:
        best_config = default_config()
    else:
        if config_file is None:
            config_file = config.TRAIN_CONFIG_FILE
        best_config = load_best_config(config_file)
    return train_final_model(
        best_config,
        split_dir,
        results_dir,
        num_epochs=num_epochs,
        checkpoint_name=checkpoint_name,
        save_epoch_checkpoints=save_epoch_checkpoints,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the MLP on a split dataset directory.")
    parser.add_argument("dataset_path", nargs="?", default=None)
    parser.add_argument("--data-dir", type=Path, default=None, help="Alias for dataset_path.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--config-file", type=Path, default=config.TRAIN_CONFIG_FILE)
    parser.add_argument("--checkpoint", type=str, default=config.CHECKPOINT_NAME)
    parser.add_argument("--use-defaults", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=config.DEFAULT_RESULTS_DIR)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--accelerator", type=str, default=config.ACCELERATOR)
    parser.add_argument("--devices", default=config.NUM_DEVICES)
    parser.add_argument("--precision", default=config.PRECISION)
    parser.add_argument("--no-epoch-checkpoints", action="store_true")
    args = parser.parse_args()
    main(
        dataset_path=args.dataset_path,
        num_epochs=args.epochs,
        config_file=args.config_file,
        checkpoint_name=args.checkpoint,
        use_defaults=args.use_defaults,
        results_dir=args.results_dir,
        data_dir=args.data_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        num_workers=args.num_workers,
        save_epoch_checkpoints=not args.no_epoch_checkpoints,
    )
