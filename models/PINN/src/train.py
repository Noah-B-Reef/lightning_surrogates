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
from data import GOW17DataModule
from model import PINN


class MetricsHistoryLogger(pl.Callback):
    def __init__(self):
        super().__init__()
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_physics_loss": [], "val_physics_loss": [],
            "train_conservation_loss": [], "val_conservation_loss": [],
        }

    def on_train_epoch_end(self, trainer, pl_module):
        for key in ("train_loss", "train_physics_loss", "train_conservation_loss"):
            value = trainer.callback_metrics.get(key)
            if value is not None:
                self.history[key].append(float(value.detach().cpu()))

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        for key in ("val_loss", "val_physics_loss", "val_conservation_loss"):
            value = trainer.callback_metrics.get(key)
            if value is not None:
                self.history[key].append(float(value.detach().cpu()))


def parse_devices(devices):
    if isinstance(devices, int):
        return devices
    if isinstance(devices, str) and devices.isdigit():
        return int(devices)
    return devices


def default_config():
    return {
        "num_hidden_layers": config.NUM_LAYERS,
        "num_neurons_per_hidden_layer": config.HIDDEN_UNITS,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
    }


def load_best_config(config_file):
    """Load hyperparameters from optimize.py's best_params.json.

    Architecture/training keys are required; the loss weights
    (physics_weight, conservation_weight) are optional so the file can also
    come from the MLP-style optimizer or be hand-written.
    """
    path = Path(config_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = json.loads(path.read_text())
    best = {
        "num_hidden_layers": int(payload["num_hidden_layers"]),
        "num_neurons_per_hidden_layer": int(payload["num_neurons_per_hidden_layer"]),
        "learning_rate": float(payload["learning_rate"]),
        "batch_size": int(payload["batch_size"]),
    }
    for key in ("physics_weight", "conservation_weight"):
        if key in payload:
            best[key] = float(payload[key])
    return best


def plot_history(history, output_path):
    output_path = Path(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for key in ("train_loss", "val_loss"):
        if history[key]:
            axes[0].plot(history[key], label=key.replace("_", " "))
    axes[0].set_title("Data loss (L1, log10 abundances)")
    for key in ("train_physics_loss", "val_physics_loss"):
        if history[key]:
            axes[1].plot(history[key], label=key.replace("_", " "))
    axes[1].set_title("Physics (ODE residual) loss")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_final_model(
    best_config,
    split_dir,
    results_dir=None,
    *,
    num_epochs=None,
    checkpoint_name=config.CHECKPOINT_NAME,
    accelerator=config.ACCELERATOR,
    devices=config.NUM_DEVICES,
    precision=config.PRECISION,
    num_workers=config.NUM_WORKERS,
    physics_weight=config.PHYSICS_WEIGHT,
    conservation_weight=config.CONSERVATION_WEIGHT,
    max_horizon=config.MAX_HORIZON,
    early_stopping=True,
    early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
    early_stopping_min_relative_improvement=(
        config.EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT
    ),
    enable_logger=True,
    seed=42,
):
    pl.seed_everything(int(seed), workers=True)
    split_dir = config.resolve_split_dir(split_dir)
    if results_dir is None:
        results_dir = config.experiment_dir(split_dir)
    results_dir = Path(results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    num_epochs = int(num_epochs or config.EPOCHS)
    batch_size = int(best_config["batch_size"])
    learning_rate = float(best_config["learning_rate"])

    log_dir = results_dir / "lightning_logs" / "pinn_gow17"
    if log_dir.exists():
        shutil.rmtree(log_dir)

    data = GOW17DataModule(
        data_dir=str(split_dir),
        batch_size=batch_size,
        num_workers=int(num_workers),
        max_horizon=int(max_horizon),
    )
    data.setup("fit")
    model_config = {
        "num_inputs": data.num_features,
        "output_size": data.num_targets,
        "num_hidden_layers": int(best_config["num_hidden_layers"]),
        "num_neurons_per_hidden_layer": int(
            best_config["num_neurons_per_hidden_layer"]
        ),
        "learning_rate": learning_rate,
        "seed": int(seed),
        "physics_weight": float(physics_weight),
        "conservation_weight": float(conservation_weight),
        "dt_ref_years": config.DT_REF_YEARS,
        "zeta_unit": config.ZETA_UNIT,
        "residual_rate_floor": config.RESIDUAL_RATE_FLOOR,
        "random_collocation": bool(config.RANDOM_COLLOCATION),
        "max_horizon": int(max_horizon),
        **data.phys_norm_config(),
    }
    model = PINN(model_config)
    num_parameters = sum(param.numel() for param in model.parameters())

    print(
        "Starting PINN training: "
        f"epochs={num_epochs}, "
        f"batch_size={batch_size}, "
        f"learning_rate={learning_rate:.6g}, "
        f"physics_weight={physics_weight}, "
        f"conservation_weight={conservation_weight}, "
        f"max_horizon={max_horizon}, "
        f"train_samples={len(data.train_ds):,}, "
        f"val_samples={len(data.val_ds):,}, "
        f"parameters={num_parameters:,}, "
        f"results_dir={results_dir}, "
        f"seed={seed}",
        flush=True,
    )

    metrics = MetricsHistoryLogger()
    callbacks = [
        EpochProgressPrinter(
            prefix="[PINN training]",
            metric_names=(
                "train_loss", "val_loss",
                "train_physics_loss", "val_physics_loss",
                "val_conservation_loss",
            ),
        ),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=str(results_dir / "checkpoints"),
            filename="pinn_best-{epoch:04d}-{val_loss:.5f}",
            save_top_k=1,
            mode="min",
        ),
        metrics,
    ]
    if early_stopping:
        callbacks.insert(
            0,
            RelativeImprovementEarlyStopping(
                monitor="val_loss",
                min_relative_improvement=early_stopping_min_relative_improvement,
                patience=early_stopping_patience,
                mode="min",
            ),
        )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        devices=parse_devices(devices),
        precision=precision,
        callbacks=callbacks,
        logger=(
            TensorBoardLogger(
                save_dir=str(results_dir / "lightning_logs"),
                name="pinn_gow17",
            )
            if enable_logger
            else False
        ),
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        # The physics loss differentiates through the autograd time
        # derivative; deterministic mode would error on the double backward
        # of some ops, so it stays off here (seeding still fixes init/order).
        deterministic=False,
        inference_mode=False,  # validation needs grad for the ODE residual
    )
    trainer.fit(model, datamodule=data)

    final_checkpoint = results_dir / checkpoint_name
    checkpoint_callback = next(
        callback for callback in callbacks if isinstance(callback, ModelCheckpoint)
    )
    best_checkpoint = checkpoint_callback.best_model_path
    if best_checkpoint:
        shutil.copy2(best_checkpoint, final_checkpoint)
    else:
        trainer.save_checkpoint(final_checkpoint)

    (results_dir / "trained_model_config.json").write_text(
        json.dumps(model_config, indent=2)
    )
    plot_history(metrics.history, results_dir / "loss_curves.png")
    print(
        f"PINN training complete: checkpoint={final_checkpoint}, "
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


def main():
    parser = argparse.ArgumentParser(
        description="Train the GOW17 PINN on a split dataset directory."
    )
    parser.add_argument("dataset_path", nargs="?", default=None)
    parser.add_argument("--data-dir", type=Path, default=None, help="Alias for dataset_path.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="best_params.json from optimize.py; overrides the architecture/"
        "training flags (and the loss-weight flags when present in the file).",
    )
    parser.add_argument("--checkpoint", type=str, default=config.CHECKPOINT_NAME)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Defaults to results/{dataset_name}/{sampler}/pinn derived from the split dir.",
    )
    parser.add_argument("--num-layers", type=int, default=config.NUM_LAYERS)
    parser.add_argument("--hidden-units", type=int, default=config.HIDDEN_UNITS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--physics-weight", type=float, default=config.PHYSICS_WEIGHT)
    parser.add_argument(
        "--conservation-weight", type=float, default=config.CONSERVATION_WEIGHT
    )
    parser.add_argument("--max-horizon", type=int, default=config.MAX_HORIZON)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--accelerator", type=str, default=config.ACCELERATOR)
    parser.add_argument("--devices", default=config.NUM_DEVICES)
    parser.add_argument("--precision", default=config.PRECISION)
    parser.add_argument(
        "--early-stopping-patience", type=int, default=config.EARLY_STOPPING_PATIENCE
    )
    parser.add_argument(
        "--early-stopping-min-relative-improvement",
        type=float,
        default=config.EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-logger", action="store_true")
    parser.add_argument("--no-early-stopping", action="store_true")
    args = parser.parse_args()

    if args.config_file is not None:
        best_config = load_best_config(args.config_file)
    else:
        best_config = {
            "num_hidden_layers": args.num_layers,
            "num_neurons_per_hidden_layer": args.hidden_units,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
        }
    return train_final_model(
        best_config,
        args.data_dir or args.dataset_path,
        args.results_dir,
        num_epochs=args.epochs,
        checkpoint_name=args.checkpoint,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        num_workers=args.num_workers,
        physics_weight=best_config.get("physics_weight", args.physics_weight),
        conservation_weight=best_config.get(
            "conservation_weight", args.conservation_weight
        ),
        max_horizon=args.max_horizon,
        early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_relative_improvement=(
            args.early_stopping_min_relative_improvement
        ),
        enable_logger=not args.no_logger,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
