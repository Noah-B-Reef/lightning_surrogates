import argparse
from pathlib import Path

import test as test_module
import train as train_module


def resolve_dataset_dir(dataset_path):
    path = Path(dataset_path).expanduser().resolve()
    if path.is_file():
        if path.name != "train.csv":
            raise ValueError("When passing a file, provide train.csv.")
        path = path.parent
    missing = [name for name in ("train.csv", "val.csv", "test.csv") if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {path}")
    return path


def main(
    dataset_path,
    epochs=None,
    config_file="best_params.txt",
    checkpoint_name="mlp_grav_collapse.ckpt",
    results_dir=None,
    use_defaults=False,
):
    dataset_dir = resolve_dataset_dir(dataset_path)
    run_results_dir = Path(results_dir).expanduser().resolve() if results_dir else dataset_dir / "results"
    checkpoint_path = run_results_dir / checkpoint_name
    train_module.main(
        num_epochs=epochs,
        config_file=config_file,
        checkpoint_path=checkpoint_name,
        use_defaults=use_defaults,
        results_dir=str(run_results_dir),
        data_dir=str(dataset_dir),
    )
    test_module.main(
        model_checkpoint=str(checkpoint_path),
        test_dir=str(dataset_dir / "test.csv"),
        output_dir=str(run_results_dir / "test_results"),
        epoch_checkpoint_dir=str(run_results_dir / "epoch_checkpoints"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--config-file", type=str, default="best_params.txt")
    parser.add_argument("--checkpoint", type=str, default="mlp_grav_collapse.ckpt")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--use-defaults", action="store_true")
    args = parser.parse_args()
    main(args.dataset_path, args.epochs, args.config_file, args.checkpoint, args.results_dir, args.use_defaults)
