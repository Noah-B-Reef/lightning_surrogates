import argparse
from pathlib import Path

from train_test_dataset import main as train_test_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampled-datasets-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["random", "density", "qr", "svd_fps", "similarity_constrained"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--config-file", type=str, default="best_params.txt")
    parser.add_argument("--checkpoint", type=str, default="mlp_grav_collapse.ckpt")
    parser.add_argument("--use-defaults", action="store_true")
    args = parser.parse_args()

    for method in args.methods:
        dataset_dir = args.sampled_datasets_dir / method
        if not (dataset_dir / "train.csv").is_file():
            print(f"Skipping {method}: missing {dataset_dir / 'train.csv'}")
            continue
        train_test_one(
            dataset_path=str(dataset_dir),
            epochs=args.epochs,
            config_file=args.config_file,
            checkpoint_name=args.checkpoint,
            results_dir=str(dataset_dir / "results"),
            use_defaults=args.use_defaults,
        )


if __name__ == "__main__":
    main()
