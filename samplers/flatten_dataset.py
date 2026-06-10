"""Flatten a raw HDF5 chemistry dataset into a vector bundle.

Reads the postprocessed gravitational-collapse chemistry HDF5 file, flattens
it with ``util.dataset_flatten()``, and writes the result via
``util.save_bundle()`` (npz format, saved with an .h5 extension by convention).

Inputs
------
--input-h5 : path to the raw HDF5 file (pandas format).
--output-path : destination for the flattened bundle.

Outputs
-------
A single ``.h5`` file (actually npz) containing the flattened bundle arrays.
"""

import argparse

import pandas as pd

from path_utils import default_data_dir, default_raw_h5, resolve_path
import util


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flatten a raw HDF5 chemistry dataset into a vector bundle.",
    )
    parser.add_argument(
        "--input-h5",
        type=str,
        default=None,
        help="Path to raw HDF5 file. Default: env SAMPLERS_RAW_H5 or "
        "datasets/grav_collapse/baseline/grav_collapse_postprocessed_chemistry_uclchem.h5",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for the flattened bundle. "
        "Default: SAMPLERS_DATA_DIR/flattened_dataset.h5",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_h5 = resolve_path(args.input_h5) if args.input_h5 else default_raw_h5()
    output_path = (
        resolve_path(args.output_path)
        if args.output_path
        else default_data_dir() / "flattened_dataset.h5"
    )

    print(f"Loading raw HDF5: {input_h5}")
    dataset = pd.read_hdf(input_h5)
    print(f"  Loaded DataFrame: {dataset.shape[0]} rows × {dataset.shape[1]} columns")

    print("Flattening dataset …")
    bundle = util.dataset_flatten(dataset)
    print(
        f"  Vectors shape: {bundle['vectors'].shape}  "
        f"({bundle['vectors'].shape[0]} tracers × {int(bundle['T'])} timesteps)"
    )

    print(f"Saving bundle: {output_path}")
    util.save_bundle(bundle, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
