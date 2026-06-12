"""Sanity-check the torch GOW17 RHS against finite differences from the data.

For random rows of a split, compares (y_{t+1} - y_t) / dt with the RHS
evaluated at the midpoint state, per species, as a relative error. Also
sweeps candidate zeta unit conversions (the dataset's zeta column is a bare
number) to see which one the data was generated with.

With ``--equilibrium`` it instead runs the production/destruction balance:
for each species it reports P/D at the dataset states (P, D = gross gain,
loss rates), which is ~1 where the dataset sits at a fixed point of our RHS
and far from 1 where our rate coefficients disagree with the generator. For
the worst species it attributes P and D to the individual reactions, which
localizes the wrong rate (this is how the OHx overproduction was traced to
R21; see gow17_rates.GOW17RHS ohx_formation_yield).

Usage:
    python validate_rhs.py [split_dir] [--split train] [--n 4096]
    python validate_rhs.py [split_dir] --equilibrium [--top 6]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gow17_network import load_reactions  # noqa: E402
from gow17_rates import GOW17RHS, SPECIES  # noqa: E402
import settings as config  # noqa: E402
from data import load_split_dataframe  # noqa: E402

SECONDS_PER_YEAR = 3.15576e7


def _reaction_labels():
    return [
        f"{'+'.join(r['reactants'])} -> {'+'.join(r['products'])}"
        for r in load_reactions()
    ]


def equilibrium_residual(df, n=4096, seed=0, top=6):
    """Per-species P/D imbalance at the dataset states, with attribution.

    P/D far from 1 for a quasi-equilibrium species means the RHS's fixed
    point disagrees with the generator: the dataset value is not where our
    rates would hold it. The per-reaction breakdown points at the culprit.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    y = torch.tensor(df.loc[idx, list(SPECIES)].to_numpy(), dtype=torch.float64)
    env = (
        torch.tensor(df.loc[idx, "gasTemp"].to_numpy(), dtype=torch.float64),
        torch.tensor(df.loc[idx, "Density"].to_numpy(), dtype=torch.float64),
        torch.tensor(df.loc[idx, "radfield"].to_numpy(), dtype=torch.float64),
        torch.tensor(df.loc[idx, "Av"].to_numpy(), dtype=torch.float64),
        torch.tensor(df.loc[idx, "zeta"].to_numpy(), dtype=torch.float64) * 1.657e-17,
    )
    rhs = GOW17RHS()
    R = rhs.rates(y, *env).detach().numpy()  # (N, 50)
    nu_pos = rhs.nu_pos.numpy()  # (18, 50)
    nu_neg = rhs.nu_neg.numpy()
    labels = _reaction_labels()

    P = R @ nu_pos.T  # (N, 18) gross production
    D = R @ nu_neg.T  # (N, 18) gross destruction
    ratio = P / np.maximum(D, 1e-300)

    rows = []
    for i, sp in enumerate(SPECIES):
        active = (P[:, i] > 0) | (D[:, i] > 0)
        med = float(np.median(ratio[active, i])) if active.any() else np.nan
        rows.append((abs(np.log10(med + 1e-300)), med, i, sp))

    print(f"{'species':6s} {'median P/D':>12s}   (~1 = dataset sits at the RHS fixed point)")
    for _, med, i, sp in sorted(rows, reverse=True):
        print(f"  {sp:6s} {med:12.3e}")

    print(f"\nper-reaction attribution for the {top} most imbalanced species:")
    for _, med, i, sp in sorted(rows, reverse=True)[:top]:
        pf = np.median(R * nu_pos[i][None, :], axis=0)  # (50,) production parts
        dfr = np.median(R * nu_neg[i][None, :], axis=0)
        print(f"\n  {sp}  (median P/D = {med:.2e})")
        for tag, contrib in (("form", pf), ("dest", dfr)):
            tot = contrib.sum()
            if tot <= 0:
                continue
            for j in np.argsort(-contrib)[:3]:
                if contrib[j] <= 0:
                    break
                print(f"    {tag} R{j + 1:<2d} {contrib[j] / tot:5.2f}  {labels[j]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split_dir", nargs="?", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--equilibrium", action="store_true",
        help="Run the per-species P/D balance + per-reaction attribution.",
    )
    parser.add_argument("--top", type=int, default=6)
    args = parser.parse_args()

    split_dir = config.resolve_split_dir(args.split_dir, required=(args.split,))
    df = load_split_dataframe(split_dir, args.split)
    df = df.sort_values(["Tracer", "Time"]).reset_index(drop=True)

    if args.equilibrium:
        equilibrium_residual(df, n=args.n, seed=args.seed, top=args.top)
        return

    rng = np.random.default_rng(args.seed)
    same_tracer = df["Tracer"].to_numpy()[:-1] == df["Tracer"].to_numpy()[1:]
    starts = np.flatnonzero(same_tracer)
    starts = rng.choice(starts, size=min(args.n, len(starts)), replace=False)

    y0 = torch.tensor(df.loc[starts, list(SPECIES)].to_numpy(), dtype=torch.float64)
    y1 = torch.tensor(df.loc[starts + 1, list(SPECIES)].to_numpy(), dtype=torch.float64)
    dt_s = torch.tensor(
        (df.loc[starts + 1, "Time"].to_numpy() - df.loc[starts, "Time"].to_numpy())
        * SECONDS_PER_YEAR,
        dtype=torch.float64,
    )
    env = {
        k: torch.tensor(df.loc[starts, c].to_numpy(), dtype=torch.float64)
        for k, c in [("nH", "Density"), ("T", "gasTemp"), ("Av", "Av"),
                     ("chi", "radfield"), ("zeta_raw", "zeta")]
    }

    fd = (y1 - y0) / dt_s[:, None]
    ymid = 0.5 * (y0 + y1)
    rhs = GOW17RHS()

    zeta_units = {
        "zeta_col * 1.3e-17 (UCLCHEM unit)": env["zeta_raw"] * 1.3e-17,
        "zeta_col * 1e-16": env["zeta_raw"] * 1e-16,
        "2e-16 (GOW17 default)": torch.full_like(env["zeta_raw"], 2e-16),
        "1.3e-17 (ignore column)": torch.full_like(env["zeta_raw"], 1.3e-17),
    }

    for label, zeta in zeta_units.items():
        pred = rhs(ymid, env["T"], env["nH"], env["chi"], env["Av"], zeta)
        rel = (pred - fd).abs() / (pred.abs() + fd.abs() + 1e-30)
        print(f"\n=== zeta = {label} ===")
        print(f"  median relative residual over all species: {rel.median():.4f}")
        for i, sp in enumerate(SPECIES):
            active = fd[:, i].abs() > 0
            med = rel[active, i].median() if active.any() else float("nan")
            print(f"    {sp:6s} median_rel={med:.4f}  (active rows: {int(active.sum())})")


if __name__ == "__main__":
    main()
