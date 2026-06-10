# PINN — physics-informed surrogate for the GOW17 network

Physics-informed neural network for the GOW17 (Gong, Ostriker & Wolfire 2017,
ApJ 843, 38) astrochemistry network, mirroring the MLP surrogate in
`models/mlp` but with the chemical ODE system as a physics-residual loss.

## Network

18 species, 50 reactions. Source files copied from
`chemistry-benchmark-surrogates/networks/gow17/` into [network/](network/):

- `species.dat` — species with charge and elemental composition
- `reactions.dat` — KIDA-format reaction list with rate coefficients

Species: e-, H, H2, H+, H2+, H3+, He, He+, C, C+, O, O+, CHx, OHx, CO, HCO+, Si, Si+
(CHx and OHx are lumped hydride pseudo-species.)

## ODE system

dn_i/dt = Σ_j ν_ij R_j with R_j = k_j Π_r n_r over reaction j's reactants.
Generate the full reaction table, rate laws, and per-species ODEs with:

```bash
python src/gow17_network.py        # prints; GOW17_ODES.txt holds last output
```

`src/gow17_network.py` also exposes `load_species()`, `load_reactions()`,
`stoichiometry()` for building the PINN residual programmatically.

Rate-law families (frml column of reactions.dat):

| frml | form |
|------|------|
| 1 | k = α ζ_CR (cosmic ray, per H or per CR-induced photon) |
| 2 | k = α χ exp(−γ A_V) (FUV photo, Draine field) |
| 3 | k = α (T/300)^β exp(−γ/T) (Kooij) |
| 4 | k = α β (0.62 + 0.4767 γ √(300/T)) (ionpol1) |
| 7 | custom — temperature/grain-dependent fits in GOW17 Appendix A, implemented in `chemistry-benchmark-surrogates/networks/gow17/kida_gow17.cpp` |

Custom (frml 7) reactions include grain-assisted H2 formation and
recombinations (R15–R19), H3+/O+/C+ abstraction branchings (R20–R24,
R28–R29), collisional dissociation/ionization of H/H2 (R41–R43), and the
O/H charge exchange pair (R49–R50). Their rate expressions must be taken
from the GOW17 paper or the C++ source when building the residual.

External parameters of the rate coefficients: gas temperature T, density n,
cosmic-ray rate ζ_CR, FUV field strength χ, visual extinction A_V, dust/grain
parameters (for frml 7/9 reactions).

## Conservation laws (free PINN constraints)

The network conserves charge and the elemental totals of H, He, C, O, Si —
five linear invariants usable as additional loss terms or hard constraints:

- charge: n(e-) = n(H+) + n(H2+) + n(H3+) + n(He+) + n(C+) + n(O+) + n(HCO+) + n(Si+)
- H: n(H) + 2n(H2) + n(H+) + 2n(H2+) + 3n(H3+) + n(CHx) + n(OHx) + n(HCO+) = const
- He: n(He) + n(He+) = const
- C: n(C) + n(C+) + n(CHx) + n(CO) + n(HCO+) = const
- O: n(O) + n(O+) + n(OHx) + n(CO) + n(HCO+) = const
- Si: n(Si) + n(Si+) = const