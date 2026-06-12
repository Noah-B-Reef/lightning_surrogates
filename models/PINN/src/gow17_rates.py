"""Differentiable (torch) right-hand side of the GOW17 chemical network.

dy_i/dt = sum_j nu_ij R_j  with y_i = n_i / n_H (fractional abundances) and
t in seconds. Reaction rates follow Gong, Ostriker & Wolfire (2017); the
"customized" (frml 7) rates are ported from the Athena++ implementation in
chemistry-benchmark-surrogates/networks/gow17/kida_gow17.cpp. The
stoichiometric matrix is built from network/reactions.dat by gow17_network.py.

Conventions (matching the C++ source):
  - cosmic-ray / photo reactions:  R = k * y_A                 [k in s^-1]
  - two-body reactions:            R = k * n_H * y_A * y_B     [k in cm^3 s^-1]
  - grain-assisted reactions:      R = k(psi, T) * n_H * y_A   (partner implicit)
  - special reactions (R5, R20-24, R28-29): full expressions from the C++.

Environment inputs per sample: T [K], nH [cm^-3], chi (FUV field, Draine
units), Av [mag], zeta (primary CR ionization rate per H, s^-1).

Caveats for the gow17_R0.05_M6.0 dataset:
  - H2/CO self-shielding of FUV rates is not modeled; with Av >= 2.4
    everywhere all photo rates are ~0, so this does not matter here.
  - The grain photoelectric field GPE0 is approximated by the (already tiny)
    attenuated chi; psi -> 0 and grain recombination sits at its psi->0 limit.
  - The physical zeta was calibrated against the dataset itself from the
    H2+ quasi-steady-state balance (see validate_rhs.py): zeta = 2.024e-17
    s^-1, i.e. zeta_column (1.2213740458) times 1.657e-17.
"""

import os

import torch

from gow17_network import load_reactions, load_species, stoichiometry

# Species order used by this module (= dataset column order). The data
# pipeline passes abundances re-indexed to this order (see data.py).
SPECIES = (
    "H", "H2", "He", "C", "O", "CHx", "OHx", "CO", "Si",
    "H+", "H2+", "H3+", "He+", "C+", "O+", "HCO+", "Si+", "e-",
)

NUM_REACTIONS = 50

# Conserved linear quantities: each row dotted with y is constant in time
# (charge neutrality and elemental H/He/C/O/Si totals).
CONSERVED = {
    "charge": {"H+": 1, "H2+": 1, "H3+": 1, "He+": 1, "C+": 1, "O+": 1,
               "HCO+": 1, "Si+": 1, "e-": -1},
    "H": {"H": 1, "H2": 2, "H+": 1, "H2+": 2, "H3+": 3, "CHx": 1, "OHx": 1,
          "HCO+": 1},
    "He": {"He": 1, "He+": 1},
    "C": {"C": 1, "C+": 1, "CHx": 1, "CO": 1, "HCO+": 1},
    "O": {"O": 1, "O+": 1, "OHx": 1, "CO": 1, "HCO+": 1},
    "Si": {"Si": 1, "Si+": 1},
}


def conservation_matrix(dtype=torch.float32):
    """(6, 18) matrix Q with Q @ y constant along trajectories."""
    Q = torch.zeros(len(CONSERVED), len(SPECIES), dtype=dtype)
    idx = {s: i for i, s in enumerate(SPECIES)}
    for row, coeffs in enumerate(CONSERVED.values()):
        for s, c in coeffs.items():
            Q[row, idx[s]] = float(c)
    return Q


def _stoichiometry_matrix():
    """(18, 50) signed stoichiometric matrix in SPECIES x reaction-ID order."""
    network_species = load_species()
    reactions = load_reactions()
    if len(reactions) != NUM_REACTIONS:
        raise ValueError(f"Expected {NUM_REACTIONS} reactions, got {len(reactions)}")
    nu_raw = stoichiometry(network_species, reactions)
    sp_index = {s: i for i, s in enumerate(network_species)}
    rxn_order = {rxn["id"]: j for j, rxn in enumerate(reactions)}
    nu = torch.zeros(len(SPECIES), NUM_REACTIONS, dtype=torch.float64)
    for i, sp in enumerate(SPECIES):
        row = nu_raw[sp_index[sp]]
        for rid in range(1, NUM_REACTIONS + 1):
            nu[i, rid - 1] = float(row[rxn_order[rid]])
    return nu


# Weingartner & Draine (2001) grain recombination fit coefficients.
_GRAIN_REC = {
    "H+":  (12.25, 8.074e-6, 1.378, 5.087e2, 1.586e-2, 0.4723, 1.102e-5),
    "C+":  (45.58, 6.089e-3, 1.128, 4.331e2, 4.845e-2, 0.8120, 1.333e-4),
    "He+": (5.572, 3.185e-7, 1.512, 5.115e3, 3.903e-7, 0.4956, 5.494e-7),
    "Si+": (2.166, 5.678e-8, 1.874, 4.375e4, 1.635e-6, 0.8964, 7.538e-5),
}

_TINY = 1e-50


def _grain_rec(c, psi, T):
    c0, c1, c2, c3, c4, c5, c6 = c
    psi = torch.clamp(psi, min=1e-10)
    return 1.0e-14 * c0 / (
        1.0 + c1 * psi**c2 * (1.0 + c3 * T**c4 * psi**(-c5 - c6 * torch.log(T)))
    )


def _cii_rec_rate(T):
    """C+ + e- radiative + dielectronic recombination (Badnell 2003, 2006)."""
    A, B, T0, T1, C, T2 = 2.995e-9, 0.7849, 6.670e-3, 1.943e6, 0.1597, 4.955e4
    BN = B + C * torch.exp(-T2 / T)
    t1 = torch.sqrt(T / T0)
    t2 = torch.sqrt(T / T1)
    alpharr = A / (t1 * (1.0 + t1) ** (1.0 - BN) * (1.0 + t2) ** (1.0 + BN))
    alphadr = T ** (-1.5) * (
        6.346e-9 * torch.exp(-1.217e1 / T)
        + 9.793e-9 * torch.exp(-7.38e1 / T)
        + 1.634e-6 * torch.exp(-1.523e4 / T)
    )
    return alpharr + alphadr


class GOW17RHS(torch.nn.Module):
    """GOW17 chemical RHS for fractional abundances y ordered as SPECIES.

    ``rates`` returns the per-reaction rates R (events per H per second);
    ``forward`` assembles dy/dt = R @ nu^T; ``production_destruction``
    splits dy/dt into the gross gain/loss terms (both >= 0), useful for
    normalizing residuals of stiff, quasi-equilibrium species.
    """

    def __init__(self, Z_d=1.0, temp_min_rates=1.0, ohx_formation_yield=None):
        super().__init__()
        self.Z_d = float(Z_d)
        self.temp_min_rates = float(temp_min_rates)
        # Effective OHx yield of the O + H3+ / H2 + O+ formation channels
        # (R21, R23), which run through the H2O+ -> H3O+ intermediate. The
        # reduced network assumes essentially all of that flux ends as the
        # lumped OHx (OH/H2O), but the dataset's full network recycles most
        # of it back to O (H3O+ dissociative recombination branches to O+H2
        # rather than OH/H2O). With yield = 1 the RHS overpredicts the OHx
        # quasi-steady-state by ~1e5 at the dataset states (see
        # validate_rhs.py --equilibrium); the default below is calibrated so
        # the RHS OHx equilibrium matches the gow17_R0.05_M6.0 dataset
        # (OHx reconstruction error 5e4 -> 0.4). It is an empirical match to
        # the data, not a first-principles rate; set OHX_FORMATION_YIELD=1
        # (or pass 1.0) to recover the original reduced-network rate.
        if ohx_formation_yield is None:
            ohx_formation_yield = float(
                os.environ.get("OHX_FORMATION_YIELD", 1.0e-5)
            )
        self.ohx_formation_yield = float(ohx_formation_yield)
        self._idx = {s: i for i, s in enumerate(SPECIES)}
        nu = _stoichiometry_matrix()
        self.register_buffer("nu", nu)
        self.register_buffer("nu_pos", torch.clamp(nu, min=0.0))
        self.register_buffer("nu_neg", torch.clamp(-nu, min=0.0))

    def rates(self, y, T, nH, chi, Av, zeta):
        """(..., 50) tensor of reaction rates, ordered by reaction ID."""
        s = self._idx
        y = torch.clamp(y, min=0.0)
        T = torch.clamp(T, min=self.temp_min_rates)
        logT = torch.log10(T)
        t300 = T / 300.0
        Zd = self.Z_d

        yH, yH2, yHe = y[..., s["H"]], y[..., s["H2"]], y[..., s["He"]]
        yC, yO, yCHx = y[..., s["C"]], y[..., s["O"]], y[..., s["CHx"]]
        yOHx, yCO, ySi = y[..., s["OHx"]], y[..., s["CO"]], y[..., s["Si"]]
        yHp, yH2p, yH3p = y[..., s["H+"]], y[..., s["H2+"]], y[..., s["H3+"]]
        yHep, yCp, yOp = y[..., s["He+"]], y[..., s["C+"]], y[..., s["O+"]]
        yHCOp, ySip, ye = y[..., s["HCO+"]], y[..., s["Si+"]], y[..., s["e-"]]

        R = [None] * (NUM_REACTIONS + 1)  # 1-based fill, drop R[0] at the end

        # --- cosmic-ray ionization (R1, R2 use the total-to-primary ratio) ---
        crfac = 2.3 * yH2 + 1.5 * yH
        R[1] = crfac * zeta * yH
        R[2] = 2.0 * crfac * zeta * yH2
        R[3] = 1.1 * zeta * yHe
        R[4] = 3.85 * zeta * yC
        # R5: CO + CR -> CO+ + e-, then CO+ + H/H2 -> HCO+ (rate ~ zeta * y_CO;
        # the H consumed does not enter the rate).
        R[5] = 6.52 * zeta * yCO
        # CR-induced FUV photons
        R[6] = 520.0 * zeta * yC
        R[7] = 92.0 * zeta * yCO
        R[8] = 8.4e3 * zeta * ySi

        # --- FUV photo reactions (Draine field, plane-parallel attenuation) ---
        R[9] = 3.5e-10 * chi * torch.exp(-3.76 * Av) * yC
        R[10] = 9.1e-10 * chi * torch.exp(-2.12 * Av) * yCHx
        R[11] = 2.4e-10 * chi * torch.exp(-3.88 * Av) * yCO
        R[12] = 3.8e-10 * chi * torch.exp(-2.66 * Av) * yOHx
        R[13] = 4.5e-9 * chi * torch.exp(-2.61 * Av) * ySi
        R[14] = 5.7e-11 * chi * torch.exp(-4.18 * Av) * yH2

        # --- grain-assisted reactions ---
        # H2 formation on dust (Jura 1975): rate per H, consumes 2 H per event.
        R[15] = 3.0e-17 * nH * Zd * yH
        psi = 1.7 * chi * torch.sqrt(T) / torch.clamp(nH * ye, min=_TINY)
        R[16] = _grain_rec(_GRAIN_REC["H+"], psi, T) * nH * Zd * yHp
        R[17] = _grain_rec(_GRAIN_REC["C+"], psi, T) * nH * Zd * yCp
        R[18] = _grain_rec(_GRAIN_REC["He+"], psi, T) * nH * Zd * yHep
        R[19] = _grain_rec(_GRAIN_REC["Si+"], psi, T) * nH * Zd * ySip

        # --- special reactions (ported from kida_gow17.cpp) ---
        # R20: C + H3+ -> CHx + H2 (Vissapragada et al. 2016)
        t1_CHx = 1.04e-9 * (300.0 / T) ** 2.31e-3
        t2_CHx = (
            3.4e-8 * torch.exp(-7.62 / T)
            + 6.97e-9 * torch.exp(-1.38 / T)
            + 1.31e-7 * torch.exp(-26.6 / T)
            + 1.51e-4 * torch.exp(-8.11e3 / T)
        )
        R[20] = (t1_CHx + T ** (-1.5) * t2_CHx) * nH * yC * yH3p
        # H2O+ branching between reaction with H2 vs recombination with e-
        h2o_ratio = 6e-10 * yH2 / torch.clamp(5.3e-6 / torch.sqrt(T) * ye, min=_TINY)
        fac_H2 = h2o_ratio / (h2o_ratio + 1.0)
        fac_e = 1.0 / (h2o_ratio + 1.0)
        # OHx-forming branches (R21, R23) carry the empirical H3O+ -> OHx
        # yield (see __init__); R22/R24 are the e- recombination branches
        # back to O and are unaffected.
        yld = self.ohx_formation_yield
        k_O_H3p = 1.99e-9 * T ** (-0.190)
        R[21] = yld * k_O_H3p * fac_H2 * nH * yO * yH3p
        R[22] = k_O_H3p * fac_e * nH * yO * yH3p
        R[23] = yld * 1.6e-9 * fac_H2 * nH * yH2 * yOp
        R[24] = 1.6e-9 * fac_e * nH * yH2 * yOp

        # --- two-body reactions (Kooij / ionpol1 from reactions.dat) ---
        R[25] = 1.7e-9 * nH * yCO * yH3p
        R[26] = 1.26e-13 * torch.exp(-22.5 / T) * nH * yH2 * yHep
        R[27] = 1.6e-9 * nH * yCO * yHep
        # R28/29: schematic C+ + H2 -> CH2+ branchings (special in cpp)
        k_Cp_H2 = T ** (-1.3) * torch.exp(-23.0 / T) * nH * yH2 * yCp
        R[28] = 2.31e-13 * k_Cp_H2
        R[29] = 0.99e-13 * k_Cp_H2
        ionpol = 0.62 + 0.4767 * 5.5 * torch.sqrt(300.0 / T)
        R[30] = 9.15e-10 * ionpol * nH * yOHx * yCp
        R[31] = 7.0e-11 * nH * yO * yCHx
        R[32] = 1.15e-10 * t300 ** (-0.339) * torch.exp(0.108 / T) * nH * yC * yOHx
        # R33: He+ + e- case-B recombination
        R[33] = (
            1e-11 * T ** (-0.5)
            * (11.19 + (-1.676 + (-0.2852 + 0.04433 * logT) * logT) * logT)
            * nH * yHep * ye
        )
        R[34] = 2.339e-8 * t300 ** (-0.52) * nH * yH3p * ye
        R[35] = 4.36e-8 * t300 ** (-0.52) * nH * yH3p * ye
        R[36] = _cii_rec_rate(T) * nH * yCp * ye
        R[37] = 2.754e-7 * t300 ** (-0.64) * nH * yHCOp * ye
        R[38] = 1.76e-9 * T**0.042 * torch.exp(-T / 46600.0) * nH * yH2 * yH2p
        R[39] = 6.4e-10 * nH * yH * yH2p
        # R40: H+ + e- case-B recombination
        R[40] = (
            2.753e-14 * (315614.0 / T) ** 1.5
            * (1.0 + (115188.0 / T) ** 0.407) ** (-2.242)
            * nH * yHp * ye
        )
        # R41-43: collisional dissociation/ionization (Glover & MacLow 2007),
        # active only above 700 K; computed at a clamped temperature and
        # masked so the cold branch contributes exactly 0 without NaN grads.
        hot = T > 700.0
        Th = torch.clamp(T, min=701.0)
        logT4 = torch.log10(Th / 1.0e4)
        lnTe = torch.log(Th * 8.6173e-5)
        k9l = 6.67e-12 * torch.sqrt(Th) * torch.exp(-(1.0 + 63590.0 / Th))
        k9h = 3.52e-9 * torch.exp(-43900.0 / Th)
        k10l = (
            5.996e-30 * Th**4.1881 / (1.0 + 6.761e-6 * Th) ** 5.6881
            * torch.exp(-54657.4 / Th)
        )
        k10h = 1.3e-9 * torch.exp(-53300.0 / Th)
        ncrH = 10.0 ** (3.0 - 0.416 * logT4 - 0.327 * logT4**2)
        ncrH2 = 10.0 ** (4.845 - 1.3 * logT4 + 1.62 * logT4**2)
        ncr = 1.0 / torch.clamp(yH / ncrH + yH2 / ncrH2, min=_TINY)
        n2ncr = nH / ncr

        def log_interp(kh, kl):
            # clamp at a float32-safe minimum: these coefficients underflow to
            # subnormals near the 700 K cutoff and log10(0) would poison the
            # gradients of the masked branch through torch.where.
            return 10.0 ** (
                torch.log10(torch.clamp(kh, min=1e-37)) * n2ncr / (1.0 + n2ncr)
                + torch.log10(torch.clamp(kl, min=1e-37)) / (1.0 + n2ncr)
            )

        zero = torch.zeros_like(T)
        R[41] = torch.where(hot, log_interp(k9h, k9l) * nH * yH2 * yH, zero)
        R[42] = torch.where(hot, log_interp(k10h, k10l) * nH * yH2 * yH2, zero)
        k43 = torch.exp(
            -3.271396786e1
            + (1.35365560e1 + (-5.73932875 + (1.56315498 + (-2.877056e-1
            + (3.48255977e-2 + (-2.63197617e-3 + (1.11954395e-4
            + (-2.03914985e-6) * lnTe) * lnTe) * lnTe) * lnTe) * lnTe)
            * lnTe) * lnTe) * lnTe
        )
        R[43] = torch.where(hot, k43 * nH * yH * ye, zero)
        R[44] = 7.2e-15 * nH * yH2 * yHep
        R[45] = 1.238e-10 * t300**0.26 * nH * yH * yCHx
        R[46] = 3.5e-11 * nH * yO * yOHx
        R[47] = 4.251e-12 * t300 ** (-0.62) * nH * ySip * ye
        R[48] = 1.35e-9 * ionpol * nH * yOHx * yHep
        # R49/50: O <-> H charge exchange (Stancil+ 1999)
        R[49] = (
            (1.1e-11 * T**0.517 + 4.0e-10 * T**6.69e-3)
            * torch.exp(-227.0 / T) * nH * yO * yHp
        )
        R[50] = (4.99e-11 * T**0.405 + 7.5e-10 * T ** (-0.458)) * nH * yH * yOp

        return torch.stack(R[1:], dim=-1)

    def forward(self, y, T, nH, chi, Av, zeta):
        R = self.rates(y, T, nH, chi, Av, zeta)
        return R @ self.nu.to(R.dtype).T

    def production_destruction(self, y, T, nH, chi, Av, zeta):
        """Gross production and destruction (..., 18), both non-negative."""
        R = self.rates(y, T, nH, chi, Av, zeta)
        prod = R @ self.nu_pos.to(R.dtype).T
        dest = R @ self.nu_neg.to(R.dtype).T
        return prod, dest
