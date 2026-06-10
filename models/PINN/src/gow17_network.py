"""GOW17 (Gong, Ostriker & Wolfire 2017) chemical network.

Parses the KIDA-format network files into a stoichiometric representation
usable for a PINN physics-residual loss:

    dn_i/dt = sum_j nu_ij * R_j,   R_j = k_j * prod_{r in reactants_j} n_r

where nu_ij = (# times species i appears as product) - (# as reactant) in
reaction j. CR / CRP / Photon are pseudo-species: they set the rate-law type
but do not enter the number-density product.

Rate formulas (frml column):
  1 cosmic-ray:        k = alpha * zeta_CR            (CRP: alpha*zeta per CR photon)
  2 photo (Draine):    k = alpha * chi * exp(-gamma * A_V)
  3 Kooij:             k = alpha * (T/300)^beta * exp(-gamma / T)
  4 ionpol1:           k = alpha * beta * (0.62 + 0.4767*gamma*sqrt(300/T))
  7 customized:        special-cased in GOW17 (see paper Appendix A / kida_gow17.cpp)
"""

from pathlib import Path

NETWORK_DIR = Path(__file__).resolve().parent.parent / "network"

PSEUDO = {"CR", "CRP", "Photon"}


def load_species(path=None):
    path = path or NETWORK_DIR / "species.dat"
    species = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("!"):
            continue
        species.append(line.split()[0])
    return species


def load_reactions(path=None):
    """Return list of dicts with reactants, products, alpha/beta/gamma, itype, frml, ID."""
    path = path or NETWORK_DIR / "reactions.dat"
    reactions = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("!"):
            continue
        # fixed-width: 3 reactant fields of 11 chars, 1 spacer, 5 product
        # fields of 11 chars, 2 spacer, then whitespace-separated numbers
        reactants = [line[i * 11:(i + 1) * 11].strip() for i in range(3)]
        prod_block = line[34:34 + 55]
        products = [prod_block[i * 11:(i + 1) * 11].strip() for i in range(5)]
        rest = line[91:].split()
        rid = int(rest[10])
        # some reactions are listed once per temperature range with the same
        # ID and identical coefficients (e.g. R30) — keep the first
        if any(r["id"] == rid for r in reactions):
            continue
        reactions.append({
            "reactants": [r for r in reactants if r],
            "products": [p for p in products if p],
            "alpha": float(rest[0]),
            "beta": float(rest[1]),
            "gamma": float(rest[2]),
            "itype": int(rest[6]),
            "frml": int(rest[9]),
            "id": rid,
        })
    return reactions


def stoichiometry(species, reactions):
    """nu[i][j] for species i, reaction j (pseudo-species excluded)."""
    idx = {s: i for i, s in enumerate(species)}
    nu = [[0] * len(reactions) for _ in species]
    for j, rxn in enumerate(reactions):
        for r in rxn["reactants"]:
            if r not in PSEUDO:
                nu[idx[r]][j] -= 1
        for p in rxn["products"]:
            if p not in PSEUDO:
                nu[idx[p]][j] += 1
    return nu


def rate_law(rxn):
    """Human-readable rate coefficient for reaction dict."""
    a, b, g = rxn["alpha"], rxn["beta"], rxn["gamma"]
    f = rxn["frml"]
    if f == 1:
        return f"{a:.3g} * zeta_CR"
    if f == 2:
        return f"{a:.3g} * chi * exp(-{g:.3g} * A_V)"
    if f == 3:
        s = f"{a:.4g}"
        if b:
            s += f" * (T/300)^{b:.3g}"
        if g:
            s += f" * exp({-g:.4g}/T)"
        return s
    if f == 4:
        return f"{a:.3g} * {b:.4g} * (0.62 + 0.4767*{g:.3g}*sqrt(300/T))"
    return "custom (GOW17 App. A / kida_gow17.cpp)"


def reaction_str(rxn):
    lhs = " + ".join(rxn["reactants"])
    rhs = " + ".join(rxn["products"])
    return f"{lhs} -> {rhs}"


def ode_strings(species, reactions):
    """dn_i/dt as signed sums of R_j terms, with R_j = k_j * product of densities."""
    nu = stoichiometry(species, reactions)
    lines = {}
    for i, s in enumerate(species):
        terms = []
        for j, c in enumerate(nu[i]):
            if c == 0:
                continue
            coef = "" if abs(c) == 1 else f"{abs(c)}*"
            terms.append(("+" if c > 0 else "-") + f" {coef}R{reactions[j]['id']}")
        lines[s] = " ".join(terms).lstrip("+ ") if terms else "0"
    return lines


if __name__ == "__main__":
    species = load_species()
    reactions = load_reactions()
    print(f"{len(species)} species, {len(reactions)} reactions\n")
    print("Reactions (R_id : reaction : rate coefficient k):")
    for r in reactions:
        dens = " * ".join(f"n({x})" for x in r["reactants"] if x not in PSEUDO)
        print(f"  R{r['id']:>2}: {reaction_str(r):55s} R = [{rate_law(r)}] * {dens}")
    print("\nODEs (dn/dt):")
    for s, rhs in ode_strings(species, reactions).items():
        print(f"  d n({s})/dt = {rhs}")