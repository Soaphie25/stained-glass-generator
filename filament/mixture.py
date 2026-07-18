#!/usr/bin/env python3
"""Sub-layer MIXTURE model: predict the colour of a Bambu-Studio colour-mix of
two transparent filaments, built ON TOP of the full-layer single-filament
calibration.

Full-layer calibration (``analyze_calibration.py``) gives each filament X its
per-channel absorption ``a_X`` (per mm) and surface term ``T0_X``.  A sub-layer
mix of A and B at fraction-B ``p``, total thickness ``T``, lit from behind by
white, transmits per channel:

    base_absorbance_c   = T * [ (1-p)*a_A,c + p*a_B,c ]          # full-layer
    scatter_absorbance_c= T * p*(1-p) * [ (1-p)*sig_A,c + p*sig_B,c ]
    ln T0_c             = (1-p)*ln T0_A,c + p*ln T0_B,c
    tau_c(p,T)          = exp( ln T0_c - base_absorbance_c - scatter_absorbance_c )

``base`` is exact at the pure ends (p=0/1: no A-B interfaces).  ``scatter`` is the
extra attenuation from the alternating-sublayer interfaces; it vanishes at the
ends and is attributed to each filament by the per-filament parameter ``sig_X``,
which the ramp's asymmetry makes identifiable.  Because ``sig_X`` is per-filament,
calibrating pairs that share a filament (A-B, B-C) lets us predict an unseen pair
(A-C) -- solved JOINTLY so a shared filament is consistent across pairs.

Posterior override: if a pair has its own measured ramp, its DIRECT fit is kept
and preferred over the generalized prediction when they disagree.

    python3 filament/mixture.py selftest
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from solve_recipe import Filament, delta_e, linear_to_hex  # noqa: E402


# --------------------------------------------------------------------------- #
# Forward model
# --------------------------------------------------------------------------- #
def mix_tau_multi(fils, sigs, fracs, T):
    """Predicted backlit-white transmittance (linear RGB, 0..1) of an N-filament
    sub-layer mix at fractions ``fracs`` (sum to 1), total thickness T.

    scatter absorbance = T * sum_i sig_i * f_i^2 * (1 - f_i).  This attributes
    the interface scatter per filament, reduces EXACTLY to the 2-filament form
    T*f_A*f_B*(f_A*sig_A + f_B*sig_B) when a fraction is 0, and needs only the
    per-filament sigmas -- so 3-filament mixes use the SAME pair-calibrated sigmas.
    """
    fracs = np.asarray(fracs, float)
    a = np.array([f.a for f in fils])
    lnT0 = np.array([np.log(f.T0) for f in fils])
    sig = np.asarray(sigs, float)
    base = T * (fracs[:, None] * a).sum(0)
    scat = T * (sig * (fracs ** 2 * (1 - fracs))[:, None]).sum(0)
    return np.clip(np.exp((fracs[:, None] * lnT0).sum(0) - base - scat), 0, 1)


def mix_tau(fX, fY, sigX, sigY, p, T):
    """Two-filament mix: fraction p of fY, (1-p) of fX (a special case)."""
    return mix_tau_multi([fX, fY], [sigX, sigY], [1 - p, p], T)


def _baseline_tau(fX, fY, p, T):
    return mix_tau_multi([fX, fY], [np.zeros(3), np.zeros(3)], [1 - p, p], T)


# --------------------------------------------------------------------------- #
# Fit per-filament sigma jointly across pairs
# --------------------------------------------------------------------------- #
def fit_sigma(pairs, fulls, T):
    """pairs: list of {"A":name,"B":name,"ratios":[p..],"tau":Nx3 measured}.
    fulls: {name: Filament}.  Returns (sigma{name:[3]}, per_pair_direct{key:...}).

    Solves, per channel, the joint least-squares system over ALL pairs so a
    shared filament's sigma is consistent; also records each pair's stand-alone
    direct fit for the posterior override.
    """
    names = sorted({p["A"] for p in pairs} | {p["B"] for p in pairs})
    idx = {n: i for i, n in enumerate(names)}
    sigma = np.zeros((len(names), 3))
    for c in range(3):
        rows, rhs = [], []
        for pr in pairs:
            fX, fY = fulls[pr["A"]], fulls[pr["B"]]
            for p, tau in zip(pr["ratios"], np.asarray(pr["tau"], float)):
                if tau[c] <= 1e-6:
                    continue
                base = _baseline_tau(fX, fY, p, T)[c]
                R = np.log(max(base, 1e-9)) - np.log(max(tau[c], 1e-9))
                row = np.zeros(len(names))
                row[idx[pr["A"]]] = T * p * (1 - p) * (1 - p)
                row[idx[pr["B"]]] = T * p * (1 - p) * p
                rows.append(row)
                rhs.append(R)
        sol, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
        sigma[:, c] = sol
    sig = {n: sigma[idx[n]] for n in names}

    # per-pair stand-alone fit (only that pair's data) -> posterior reference
    direct = {}
    for pr in pairs:
        one = fit_sigma_pair(pr, fulls, T)
        direct[_key(pr["A"], pr["B"])] = one
    return sig, direct


def fit_sigma_pair(pr, fulls, T):
    """Stand-alone (sig_A, sig_B) from a single pair's ramp."""
    fX, fY = fulls[pr["A"]], fulls[pr["B"]]
    out = np.zeros((2, 3))
    for c in range(3):
        rows, rhs = [], []
        for p, tau in zip(pr["ratios"], np.asarray(pr["tau"], float)):
            if tau[c] <= 1e-6:
                continue
            base = _baseline_tau(fX, fY, p, T)[c]
            rows.append([T * p * (1 - p) * (1 - p), T * p * (1 - p) * p])
            rhs.append(np.log(max(base, 1e-9)) - np.log(max(tau[c], 1e-9)))
        sol, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
        out[:, c] = sol
    return {"A": pr["A"], "B": pr["B"], "sigA": out[0], "sigB": out[1]}


def _key(a, b):
    return "+".join(sorted((a, b)))


# --------------------------------------------------------------------------- #
# Predict a pair, with posterior override
# --------------------------------------------------------------------------- #
def predict_mix(fX, fY, nameX, nameY, sigma, p, T, direct=None, tol_de=2.0):
    """Predicted transmittance for a mix; uses this pair's DIRECT fit instead of
    the generalized sigma when a measured direct fit exists and disagrees."""
    sigX, sigY = sigma[nameX], sigma[nameY]
    src = "generalized"
    if direct and _key(nameX, nameY) in direct:
        d = direct[_key(nameX, nameY)]
        dX, dY = (d["sigA"], d["sigB"]) if d["A"] == nameX else (d["sigB"], d["sigA"])
        gen = mix_tau(fX, fY, sigX, sigY, p, T)
        dpred = mix_tau(fX, fY, dX, dY, p, T)
        if delta_e(gen, dpred) > tol_de:            # posterior wins if far off
            sigX, sigY, src = dX, dY, "direct(posterior)"
    return mix_tau(fX, fY, sigX, sigY, p, T), src


# --------------------------------------------------------------------------- #
# Self-test: synthetic 3 filaments, calibrate A-B & B-C, predict A-C
# --------------------------------------------------------------------------- #
def _gt():
    fils = {
        "A": Filament("A", a=[0.30, 0.70, 1.60], T0=[0.94, 0.94, 0.92]),
        "B": Filament("B", a=[1.50, 0.35, 0.45], T0=[0.93, 0.94, 0.94]),
        "C": Filament("C", a=[0.40, 1.45, 0.55], T0=[0.94, 0.92, 0.94]),
    }
    sig = {  # ground-truth per-filament mixture scatter (per mm)
        "A": np.array([0.15, 0.35, 0.55]),
        "B": np.array([0.45, 0.18, 0.28]),
        "C": np.array([0.28, 0.50, 0.14]),
    }
    return fils, sig


def _ramp(fX, fY, sgX, sgY, T, steps=10, noise=0.004, seed=0):
    rng = np.random.default_rng(seed)
    ratios = [i / steps for i in range(steps + 1)]
    tau = np.array([mix_tau(fX, fY, sgX, sgY, p, T) for p in ratios])
    tau = np.clip(tau + rng.normal(0, noise, tau.shape), 0, 1)
    return ratios, tau


def run_selftest():
    fils, gt = _gt()
    T = 1.0
    # measure A-B and B-C ramps (NOT A-C)
    rAB, tAB = _ramp(fils["A"], fils["B"], gt["A"], gt["B"], T, seed=1)
    rBC, tBC = _ramp(fils["B"], fils["C"], gt["B"], gt["C"], T, seed=2)
    pairs = [{"A": "A", "B": "B", "ratios": rAB, "tau": tAB},
             {"A": "B", "B": "C", "ratios": rBC, "tau": tBC}]

    sigma, direct = fit_sigma(pairs, fils, T)
    print("self-test: recovered per-filament sigma (internal nuisance params)")
    for n in ("A", "B", "C"):
        print("  %-3s true %-20s  recovered %s" % (
            n, np.round(gt[n], 3).tolist(), np.round(sigma[n], 3).tolist()))

    ok = True

    def check(label, fils_l, names_l, gtsig_l, fracs, base_cmp=None):
        nonlocal ok
        pred = mix_tau_multi(fils_l, [sigma[nm] for nm in names_l], fracs, T)
        truth = mix_tau_multi(fils_l, gtsig_l, fracs, T)
        de = delta_e(pred, truth)
        base = ("" if base_cmp is None else
                "  (baseline dE=%.2f)" % delta_e(base_cmp, truth))
        ok = ok and de < 1.5
        print("  %-26s pred #%-8s truth #%-8s  dE=%.2f%s%s"
              % (label, linear_to_hex(pred), linear_to_hex(truth), de, base,
                 "" if de < 1.5 else "  <-- FAIL"))

    print("\n  color prediction vs ground truth (pass if dE<1.5):")
    for p in (0.3, 0.6):                         # calibrated pairs
        check("A-B  p=%.1f (calibrated)" % p, [fils["A"], fils["B"]], ["A", "B"],
              [gt["A"], gt["B"]], [1 - p, p])
    print("  -- the payoff: the UNSEEN A-C pair, from A-B + B-C sigmas --")
    for p in (0.25, 0.5, 0.75):
        check("A-C  p=%.2f (UNSEEN)" % p, [fils["A"], fils["C"]], ["A", "C"],
              [gt["A"], gt["C"]], [1 - p, p],
              base_cmp=_baseline_tau(fils["A"], fils["C"], p, T))
    print("  -- 3-filament mixes, from the same pair-calibrated sigmas --")
    for fr in ([0.4, 0.35, 0.25], [0.34, 0.33, 0.33]):
        check("A+B+C %s" % fr, [fils["A"], fils["B"], fils["C"]], ["A", "B", "C"],
              [gt["A"], gt["B"], gt["C"]], fr)

    print("SELF-TEST %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest", help="synthetic sigma-recovery + A-C generalization")
    opts = p.parse_args(argv)
    if opts.cmd == "selftest":
        return run_selftest()


if __name__ == "__main__":
    raise SystemExit(main())
