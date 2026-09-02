"""
Equivalence testing for the degradation claim.

The submitted manuscript operationalised "graceful degradation" as a
non-significant Welch contrast between rho = 0.10 and rho = 0.30. That is a
misuse of a significance test: a large p-value is absence of evidence, not
evidence of absence, and with n = 3 the design has almost no power, so a
method can be declared graceful simply by being noisy. The manuscript already
concedes this for FedAvg under Gaussian noise (the dagger footnote), which
shows the criterion does not do the work asked of it.

This script replaces the criterion with two one-sided tests (TOST). A method
degrades gracefully over [rho_1, rho_2] when the accuracy drop is
*statistically contained* inside a pre-declared practical margin epsilon:

    H01: drop >= epsilon        H02: drop <= -epsilon
    reject both  =>  equivalence established at level alpha

Only means, standard deviations and n are required, all of which the
manuscript already reports, so no re-run is needed to produce this table.

Choice of epsilon
-----------------
epsilon = 5.0 percentage points. Justification stated in the manuscript: at
the reported class balance (54.1% benign) and an operating point of roughly
90% accuracy on a link carrying 1e6 flows/day, 5 pp of accuracy is of the
order of 5e4 additional misclassified flows per day, which is the scale at
which an analyst team's triage capacity is affected. It is declared before
the tests are run and applied uniformly to every method.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

EPSILON = 5.0     # practical margin, percentage points
ALPHA = 0.05
N = 3

# (mean, sd) at rho = 0.10 and rho = 0.30. MEASURED - manuscript Tables 4, 5.
DATA = {
    "label flipping": {
        "FedAvg":       ((94.3, 0.7), (73.8, 0.8)),
        "Trimmed Mean": ((92.1, 1.3), (83.0, 0.7)),
        "Krum":         ((89.7, 0.6), (87.3, 3.3)),
        "FLTrust":      ((77.4, 1.5), (72.1, 1.1)),
        "FedDBC":       ((92.1, 1.3), (72.0, 2.4)),
        "AMFTA":        ((93.1, 1.2), (80.3, 9.6)),
        "AMFTA-ND":     ((92.5, 1.1), (91.7, 1.0)),
    },
    "Gaussian noise": {
        "FedAvg":       ((71.3, 1.8), (42.8, 21.8)),
        "Trimmed Mean": ((94.4, 0.8), (41.5, 19.9)),
        "Krum":         ((90.0, 0.4), (90.3, 0.2)),
        "FLTrust":      ((77.3, 1.6), (70.3, 1.9)),
        "FedDBC":       ((93.9, 1.0), (65.3, 9.2)),
        "AMFTA":        ((90.3, 0.9), (89.3, 0.7)),
        "AMFTA-ND":     ((90.9, 0.9), (90.6, 0.9)),
    },
}


def welch(m1, s1, n1, m2, s2, n2):
    """Welch t statistic, df and two-sided p for m1 - m2."""
    se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    if se == 0:
        return float("inf"), n1 + n2 - 2, 0.0, 0.0
    t = (m1 - m2) / se
    df = se ** 4 / ((s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1))
    p = 2 * stats.t.sf(abs(t), df)
    return t, df, p, se


def tost(m1, s1, n1, m2, s2, n2, eps=EPSILON):
    """Two one-sided tests for equivalence of m1 and m2 within +/- eps.

    Returns the larger of the two one-sided p-values, which is the TOST
    p-value: equivalence is established iff it is below alpha.
    """
    diff = m1 - m2
    _, df, _, se = welch(m1, s1, n1, m2, s2, n2)
    if se == 0:
        return diff, 0.0, df, 0.0, 0.0
    t_lo = (diff + eps) / se        # H01: diff <= -eps
    t_hi = (diff - eps) / se        # H02: diff >= +eps
    p_lo = stats.t.sf(t_lo, df)     # want small: diff is above -eps
    p_hi = stats.t.cdf(t_hi, df)    # want small: diff is below +eps
    return diff, max(p_lo, p_hi), df, p_lo, p_hi


def ci90(m1, s1, n1, m2, s2, n2):
    """90% CI on the drop; equivalent decision rule to TOST at alpha=0.05."""
    diff = m1 - m2
    _, df, _, se = welch(m1, s1, n1, m2, s2, n2)
    h = stats.t.ppf(0.95, df) * se
    return diff - h, diff + h


def main():
    results = {}
    print(f"Equivalence testing, margin epsilon = {EPSILON} pp, alpha = {ALPHA}, n = {N}")
    print("Drop is accuracy at rho=0.10 minus accuracy at rho=0.30 (positive = degradation).\n")
    for attack, table in DATA.items():
        print(f"--- {attack} ---")
        print(f"  {'method':<14s}{'drop':>7s}{'90% CI':>18s}"
              f"{'p_NHST':>9s}{'p_TOST':>9s}  verdict")
        results[attack] = {}
        for m, ((m1, s1), (m2, s2)) in table.items():
            _, _, p_nhst, _ = welch(m1, s1, N, m2, s2, N)
            drop, p_tost, df, _, _ = tost(m1, s1, N, m2, s2, N)
            lo, hi = ci90(m1, s1, N, m2, s2, N)
            equivalent = p_tost < ALPHA
            different = p_nhst < ALPHA
            if equivalent and not different:
                verdict = "graceful (equivalent)"
            elif different and not equivalent:
                verdict = "degrades"
            elif equivalent and different:
                verdict = "degrades but trivially"
            else:
                verdict = "INCONCLUSIVE (underpowered)"
            results[attack][m] = {
                "drop": drop, "ci90": [lo, hi], "p_nhst": p_nhst,
                "p_tost": p_tost, "df": df, "verdict": verdict,
            }
            print(f"  {m:<14s}{drop:>7.1f}  [{lo:6.1f}, {hi:6.1f}]"
                  f"{p_nhst:>9.3f}{p_tost:>9.3f}  {verdict}")
        print()

    print("Summary of what changes relative to the submitted criterion:")
    for attack, table in results.items():
        for m, v in table.items():
            if v["verdict"].startswith("INCONCLUSIVE") and v["p_nhst"] >= ALPHA:
                print(f"  {attack:<16s}{m:<14s} was reported 'graceful' on p>0.05;"
                      f" equivalence is NOT established (p_TOST={v['p_tost']:.3f})")

    (OUT / "equivalence.json").write_text(json.dumps(
        {"epsilon": EPSILON, "alpha": ALPHA, "n": N, "results": results}, indent=2))
    print(f"\nwrote {OUT / 'equivalence.json'}")


if __name__ == "__main__":
    main()
