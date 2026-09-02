"""
Turn the study CSVs into the manuscript's tables and statistics.
================================================================

Reads whatever ``run_paper_study.py`` has produced under ``--results`` and
emits, for each block, a LaTeX table plus the statistical tests the
manuscript reports.

    python experiments/paper_stats.py --results results_paper --out tables

Reporting convention (matches the manuscript)
---------------------------------------------
Per seed: mean over the final ``--last-rounds`` communication rounds.
Across seeds: mean and standard deviation of those per-seed means.
Every reported +/- is therefore an across-seed standard deviation, and the
round averaging is a smoothing step applied before the seed statistic, not a
source of replication.

Statistics
----------
  * Welch's t for pairwise contrasts, with Holm-Bonferroni across the family.
  * Two one-sided tests (TOST) for the degradation claim, against a margin
    declared on the command line (``--epsilon``, default 5.0 pp).
    A non-significant difference is NOT evidence of stability; only a
    significant TOST establishes containment. Cells where neither test
    resolves are reported as inconclusive rather than graceful.
  * Inverse-variance weighted trend test for the validation-buffer effect
    across attacker fractions.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

PRETTY = {
    "fedavg": "FedAvg", "trimmed_mean": "Trimmed Mean", "krum": "Krum",
    "fltrust": "FLTrust", "feddbc": "FedDBC", "amfta": "AMFTA",
    "amfta_noq": "AMFTA-ND", "normclip": "NormClip-Only",
    "median": "Coordinate-wise Median", "multikrum": "Multi-Krum",
    "foolsgold": "FoolsGold",
}
ORDER = ["fedavg", "trimmed_mean", "krum", "fltrust", "feddbc",
         "normclip", "median", "multikrum", "foolsgold", "amfta", "amfta_noq"]


# ---------------------------------------------------------------------------
# Loading and reduction
# ---------------------------------------------------------------------------

def load(results_dir: Path) -> list:
    rows = []
    for p in sorted(results_dir.glob("*.csv")):
        with p.open(newline="", encoding="utf8") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def per_seed_means(rows, last_rounds: int, metric: str = "accuracy") -> dict:
    """(block, method, attack, rho, alpha) -> {seed: mean of final rounds}."""
    grouped = defaultdict(list)
    for r in rows:
        try:
            v = float(r[metric])
        except (ValueError, KeyError, TypeError):
            continue
        key = (r["block"], r["method"], r["attack"], r["rho"], r["alpha"], r["seed"])
        grouped[key].append((int(r["round"]), v))

    out = defaultdict(dict)
    for (block, method, attack, rho, alpha, seed), vals in grouped.items():
        vals.sort()
        tail = [v for _, v in vals[-last_rounds:]]
        out[(block, method, attack, rho, alpha)][int(seed)] = float(np.mean(tail))
    return out


def cell(ps, key):
    """-> (mean%, sd%, n) with accuracy scaled to percent."""
    d = ps.get(key)
    if not d:
        return None
    v = np.array(sorted(d.items()))[:, 1] * 100.0
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return float(v.mean()), sd, len(v)


def fmt(c, bold=False):
    if c is None:
        return "---"
    s = f"{c[0]:.1f}\\pm{c[1]:.1f}"
    return f"$\\mathbf{{{c[0]:.1f}}}\\pm{c[1]:.1f}$" if bold else f"${s}$"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def welch(m1, s1, n1, m2, s2, n2):
    """Welch's t. Returns NaN when the standard error is zero.

    Zero across-seed spread in both cells means the test is undefined, not
    that the difference is infinitely significant. Callers must treat a NaN
    p-value as 'no verdict', never as a rejection.
    """
    se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    if se == 0 or n1 < 2 or n2 < 2:
        return float("nan"), float("nan"), float("nan"), 0.0
    t = (m1 - m2) / se
    df = se ** 4 / ((s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1))
    return t, df, 2 * stats.t.sf(abs(t), df), se


def tost(m1, s1, n1, m2, s2, n2, eps):
    diff = m1 - m2
    _, df, _, se = welch(m1, s1, n1, m2, s2, n2)
    if se == 0 or not math.isfinite(df):
        return diff, float("nan"), (float("nan"), float("nan"))
    p = max(stats.t.sf((diff + eps) / se, df), stats.t.cdf((diff - eps) / se, df))
    h = stats.t.ppf(0.95, df) * se
    return diff, p, (diff - h, diff + h)


def holm(pvals):
    idx = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------

def table_sweep(ps, block, attack, out, label, caption):
    rhos = ["0.10", "0.20", "0.30", "0.40"]
    methods = [m for m in ORDER
               if any((block, m, attack, r, "0.50") in ps for r in rhos)]
    lines = ["\\begin{table}[H]", f"\t\\caption{{{caption}}}", f"\t\\label{{{label}}}",
             "\t\\centering", "\t\\begin{tabular}{@{}lcccc@{}}", "\t\t\\toprule",
             "\t\t\\textbf{Method} & " + " & ".join(f"$\\rho={r}$" for r in rhos) + "\\\\",
             "\t\t\\midrule"]
    best = {}
    for r in rhos:
        vals = [(m, cell(ps, (block, m, attack, r, "0.50"))) for m in methods]
        vals = [(m, c) for m, c in vals if c]
        best[r] = max(vals, key=lambda t: t[1][0])[0] if vals else None
    for m in methods:
        cells = [fmt(cell(ps, (block, m, attack, r, "0.50")), bold=(best[r] == m))
                 for r in rhos]
        name = f"\\textbf{{{PRETTY[m]}}}" if m == "amfta_noq" else PRETTY[m]
        lines.append(f"\t\t{name} & " + " & ".join(cells) + "\\\\")
    lines += ["\t\t\\bottomrule", "\t\\end{tabular}", "\\end{table}", ""]
    out.write_text("\n".join(lines), encoding="utf8")
    print(f"  wrote {out}")


def degradation(ps, block, attack, eps, out):
    rhos = ("0.10", "0.30")
    lines = [f"% Degradation over rho in [{rhos[0]}, {rhos[1]}], attack={attack}, "
             f"epsilon={eps} pp", "% verdicts: degrades / graceful (equivalence "
             "established) / inconclusive"]
    print(f"\n  degradation, {attack} (epsilon = {eps} pp)")
    print(f"    {'method':<24s}{'drop':>7s}{'90% CI':>18s}{'p_NHST':>9s}{'p_TOST':>9s}  verdict")
    for m in ORDER:
        a = cell(ps, (block, m, attack, rhos[0], "0.50"))
        b = cell(ps, (block, m, attack, rhos[1], "0.50"))
        if not a or not b:
            continue
        _, _, p_n, _ = welch(a[0], a[1], a[2], b[0], b[1], b[2])
        drop, p_t, ci = tost(a[0], a[1], a[2], b[0], b[1], b[2], eps)
        if not (math.isfinite(p_n) and math.isfinite(p_t)):
            # Zero across-seed spread: the tests are undefined. This signals a
            # collapsed or degenerate run, not stability.
            print(f"    {PRETTY[m]:<24s}{drop:>7.1f}  {'   undefined   ':>18s}"
                  f"{'   n/a':>9s}{'   n/a':>9s}  DEGENERATE (zero variance)")
            lines.append(f"% {PRETTY[m]}: drop={drop:.1f} DEGENERATE, zero "
                         f"across-seed variance, tests undefined")
            continue
        eq, diff = p_t < 0.05, p_n < 0.05
        verdict = ("degrades" if diff and not eq else
                   "graceful" if eq and not diff else
                   "degrades but trivially" if eq and diff else
                   "INCONCLUSIVE")
        print(f"    {PRETTY[m]:<24s}{drop:>7.1f}  [{ci[0]:6.1f},{ci[1]:6.1f}]"
              f"{p_n:>9.3f}{p_t:>9.3f}  {verdict}")
        lines.append(f"% {PRETTY[m]}: drop={drop:.1f} CI=[{ci[0]:.1f},{ci[1]:.1f}] "
                     f"p_NHST={p_n:.3f} p_TOST={p_t:.3f} {verdict}")
    out.write_text("\n".join(lines), encoding="utf8")


def buffer_trend(ps, out):
    """AMFTA-ND minus AMFTA across attacker fractions, with a weighted trend test."""
    rhos = ["0.10", "0.20", "0.30", "0.40"]
    xs, ds, ws, rows = [], [], [], []
    for r in rhos:
        a = cell(ps, ("labelflip", "amfta", "label_flipping", r, "0.50"))
        b = cell(ps, ("labelflip", "amfta_noq", "label_flipping", r, "0.50"))
        if not a or not b:
            continue
        _, _, p, se = welch(b[0], b[1], b[2], a[0], a[1], a[2])
        d = b[0] - a[0]
        rows.append((r, a, b, d, p))
        if se > 0:
            xs.append(float(r)); ds.append(d); ws.append(1.0 / se ** 2)
    if len(xs) < 3:
        print("\n  buffer trend: not enough cells yet")
        return
    x, d, w = np.array(xs), np.array(ds), np.array(ws)
    xb = np.sum(w * x) / np.sum(w)
    sxx = np.sum(w * (x - xb) ** 2)
    slope = np.sum(w * (x - xb) * d) / sxx
    se_s = math.sqrt(1.0 / sxx)
    z = slope / se_s
    p = 2 * stats.norm.sf(abs(z))
    rho_s = stats.spearmanr(x, d).statistic
    print("\n  validation-buffer removal (AMFTA-ND minus AMFTA), label flipping")
    for r, a, b, dd, pp in rows:
        print(f"    rho={r}  AMFTA {a[0]:5.1f}+-{a[1]:.1f}   "
              f"AMFTA-ND {b[0]:5.1f}+-{b[1]:.1f}   delta {dd:+5.1f}  p={pp:.3f}")
    print(f"    weighted trend: slope {slope:.1f} +- {se_s:.1f} pp per unit rho, "
          f"z={z:.2f}, p={p:.3f}, Spearman r={rho_s:.2f}")
    out.write_text(
        f"% buffer trend: slope={slope:.2f}+-{se_s:.2f} z={z:.2f} p={p:.4f} "
        f"spearman={rho_s:.2f}\n", encoding="utf8")


def pairwise(ps, eps, out):
    """The contrasts the positioning depends on, Holm-corrected."""
    fam = [
        ("amfta_noq", "amfta", "labelflip", "label_flipping", "0.30"),
        ("amfta_noq", "amfta", "gaussian", "gaussian_noise", "0.30"),
        ("amfta_noq", "krum", "labelflip", "label_flipping", "0.30"),
        ("amfta_noq", "krum", "gaussian", "gaussian_noise", "0.30"),
        ("amfta_noq", "feddbc", "labelflip", "label_flipping", "0.30"),
        ("amfta_noq", "feddbc", "gaussian", "gaussian_noise", "0.30"),
    ]
    res, ps_raw = [], []
    for m1, m2, block, attack, r in fam:
        a = cell(ps, (block, m1, attack, r, "0.50"))
        b = cell(ps, (block, m2, attack, r, "0.50"))
        if not a or not b:
            continue
        t, df, p, _ = welch(a[0], a[1], a[2], b[0], b[1], b[2])
        if not math.isfinite(p):
            print(f"    {PRETTY[m1]} vs {PRETTY[m2]} ({attack}): undefined "
                  f"(zero across-seed variance), skipped")
            continue
        res.append((m1, m2, attack, a[0] - b[0], t, p))
        ps_raw.append(p)
    if not res:
        print("\n  pairwise: no cells yet")
        return
    adj = holm(np.array(ps_raw))
    print("\n  pairwise contrasts at rho=0.30 (Holm-corrected)")
    lines = []
    for (m1, m2, attack, d, t, p), pa in zip(res, adj):
        verdict = "significant" if pa < 0.05 else "n.s."
        print(f"    {PRETTY[m1]} vs {PRETTY[m2]:<22s} {attack:<15s} "
              f"delta={d:+6.1f} t={t:6.2f} p={p:.3f} p_holm={pa:.3f}  {verdict}")
        lines.append(f"% {PRETTY[m1]} vs {PRETTY[m2]} ({attack}): delta={d:+.1f} "
                     f"t={t:.2f} p={p:.4f} p_holm={pa:.4f} {verdict}")
    out.write_text("\n".join(lines), encoding="utf8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_paper")
    ap.add_argument("--out", default="tables")
    ap.add_argument("--last-rounds", type=int, default=5)
    ap.add_argument("--epsilon", type=float, default=5.0,
                    help="equivalence margin in percentage points, declared "
                         "before testing")
    args = ap.parse_args()

    rd, od = Path(args.results), Path(args.out)
    if not rd.exists():
        raise SystemExit(f"no results at {rd}; run experiments/run_paper_study.py first")
    od.mkdir(parents=True, exist_ok=True)

    rows = load(rd)
    if not rows:
        raise SystemExit(f"no CSV rows under {rd}")
    ps = per_seed_means(rows, args.last_rounds)

    seeds = sorted({int(r["seed"]) for r in rows})
    print(f"loaded {len(rows)} rows, {len(ps)} cells, seeds {seeds}\n")
    print("LaTeX tables")

    if any(k[0] == "labelflip" for k in ps):
        table_sweep(ps, "labelflip", "label_flipping", od / "tab_labelflip.tex",
                    "tab:labelflip",
                    "Accuracy (\\%) under label flipping, mean $\\pm$ across-seed "
                    "standard deviation. Best per column in \\textbf{bold}.")
    if any(k[0] == "gaussian" for k in ps):
        table_sweep(ps, "gaussian", "gaussian_noise", od / "tab_gaussian.tex",
                    "tab:gaussian",
                    "Accuracy (\\%) under Gaussian-noise model poisoning, mean "
                    "$\\pm$ across-seed standard deviation.")

    if any(k[0] == "labelflip" for k in ps):
        degradation(ps, "labelflip", "label_flipping", args.epsilon,
                    od / "degradation_labelflip.tex")
    if any(k[0] == "gaussian" for k in ps):
        degradation(ps, "gaussian", "gaussian_noise", args.epsilon,
                    od / "degradation_gaussian.tex")

    buffer_trend(ps, od / "buffer_trend.tex")
    pairwise(ps, args.epsilon, od / "pairwise.tex")
    print(f"\ntables and test output -> {od}")


if __name__ == "__main__":
    main()
