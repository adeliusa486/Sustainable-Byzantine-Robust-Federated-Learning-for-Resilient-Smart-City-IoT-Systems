"""
Rebuild every manuscript table that the real run logs can support.

Points at the real project's `results/` directory and emits LaTeX for the
tables whose numbers exist. Tables the logs cannot support are listed at the
end rather than fabricated.

    python rebuild_tables_from_logs.py \
        --results "C:/Users/adeel/.gemini/antigravity/scratch/amfta-fl/amfta-fl/results" \
        --out tables_from_logs

Reporting convention: per seed, the mean over the final five rounds; across
seeds, the mean and SAMPLE standard deviation (ddof=1). The manuscript
currently uses population SD (ddof=0), which understates every interval by
about 18%; this script reports both so the change is visible.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
from pathlib import Path

METHODS = ["fedavg", "trimmed_mean", "krum", "fltrust", "feddbc", "amfta", "amfta_noq"]
PRETTY = {"fedavg": "FedAvg", "trimmed_mean": "Trimmed Mean", "krum": "Krum",
          "fltrust": "FLTrust", "feddbc": "FedDBC", "amfta": "AMFTA",
          "amfta_noq": r"\textbf{AMFTA-ND}"}
CITE = {"fedavg": r"~\cite{mcmahan2017communication}",
        "trimmed_mean": r"~\cite{yin2018byzantine}",
        "krum": r"~\cite{blanchard2017byzantine}",
        "fltrust": r"~\cite{cao2021fltrust}", "feddbc": r"~\cite{feddbc2026}",
        "amfta": "", "amfta_noq": ""}
RHOS = ["0.1", "0.2", "0.3", "0.4"]
SEEDS = (42, 123, 456)


def per_seed(results, method, byz, attack, key, last=5):
    out = {}
    for s in SEEDS:
        fs = sorted(glob.glob(os.path.join(
            results, f"{method}_byz{byz}_{attack}_seed{s}_*.json")))
        for f in reversed(fs):                     # newest non-empty wins
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if not d:
                continue
            v = [r[key] for r in d if key in r][-last:]
            if v:
                out[s] = sum(v) / len(v)
                break
    return out


def cell(results, method, byz, attack, key="accuracy", scale=100.0):
    d = per_seed(results, method, byz, attack, key)
    if not d:
        return None
    v = [x * scale for x in d.values()]
    return (st.mean(v),
            st.stdev(v) if len(v) > 1 else 0.0,      # sample SD, ddof=1
            st.pstdev(v),                            # population SD, ddof=0
            len(v))


def fmt(c, dec=1):
    return "---" if c is None else f"${c[0]:.{dec}f}\\pm{c[1]:.{dec}f}$"


def sweep_table(results, attack, label, caption, rhos):
    rows = []
    for m in METHODS:
        cells = [cell(results, m, r, attack) for r in rhos]
        if all(c is None for c in cells):
            continue
        rows.append((m, cells))
    best = {}
    for j, r in enumerate(rhos):
        cand = [(m, c[j]) for m, c in rows if c[j]]
        best[j] = max(cand, key=lambda t: t[1][0])[0] if cand else None

    L = [r"\begin{table}[H]", f"\t\\caption{{{caption}}}", f"\t\\label{{{label}}}",
         r"	\centering",
         "\t\\begin{tabular}{@{}l" + "c" * len(rhos) + "@{}}", r"		\toprule",
         "\t\t\\textbf{Method} & " + " & ".join(f"$\\rho={float(r):.2f}$" for r in rhos) + r"\\",
         r"		\midrule"]
    for m, cells in rows:
        cs = []
        for j, c in enumerate(cells):
            s = fmt(c)
            if c and best[j] == m:
                s = f"$\\mathbf{{{c[0]:.1f}}}\\pm{c[1]:.1f}$"
            cs.append(s)
        L.append(f"\t\t{PRETTY[m]}{CITE[m]} & " + " & ".join(cs) + r"\\")
    L += [r"		\bottomrule", r"	\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def f1_table(results):
    L = [r"\begin{table}[H]",
         "\t\\caption{F1 score at $\\rho=0.30$, mean $\\pm$ across-seed sample "
         "standard deviation over three seeds. Best per column in \\textbf{bold}.}",
         r"	\label{tab:f1}", r"	\centering", r"	\begin{tabular}{@{}lcc@{}}",
         r"		\toprule",
         r"		\textbf{Method} & \textbf{Label flipping} & \textbf{Gaussian noise}\\",
         r"		\midrule"]
    rows = [(m, cell(results, m, "0.3", "label_flipping", "f1", 1.0),
             cell(results, m, "0.3", "gaussian_noise", "f1", 1.0)) for m in METHODS]
    for j in (1, 2):
        cand = [(m, r[j]) for m, r in [(x[0], x) for x in rows] if r[j]]
    bl = max([r for r in rows if r[1]], key=lambda r: r[1][0])[0]
    bg = max([r for r in rows if r[2]], key=lambda r: r[2][0])[0]
    for m, a, b in rows:
        sa = f"$\\mathbf{{{a[0]:.3f}}}\\pm{a[1]:.3f}$" if a and m == bl else fmt(a, 3)
        sb = f"$\\mathbf{{{b[0]:.3f}}}\\pm{b[1]:.3f}$" if b and m == bg else fmt(b, 3)
        L.append(f"\t\t{PRETTY[m]} & {sa} & {sb}\\\\")
    L += [r"		\bottomrule", r"	\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def detection_table(results):
    """Precision and recall only. FPR and ROC-AUC are not stored in the logs
    and cannot be recovered without re-running with extra instrumentation."""
    L = [r"\begin{table}[H]",
         "\t\\caption{Detection metrics at $\\rho=0.30$ under label flipping, "
         "mean $\\pm$ across-seed sample standard deviation over three seeds. "
         "False-positive rate and ROC-AUC are omitted: the stored per-round logs "
         "record accuracy, F1, precision and recall only.}",
         r"	\label{tab:idsmetrics}", r"	\centering",
         r"	\begin{tabular}{@{}lcc@{}}", r"		\toprule",
         r"		\textbf{Method} & \textbf{Precision} & \textbf{Recall}\\",
         r"		\midrule"]
    for m in METHODS:
        p = cell(results, m, "0.3", "label_flipping", "precision", 1.0)
        r_ = cell(results, m, "0.3", "label_flipping", "recall", 1.0)
        L.append(f"\t\t{PRETTY[m]} & {fmt(p,3)} & {fmt(r_,3)}\\\\")
    L += [r"		\bottomrule", r"	\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def coverage(results):
    print("\nCoverage against the manuscript's tables")
    print(f"  {'cell':<44}{'n':>4}  status")
    missing = []
    for attack in ("label_flipping", "gaussian_noise"):
        for m in METHODS:
            for r in RHOS:
                c = cell(results, m, r, attack)
                tag = f"{m} | rho={r} | {attack}"
                if c is None:
                    missing.append(tag)
                    print(f"  {tag:<44}{'--':>4}  NO RUN")
                else:
                    print(f"  {tag:<44}{c[3]:>4}  ok  acc={c[0]:.1f} "
                          f"sd(1)={c[1]:.1f} sd(0)={c[2]:.1f}")
    for extra in ("none", "adaptive", "sign_flipping"):
        got = [m for m in METHODS if cell(results, m, "0.0" if extra == "none" else "0.3", extra)]
        print(f"  {extra:<44}{len(got):>4}  {'partial' if got else 'NO RUN'}")
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="tables_from_logs")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rhos = ["0.1", "0.2", "0.3"]
    (out / "tab_labelflip.tex").write_text(sweep_table(
        a.results, "label_flipping", "tab:labelflip",
        "Accuracy (\\%) under label flipping, mean $\\pm$ across-seed sample "
        "standard deviation over three seeds. Best per column in \\textbf{bold}.",
        rhos), encoding="utf8")
    (out / "tab_gaussian.tex").write_text(sweep_table(
        a.results, "gaussian_noise", "tab:gaussian",
        "Accuracy (\\%) under Gaussian-noise model poisoning, mean $\\pm$ "
        "across-seed sample standard deviation over three seeds.",
        rhos), encoding="utf8")
    (out / "tab_f1.tex").write_text(f1_table(a.results), encoding="utf8")
    (out / "tab_idsmetrics.tex").write_text(detection_table(a.results), encoding="utf8")

    missing = coverage(a.results)
    print(f"\nwrote 4 tables to {out}")
    print(f"{len(missing)} cells have no run and were left out of the tables.")


if __name__ == "__main__":
    main()
