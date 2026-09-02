"""Audit every manuscript cell against the aggregated run table."""
import json

D = json.load(open(r"C:\Users\adeel\.gemini\antigravity\scratch\amfta-fl\amfta-fl"
                   r"\results\paper_tables.json"))["table"]

PAPER = {
 "label_flipping": {
  "fedavg":       [94.3, 89.3, 73.8],
  "trimmed_mean": [92.1, 91.4, 83.0],
  "krum":         [89.7, 89.5, 87.3],
  "fltrust":      [77.4, 74.6, 72.1],
  "feddbc":       [92.1, 85.6, 72.0],
  "amfta":        [93.1, 92.8, 80.3],
  "amfta_noq":    [92.5, 92.3, 91.7]},
 "gaussian_noise": {
  "fedavg":       [71.3, 40.8, 42.8],
  "trimmed_mean": [94.4, 57.0, 41.5],
  "krum":         [90.0, 89.9, 90.3],
  "fltrust":      [77.3, 73.6, 70.3],
  "feddbc":       [93.9, 92.8, 65.3],
  "amfta":        [90.3, 89.5, 89.3],
  "amfta_noq":    [90.9, 90.6, 90.6]},
}
RHOS = ["0.10", "0.20", "0.30"]

ok = miss = bad = 0
print(f"{'attack':<16}{'method':<14}{'rho':<7}{'paper':>8}{'logs':>10}{'seeds':>7}  verdict")
for attack, table in PAPER.items():
    for m, vals in table.items():
        for r, want in zip(RHOS, vals):
            k = f"{m}|byz{r}|{attack}"
            e = D.get(k)
            if e is None:
                print(f"{attack:<16}{m:<14}{r:<7}{want:>8.1f}{'--':>10}{'--':>7}  NO RUN")
                miss += 1
                continue
            got = 100 * e["acc_mean"]
            n = e["n_seeds"]
            match = abs(got - want) < 0.15
            ok += match
            bad += (not match)
            print(f"{attack:<16}{m:<14}{r:<7}{want:>8.1f}{got:>10.1f}{n:>7}  "
                  f"{'match' if match else '*** MISMATCH ***'}")

print(f"\nmatched {ok}   mismatched {bad}   no run {miss}   of {ok+bad+miss}")

print("\n--- what else is in the run table but not in these two paper tables ---")
used = {f"{m}|byz{r}|{a}" for a, t in PAPER.items() for m in t for r in RHOS}
for k in sorted(set(D) - used):
    e = D[k]
    print(f"  {k:<42} acc={100*e['acc_mean']:.1f}  n_seeds={e['n_seeds']}")
