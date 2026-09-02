# Sustainable Byzantine-Robust Federated Learning for Smart-City IoT

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-ee4c2c.svg)](https://pytorch.org/)

Reference implementation for **"Sustainable Byzantine-Robust Federated Learning
for Resilient Smart-City IoT Systems."**

Federated intrusion detection lets a city train on its own traffic without
centralising it — and hands the aggregation server a problem it cannot solve
from a single round: telling a poisoned update apart from an unusual but honest
one. This repository studies that problem together with the one nobody prices:
what the defence costs the devices that run it.

---

## The result in one figure

Once a deployment accuracy floor is applied, robustness and sustainability do
not trade off across the rules evaluated here. The linear-cost stateful trust
rule is the sole occupant of the frontier at moderate attacker fractions, and
the two quadratic-cost rules spend 38–40% more system energy without buying
admissible robustness.

<p align="center"><img src="figures/fig_pareto.png" width="620"></p>

---

## Headline findings

| | |
|---|---|
| **Certified stability** | AMFTA-ND is the only rule whose degradation over ρ ∈ [0.10, 0.30] is *statistically contained* within a pre-declared 5 pp margin in both attack families (TOST, not a null result) |
| **No security/sustainability trade-off** | Gated Pareto frontier at ρ = 0.20 and 0.30 has one occupant; Krum and FedDBC cost 38–40% more energy for no admissible gain |
| **Where trust earns its place** | Median-norm rescaling supplies most resistance to magnitude attacks; trust weighting carries label flipping (+15.3 pp) |
| **Trusted server data is unnecessary** | Removing the validation buffer never hurts and helps under pressure (trend test *p* = 0.009), while costing 47% of server energy |
| **A hard boundary** | Between ρ = 0.30 and 0.40 no rule stays usable — the frontier is empty, so energy spent there buys nothing |
| **Scheduling relieves the constrained tier** | −24.3% client energy overall, −66.7% for battery nodes, at a bounded 4.0× influence amplification |

---

## Method

Two layers, deliberately kept apart.

**Security layer** — answers *can this update be believed?* Gradient similarity
to the population centroid, an EMA reputation term that accumulates evidence
across rounds, and median-norm rescaling. No server-side dataset of any kind.

**Sustainability layer** — answers *what did this update cost to obtain?*
Energy, communication, computation and latency, each expressed as a fraction of
the client's **own** energy budget. A battery sensor's joule is not a mains
gateway's joule.

They meet only in the controller, which acts on *participation and local effort*
— never on the trust weight. Folding a resource score into trust would
systematically silence the battery tier, which in a city is also the tier
holding the most distinctive traffic. Instead an importance correction keeps a
resource-poor honest client's expected influence intact (Proposition 1), at the
cost of a new adversarial channel we bound rather than hide (Proposition 2).

---

## Quick start

```bash
git clone https://github.com/adeliusa486/sustainable-byzantine-robust-fl
cd sustainable-byzantine-robust-fl
pip install -r requirements.txt

# NF-TON-IoT (real data; ~209 MB parquet)
python scripts/setup_data.py --dataset nf-ton-iot

# One configuration
python experiments/run_main.py --method amfta_noq --byzantine_fraction 0.30

# Every table in the paper
python experiments/run_paper_study.py --block all --model logistic
python experiments/paper_stats.py --results results_paper --out tables
```

> **Data note.** The experiments use **NF-TON-IoT**: 2,627,177 flows, 41 NetFlow
> features, binary normal/attack. `--synthetic` exists for CI smoke tests only
> and reproduces nothing in the paper.

---

## Reproducing the paper

Each block maps to one table. Blocks are resumable — a configuration already
present in the block's CSV is skipped, so long runs can be interrupted safely.

| Block | Produces |
|---|---|
| `clean` | Clean accuracy, ρ = 0 |
| `labelflip` | Accuracy under label flipping, four attacker fractions |
| `gaussian` | Accuracy under Gaussian-noise model poisoning |
| `adaptive` | AGR-tailored adaptive attack at ρ = 0.30 |
| `extras` | Coordinate-wise Median, Multi-Krum, FoolsGold |
| `normclip` | NormClip-Only against AMFTA-ND |
| `alpha01` | Severe heterogeneity, Dirichlet α = 0.1 |
| `scalability` | Server aggregation time against client population |

`paper_stats.py` emits the LaTeX tables plus Welch contrasts with
Holm–Bonferroni, the inverse-variance trend test, and **two one-sided
equivalence tests** for the degradation claim. Degradation is reported as
*graceful* only when equivalence is established — a non-significant difference
is absence of evidence, not evidence of stability, and cells the design cannot
resolve are labelled inconclusive.

Reporting convention: per seed, the mean over the final five rounds; across
seeds, the mean and **sample** standard deviation of those per-seed means.

---

## Sustainability model

`sustainability/resource_model.py` derives every energy, communication and
computation figure. It separates three kinds of quantity and labels them,
because they carry different evidential weight:

- **Exact** — bytes on the wire, FLOPs of a convex model. No assumptions.
- **Measured** — server aggregation wall-clock on the stated hardware.
- **Estimated** — energy, from a documented device power model, swept a decade
  in each direction (`sensitivity()`). Described as *estimated*, never
  *measured*.

Three Smart-City device tiers — gateway, concentrator, battery sensor —
spanning three orders of magnitude in energy budget.

```bash
python sustainability/resource_model.py      # every number in §7–§8
python sustainability/equivalence_tests.py   # the TOST table
python sustainability/make_figures.py        # Figures 3–5
```

---

## Layout

```
amfta/
  aggregation/   AMFTA, AMFTA-ND, FedAvg, Krum, Multi-Krum, Trimmed Mean,
                 coordinate-wise Median, FLTrust, FedDBC, FoolsGold, NormClip-Only
  attacks/       label flipping, Gaussian noise, sign flipping, mimicry,
                 AGR-tailored adaptive
  models/        logistic regression (paper configuration), MLP
  data/          NF-TON-IoT preprocessing, Dirichlet partitioning
training/        federated orchestration
experiments/     run_paper_study.py, paper_stats.py, scalability
sustainability/  resource model, equivalence tests, figures
```

---

## Scope

One dataset, one convex model class, three seeds, a simulated device
population, and energy that is modelled rather than instrumented. The
manuscript's Limitations section states each of these and what it bounds. We
report the region where the method works and the region where nothing we tested
does; knowing where a defence stops working is as useful to an operator as
knowing where it holds.

## Citation

```bibtex
@article{ahmad2026sustainable,
  title   = {Sustainable Byzantine-Robust Federated Learning for
             Resilient Smart-City IoT Systems},
  author  = {Ahmad, Adeel and Akarma, Ali and Mohmand, Muhammad Ismail and
             Syed, Toqeer Ali and Jan, Salman},
  journal = {Smart Cities},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
