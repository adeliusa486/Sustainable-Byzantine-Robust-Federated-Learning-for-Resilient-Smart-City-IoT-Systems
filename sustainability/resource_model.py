"""
Resource, Communication and Energy Model for Sustainable Byzantine-Robust FL
============================================================================

Produces every number in the sustainability sections of the manuscript
"Sustainable Byzantine-Robust Federated Learning for Resilient Smart-City
IoT Systems".

The model is a *Level-B* evaluation in the sense of the revision blueprint:
energy is not measured with a power meter, it is derived from a documented
device-level power model whose inputs are stated explicitly and whose
conclusions are checked against a sensitivity sweep.

Three classes of quantity appear here and they must not be confused:

  EXACT      - determined by the protocol and the declared configuration
               (bytes on the wire, FLOPs of a convex model). No modelling
               assumption is involved.
  MEASURED   - taken from wall-clock timings already reported in the
               manuscript (server aggregation time, Table "runtime").
  MODELLED   - obtained by combining EXACT/MEASURED quantities with a
               device power profile. Every such number is reported together
               with the sensitivity sweep in `sensitivity()`.

Usage:
    python resource_model.py            # print report + write LaTeX tables
    python resource_model.py --json     # dump machine-readable results
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Configuration declared in the manuscript (Table: Experimental configuration)
# ---------------------------------------------------------------------------

D_PARAMS = 41          # global model dimension (logistic regression)
BYTES_PER_PARAM = 4    # float32
N_CLIENTS = 100
T_ROUNDS = 25
E_EPOCHS = 5
BATCH = 32
SEEDS = (42, 123, 456)

# Server-side aggregation wall-clock time per round at N=100, milliseconds.
# MEASURED - reproduced verbatim from the manuscript's scalability table.
AGG_MS_N100 = {
    "FedAvg": 3.4,
    "FLTrust": 5.1,
    "Trimmed Mean": 8.3,
    "AMFTA-ND": 8.6,
    "AMFTA": 12.1,
    "Krum": 112.4,
    "FedDBC": 119.6,
}
AGG_MS_BY_N = {   # MEASURED - full scalability table
    50:  {"FedAvg": 1.8, "FLTrust": 2.6, "Trimmed Mean": 4.1, "AMFTA-ND": 4.3,
          "AMFTA": 6.2, "Krum": 28.6, "FedDBC": 31.1},
    100: AGG_MS_N100,
    200: {"FedAvg": 6.9, "FLTrust": 10.3, "Trimmed Mean": 17.0, "AMFTA-ND": 17.4,
          "AMFTA": 24.3, "Krum": 447.9, "FedDBC": 470.2},
    500: {"FedAvg": 17.2, "FLTrust": 25.8, "Trimmed Mean": 43.5, "AMFTA-ND": 43.1,
          "AMFTA": 60.7, "Krum": 2795.3, "FedDBC": 2903.7},
}

# Robust accuracy (%) by attacker fraction. MEASURED - manuscript Tables 4 and 5.
RHOS = (0.10, 0.20, 0.30)
ACC_LF = {   # label flipping
    "FedAvg":       (94.3, 89.3, 73.8),
    "Trimmed Mean": (92.1, 91.4, 83.0),
    "Krum":         (89.7, 89.5, 87.3),
    "FLTrust":      (77.4, 74.6, 72.1),
    "FedDBC":       (92.1, 85.6, 72.0),
    "AMFTA":        (93.1, 92.8, 80.3),
    "AMFTA-ND":     (92.5, 92.3, 91.7, 70.9),
}
ACC_GN = {   # Gaussian-noise model poisoning
    "FedAvg":       (71.3, 40.8, 42.8),
    "Trimmed Mean": (94.4, 57.0, 41.5),
    "Krum":         (90.0, 89.9, 90.3),
    "FLTrust":      (77.3, 73.6, 70.3),
    "FedDBC":       (93.9, 92.8, 65.3),
    "AMFTA":        (90.3, 89.5, 89.3),
    "AMFTA-ND":     (90.9, 90.6, 90.6, 79.2),
}
ACC_RHO30 = {m: (ACC_LF[m][2], ACC_GN[m][2]) for m in ACC_LF}

# Server-side extra work beyond aggregation, expressed as an equivalent
# number of additional local-training passes per round. EXACT by construction.
#   FLTrust trains on its root set (1% of train) for 5 epochs every round.
#   AMFTA runs leave-one-out validation passes for the borderline set.
ROOT_FRACTION = 0.01
FLTRUST_SERVER_EPOCHS = 5
AMFTA_BORDERLINE_FRAC = 0.5   # interquartile range => at most half the cohort
AMFTA_VAL_FRACTION = 0.01     # validation buffer is 1% of train

# ---------------------------------------------------------------------------
# 2. Device power profiles (MODELLED inputs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceClass:
    """A Smart-City edge device profile.

    throughput_flops : sustained achievable FLOP/s for dense float math
    p_active_w       : active-compute power draw, watts
    e_bit_j          : radio energy per transmitted bit, joules
    e_fix_j          : fixed per-round radio energy (wake-up, sync, tail),
                       joules; dominates for small payloads
    budget_j         : energy available to the device over the deployment
                       window. Mains-powered tiers are given a large but
                       finite budget; the battery tier is a real cell budget.
                       Sustainability cost is expressed as a FRACTION of this,
                       which is what makes the score fair: a joule spent by a
                       battery sensor is not the same object as a joule spent
                       by a mains-powered gateway.
    share            : fraction of the client population in this class
    """
    name: str
    label: str
    throughput_flops: float
    p_active_w: float
    e_bit_j: float
    e_fix_j: float
    budget_j: float
    share: float
    radio: str


# Representative values for three widely deployed Smart-City edge tiers.
# These are order-of-magnitude representative, not device-specific claims;
# `sensitivity()` sweeps each by a decade in both directions.
# Class C budget: 2 x AA lithium, 3000 mAh at 3.0 V = 3.24e4 J.
CLASSES = [
    DeviceClass("A", "Gateway / roadside unit (Cortex-A72 class)",
                throughput_flops=1.0e10, p_active_w=6.0,
                e_bit_j=5.0e-8, e_fix_j=5.0e-2, budget_j=1.0e7,
                share=0.20, radio="Wi-Fi / Ethernet"),
    DeviceClass("B", "Building or meter concentrator (Cortex-A7 class)",
                throughput_flops=5.0e8, p_active_w=1.5,
                e_bit_j=1.0e-6, e_fix_j=3.0e-1, budget_j=1.0e6,
                share=0.45, radio="NB-IoT / LTE-M"),
    DeviceClass("C", "Battery sensor node (Cortex-M4 class)",
                throughput_flops=2.0e7, p_active_w=0.09,
                e_bit_j=2.0e-5, e_fix_j=1.5e-1, budget_j=3.24e4,
                share=0.35, radio="LoRaWAN"),
]

ACC_FLOOR = 85.0   # deployment accuracy floor for the gated Pareto frontier
P_SERVER_W = 82.0   # Xeon Gold 6248R, 205 W TDP at ~40% sustained utilisation


# ---------------------------------------------------------------------------
# 3. Exact quantities
# ---------------------------------------------------------------------------

def load_partition_sizes(seed: int = 42) -> np.ndarray:
    """Real Dirichlet(alpha=0.5) partition sizes over 100 clients (TON_IoT)."""
    sizes = json.loads((HERE / "partition_sizes.json").read_text())
    return np.array(sizes[str(seed)], dtype=np.int64)


def client_flops_per_round(n_i: np.ndarray, d: int = D_PARAMS,
                           epochs: int = E_EPOCHS, batch: int = BATCH) -> np.ndarray:
    """EXACT FLOPs for E local epochs of SGD-with-momentum on logistic regression.

    Per sample : 2d (forward dot product) + d (gradient) = 3d
    Per batch  : 2d (momentum buffer) + 2d (parameter update) = 4d
    """
    n_batches = np.ceil(n_i / batch)
    return epochs * d * (3.0 * n_i + 4.0 * n_batches)


def bytes_per_round_per_client(d: int = D_PARAMS, metadata_floats: int = 0) -> tuple:
    """EXACT bytes on the wire: (uplink, downlink)."""
    up = d * BYTES_PER_PARAM + metadata_floats * BYTES_PER_PARAM
    down = d * BYTES_PER_PARAM
    return up, down


# ---------------------------------------------------------------------------
# 4. Modelled energy
# ---------------------------------------------------------------------------

def assign_classes(n_clients: int, rng: np.random.Generator) -> np.ndarray:
    """Assign each client to a device class by the declared population shares.

    Assignment is independent of partition size, so device heterogeneity and
    data heterogeneity are uncorrelated by construction. This is the
    conservative choice: correlating them would amplify every effect reported.
    """
    shares = np.array([c.share for c in CLASSES])
    counts = np.floor(shares * n_clients).astype(int)
    counts[-1] += n_clients - counts.sum()
    idx = np.concatenate([np.full(c, k) for k, c in enumerate(counts)])
    rng.shuffle(idx)
    return idx


def client_energy(n_i: np.ndarray, cls_idx: np.ndarray, d: int = D_PARAMS,
                  rounds: int = T_ROUNDS, epochs: int = E_EPOCHS,
                  metadata_floats: int = 0, classes=None,
                  participation: np.ndarray | None = None,
                  epochs_per_client: np.ndarray | None = None) -> dict:
    """MODELLED per-client energy over the whole federation run.

    participation      : per-client expected participation rate in [0,1]
    epochs_per_client  : per-client local epoch count (overrides `epochs`)
    """
    classes = classes or CLASSES
    thr = np.array([classes[k].throughput_flops for k in cls_idx])
    pw = np.array([classes[k].p_active_w for k in cls_idx])
    ebit = np.array([classes[k].e_bit_j for k in cls_idx])
    efix = np.array([classes[k].e_fix_j for k in cls_idx])

    if participation is None:
        participation = np.ones(len(n_i))
    if epochs_per_client is None:
        epochs_per_client = np.full(len(n_i), float(epochs))

    # ---- compute -----------------------------------------------------------
    flops_round = np.array([
        client_flops_per_round(np.array([n]), d=d, epochs=int(round(e)))[0]
        for n, e in zip(n_i, epochs_per_client)
    ])
    t_compute_round = flops_round / thr                 # seconds
    e_compute_round = t_compute_round * pw              # joules
    e_compute = e_compute_round * rounds * participation

    # ---- communication -----------------------------------------------------
    up, down = bytes_per_round_per_client(d, metadata_floats)
    bits = (up + down) * 8
    e_comm_round = efix + ebit * bits
    e_comm = e_comm_round * rounds * participation

    budget = np.array([classes[k].budget_j for k in cls_idx])
    e_total = e_compute + e_comm
    return {
        "flops_round": flops_round,
        "t_compute_round": t_compute_round,
        "e_compute_round": e_compute_round,
        "e_comm_round": e_comm_round,
        "e_compute": e_compute,
        "e_comm": e_comm,
        "e_total": e_total,
        "budget_frac": e_total / budget,          # share of the device's own budget
        "bytes_total": (up + down) * rounds * participation,
    }


def server_energy(method: str, n_i: np.ndarray, d: int = D_PARAMS,
                  rounds: int = T_ROUNDS, n_clients: int = N_CLIENTS) -> dict:
    """MODELLED server energy = MEASURED aggregation time x server power,
    plus the EXACT extra training/validation work a method requires."""
    t_agg = AGG_MS_N100[method] / 1000.0 * rounds       # seconds, MEASURED

    n_total = int(n_i.sum())
    extra_flops = 0.0
    if method == "FLTrust":
        root_n = ROOT_FRACTION * n_total
        extra_flops = rounds * FLTRUST_SERVER_EPOCHS * d * (3 * root_n + 4 * math.ceil(root_n / BATCH))
    elif method == "AMFTA":
        # leave-one-out validation passes: forward only, 2d per sample
        val_n = AMFTA_VAL_FRACTION * n_total
        n_loo = AMFTA_BORDERLINE_FRAC * n_clients
        extra_flops = rounds * n_loo * 2 * d * val_n

    # server throughput: one Xeon core sustained, conservative
    t_extra = extra_flops / 5.0e9
    e_server = (t_agg + t_extra) * P_SERVER_W
    return {"t_agg_s": t_agg, "t_extra_s": t_extra,
            "extra_flops": extra_flops, "e_server_j": e_server}


# ---------------------------------------------------------------------------
# 5. The resource-aware scheduling layer (AMFTA-S)
# ---------------------------------------------------------------------------

def sustainability_score(n_i: np.ndarray, cls_idx: np.ndarray, classes=None,
                         lam=(0.4, 0.2, 0.2, 0.2), s_min: float = 0.05) -> np.ndarray:
    """S_i in [s_min, 1]: 1 means cheap to obtain an update from, 0 means costly.

    The energy and communication components are expressed as a fraction of the
    client's OWN energy budget rather than in absolute joules. This is the
    design decision that keeps the score fair: without it the score is
    dominated by whichever tier happens to draw the most absolute power,
    which for a Smart-City deployment is the mains-fed gateway, exactly the
    tier that needs no relief.

    Quantities spanning several decades are min-max normalised on a log scale;
    a linear min-max would collapse every client but the single worst one.
    """
    classes = classes or CLASSES
    ce = client_energy(n_i, cls_idx, classes=classes)
    lam_e, lam_c, lam_f, lam_t = lam
    budget = np.array([classes[k].budget_j for k in cls_idx])

    def mm(x, log=False):
        x = np.asarray(x, dtype=float)
        if log:
            x = np.log10(np.maximum(x, 1e-30))
        rng = x.max() - x.min()
        return np.zeros_like(x) if rng == 0 else (x - x.min()) / rng

    cost = (lam_e * mm((ce["e_compute_round"] + ce["e_comm_round"]) / budget, log=True)
            + lam_c * mm(ce["e_comm_round"] / budget, log=True)
            + lam_f * mm(ce["flops_round"], log=True)
            + lam_t * mm(ce["t_compute_round"], log=True))
    return np.clip(1.0 - cost, s_min, 1.0)


def ras_schedule(S: np.ndarray, p_min: float = 0.25, gamma: float = 1.0,
                 e_base: int = E_EPOCHS, e_min: int = 1) -> tuple:
    """Resource-Aware Scheduling: participation probabilities and local epochs.

    p_i is server-assigned, never client-declared, and floored at p_min so
    that (a) the reputation EMA of every client keeps receiving evidence and
    (b) the importance-weight amplification 1/p_i is bounded by 1/p_min.
    """
    p = np.clip((S / S.mean()) ** gamma, 0.0, 1.0)
    p = np.clip(p, p_min, 1.0)
    epochs = np.clip(np.round(e_base * S / S.mean()), e_min, e_base).astype(float)
    return p, epochs


# ---------------------------------------------------------------------------
# 6. Report
# ---------------------------------------------------------------------------

def fmt(x, unit=""):
    if x >= 1e6:
        return f"{x/1e6:.2f} M{unit}"
    if x >= 1e3:
        return f"{x/1e3:.2f} k{unit}"
    if x >= 1:
        return f"{x:.2f} {unit}".strip()
    if x >= 1e-3:
        return f"{x*1e3:.2f} m{unit}"
    return f"{x*1e6:.2f} u{unit}"


def build(seed: int = 42, d: int = D_PARAMS):
    rng = np.random.default_rng(seed)
    n_i = load_partition_sizes(seed)
    cls_idx = assign_classes(len(n_i), rng)
    res = {"seed": seed, "d": d, "n_total": int(n_i.sum())}

    # ---- baseline (no scheduling) -----------------------------------------
    base = client_energy(n_i, cls_idx, d=d)
    res["client"] = {
        "e_compute_total_j": float(base["e_compute"].sum()),
        "e_comm_total_j": float(base["e_comm"].sum()),
        "e_total_j": float(base["e_total"].sum()),
        "bytes_total": float(base["bytes_total"].sum()),
        "comm_share": float(base["e_comm"].sum() / base["e_total"].sum()),
    }

    # per-class breakdown
    per_class = {}
    for k, c in enumerate(CLASSES):
        m = cls_idx == k
        per_class[c.name] = {
            "label": c.label, "radio": c.radio, "n_clients": int(m.sum()),
            "mean_samples": float(n_i[m].mean()),
            "e_compute_j": float(base["e_compute"][m].sum()),
            "e_comm_j": float(base["e_comm"][m].sum()),
            "e_total_j": float(base["e_total"][m].sum()),
            "mean_latency_s": float(base["t_compute_round"][m].mean()),
            "max_latency_s": float(base["t_compute_round"][m].max()),
            "mean_budget_pct": float(100 * base["budget_frac"][m].mean()),
            "max_budget_pct": float(100 * base["budget_frac"][m].max()),
            "runs_per_budget": float(1.0 / base["budget_frac"][m].max()),
        }
    res["per_class"] = per_class

    # Synchronous FL is straggler-bound: the round takes as long as its
    # slowest participant, so the max is the operationally relevant statistic.
    res["latency"] = {
        "round_s": float(base["t_compute_round"].max()),
        "median_client_s": float(np.median(base["t_compute_round"])),
        "straggler_ratio": float(base["t_compute_round"].max()
                                 / np.median(base["t_compute_round"])),
    }

    # ---- per-method totals -------------------------------------------------
    methods = {}
    for m in AGG_MS_N100:
        se = server_energy(m, n_i, d=d)
        client_j = base["e_total"].sum()
        total = client_j + se["e_server_j"]
        lf, gn = ACC_RHO30[m]
        methods[m] = {
            "client_j": float(client_j),
            "server_j": float(se["e_server_j"]),
            "server_agg_s": float(se["t_agg_s"]),
            "server_extra_s": float(se["t_extra_s"]),
            "total_j": float(total),
            "bytes_total": float(base["bytes_total"].sum()),
            "acc_lf": lf, "acc_gn": gn,
            "acc_worst": min(lf, gn),
            "j_per_pp_worst": float(total / min(lf, gn)),
            "worst_acc_per_kj": float(min(lf, gn) / (total / 1000.0)),
        }
    res["methods"] = methods

    # Pareto set on (total energy, worst-case robust accuracy at rho=0.30)
    pts = [(k, v["total_j"], v["acc_worst"]) for k, v in methods.items()]
    pareto = []
    for k, e, a in pts:
        dominated = any((e2 <= e and a2 >= a and (e2 < e or a2 > a))
                        for k2, e2, a2 in pts if k2 != k)
        if not dominated:
            pareto.append(k)
    res["pareto"] = sorted(pareto, key=lambda k: methods[k]["total_j"])

    # ---- resource-aware scheduling ----------------------------------------
    S = sustainability_score(n_i, cls_idx)
    p, ep = ras_schedule(S)
    sched = client_energy(n_i, cls_idx, d=d, participation=p, epochs_per_client=ep)
    md = client_energy(n_i, cls_idx, d=d, metadata_floats=4)   # metadata overhead
    res["ras"] = {
        "S_mean": float(S.mean()), "S_min": float(S.min()), "S_max": float(S.max()),
        "mean_participation": float(p.mean()),
        "mean_epochs": float(ep.mean()),
        "e_total_j": float(sched["e_total"].sum()),
        "energy_saving_pct": float(100 * (1 - sched["e_total"].sum() / base["e_total"].sum())),
        "bytes_total": float(sched["bytes_total"].sum()),
        "comm_saving_pct": float(100 * (1 - sched["bytes_total"].sum() / base["bytes_total"].sum())),
        "max_amplification": float(1.0 / p.min()),
        "metadata_overhead_pct_uplink": float(100 * 4 * BYTES_PER_PARAM / (d * BYTES_PER_PARAM)),
        "metadata_energy_overhead_pct": float(100 * (md["e_total"].sum() / base["e_total"].sum() - 1)),
    }
    # per-class relief. The fairness-relevant number is the change in
    # budget fraction consumed by the battery tier, not absolute joules.
    for k, c in enumerate(CLASSES):
        m = cls_idx == k
        res["ras"][f"class{c.name}_energy_saving_pct"] = float(
            100 * (1 - sched["e_total"][m].sum() / base["e_total"][m].sum()))
        res["ras"][f"class{c.name}_budget_pct_before"] = float(100 * base["budget_frac"][m].mean())
        res["ras"][f"class{c.name}_budget_pct_after"] = float(100 * sched["budget_frac"][m].mean())
        res["ras"][f"class{c.name}_mean_p"] = float(p[m].mean())
    res["ras"]["latency_round_s_after"] = float(
        (sched["t_compute_round"] * (p > 0)).max())

    # ---- data-heterogeneity driven compute spread -------------------------
    res["spread"] = {
        "n_min": int(n_i.min()), "n_max": int(n_i.max()),
        "n_median": float(np.median(n_i)),
        "sample_ratio": float(n_i.max() / n_i.min()),
        "latency_ratio": float(base["t_compute_round"].max() / base["t_compute_round"].min()),
        "energy_ratio": float(base["e_total"].max() / base["e_total"].min()),
    }
    return res


def scale_projection(seed: int = 42):
    """How the compute/communication balance shifts with model dimension."""
    rows = []
    for d in (41, 5_057, 100_000):
        r = build(seed, d=d)
        pc = r["per_class"]["C"]
        rows.append({
            "d": d,
            "comm_share": r["client"]["comm_share"],
            "client_j": r["client"]["e_total_j"],
            "bytes_mb": r["client"]["bytes_total"] / 1e6,
            "classC_budget_pct": pc["mean_budget_pct"],
            "classC_runs_affordable": pc["runs_per_budget"],
            "classC_days_daily_retrain": pc["runs_per_budget"],
            "ras_saving_pct": r["ras"]["energy_saving_pct"],
            "ras_classC_saving_pct": r["ras"]["classC_energy_saving_pct"],
            "metadata_overhead_pct_uplink": r["ras"]["metadata_overhead_pct_uplink"],
            "metadata_energy_overhead_pct": r["ras"]["metadata_energy_overhead_pct"],
        })
    return rows


def pareto_by_rho(seed: int = 42, d: int = D_PARAMS):
    """Robustness-sustainability Pareto set at each attacker fraction.

    Energy does not depend on rho in this model: the round budget and the
    participation schedule are fixed, so adversarial pressure changes what a
    joule buys, not how many joules are spent. The informative quantity is
    therefore energy per retained accuracy point, which does move with rho.
    """
    r = build(seed, d=d)
    out = {}
    for j, rho in enumerate(RHOS):
        rows = {}
        for m, v in r["methods"].items():
            worst = min(ACC_LF[m][j], ACC_GN[m][j])
            rows[m] = {"total_j": v["total_j"], "acc_worst": worst,
                       "j_per_pp": v["total_j"] / worst,
                       "acc_per_kj": worst / (v["total_j"] / 1000.0)}
        pts = [(k, v["total_j"], v["acc_worst"]) for k, v in rows.items()]

        def frontier(candidates):
            return [k for k, e, a in candidates
                    if not any(e2 <= e and a2 >= a and (e2 < e or a2 > a)
                               for k2, e2, a2 in candidates if k2 != k)]

        par = frontier(pts)
        # An unconstrained frontier admits a method purely for being cheap,
        # even at 42.8% accuracy on a binary task. The operationally
        # meaningful object is the frontier restricted to rules that clear a
        # deployment accuracy floor.
        gated = [p_ for p_ in pts if p_[2] >= ACC_FLOOR]
        out[rho] = {"rows": rows,
                    "pareto": sorted(par, key=lambda k: rows[k]["total_j"]),
                    "pareto_gated": sorted(frontier(gated), key=lambda k: rows[k]["total_j"]),
                    "admissible": sorted([k for k, _, _ in gated])}
    return out


def sensitivity(seed: int = 42):
    """Sweep every modelled device parameter by a decade in both directions
    and report whether the qualitative conclusions survive."""
    rng = np.random.default_rng(seed)
    n_i = load_partition_sizes(seed)
    cls_idx = assign_classes(len(n_i), rng)
    out = []
    for factor in (0.1, 0.316, 1.0, 3.16, 10.0):
        for knob in ("e_bit", "e_fix", "power", "throughput"):
            cs = []
            for c in CLASSES:
                kw = asdict(c)
                if knob == "e_bit":
                    kw["e_bit_j"] *= factor
                elif knob == "e_fix":
                    kw["e_fix_j"] *= factor
                elif knob == "power":
                    kw["p_active_w"] *= factor
                elif knob == "throughput":
                    kw["throughput_flops"] *= factor
                cs.append(DeviceClass(**kw))
            ce = client_energy(n_i, cls_idx, classes=cs)
            out.append({
                "knob": knob, "factor": factor,
                "comm_share": float(ce["e_comm"].sum() / ce["e_total"].sum()),
                "total_j": float(ce["e_total"].sum()),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    per_seed = {s: build(s) for s in SEEDS}
    r = per_seed[42]

    if args.json:
        payload = {"per_seed": per_seed, "scale": scale_projection(),
                   "pareto_by_rho": {str(k): v for k, v in pareto_by_rho().items()},
                   "sensitivity": sensitivity()}
        (OUT / "resource_model.json").write_text(json.dumps(payload, indent=2))
        print(f"wrote {OUT / 'resource_model.json'}")
        return

    print("=" * 78)
    print("RESOURCE MODEL - seed 42, d =", r["d"], ", N =", N_CLIENTS,
          ", T =", T_ROUNDS, ", E =", E_EPOCHS)
    print("=" * 78)

    print("\n[EXACT] Data heterogeneity -> compute heterogeneity")
    sp = r["spread"]
    print(f"  client samples: min {sp['n_min']}  median {sp['n_median']:.0f}  max {sp['n_max']}"
          f"   ratio {sp['sample_ratio']:.0f}x")
    print(f"  local latency ratio across cohort : {sp['latency_ratio']:.0f}x")
    print(f"  per-client energy ratio           : {sp['energy_ratio']:.0f}x")

    print("\n[EXACT] Communication")
    up, down = bytes_per_round_per_client()
    print(f"  per client per round: {up} B up + {down} B down = {up+down} B")
    print(f"  whole federation    : {r['client']['bytes_total']/1e6:.2f} MB over {T_ROUNDS} rounds")

    print("\n[MODELLED] Client energy split")
    c = r["client"]
    print(f"  compute {fmt(c['e_compute_total_j'],'J')}   communication {fmt(c['e_comm_total_j'],'J')}"
          f"   total {fmt(c['e_total_j'],'J')}")
    print(f"  communication share of client energy: {100*c['comm_share']:.1f}%")

    print("\n[MODELLED] Per device class")
    for k, v in r["per_class"].items():
        print(f"  Class {k} ({v['radio']:>18s}) n={v['n_clients']:3d}"
              f"  E={fmt(v['e_total_j'],'J'):>10s}"
              f"  budget/run {v['mean_budget_pct']:7.4f}%"
              f"  runs affordable {v['runs_per_budget']:9.0f}"
              f"  max local latency {v['max_latency_s']*1e3:9.2f} ms")
    lat = r["latency"]
    print(f"  straggler-bound round latency {lat['round_s']*1e3:.2f} ms"
          f"  (median client {lat['median_client_s']*1e3:.3f} ms,"
          f" ratio {lat['straggler_ratio']:.0f}x)")

    print("\n[MEASURED + MODELLED] Per aggregation rule, whole run")
    print(f"  {'method':<14s}{'client J':>10s}{'server J':>10s}{'total J':>10s}"
          f"{'worst acc':>11s}{'J / pp':>9s}{'acc/kJ':>9s}")
    for m, v in sorted(r["methods"].items(), key=lambda kv: kv[1]["total_j"]):
        star = " *" if m in r["pareto"] else "  "
        print(f"  {m:<14s}{v['client_j']:>10.1f}{v['server_j']:>10.2f}{v['total_j']:>10.1f}"
              f"{v['acc_worst']:>11.1f}{v['j_per_pp_worst']:>9.2f}{v['worst_acc_per_kj']:>9.2f}{star}")
    print(f"  Pareto-efficient (* above): {', '.join(r['pareto'])}")

    print("\n[MODELLED] Resource-aware scheduling (AMFTA-S)")
    ras = r["ras"]
    print(f"  S in [{ras['S_min']:.3f}, {ras['S_max']:.3f}], mean {ras['S_mean']:.3f}")
    print(f"  mean participation {ras['mean_participation']:.3f}, mean local epochs {ras['mean_epochs']:.2f}")
    print(f"  client energy saving overall : {ras['energy_saving_pct']:.1f}%")
    for c in CLASSES:
        print(f"    Class {c.name} mean p={ras[f'class{c.name}_mean_p']:.2f}"
              f"  energy {ras[f'class{c.name}_energy_saving_pct']:5.1f}%"
              f"  budget/run {ras[f'class{c.name}_budget_pct_before']:.4f}%"
              f" -> {ras[f'class{c.name}_budget_pct_after']:.4f}%")
    print(f"  communication saving         : {ras['comm_saving_pct']:.1f}%")
    print(f"  worst-case influence amplification 1/p_min : {ras['max_amplification']:.2f}x")
    print(f"  resource metadata overhead: {ras['metadata_overhead_pct_uplink']:.1f}% of uplink bytes,"
          f" {ras['metadata_energy_overhead_pct']:.2f}% of client energy")

    print("\n[ANALYSIS] Energy per retained accuracy point vs attacker fraction")
    pbr = pareto_by_rho()
    hdr = "  " + "method".ljust(14) + "".join(f"{'rho='+str(rho):>10s}" for rho in RHOS)
    print(hdr + "     (J per pp of worst-case robust accuracy)")
    for m in AGG_MS_N100:
        cells = "".join(f"{pbr[rho]['rows'][m]['j_per_pp']:>10.2f}" for rho in RHOS)
        print(f"  {m:<14s}{cells}")
    for rho in RHOS:
        g = pbr[rho]["pareto_gated"]
        print(f"  rho={rho:.2f}  Pareto: {', '.join(pbr[rho]['pareto']):<38s}"
              f" | above {ACC_FLOOR:.0f}% floor: {', '.join(pbr[rho]['admissible']) or 'none'}"
              f" | gated Pareto: {', '.join(g) or 'none'}")

    print("\n[PROJECTION] Model dimension sweep")
    for row in scale_projection():
        print(f"  d = {row['d']:>7d}   comm share {100*row['comm_share']:5.1f}%"
              f"   client energy {fmt(row['client_j'],'J'):>10s}   {row['bytes_mb']:8.2f} MB"
              f"   ClassC budget/run {row['classC_budget_pct']:7.4f}%"
              f"   daily-retrain days {row['classC_runs_affordable']:9.0f}"
              f"   metadata {row['metadata_overhead_pct_uplink']:5.2f}% uplink")

    print("\n[SENSITIVITY] communication share of client energy under +/- 1 decade")
    sens = sensitivity()
    for knob in ("e_bit", "e_fix", "power", "throughput"):
        vals = [s["comm_share"] for s in sens if s["knob"] == knob]
        print(f"  {knob:<11s} comm share ranges {100*min(vals):5.1f}% .. {100*max(vals):5.1f}%")

    print("\n[SEED CHECK] totals across the three partition seeds")
    for s, rr in per_seed.items():
        print(f"  seed {s}: client {fmt(rr['client']['e_total_j'],'J'):>10s}"
              f"  comm share {100*rr['client']['comm_share']:5.1f}%"
              f"  RAS saving {rr['ras']['energy_saving_pct']:5.1f}%"
              f"  Class-C saving {rr['ras']['classC_energy_saving_pct']:5.1f}%")

    payload = {"per_seed": per_seed, "scale": scale_projection(),
               "pareto_by_rho": {str(k): v for k, v in pareto_by_rho().items()},
               "sensitivity": sens}
    (OUT / "resource_model.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT / 'resource_model.json'}")


if __name__ == "__main__":
    main()
