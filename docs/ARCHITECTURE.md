# AMFTA Architecture Documentation

## Overview

AMFTA (Adaptive Multi-Factor Trust Aggregation) is a federated learning
aggregation framework designed for Byzantine-resilient intrusion detection
in non-IID smart city IoT deployments.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                     AMFTA Federated Learning System                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   ┌─────────────────── GLOBAL SERVER ─────────────────────────────┐   │
│   │                                                                │   │
│   │  ┌──────────────────────────────────────────────────────────┐ │   │
│   │  │              AMFTA Trust Engine (Algorithm 1)            │ │   │
│   │  │                                                          │ │   │
│   │  │  [Factor I]  Gradient Similarity   S_i (cosine, [0,1])  │ │   │
│   │  │      ↓                                                   │ │   │
│   │  │  [Factor II] Reputation Tracker    H_i (EMA, β=0.9)     │ │   │
│   │  │      ↓                                                   │ │   │
│   │  │  [Borderline Detection] τ_l=0.35, τ_u=0.55              │ │   │
│   │  │      ↓ (borderline only)                                 │ │   │
│   │  │  [Factor III] Quality Evaluation   Q_i (LOO δAcc)       │ │   │
│   │  │      ↓                                                   │ │   │
│   │  │  T_i = 0.4·S_i + 0.3·H_i + 0.3·Q_i                    │ │   │
│   │  │      ↓                                                   │ │   │
│   │  │  ḡ = Σ T_i·g_i / Σ T_i  (soft trust-weighted avg)      │ │   │
│   │  └──────────────────────────────────────────────────────────┘ │   │
│   │                                                                │   │
│   │  ┌──────────────────┐   ┌────────────────────────────────┐   │   │
│   │  │  Global Model    │   │  Server Validation Buffer      │   │   │
│   │  │  LocalMLP        │   │  500 samples (quality eval)    │   │   │
│   │  │  45→64→32→1      │   │  (no label info shared out)    │   │   │
│   │  └──────────────────┘   └────────────────────────────────┘   │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                 ↑ updates g_i           ↓ global model w(t)            │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                   FL CLIENTS  (N=100)                           │  │
│   │                                                                 │  │
│   │   Benign (70%)              Byzantine (30%)                    │  │
│   │   ┌────────────┐            ┌────────────────────────────┐     │  │
│   │   │ IoT Device │            │ Label Flipping Attack      │     │  │
│   │   │ Local SGD  │            │ or Gaussian Noise Attack   │     │  │
│   │   │ E=5 epochs │            │ (produces malicious g_i)   │     │  │
│   │   └────────────┘            └────────────────────────────┘     │  │
│   └─────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Local Model — `amfta/models/local_mlp.py`

```
Input (45 features)
      ↓
Linear(45→64) + ReLU
      ↓
Linear(64→32) + ReLU
      ↓
Linear(32→1) + Sigmoid
      ↓
P(attack) ∈ [0,1]
```

- **Parameters:** 5,057 trainable weights
- **Activation:** ReLU (hidden), Sigmoid (output)
- **Initialisation:** Xavier uniform
- **Loss function:** Binary Cross-Entropy (BCELoss)
- **Optimiser:** SGD with momentum=0.9

---

### 2. Trust Factor Computation — `amfta/aggregation/trust_factors.py`

#### Factor I — Gradient Similarity

```
S_i(t) = (cosine(g_i, ḡ_mean) + 1) / 2  ∈ [0,1]
```

- `ḡ_mean` = simple mean of all N client updates (not trust-weighted)
- Known limitation: with >40% Byzantine fraction and large Byzantine magnitude,
  the centroid tilts toward Byzantine direction; Factor II compensates

#### Factor II — Historical Reputation

```
H_i(t) = β · H_i(t-1) + (1-β) · S_i(t)
β = 0.9  →  ~10-round effective memory
H_i(0)  = 0.5 (neutral initialisation)
```

- Exponential Moving Average — clients cannot "reset" a bad history quickly
- Detects persistent Byzantine behaviour that single-round similarity misses

#### Borderline Detection

```
T̂_i = (α_s · S_i + α_h · H_i) / (α_s + α_h)

if T̂_i > τ_u = 0.55  → clearly BENIGN   (Q_i = 1.0)
if T̂_i < τ_l = 0.35  → clearly SUSPECT  (Q_i = 0.0)
otherwise             → BORDERLINE       (evaluate Factor III)
```

#### Factor III — Contribution Quality (Leave-One-Out)

```
Q_i(t) = A(w + ḡ_all) − A(w + ḡ_{-i})   (clipped to [0,1])
```

- `A(·)` = accuracy on 500-sample server validation buffer
- Only evaluated for borderline clients → reduces compute to O(|borderline|) evaluations per round
- Uses uniform mean ḡ (not trust-weighted) to avoid circular dependency

#### Final Trust Score

```
T_i(t) = α_s · S_i(t) + α_h · H_i(t) + α_q · Q_i(t)
        = 0.4 · S_i   + 0.3 · H_i   + 0.3 · Q_i
```

#### Aggregation

```
ḡ(t) = Σ_i T_i(t) · g_i(t) / Σ_i T_i(t)
w(t) = w(t-1) + η · ḡ(t)    (η=1.0)
```

No hard exclusion — all clients contribute proportionally to their trust score.

---

### 3. Baseline Aggregators — `amfta/aggregation/baselines.py`

| Method | Key Property | Limitation vs AMFTA |
|--------|-------------|---------------------|
| **FedAvg** | Uniform averaging | No Byzantine protection |
| **Krum** | Nearest-neighbour selection | Selects 1 update; fails under non-IID |
| **Trimmed Mean** | Coordinate-wise trim | Requires known Byzantine fraction |
| **FLTrust** | Server reference update | Requires clean server-side labelled data |

---

### 4. Byzantine Attacks — `amfta/attacks/`

#### Label Flipping (`label_flipping.py`)

```
y_poisoned = 1 - y_true  (full flip, fraction=1.0)
g_i = LocalSGD(w, X, y_poisoned) - w
```

Detection: produces updates in opposite direction to honest gradient → very
low cosine similarity → Factor I correctly assigns low S_i score.

#### Gaussian Noise (`gaussian_noise.py`)

```
g_i ~ N(0, σ² · I)
```

Detection: random noise is inconsistent across rounds → Factor II reputation
decays for Byzantine clients producing noise.

---

### 5. Data Pipeline — `amfta/data/`

```
NF-TON-IoT.csv
      ↓  preprocessing.py
MinMax normalise → binary labels → deduplicate → train/val/test split
      ↓  partitioning.py
Dirichlet(α=0.5) → N=100 non-IID client partitions
      ↓
data/partitions/seed_42/client_000.npz ... client_099.npz
data/server/val_buffer.npz  (500 samples for quality evaluation)
```

---

### 6. Federated Runner — `training/federated_runner.py`

Single-machine FL simulation:

```python
for round_t in range(1, T+1):

    # Phase A: Parallel client training (simulated sequentially)
    updates = {}
    for client_id in range(N_clients):
        if client_id in byzantine_ids:
            updates[client_id] = attack.get_update(global_model, local_data)
        else:
            updates[client_id] = local_train(global_model, local_data)

    # Phase B: Trust-weighted aggregation
    agg_update = amfta.aggregate(global_model, updates, val_buffer)

    # Phase C: Global model update
    global_model.apply_update(agg_update, lr=η_global)

    # Phase D: Evaluation on held-out test set
    metrics = evaluate_model(global_model, X_test, y_test)
```

---

## Design Decisions & Assumptions

### Parameter Values (from paper)

| Parameter | Value | Source |
|-----------|-------|--------|
| N (clients) | 100 | Paper §IV-A |
| T (rounds) | 100 | Paper §IV-B |
| E (local epochs) | 5 | Paper §IV-A |
| η_local | 0.01 | Paper §IV-A |
| η_global | 1.0 | Paper §IV-A |
| α_s | 0.4 | Paper §III-D |
| α_h | 0.3 | Paper §III-D |
| α_q | 0.3 | Paper §III-D |
| β | 0.9 | Paper §III-B |
| τ_l | 0.35 | Paper §III-C |
| τ_u | 0.55 | Paper §III-C |
| Dirichlet α | 0.5 | Paper §IV-A |

### Ambiguities Resolved

1. **Centroid for Gradient Similarity:** Paper does not specify whether the
   centroid uses trust-weighted or uniform mean. We use **uniform mean** to
   avoid circular dependency (trust depends on similarity; similarity depends
   on centroid; centroid depends on trust).

2. **Quality Evaluation Reference Aggregation:** Same circular dependency
   applies to Q_i. We use **uniform mean** of all updates as the reference
   for LOO evaluation.

3. **Model Parameter Count:** Paper states 3,393 parameters; our 45→64→32→1
   architecture yields 5,057. We implement the stated layer dimensions and
   expose the actual count via `model.num_parameters()`.

4. **Server Validation Buffer Stratification:** Paper states 500 samples;
   does not specify class balance. We draw from the validation split
   (stratified at preprocessing time, so ~38% normal / 62% attack).

5. **Gaussian Noise σ:** Not specified in paper. Default σ=1.0 with an
   optional `scale_to_honest` flag for magnitude-matched noise.

---

## Scalability Notes

- **Single-machine simulation:** All 100 clients run sequentially on one GPU/CPU.
  For true distributed execution, replace `local_train()` with gRPC/MQTT calls.
- **Quality evaluation cost:** O(|borderline|) forward passes per round.
  At 30% Byzantine fraction, ~20-30% of clients are typically borderline.
- **Memory:** Stores N full model state_dicts simultaneously → ~1MB per client
  for the default 5K-parameter model → 100MB for N=100. Scales well.
- **Throughput bottleneck:** Factor III LOO evaluation is the slowest step.
  Can be parallelised with multiprocessing (not yet implemented).
