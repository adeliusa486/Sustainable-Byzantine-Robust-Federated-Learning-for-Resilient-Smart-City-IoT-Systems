# Reproducing the manuscript's tables

This repository could not produce the configuration the manuscript describes.
The gaps are listed below, together with what was added to close them and how
to run the study.

## What was missing, and what now exists

| Manuscript element | Status before | Added |
|---|---|---|
| Logistic-regression model | absent (only `LocalMLP`) | `amfta/models/logistic.py` |
| NormClip-Only ablation arm | absent | `amfta/aggregation/extra_baselines.py` |
| Coordinate-wise Median | absent | same |
| Multi-Krum (as a selectable method) | flag only | same |
| FoolsGold | absent | same |
| Adaptive / AGR-tailored attack | absent | `amfta/attacks/adaptive.py` |
| Dirichlet α = 0.1 condition | not runnable (fixed α = 0.5 partitions) | `repartition` flag on `RunConfig` |
| Study driver for every table | absent | `experiments/run_paper_study.py` |
| Equivalence tests, Holm, trend test | absent | `experiments/paper_stats.py` |

AMFTA-ND (`amfta_noq`) and median-norm rescaling were already implemented; they
had simply never been run, since no result file for either exists in the repo
history.

The two runner patches are idempotent and were applied by `patch_runner.py`
and `patch_runner2.py`. Both default to the previous behaviour
(`model_class="mlp"`, `repartition=False`), so nothing that worked before
changes.

## Running it

```bash
python experiments/run_paper_study.py --block clean       --model logistic
python experiments/run_paper_study.py --block labelflip   --model logistic
python experiments/run_paper_study.py --block gaussian    --model logistic
python experiments/run_paper_study.py --block adaptive    --model logistic
python experiments/run_paper_study.py --block extras      --model logistic
python experiments/run_paper_study.py --block normclip    --model logistic
python experiments/run_paper_study.py --block alpha01     --model logistic
python experiments/run_paper_study.py --block scalability --model logistic
```

Then:

```bash
python experiments/paper_stats.py --results results_paper --out tables
```

Blocks are resumable: a configuration already present in the block's CSV is
skipped, so the study can be run in pieces and interrupted safely. The default
seed list has ten entries and begins with the manuscript's three
(42, 123, 456), so a three-seed run is a strict prefix of a ten-seed one.

## Three things to settle before the tables are trustworthy

**1. The model dimension.** The manuscript states d = 41 for a logistic
regression. The processed tensors in `data/processed` carry 45 features, so a
logistic regression on them has d = 46. Either the feature-selection step that
reduces 45 columns to 40 is missing from the preprocessing pipeline, or the
manuscript's d is wrong. `LogisticRegression.num_parameters()` reports the true
count for whatever is actually built; the table should be filled from that
rather than asserted separately. Note that an earlier version of the manuscript
recovered from git history states d = 7,681 for the same "logistic regression",
so this number has already changed twice.

**2. The class balance.** The manuscript's experimental-configuration table
states "Class balance (train) 54.1% normal / 45.9% attack", and a passage in
the detection-metrics section reasons from those figures about what a constant
classifier would score. The processed test set in this repository is **62.0%
class 1**. A zero-initialised model that predicts the positive class everywhere
scores exactly 0.620 accuracy with recall 1.0 on it, which is the floor every
run starts from. Whatever the correct balance is, the manuscript's arithmetic
about constant predictors does not hold on this data.

**3. The dataset may be close to separable.** With this preprocessing, methods
reach 99%+ accuracy on the released logs even under 30% label flipping. A
benchmark on which undefended FedAvg is barely degraded does not discriminate
between aggregation rules, and reviewers will notice. Before re-running the
full study it is worth checking how much of the label information sits in one
or two features, and whether the deduplication step has left near-duplicate
rows spanning the train/test split.

## What the new components do and do not guarantee

The aggregators were unit-tested on controlled updates: with 70 honest clients
sharing a direction and 30 colluding clients pushing against it, FoolsGold,
coordinate-wise Median and NormClip-Only all recover the honest direction
(cosine +0.99 or better) and reject the coalition. That verifies the mechanics,
not the empirical claims.

The adaptive attack implements the coalition-level formulation the manuscript
describes, including a binary search on the perturbation magnitude against an
acceptance oracle. Without an oracle it defaults to the largest magnitude that
median-norm rescaling leaves untouched. Its strength depends on `gamma_init`,
`jitter` and `knowledge`, all of which must be reported in the paper — the
manuscript's own limitations section asks for exactly that specification.

Nothing here reproduces the numbers currently in the manuscript. It makes it
possible to generate numbers, which was not previously the case.
