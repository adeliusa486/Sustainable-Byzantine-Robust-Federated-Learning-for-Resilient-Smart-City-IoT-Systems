# Data

The experiments use **NF-TON-IoT**, the NetFlow-formatted version of TON_IoT
released by UQ/UNSW. The dataset files are not in this repository — the raw
parquet is 209 MB and the processed tensors are roughly 580 MB, both far over
GitHub's limits.

## What the study uses

| | |
|---|---|
| Source file | `NF-TON-IoT.parquet` (209 MB) |
| Features | 41 NetFlow fields (see `data/processed/feature_names.txt` after preprocessing) |
| Task | binary, normal vs attack |
| Split | 9,195,116 train / 1,313,588 validation / 2,627,177 test |
| Scaling | min–max to [0,1], fitted on the training split only |
| Partition | Dirichlet, α = 0.5, 100 clients |

## Getting it

Download NF-TON-IoT from the University of Queensland's machine-learning
datasets collection (`staff.itee.uq.edu.au/marius/NIDS_datasets/`), place
`NF-TON-IoT.parquet` in `data/raw/`, then:

```bash
python scripts/setup_data.py --dataset nf-ton-iot
```

This writes `data/processed/{train,val,test}.npz`, the server validation buffer,
and the per-seed Dirichlet partitions.

## A warning about the synthetic path

`scripts/setup_data.py` also has a `--synthetic` flag that generates
Beta-distributed data for CI smoke tests. **It reproduces nothing in the paper.**
A logistic regression reaches 100% accuracy on it with no attacker present,
because two well-separated Beta distributions over 45 independent features are
almost perfectly linearly separable. An earlier published version of this
repository was accidentally built on that path, which is why the warning is
here. Use it only to check that the code runs.

## Verifying you have the right data

```python
import numpy as np
z = np.load("data/processed/test.npz")
assert z["X"].shape == (2627177, 41)
# real NetFlow features are correlated; the synthetic ones are not
r = np.corrcoef(z["X"][:5000].T)
print(abs(r[np.triu_indices(41, 1)]).mean())   # ~0.26 real, ~0.005 synthetic
```
