# Setup & Installation Guide

## Requirements

| Component | Version |
|-----------|---------|
| Python    | ≥ 3.9   |
| PyTorch   | ≥ 2.1.0 |
| CUDA (optional) | 11.8 / 12.1 |
| RAM       | ≥ 8 GB  |
| Disk      | ≥ 5 GB  (15 GB with full dataset) |

---

## Installation

### Option A — Standard Install (Recommended)

```bash
git clone https://github.com/amfta-research/amfta-fl.git
cd amfta-fl

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option B — GPU (CUDA 12.1)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Option C — Docker

```bash
docker build -t amfta-fl:latest .
docker run --rm -it amfta-fl:latest bash
# Inside container: make smoke
```

---

## Data Setup

### Quick Start — Synthetic Data (No Download)

```bash
python scripts/setup_data.py --synthetic
```

This generates 100,000 synthetic samples partitioned across 100 non-IID
clients. Results will differ from paper figures but verify the pipeline works.

### Full Experiment — TON_IoT Dataset

1. **Download** the Network Flow variant from UNSW:
   https://research.unsw.edu.au/projects/toniot-datasets

2. **Place** `NF-TON-IoT.csv` in `data/raw/`

3. **Run setup:**
   ```bash
   python scripts/setup_data.py --preprocess
   # Runs: preprocessing → partitioning → server buffer creation
   ```

4. **Verify:**
   ```bash
   python scripts/setup_data.py --verify
   ```

---

## Running Experiments

### Smoke Test (< 2 minutes)

```bash
make smoke
# Equivalent to:
python experiments/run_main.py \
  --method amfta --use_synthetic \
  --num_rounds 5 --num_clients 10 --seeds 42
```

### Single Method Run

```bash
python experiments/run_main.py \
  --method amfta \
  --byzantine_fraction 0.30 \
  --attack label_flipping \
  --num_rounds 100 \
  --seeds 42 123 456 789 1024
```

### Full Paper Reproduction (4-8 hours on GPU)

```bash
python experiments/run_main.py --all
python experiments/run_ablation.py
python experiments/run_scalability.py
python evaluation/visualize.py --results_dir results --output_dir figures
```

---

## API Server

```bash
# Start development server
make api
# Open: http://localhost:8000/docs

# Test with a sample request
make api-test
```

---

## Running Tests

```bash
make test-unit          # Fast unit tests (~15 seconds)
make test-integration   # Full FL simulation (~2 minutes)
make test-coverage      # With HTML coverage report
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'amfta'`
```bash
export PYTHONPATH=/path/to/amfta-fl
# or: pip install -e .
```

### CUDA out of memory
```bash
export CUDA_VISIBLE_DEVICES=""   # Force CPU
# or reduce num_clients / batch_size
```

### `FileNotFoundError: data/partitions/seed_42/`
```bash
python scripts/setup_data.py --synthetic   # or --preprocess
```

### Slow quality evaluation (Factor III)
The LOO evaluation runs on CPU. Set `use_quality_eval=False` to run the
AMFTA-SH ablation variant (similarity + history only) for faster experiments.

### Test failures with `pytest --timeout`
```bash
pip install pytest-timeout
```
