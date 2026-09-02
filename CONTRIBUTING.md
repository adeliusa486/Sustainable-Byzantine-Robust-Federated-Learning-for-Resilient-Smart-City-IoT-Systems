# Contributing to AMFTA-FL

Thank you for your interest in contributing! This document outlines the process
for contributing code, documentation, and research extensions.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/amfta-fl.git`
3. **Create a branch**: `git checkout -b feature/my-new-aggregator`
4. **Install dev dependencies**: `make install-dev`
5. **Make your changes** with tests
6. **Run tests**: `make test-unit`
7. **Format code**: `make format`
8. **Push and create a Pull Request**

## Development Setup

```bash
git clone https://github.com/amfta-research/amfta-fl.git
cd amfta-fl
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install-dev
```

## Adding a New Aggregation Method

1. Create `amfta/aggregation/my_method.py` implementing `BaseAggregator`
2. Add to `AGGREGATOR_REGISTRY` in `amfta/aggregation/baselines.py`
3. Add to `FederatedRunner._build_aggregator()` in `training/federated_runner.py`
4. Write tests in `tests/test_my_method.py`
5. Document assumptions and limitations in the module docstring

## Adding a New Attack

1. Create `amfta/attacks/my_attack.py` inheriting `BaseAttack`
2. Decorate with `@register_attack("my_attack")`
3. Implement `get_update(global_model, local_data, ...)`
4. Add to `amfta/attacks/__init__.py`

## Code Style

- **Black** formatter (line length 95): `make format`
- **isort** imports
- **Type hints** on all public functions
- **Docstrings** (NumPy style) on all classes and public methods
- **No bare `except:`** — always specify exception type

## Pull Request Guidelines

- PRs should target the `develop` branch
- Include tests for all new functionality (>70% coverage)
- Update `IMPLEMENTATION_STATUS.md` if adding major components
- Provide a clear description of what changed and why

## Reporting Issues

Use GitHub Issues with:
- A minimal reproducible example
- Environment details (`python --version`, `torch.__version__`)
- Expected vs actual behaviour

## Research Extensions Welcome

We especially welcome:
- New trust aggregation mechanisms
- New Byzantine attack strategies
- Support for additional IoT datasets
- Distributed (multi-machine) FL simulation
- Communication compression
- Differential privacy integration
