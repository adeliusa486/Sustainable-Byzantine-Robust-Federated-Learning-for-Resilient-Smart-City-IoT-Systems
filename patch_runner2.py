"""
Second runner patch: allow re-partitioning at an arbitrary Dirichlet alpha.

``load_partitions`` reads the fixed partitions in ``data/partitions/seed_*``,
which were generated at alpha = 0.5. The manuscript also reports a severe
heterogeneity condition at alpha = 0.1, which therefore could not be run.

This adds ``repartition`` to RunConfig. When True the client split is
regenerated from the processed training set at ``cfg.alpha_dirichlet``,
seeded by ``cfg.seed`` so it stays reproducible. Default False, so existing
call sites keep loading the stored partitions and nothing changes for them.

Idempotent.
"""

from pathlib import Path

P = Path("training/federated_runner.py")
src = P.read_text(encoding="utf8")

if "repartition" in src:
    print("already patched; nothing to do")
    raise SystemExit(0)

old_cfg = '    model_class: str = "mlp"   # \'logistic\' reproduces the manuscript'
new_cfg = ('    repartition: bool = False  # regenerate the split at alpha_dirichlet\n'
           + old_cfg)
assert old_cfg in src, "run patch_runner.py first"
src = src.replace(old_cfg, new_cfg, 1)

old_load = """            self.client_data = load_partitions(
                partition_dir=Path(cfg.partition_dir),
                seed=cfg.seed,
                num_clients=cfg.num_clients,
            )"""
new_load = """            if cfg.repartition:
                # Regenerate the client split at the requested concentration.
                # The stored partitions under data/partitions were generated at
                # alpha=0.5 only, so any other heterogeneity level must be
                # produced here.
                logger.info(
                    "Repartitioning %d clients at Dirichlet alpha=%.3f (seed %d)",
                    cfg.num_clients, cfg.alpha_dirichlet, cfg.seed,
                )
                self.client_data = dirichlet_partition(
                    data["X_train"], data["y_train"],
                    num_clients=cfg.num_clients,
                    alpha=cfg.alpha_dirichlet,
                    seed=cfg.seed,
                )
            else:
                self.client_data = load_partitions(
                    partition_dir=Path(cfg.partition_dir),
                    seed=cfg.seed,
                    num_clients=cfg.num_clients,
                )"""
assert old_load in src
src = src.replace(old_load, new_load, 1)

P.write_text(src, encoding="utf8")
print("patched training/federated_runner.py (repartition)")
