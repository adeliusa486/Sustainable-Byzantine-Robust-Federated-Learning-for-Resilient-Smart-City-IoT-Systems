"""
Patch training/federated_runner.py to support the manuscript's configuration.

Adds, without disturbing existing behaviour:
  * ``model_class`` on RunConfig ('logistic' | 'mlp'), default 'mlp' so that
    every existing call site keeps its current semantics.
  * dispatch for the four aggregators added in amfta/aggregation/extra_baselines.
  * coalition-level crafting for the adaptive attack.

Idempotent: running it twice is a no-op.
"""

from pathlib import Path

P = Path("training/federated_runner.py")
src = P.read_text(encoding="utf8")

if "model_class" in src:
    print("already patched; nothing to do")
    raise SystemExit(0)

# --- 1. imports -----------------------------------------------------------
old_imp = "from amfta.models.local_mlp import LocalMLP"
new_imp = (
    "from amfta.models.local_mlp import LocalMLP\n"
    "from amfta.models.logistic import build_global_model\n"
    "import amfta.aggregation.extra_baselines  # noqa: F401  (registers rules)\n"
    "import amfta.attacks.adaptive  # noqa: F401  (registers 'adaptive')\n"
    "from amfta.aggregation.baselines import build_aggregator as _build_by_name"
)
assert old_imp in src
src = src.replace(old_imp, new_imp, 1)

# --- 2. RunConfig field ---------------------------------------------------
old_cfg = "    model_input_dim: int = 41"
new_cfg = (
    "    model_class: str = \"mlp\"   # 'logistic' reproduces the manuscript\n"
    "    model_input_dim: int = 41"
)
assert old_cfg in src
src = src.replace(old_cfg, new_cfg, 1)

# --- 3. model construction -----------------------------------------------
old_model = """        self.global_model = LocalMLP(
            input_dim=config.model_input_dim,
            hidden1=config.model_hidden1,
            hidden2=config.model_hidden2,
        ).to(self.device)"""
new_model = """        self.global_model = build_global_model(
            config.model_class,
            input_dim=config.model_input_dim,
            hidden1=config.model_hidden1,
            hidden2=config.model_hidden2,
        ).to(self.device)
        logger.info(
            "Global model: %r (d=%d)",
            self.global_model, self.global_model.num_parameters(),
        )"""
assert old_model in src
src = src.replace(old_model, new_model, 1)

# --- 4. aggregator dispatch ----------------------------------------------
old_disp = """        else:
            raise ValueError(f"Unknown method: {method}")"""
new_disp = """        elif method in ("normclip", "normclip_only", "median", "foolsgold"):
            return _build_by_name(method)

        elif method == "multikrum":
            f = int(cfg.num_clients * cfg.byzantine_fraction)
            return _build_by_name("multikrum", {"num_byzantine": f})

        else:
            raise ValueError(f"Unknown method: {method}")"""
assert old_disp in src
src = src.replace(old_disp, new_disp, 1)

# --- 5. coalition-level adaptive attack ----------------------------------
old_agg = "                # ── Phase B: Aggregation ───────────────────────────────────"
new_agg = """                # ── Phase A2: coalition-level attacks ──────────────────────
                # The adaptive attack is defined over the whole coalition: it
                # needs the round's honest updates to choose a direction and
                # to scale the perturbation. Per-client attacks are already
                # applied in Phase A.
                if cfg.attack_type == "adaptive" and self.byzantine_ids:
                    from amfta.attacks.adaptive import AdaptiveAGRAttack
                    honest = {c: u for c, u in updates.items()
                              if c not in self.byzantine_ids}
                    if honest:
                        crafter = self.attack if isinstance(
                            self.attack, AdaptiveAGRAttack) else AdaptiveAGRAttack()
                        updates.update(
                            crafter.craft_coalition(
                                self.global_model, honest,
                                sorted(self.byzantine_ids),
                            )
                        )

                # ── Phase B: Aggregation ───────────────────────────────────"""
assert old_agg in src
src = src.replace(old_agg, new_agg, 1)

P.write_text(src, encoding="utf8")
print("patched training/federated_runner.py")
