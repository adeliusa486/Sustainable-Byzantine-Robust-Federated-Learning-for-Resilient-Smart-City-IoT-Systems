"""
Wire AMFTA-S into the federated runner.

Adds partial participation with per-client local epochs, both set by the
sustainability layer, and hands the sampled probabilities to the aggregator so
it can apply the importance correction. Reputation of clients the scheduler did
not invite is held rather than decayed: absence is scheduled, not suspicious.

Also adds ``spoof_fraction``, which makes a share of the Byzantine clients
understate their resource cost, so Experiment 8 can be run.

Idempotent; defaults leave every existing call site unchanged.
"""

from pathlib import Path

P = Path("training/federated_runner.py")
s = P.read_text(encoding="utf8")

if "amfta_s" in s:
    print("already patched")
    raise SystemExit(0)

# --- config fields ----------------------------------------------------------
anchor = "    repartition: bool = False  # regenerate the split at alpha_dirichlet"
add = (anchor + "\n"
       "    p_min: float = 0.25          # participation floor for AMFTA-S\n"
       "    sched_gamma: float = 1.0     # participation exponent\n"
       "    spoof_fraction: float = 0.0  # share of Byzantine clients understating cost\n"
       "    spoof_factor: float = 0.05   # how far they understate it")
assert anchor in s, "run patch_runner2.py first"
s = s.replace(anchor, add, 1)

# --- imports ----------------------------------------------------------------
imp = "from amfta.aggregation.amfta import AMFTAAggregator"
s = s.replace(imp, imp + "\nfrom amfta.aggregation.amfta_s import (\n"
              "    AMFTASAggregator, ResourceProfile, compute_scores, schedule,\n)", 1)

# --- aggregator dispatch ----------------------------------------------------
disp = '        if method in ("amfta", "amfta_noq"):'
new_disp = ('        if method == "amfta_s":\n'
            '            return AMFTASAggregator(num_clients=cfg.num_clients,\n'
            '                                    beta=cfg.beta, p_min=cfg.p_min)\n\n'
            + disp)
assert disp in s
s = s.replace(disp, new_disp, 1)

# --- build the schedule once the partitions are known -----------------------
anchor2 = "        # Assign Byzantine clients"
sched = '''        # Resource-aware schedule (AMFTA-S only). The sustainability layer
        # sees client resource reports and nothing else; it never sees trust.
        self.participation = {}
        self.local_epochs_per_client = {}
        if config.method == "amfta_s":
            n_samples = {cid: len(y) for cid, (_, y) in self.client_tensors.items()}
            self.resource_profile = ResourceProfile(config.num_clients, seed=config.seed)
            misreport = None
            if config.spoof_fraction > 0:
                byz = sorted(assign_byzantine_clients(
                    config.num_clients, config.byzantine_fraction, config.seed))
                k = int(round(config.spoof_fraction * len(byz)))
                misreport = {c: config.spoof_factor for c in byz[:k]}
            scores = compute_scores(
                n_samples, self.resource_profile,
                d=self.global_model.num_parameters(),
                epochs=config.local_epochs, batch=config.local_batch_size,
                misreport=misreport)
            self.participation, self.local_epochs_per_client = schedule(
                scores, p_min=config.p_min, gamma=config.sched_gamma,
                base_epochs=config.local_epochs)
            logger.info("AMFTA-S schedule: mean p=%.3f, mean local epochs=%.2f",
                        sum(self.participation.values()) / len(self.participation),
                        sum(self.local_epochs_per_client.values()) / len(self.local_epochs_per_client))

'''
assert anchor2 in s
s = s.replace(anchor2, sched + anchor2, 1)

P.write_text(s, encoding="utf8")
print("patched RunConfig, imports, dispatch and schedule construction")
print("NOTE: the per-round sampling hook is applied by patch_runner4.py")
