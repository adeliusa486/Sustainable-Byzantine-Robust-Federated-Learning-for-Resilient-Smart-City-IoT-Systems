"""
Per-round hook for AMFTA-S: sample participation and use per-client epochs.

Under every other method the loop trains all N clients for the same number of
epochs. Under AMFTA-S the controller decides, each round, which clients are
invited and how much local work each does, and hands the realised probabilities
to the aggregator for the importance correction.

Idempotent.
"""

from pathlib import Path

P = Path("training/federated_runner.py")
s = P.read_text(encoding="utf8")

if "self._sampled_this_round" in s:
    print("already patched")
    raise SystemExit(0)

old = """                for cid in range(cfg.num_clients):
                    X_c, y_c = self.client_tensors[cid]

                    if cid in self.byzantine_ids:
                        update = self.attack.get_update(
                            self.global_model, (X_c, y_c),
                            epochs=cfg.local_epochs,
                            lr=cfg.local_lr,
                            batch_size=cfg.local_batch_size,
                        )
                    else:
                        update = local_train(
                            self.global_model, X_c, y_c,
                            epochs=cfg.local_epochs,
                            lr=cfg.local_lr,
                            batch_size=cfg.local_batch_size,
                            device=self.device,
                        )

                    updates[cid] = update"""

new = """                # Under AMFTA-S the sustainability layer decides who is asked
                # to work this round and how much work each does. Every other
                # method trains the full cohort for a fixed number of epochs.
                if self.participation:
                    invited = [c for c in range(cfg.num_clients)
                               if self._rng.random() < self.participation.get(c, 1.0)]
                    if not invited:                      # never skip a whole round
                        invited = [max(self.participation,
                                       key=self.participation.get)]
                    self._sampled_this_round = invited
                else:
                    self._sampled_this_round = list(range(cfg.num_clients))

                for cid in self._sampled_this_round:
                    X_c, y_c = self.client_tensors[cid]
                    epochs = self.local_epochs_per_client.get(cid, cfg.local_epochs)

                    if cid in self.byzantine_ids:
                        update = self.attack.get_update(
                            self.global_model, (X_c, y_c),
                            epochs=epochs,
                            lr=cfg.local_lr,
                            batch_size=cfg.local_batch_size,
                        )
                    else:
                        update = local_train(
                            self.global_model, X_c, y_c,
                            epochs=epochs,
                            lr=cfg.local_lr,
                            batch_size=cfg.local_batch_size,
                            device=self.device,
                        )

                    updates[cid] = update

                if self.participation:
                    # the aggregator needs the assigned probabilities, not the
                    # realised draw, to weight by 1/p_i
                    self.aggregator.participation = {
                        c: self.participation[c] for c in self._sampled_this_round}"""

assert old in s, "round loop not in the expected form"
s = s.replace(old, new, 1)

# a dedicated RNG so participation sampling does not disturb training seeds
anchor = "        self.participation = {}"
s = s.replace(anchor,
              "        import numpy as _np\n"
              "        self._rng = _np.random.default_rng(config.seed + 9973)\n"
              "        self._sampled_this_round = []\n" + anchor, 1)

P.write_text(s, encoding="utf8")
print("patched the per-round loop for participation sampling and per-client epochs")
