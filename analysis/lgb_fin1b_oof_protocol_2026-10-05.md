# LGB5 FIN1b component OOF

Registration 283 reproduces the five-seed production regressor configuration on six month-pair splits. It is retrospective component validation: the fixed 2319 rounds were originally selected using January/July validation. It is not independent nested validation, a complete v12048 OOF prediction, or a new submission.

The unchanged actual Hetzner trainer is snapshotted. Fixed seed sequences are 1000–1004, bagging 2000–2004, and feature-fraction 3000–3004. Only 2025 rows enter each view; heldout targets become null. Encodings and the NM proxy are recomputed from ten training months. Source, input, production evidence, runtime, feature names, parameters, and output hashes are recorded and checked.

Each fold publishes only after all five fits complete, with exact mean, per-seed predictions, ID, target, and month. Interrupted attempts are preserved in quarantine. Completed outputs are validated without overwriting. There is no quality-based stopping, tuning, or automatic upload.

Use the server `.venv/bin/python` with `scripts/lgb_fin1b_oof.py prepare` then `run`, `--root /root/prc-2026`, `--work /root/prc-2026/data/lgb_fin1b_oof_20261005`, and the full registration commit. Both commands run in one systemd service with CPUAffinity=0,2,4,6,8,10, Nice=10, CPUQuota=600%, MemoryMax=24G, MemorySwapMax=0. The existing `REG_OVERRIDE` hook changes only `num_threads` to six; OpenMP is also capped at six with passive waiting. Shared source files and FlightFinder are untouched.

Ten synthetic contracts pass. Independent review found no material blocker. The historical production log confirms all five fits and exact completion format. Initial duration estimate is 4–7 hours; the first completed fold will provide measured timing.
