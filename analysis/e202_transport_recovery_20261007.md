# E202 transport-only recovery before remote preparation or fit

Original scientific registration: `55fe9b68f7cc8405681f6c85d576979e067b55c3`,
journal350. Pack completed13:10:12UTC, input spec
`6d604eaaca830fa83c18ebb78986b028d338869123d5f8c10b2450dde1ec4309`.
Original controller PID78189 stopped immediately at its first nested SSH:
`ssh: connect to host 192.168.1.148 port 22: Operation not permitted`.
The sandbox inherited by the background Python process blocked transport.
Failed state/logs are preserved. No remote input, preparation or target fit
had started; this is not a fit retry.

Purpose-specific launcher `scripts/e202_transport_recovery_20261007.py` retains
the original controller except: new state/log filenames; reuse and verify the
already sealed bundle instead of packing again; prove original transport-only
failure and original controller SHA; require that the input path is absent on
all three remote hosts; provide `--launch` for one outside-sandbox launch.
All original scientific source/protocol/input spec are unchanged. Remote
commands, resource limits, admission, laptop release and all24fit are unchanged.
No automatic retries. The recovery is separately recorded before any remote fit.

The launcher must run with the platform's required network permission, after
the observed sandbox failure. If approval is pending, report it; do not claim
that remote preparation or GPU fitting has begun.
