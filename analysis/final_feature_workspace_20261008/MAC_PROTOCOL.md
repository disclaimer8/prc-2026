# Mac execution adaptation, 8 October 2026

Protocol `final-feature-Mac12threads-24GiB-4GiBdisk-2h-v1`.
The user's explicit request to use the Mac prioritizes this execution path.
Unchanged reviewed feature producers and data recipe from the main protocol;
same seal/metadata/IDs/labels/outputs, no fits and no new model design.

Use a fresh Mac WORK. Parent runs registered metadata seal, then one child
process group with12 Polars/OMP threads, BLAS1. Parent checks group RSS every
second (24GiB ceiling), disk4GiB and wall2h, kills only its owned child group
on failure, preserves partials, never retries. No live environment changes.
Seal retains its existing 900s/3GiB phase checks. Root runs parent under
caffeinate/nice10; live PID and per-stage receipts establish actual work.

Disk snapshot11:52UTC:13,803,520,000 bytes available; allowlisted raw/static
inputs455,718,989 bytes. Historical E106/E113 are each~914MB, FIN1b971MB;
final population grows13.4%. Preserve all four final tables. Delete only fresh
intermediates after their last consumer: features_final afterE28, E28 afterE77,
E77 afterE106, E163/E51/E157/E165 afterFIN1b. Their SHA receipts remain. This
keeps the estimated peak~8.5GB below available capacity while preserving4GiB;
the live disk guard, not the estimate, decides. Existing inputs/outputs are
never deleted or modified. No symlink/hardlink staging. The25GiB Linux
admission in the original generic helper is superseded only for this explicit
pruned Mac schedule; the original source remains frozen.

Root preregisters this adapter together with the feature recipe before seal.
One fixed all-four-month feature construction, no CV/RMSE prior or fit. Complete
requires the same four schema/ID/label/hash checks as the reviewed generic
feature builder and a successful monitored child exit. The scientific result
is not assessed at this engineering step. Actual runtime remains to be measured.
