"""Gate print for two CV prediction dirs: python src/gate.py <control dir> <candidate dir> (FEATURES = the table for decided rows).
Per-fold lines and the summary of fold_metrics.compare (LB-equivalent seconds; decided rows = the gate since journal 192)."""
import sys

sys.path.insert(0, "src")
from fold_metrics import compare  # noqa: E402

dd, tot, small, trim, lower, n, st = compare(sys.argv[1], sys.argv[2])
print(f"SUMMARY {sys.argv[2]} vs {sys.argv[1]}: decided {dd:+.3f} (lower in {lower}/{n}) | strict {st:+.3f} | all {tot:+.3f} | "
      f"y<=7200 {small:+.3f} | trimmed {trim:+.3f}")
