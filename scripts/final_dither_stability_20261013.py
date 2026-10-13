"""Journal 468: stability of the final file to a fresh +/-12 s dither draw.

Label-free, 0 fits. Combines the measured 8 Oct sensitivity of the v870
premix (RMS change 16.20 s outside the 43 tail rows; board band +/-0.08 s
at 1 sigma on January/July) with registered scenarios for the ordered-
trajectory neural member (weight 0.35 in the final premix; its prepared
tensors and OOF checkpoints are on the Home machine, so a direct
re-inference under shifted token times is not possible on the Mac).
The tabular part carries weight 0.65. Post-premix stages pass a premix
shift through at 0.76 (v12076: premix RMS shift 6.3 s -> file 4.849 s).
The RMSE band scales in proportion to the final-file RMS change from the
measured 16.20 s / +/-0.08 s pair, rescaled from 344,841 to 670,790 rows.
The 62 served E156 rows are priced exactly at scenario RMS changes.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "analysis/final_dither_stability_20261013"
MEASURED_PREMIX_RMS = 16.20  # s, 8 Oct audit section 3, excl. 43 tail rows
MEASURED_BAND = 0.08  # s at 1 sigma, January/July, 344,841 rows
PASS_THROUGH = 0.76  # post-premix stages, v12076 receipt
W_TAB, W_NN = 0.65, 0.35
N_FINAL = 670_790
N_JANJUL = 344_841
TAIL_ROWS = 62


def main() -> None:
    scale_rows = (N_JANJUL / N_FINAL) ** 0.5
    scenarios = []
    for nn_rms in (0.0, 8.0, 16.0, 32.0, 64.0):
        premix = ((W_TAB * MEASURED_PREMIX_RMS) ** 2 + (W_NN * nn_rms) ** 2) ** 0.5
        final_rms = PASS_THROUGH * premix
        band = MEASURED_BAND * (final_rms / MEASURED_PREMIX_RMS) * scale_rows
        scenarios.append(
            {
                "nn_response_rms_s": nn_rms,
                "premix_rms_change_s": premix,
                "final_rms_change_s": final_rms,
                "rmse_band_1sigma_s": band,
            }
        )
    # NN response at which the 1-sigma band reaches 0.05 s (registered rule).
    target_final = (
        0.05 / MEASURED_BAND * MEASURED_PREMIX_RMS / scale_rows
    )
    target_premix = target_final / PASS_THROUGH
    nn_threshold = (
        max(target_premix**2 - (W_TAB * MEASURED_PREMIX_RMS) ** 2, 0.0) ** 0.5
        / W_NN
    )
    # Tail: 62 served rows; even a large per-row change costs little overall.
    tail = {}
    for row_rms in (100.0, 300.0, 700.0, 1627.0):
        dmse = TAIL_ROWS * row_rms**2 / N_FINAL
        tail[str(int(row_rms))] = {
            "dmse_per_row": dmse,
            "drmse_at_231": (231.0**2 + dmse) ** 0.5 - 231.0,
        }
    results = {
        "protocol": "final-dither-stability-20261013-v1",
        "measured_inputs": {
            "premix_rms_change_s": MEASURED_PREMIX_RMS,
            "board_band_1sigma_s": MEASURED_BAND,
            "pass_through": PASS_THROUGH,
            "weights": {"tabular": W_TAB, "neural": W_NN},
        },
        "scenarios": scenarios,
        "nn_response_rms_at_band_0.05_s": nn_threshold,
        "tail_62_rows": tail,
        "direct_nn_reinference": (
            "not possible on the Mac: E197 prepared tensors (19 GB) and the "
            "OOF checkpoints are on the Home machine; the NN term above is "
            "therefore scenario-based, not measured"
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False) + "\n"
    )
    for s in scenarios:
        print(
            f"NN {s['nn_response_rms_s']:5.1f} -> premix {s['premix_rms_change_s']:5.2f} "
            f"-> final {s['final_rms_change_s']:5.2f} -> band +/-{s['rmse_band_1sigma_s']:.4f}"
        )
    print(f"NN response at band 0.05 s: {nn_threshold:.1f} s")
    print(f"tail 700 s/row: drmse {tail['700']['drmse_at_231']:.4f}")


if __name__ == "__main__":
    main()
