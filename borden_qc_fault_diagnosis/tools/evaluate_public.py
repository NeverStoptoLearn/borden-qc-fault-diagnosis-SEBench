import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ALLOWED = {
    "normal", "spike", "drift", "stuck_zero", "missing", "unit_error",
    "time_shift", "coordinate_or_depth_error", "true_plume_arrival", "negative"
}
FAULT_LABELS = ALLOWED - {"normal", "true_plume_arrival"}


def macro_f1(y_true, y_pred, labels):
    vals = []
    for lab in labels:
        yt = np.asarray([x == lab for x in y_true])
        yp = np.asarray([x == lab for x in y_pred])
        tp = float(np.sum(yt & yp))
        fp = float(np.sum(~yt & yp))
        fn = float(np.sum(yt & ~yp))
        if tp + fp + fn == 0:
            continue
        vals.append(2 * tp / max(2 * tp + fp + fn, 1e-9))
    return float(np.mean(vals)) if vals else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answer", default="answer.csv")
    p.add_argument("--cleaned", default="cleaned_monitoring_data.csv")
    args = p.parse_args()
    labels = pd.read_csv("public_fault_labels_small.csv")
    ans = pd.read_csv(args.answer)
    merged = labels.merge(ans, on="record_id", how="left", suffixes=("_true", "_pred"))
    pred = merged["anomaly_label_pred"].fillna("normal").where(lambda s: s.isin(ALLOWED), "normal")
    true = merged["anomaly_label_true"]
    y_true_fault = true.isin(FAULT_LABELS)
    y_pred_fault = pred.isin(FAULT_LABELS)
    tp = float(np.sum(y_true_fault & y_pred_fault))
    fp = float(np.sum(~y_true_fault & y_pred_fault))
    fn = float(np.sum(y_true_fault & ~y_pred_fault))
    f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
    mf1 = macro_f1(true, pred, sorted(ALLOWED))
    plume = true.eq("true_plume_arrival")
    plume_fp = float(np.mean(pred[plume].isin(FAULT_LABELS))) if plume.any() else 0.0
    print(f"PUBLIC_RECORD_FAULT_F1 {f1:.3f}")
    print(f"PUBLIC_TYPE_MACRO_F1 {mf1:.3f}")
    print(f"PUBLIC_TRUE_PLUME_FALSE_FAULT_RATE {plume_fp:.3f}")
    if Path(args.cleaned).exists():
        clean_ref = pd.read_csv("public_clean_reference_subset.csv")
        cleaned = pd.read_csv(args.cleaned)
        m = clean_ref.merge(cleaned, on="record_id", how="left")
        rmse = np.sqrt(np.nanmean((m["cleaned_concentration_mg_L"] - m["clean_concentration_mg_L"]) ** 2))
        raw_rmse = np.sqrt(np.nanmean((m["observed_concentration_mg_L"] - m["clean_concentration_mg_L"]) ** 2))
        print(f"PUBLIC_CLEAN_RMSE {rmse:.6f}")
        print(f"PUBLIC_RAW_RMSE {raw_rmse:.6f}")


if __name__ == "__main__":
    main()
