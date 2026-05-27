import argparse

import numpy as np
import pandas as pd

ALLOWED = {
    "normal", "spike", "drift", "stuck_zero", "missing", "unit_error",
    "time_shift", "coordinate_or_depth_error", "true_plume_arrival", "negative"
}


def robust_flags(g):
    vals = g["observed_concentration_mg_L"].astype(float)
    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med)) + 1e-9
    z = np.abs(vals - med) / (1.4826 * mad + 1e-9)
    roll = vals.rolling(5, center=True, min_periods=1).median()
    jump = np.abs(vals - roll)
    jmad = np.nanmedian(np.abs(jump - np.nanmedian(jump))) + 1e-9
    labels = []
    cleaned = []
    actions = []
    for v, zz, jj, rr in zip(vals, z, jump, roll):
        if not np.isfinite(v):
            labels.append("missing"); cleaned.append(float(rr) if np.isfinite(rr) else 0.0); actions.append("rolling_median_fill")
        elif v < -1e-12:
            labels.append("negative"); cleaned.append(max(float(rr), 0.0)); actions.append("clip_negative")
        elif abs(v) < 1e-12 and med > 0.01:
            labels.append("stuck_zero"); cleaned.append(float(rr)); actions.append("rolling_median_fill")
        elif v > max(5.0, 500 * max(med, 1e-5)):
            labels.append("unit_error"); cleaned.append(float(v) / 100.0); actions.append("scale_down")
        else:
            labels.append("normal"); cleaned.append(max(float(v), 0.0)); actions.append("none")
    return labels, cleaned, actions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="eval_monitoring_noisy.csv")
    p.add_argument("--answer", default="answer.csv")
    p.add_argument("--cleaned", default="cleaned_monitoring_data.csv")
    p.add_argument("--report", default="fault_report.md")
    args = p.parse_args()
    df = pd.read_csv(args.input)
    answers = []
    cleans = []
    for wid, g in df.sort_values(["well_id", "time_days"]).groupby("well_id", sort=False):
        labels, cleaned, actions = robust_flags(g)
        for (_, r), lab, cv, act in zip(g.iterrows(), labels, cleaned, actions):
            answers.append({
                "record_id": r["record_id"],
                "well_id": r["well_id"],
                "time_days": r["time_days"],
                "anomaly_label": lab if lab in ALLOWED else "normal",
                "confidence": 0.75 if lab != "normal" else 0.55,
                "event_id": "E_AUTO",
                "root_cause": "robust univariate screen" if lab != "normal" else "none",
            })
            cleans.append({
                "record_id": r["record_id"],
                "well_id": r["well_id"],
                "time_days": r["time_days"],
                "cleaned_concentration_mg_L": max(float(cv), 0.0),
                "cleaning_action": act,
            })
    pd.DataFrame(answers).to_csv(args.answer, index=False)
    pd.DataFrame(cleans).to_csv(args.cleaned, index=False)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("# Fault diagnosis report\n\nBaseline robust univariate detector. Improve with Borden travel-time and neighbor-well consistency.\n")
    print("Wrote answer.csv, cleaned_monitoring_data.csv, fault_report.md")


if __name__ == "__main__":
    main()
