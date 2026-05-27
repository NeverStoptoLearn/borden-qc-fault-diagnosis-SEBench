import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ALLOWED = {
    "normal", "spike", "drift", "stuck_zero", "missing", "unit_error",
    "time_shift", "coordinate_or_depth_error", "true_plume_arrival", "negative"
}
FAULT_LABELS = ALLOWED - {"normal", "true_plume_arrival"}


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def macro_f1(y_true, y_pred, labels):
    vals = []
    by_label = {}
    for lab in labels:
        yt = np.asarray([x == lab for x in y_true])
        yp = np.asarray([x == lab for x in y_pred])
        tp = float(np.sum(yt & yp))
        fp = float(np.sum(~yt & yp))
        fn = float(np.sum(yt & ~yp))
        f1 = 0.0 if tp + fp + fn == 0 else 2 * tp / max(2 * tp + fp + fn, 1e-9)
        if tp + fp + fn > 0:
            vals.append(f1)
        by_label[lab] = f1
    return (float(np.mean(vals)) if vals else 0.0), by_label


def binary_f1(y_true_bool, y_pred_bool):
    yt = np.asarray(y_true_bool, dtype=bool)
    yp = np.asarray(y_pred_bool, dtype=bool)
    tp = float(np.sum(yt & yp))
    fp = float(np.sum(~yt & yp))
    fn = float(np.sum(yt & ~yp))
    return 0.0 if tp + fp + fn == 0 else 2 * tp / max(2 * tp + fp + fn, 1e-9)


def task_result(score, max_score, threshold=0.5):
    score = round(float(score), 3)
    max_score = round(float(max_score), 3)
    rate = 0.0 if max_score <= 0 else round(max(0.0, min(1.0, score / max_score)), 3)
    return {"score": score, "max_score": max_score, "pass_rate": rate, "passed": bool(rate >= threshold)}


def scaled_score(metric, baseline, target, max_score, gamma=1.5):
    metric = float(metric)
    if not np.isfinite(metric) or metric <= baseline:
        return 0.0
    if metric >= target:
        return float(max_score)
    x = (metric - baseline) / max(target - baseline, 1e-9)
    return float(max_score) * (max(0.0, min(1.0, x)) ** gamma)



def release_gate(metric, baseline, target):
    """Return 0 before the scoring baseline, 1 after target, and linear release in between."""
    metric = float(metric)
    if not np.isfinite(metric) or metric <= baseline:
        return 0.0
    if metric >= target:
        return 1.0
    return float((metric - baseline) / max(target - baseline, 1e-9))


def event_score(labels, pred):
    merged = labels[["record_id", "event_id", "anomaly_label"]].merge(pred[["record_id", "anomaly_label_pred"]], on="record_id", how="left")
    scores = []
    event_rows = []
    for eid, g in merged.groupby("event_id"):
        true_lab = str(g["anomaly_label"].mode().iloc[0])
        if true_lab == "normal":
            continue
        if true_lab in {"spike", "negative"}:
            continue
        pred_labs = g["anomaly_label_pred"].fillna("normal").astype(str)
        if true_lab == "true_plume_arrival":
            coverage = float(np.mean(pred_labs.eq("true_plume_arrival")))
        else:
            coverage = float(np.mean(pred_labs.isin(FAULT_LABELS)))
        majority_hit = float(pred_labs.mode().iloc[0] == true_lab) if len(pred_labs) else 0.0
        type_fraction = float(np.mean(pred_labs.eq(true_lab)))
        # Event credit is deliberately stricter than record F1: weak spike
        # screens should not get high event-level credit for long drift,
        # time-shift, coordinate/depth, or true-plume-arrival events.
        s = 0.45 * min(1.0, coverage / 0.55) + 0.35 * majority_hit + 0.20 * min(1.0, type_fraction / 0.45)
        scores.append(s)
        event_rows.append({"event_id": eid, "true_label": true_lab, "event_score": s})
    return (float(np.mean(scores)) if scores else 0.0), event_rows


def evaluate(submission_dir, case_dir, scoring_dir, output):
    sub = Path(submission_dir)
    scoring = Path(scoring_dir)
    labels = safe_read_csv(scoring / "hidden_fault_labels.csv")
    clean = safe_read_csv(scoring / "hidden_clean_concentrations.csv")
    ans = safe_read_csv(sub / "answer.csv")
    cleaned = safe_read_csv(sub / "cleaned_monitoring_data.csv")
    events = safe_read_csv(sub / "fault_events.csv")
    types = safe_read_csv(sub / "fault_types.csv")
    detail = {"total_score": 0.0, "errors": [], "warnings": [], "metrics": {}}

    required = {"record_id", "well_id", "time_days", "anomaly_label", "confidence", "event_id", "root_cause"}
    clean_required = {"record_id", "well_id", "time_days", "cleaned_concentration_mg_L", "cleaning_action"}
    event_required = {"event_id", "anomaly_label", "well_id", "start_day", "end_day", "confidence", "evidence"}
    type_required = {"anomaly_label", "description", "detection_rule", "physical_rationale"}
    valid_answer = required.issubset(ans.columns)
    valid_cleaned = clean_required.issubset(cleaned.columns)
    valid_events = event_required.issubset(events.columns) and len(events) >= 3
    valid_types = type_required.issubset(types.columns) and len(types) >= 5
    format_score = 0.0
    if valid_answer:
        format_score += 2.0
    else:
        detail["errors"].append("answer.csv missing required columns")
    if valid_cleaned:
        format_score += 0.5
    else:
        detail["warnings"].append("cleaned_monitoring_data.csv missing required columns")
    if valid_events:
        format_score += 0.3
    else:
        detail["warnings"].append("fault_events.csv missing or incomplete")
    if valid_types:
        format_score += 0.2
    else:
        detail["warnings"].append("fault_types.csv missing or incomplete")
    format_score = min(3.0, format_score)
    if format_score < 2.0:
        detail["format_score"] = format_score
        detail["task_results"] = {"format": task_result(format_score, 3.0)}
        Path(output).write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"CASE borden_qc_fault_diagnosis OK score=0.000")
        print("TOTAL_SCORE 0.000")
        print("TASK_RESULT format score=0.000 max=3.000 pass_rate=0.000")
        return detail

    ans = ans.copy()
    ans["anomaly_label_pred"] = ans["anomaly_label"].astype(str).where(ans["anomaly_label"].astype(str).isin(ALLOWED), "normal")
    merged = labels.merge(ans, on="record_id", how="left")
    pred = merged["anomaly_label_pred"].fillna("normal").astype(str)
    true = merged["anomaly_label_x"].astype(str) if "anomaly_label_x" in merged.columns else merged["anomaly_label"].astype(str)
    y_true_fault = true.isin(FAULT_LABELS)
    y_pred_fault = pred.isin(FAULT_LABELS)

    record_f1 = binary_f1(y_true_fault, y_pred_fault)
    record_score = scaled_score(record_f1, 0.65, 0.92, 15.0, gamma=1.5)
    type_mf1, by_label = macro_f1(true, pred, sorted(ALLOWED - {"normal"}))
    type_score = scaled_score(type_mf1, 0.55, 0.88, 20.0, gamma=1.8)
    ev_quality, event_rows = event_score(labels, ans.rename(columns={"anomaly_label": "anomaly_label_pred"}) if "anomaly_label_pred" not in ans.columns else ans)
    event_score_val = scaled_score(ev_quality, 0.60, 0.90, 15.0, gamma=1.5)

    # Release gates are stricter than raw metrics. Downstream modules such as
    # physical consistency and cleaning only release credit after the core
    # detection/event/type metrics pass their baselines.
    record_gate = release_gate(record_f1, 0.65, 0.92)
    event_gate = release_gate(ev_quality, 0.60, 0.90)
    type_gate = release_gate(type_mf1, 0.55, 0.88)
    diagnosis_release_gate = min(record_gate, event_gate, type_gate)

    plume_mask = true.eq("true_plume_arrival")
    plume_fp = float(np.mean(pred[plume_mask].isin(FAULT_LABELS))) if plume_mask.any() else 0.0
    drift_mask = true.eq("drift")
    drift_recall = float(np.mean(pred[drift_mask].eq("drift"))) if drift_mask.any() else 0.0
    shift_mask = true.eq("time_shift")
    shift_recall = float(np.mean(pred[shift_mask].eq("time_shift"))) if shift_mask.any() else 0.0
    coord_mask = true.eq("coordinate_or_depth_error")
    coord_recall = float(np.mean(pred[coord_mask].eq("coordinate_or_depth_error"))) if coord_mask.any() else 0.0
    downstream_order = 1.0 - plume_fp
    complex_recall = (drift_recall + shift_recall + coord_recall) / 3.0
    physical_quality = (0.35 * downstream_order + 0.25 * drift_recall + 0.20 * shift_recall + 0.20 * coord_recall) * min(1.0, complex_recall / 0.20)
    # Physical-consistency credit should not be released when the model cannot
    # classify the relevant physical fault types. Otherwise simple spatial rules
    # can earn medium scores despite failed fault typing.
    physical_score = 10.0 * max(0.0, min(1.0, physical_quality)) * diagnosis_release_gate

    clean_score = 0.0
    clean_metrics = {}
    clean_raw = 0.0
    if valid_cleaned:
        m = clean.merge(cleaned, on="record_id", how="left")
        pred_clean = pd.to_numeric(m["cleaned_concentration_mg_L"], errors="coerce")
        true_clean = pd.to_numeric(m["clean_concentration_mg_L"], errors="coerce")
        raw_obs = pd.to_numeric(m["observed_concentration_mg_L"], errors="coerce")
        rmse = float(np.sqrt(np.nanmean((pred_clean - true_clean) ** 2))) if len(m) else float("inf")
        raw_rmse = float(np.sqrt(np.nanmean((raw_obs - true_clean) ** 2))) if len(m) else float("inf")
        improvement = max(0.0, min(1.0, (raw_rmse - rmse) / max(raw_rmse, 1e-9)))
        clean_raw = improvement
        clean_score = scaled_score(improvement, 0.05, 0.45, 25.0, gamma=1.5)
        clean_metrics = {"clean_rmse": rmse, "raw_rmse": raw_rmse, "rmse_improvement": improvement}

    report_score = 0.0
    report = (sub / "fault_report.md").read_text(encoding="utf-8", errors="ignore") if (sub / "fault_report.md").exists() else ""
    long_report = len(report.strip()) >= 700
    if long_report:
        report_score += 1.0
    lower = report.lower()
    for group in [
        ["borden", "groundwater", "plume"],
        ["neighbor", "spatial", "downstream", "upstream"],
        ["travel", "arrival", "time shift"],
        ["drift", "sensor", "fault"],
        ["clean", "repair", "interpolate"],
    ]:
        if long_report and any(k in lower for k in group):
            report_score += 0.45
    report_score = min(5.0, report_score)
    robustness_raw = min(record_f1, ev_quality, type_mf1, clean_raw if clean_raw > 0 else 0.0)
    robustness_score = scaled_score(robustness_raw, 0.50, 0.85, 7.0, gamma=1.5)

    raw_total = format_score + record_score + event_score_val + type_score + physical_score + clean_score + report_score + robustness_score
    caps = []
    cap_reasons = []
    if not valid_events:
        caps.append(15.0); cap_reasons.append("missing_or_invalid_fault_events")
    if not valid_cleaned:
        caps.append(20.0); cap_reasons.append("missing_or_invalid_cleaned_data")
    if clean_raw <= 0.05:
        caps.append(30.0); cap_reasons.append("cleaned_data_no_meaningful_improvement")
    if type_mf1 < 0.55:
        caps.append(35.0); cap_reasons.append("type_macro_f1_below_release_threshold")
    if report_score <= 0.0:
        caps.append(60.0); cap_reasons.append("missing_or_uninformative_report")
    total = min(raw_total, min(caps) if caps else 100.0)
    detail.update({
        "format_score": round(format_score, 3),
        "record_anomaly_detection_score": round(record_score, 3),
        "event_detection_score": round(event_score_val, 3),
        "type_classification_score": round(type_score, 3),
        "physical_consistency_score": round(physical_score, 3),
        "cleaned_data_quality_score": round(clean_score, 3),
        "robustness_score": round(robustness_score, 3),
        "report_score": round(report_score, 3),
        "raw_total_before_caps": round(raw_total, 3),
        "score_caps": caps,
        "cap_reasons": cap_reasons,
        "total_score": round(max(0.0, min(100.0, total)), 3),
    })
    detail["metrics"] = {
        "score_policy": "baseline-subtracted nonlinear scoring with strict release gates for cleaning and physics",
        "release_gates": {
            "record_gate": round(record_gate, 3),
            "event_gate": round(event_gate, 3),
            "type_gate": round(type_gate, 3),
            "diagnosis_release_gate": round(diagnosis_release_gate, 3),
        },
        "clean_metrics": clean_metrics,
        "quality_bands": {
            "detection": "low" if record_f1 < 0.65 else ("medium" if record_f1 < 0.82 else "high"),
            "classification": "low" if type_mf1 < 0.55 else ("medium" if type_mf1 < 0.75 else "high"),
            "event": "low" if ev_quality < 0.60 else ("medium" if ev_quality < 0.78 else "high"),
            "cleaning": "missing_or_poor" if clean_raw <= 0.05 else ("medium" if clean_raw < 0.30 else "high"),
            "physics": "low" if physical_score < 3.0 else ("medium" if physical_score < 7.0 else "high"),
        },
    }
    detail["task_results"] = {
        "format": task_result(format_score, 3.0, 0.8),
        "record_anomaly_detection": task_result(record_score, 15.0, 0.5),
        "event_detection": task_result(event_score_val, 15.0, 0.45),
        "type_classification": task_result(type_score, 20.0, 0.45),
        "physical_consistency": task_result(physical_score, 10.0, 0.45),
        "cleaned_data_quality": task_result(clean_score, 25.0, 0.4),
        "robustness_generalization": task_result(robustness_score, 7.0, 0.4),
        "report": task_result(report_score, 5.0, 0.4),
        "overall": task_result(detail["total_score"], 100.0, 0.5),
    }
    Path(output).write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"CASE borden_qc_fault_diagnosis OK score={detail['total_score']:.3f}")
    print(f"TOTAL_SCORE {detail['total_score']:.3f}")
    for name, res in detail["task_results"].items():
        print(f"TASK_RESULT {name} score={res['score']:.3f} max={res['max_score']:.3f} pass_rate={res['pass_rate']:.3f}")
    feedback = {
        "total_score": detail["total_score"],
        "task_results": detail["task_results"],
        "quality_bands": detail["metrics"].get("quality_bands", {}),
        "cap_reasons": cap_reasons,
    }
    print("SCORE_BREAKDOWN_JSON " + json.dumps(feedback, ensure_ascii=False, sort_keys=True))
    print(
        "QC_FEEDBACK "
        f"detection={feedback['quality_bands'].get('detection')} "
        f"classification={feedback['quality_bands'].get('classification')} "
        f"event={feedback['quality_bands'].get('event')} "
        f"cleaning={feedback['quality_bands'].get('cleaning')}"
    )
    print(f"PHYSICS_FEEDBACK consistency={feedback['quality_bands'].get('physics')}")
    return detail


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--submission_dir", required=True)
    p.add_argument("--case_dir", required=True)
    p.add_argument("--scoring_dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    evaluate(args.submission_dir, args.case_dir, args.scoring_dir, args.output)


if __name__ == "__main__":
    main()
