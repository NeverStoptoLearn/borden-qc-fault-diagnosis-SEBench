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


def _band_from_score(score, max_score):
    try:
        ratio = max(0.0, min(1.0, float(score) / max(float(max_score), 1e-9)))
    except Exception:
        ratio = 0.0
    if ratio >= 0.85:
        return "strong"
    if ratio >= 0.65:
        return "adequate"
    if ratio >= 0.35:
        return "partial"
    if ratio > 0.0:
        return "weak"
    return "missing"


def _normalize_quality(value):
    value = str(value or "missing")
    return {
        "high": "strong",
        "medium": "partial",
        "low": "weak",
        "missing_or_poor": "missing_or_poor",
    }.get(value, value)


def qc_review_state(detail):
    total = float(detail.get("total_score", 0.0) or 0.0)
    caps = set(detail.get("cap_reasons", []) or [])
    artifacts = detail.get("artifact_status", {}) or {}
    missing_events = ("missing_or_invalid_fault_events" in caps) or (artifacts.get("valid_events") is False)
    missing_cleaned = ("missing_or_invalid_cleaned_data" in caps) or (artifacts.get("valid_cleaned") is False)
    metrics = detail.get("metrics", {}) or {}
    bands = metrics.get("quality_bands", {}) or {}
    gates = metrics.get("release_gates", {}) or {}

    if detail.get("errors") or total <= 0.0:
        review_status = "invalid"
    elif missing_events or missing_cleaned:
        review_status = "needs_revision"
    elif total >= 85.0:
        review_status = "defensible"
    elif total >= 65.0:
        review_status = "acceptable"
    elif total >= 35.0:
        review_status = "provisional"
    else:
        review_status = "needs_revision"

    data_screening = _normalize_quality(bands.get("detection"))
    event_grouping = _normalize_quality(bands.get("event"))
    fault_typing = _normalize_quality(bands.get("classification"))
    cleaning_repair = _normalize_quality(bands.get("cleaning"))
    physical_rationale = _normalize_quality(bands.get("physics"))
    if data_screening in {"adequate", "strong"} and float(gates.get("record_dependency_gate", 1.0) or 0.0) < 0.50:
        data_screening = "dependency_blocked"
    if cleaning_repair in {"adequate", "strong"} and float(gates.get("cleaning_dependency_gate", 1.0) or 0.0) < 0.50:
        cleaning_repair = "dependency_blocked"
    report_quality = _band_from_score(detail.get("report_score", 0.0), 5.0)

    if detail.get("format_score", 0.0) < 2.0:
        process_stage = "setup"
    elif data_screening in {"missing", "weak", "dependency_blocked"}:
        process_stage = "screening"
    elif event_grouping in {"missing", "weak"}:
        process_stage = "screening+events"
    elif fault_typing in {"missing", "weak"}:
        process_stage = "screening+events+typing"
    elif cleaning_repair in {"missing", "missing_or_poor", "weak", "dependency_blocked"}:
        process_stage = "screening+events+typing+repair"
    elif physical_rationale in {"missing", "weak"} or report_quality in {"missing", "weak"}:
        process_stage = "screening+events+typing+repair+physics_review"
    else:
        process_stage = "final_qc_package"

    if review_status in {"defensible", "acceptable"}:
        validation_status = review_status
    elif total >= 35.0:
        validation_status = "provisional"
    else:
        validation_status = "not_defensible"

    if detail.get("format_score", 0.0) < 2.0:
        next_review = "Complete the required answer.csv schema before QC review."
    elif missing_events:
        next_review = "Group record-level detections into event intervals with evidence."
    elif missing_cleaned:
        next_review = "Provide repaired concentrations and cleaning actions for every evaluation record."
    elif data_screening in {"missing", "weak", "dependency_blocked"}:
        next_review = "Improve initial anomaly screening before fault typing."
    elif event_grouping in {"missing", "weak"}:
        next_review = "Convert isolated record detections into physically consistent fault events."
    elif fault_typing in {"missing", "weak"}:
        next_review = "Distinguish drift, time shift, coordinate/depth error, unit error, and true plume arrival."
    elif cleaning_repair in {"missing", "missing_or_poor", "weak", "dependency_blocked"}:
        next_review = "Improve concentration repair using neighbors, time-series continuity, and detection limits."
    elif physical_rationale in {"missing", "weak"}:
        next_review = "Recheck Borden plume travel-time ordering, downstream consistency, and true plume arrivals."
    elif report_quality in {"missing", "weak"}:
        next_review = "Document detection rules, uncertainty, cleaning rationale, and physical evidence."
    else:
        next_review = "Review remaining public validation residuals and uncertainty before finalizing the QC package."

    if validation_status == "defensible":
        validation_review = "withheld QC review is defensible for the submitted fault-diagnosis package"
    elif validation_status == "acceptable":
        validation_review = "withheld QC review is acceptable but still has material uncertainty"
    elif validation_status == "provisional":
        validation_review = "withheld QC review is provisional; keep fault rules conservative"
    else:
        validation_review = "withheld QC review not defensible"

    return {
        "review_status": review_status,
        "process_stage": process_stage,
        "data_qc_status": data_screening,
        "event_review_status": event_grouping,
        "fault_typing_status": fault_typing,
        "cleaning_status": cleaning_repair,
        "physical_consistency_status": physical_rationale,
        "validation_status": validation_status,
        "next_review": next_review,
        "data_screening": data_screening,
        "event_grouping": event_grouping,
        "fault_typing": fault_typing,
        "cleaning_repair": cleaning_repair,
        "physical_rationale": physical_rationale,
        "report_quality": report_quality,
        "public_validation_review": "Use tools/evaluate_public.py to check public labels, true-plume false-faulting, and cleaning behavior.",
        "withheld_qc_review": validation_review,
    }


QC_REVIEW_STATUS_TEXT = {
    "invalid": "This submission cannot be reviewed yet because required QC files, schemas, or runnable components are missing or invalid.",
    "needs_revision": "This submission is not yet a defensible monitoring-data QC result and needs revision before it can support a project conclusion.",
    "provisional": "This submission provides a provisional QC interpretation, but the fault evidence and repair quality are still insufficient for a final conclusion.",
    "acceptable": "This submission is broadly acceptable for interim QC review, but uncertainty, event evidence, or cleaning rationale still need attention.",
    "defensible": "This submission is technically defensible as a monitoring-data QC and fault-diagnosis result, subject to the documented assumptions and uncertainty.",
}

QC_PROCESS_STAGE_TEXT = {
    "setup": "The submission is still in the setup stage; fix required CSV schemas and output files before fault-diagnosis review.",
    "screening": "The workflow is at the record-screening stage; improve initial anomaly detection before relying on event grouping or fault typing.",
    "screening+events": "The workflow has reached event review; convert record-level detections into physically consistent fault intervals with evidence.",
    "screening+events+typing": "The workflow has reached fault typing; distinguish sensor faults, timing errors, coordinate/depth errors, unit errors, and true plume arrivals.",
    "screening+events+typing+repair": "The workflow has reached repair review; improve cleaned concentrations and document cleaning actions for each evaluation record.",
    "screening+events+typing+repair+physics_review": "The workflow has reached physical-consistency review; check Borden plume timing, downstream order, neighbor wells, and true plume arrivals.",
    "final_qc_package": "The submission has reached the final QC package stage; review uncertainty, reproducibility, and project defensibility.",
}

QC_DATA_QC_TEXT = {
    "missing": "No meaningful record-level anomaly screening was produced.",
    "weak": "Record-level anomaly screening is weak; improve initial detection before fault typing.",
    "dependency_blocked": "Record-level screening finds candidate anomalies, but it is not yet creditable as a completed QC result because event grouping, fault typing, or physical closure is weak.",
    "partial": "Record-level anomaly screening is partially useful, but false positives or missed faults still need review.",
    "adequate": "Record-level anomaly screening is adequate for event-level review.",
    "strong": "Record-level anomaly screening is strongly supported by the submitted outputs.",
}

QC_EVENT_TEXT = {
    "missing": "No meaningful fault-event grouping was produced.",
    "weak": "Fault-event grouping is weak; isolated record labels need to be organized into coherent event intervals.",
    "partial": "Fault-event grouping is partially useful, but event boundaries or evidence remain incomplete.",
    "adequate": "Fault-event grouping is adequate for project review.",
    "strong": "Fault-event grouping is strongly supported by interval evidence and consistent labels.",
}

QC_TYPING_TEXT = {
    "missing": "No meaningful fault-type diagnosis was produced.",
    "weak": "Fault typing is weak; distinguish drift, time shift, coordinate/depth error, unit error, stuck-zero, spike, negative, and true plume arrival more carefully.",
    "partial": "Fault typing is partially useful, but important physical fault types remain confused.",
    "adequate": "Fault typing is adequate for QC review.",
    "strong": "Fault typing is strongly supported across the relevant anomaly classes.",
}

QC_CLEANING_TEXT = {
    "missing": "No cleaned-concentration repair was produced.",
    "missing_or_poor": "Cleaning and repair are not yet meaningful; repaired concentrations do not sufficiently improve on the raw observations.",
    "weak": "Cleaning and repair are weak; use neighbors, time-series continuity, detection limits, and fault type to guide repaired concentrations.",
    "dependency_blocked": "The cleaned concentrations may improve the series numerically, but the repair is not yet creditable because upstream fault typing and physical diagnosis are not defensible.",
    "partial": "Cleaning and repair are partially useful, but concentration repair quality or cleaning actions need more evidence.",
    "adequate": "Cleaning and repair are adequate for technical review.",
    "strong": "Cleaning and repair are strongly supported by the submitted cleaned series and action rationale.",
}

QC_PHYSICS_TEXT = {
    "missing": "No physical consistency evidence was produced.",
    "weak": "Physical consistency is weak; recheck plume travel-time ordering, downstream behavior, vertical screen depth, and true plume arrivals.",
    "partial": "Physical consistency is partially supported, but plume dynamics and sensor-fault distinctions need stronger evidence.",
    "adequate": "Physical consistency is adequate for QC review.",
    "strong": "Physical consistency is strongly supported by Borden plume timing, neighbor wells, and fault-event evidence.",
}

QC_REPORT_TEXT = {
    "missing": "No adequate fault report was submitted.",
    "weak": "The fault report is weak; it does not yet explain detection rules, evidence, cleaning rationale, uncertainty, and physical interpretation.",
    "partial": "The fault report is partially useful, but evidence, uncertainty, or physical rationale remain incomplete.",
    "adequate": "The fault report is adequate for technical review.",
    "strong": "The fault report strongly documents rules, physical rationale, uncertainty, event evidence, and cleaning actions.",
}

QC_VALIDATION_TEXT = {
    "not_defensible": "The submitted QC package is not yet defensible under withheld review; treat it as an unvalidated working diagnosis.",
    "provisional": "The submitted QC package has limited withheld-review support; keep the detection and repair rules conservative.",
    "acceptable": "The submitted QC package has acceptable withheld-review support, but uncertainty and edge cases still need documentation.",
    "defensible": "The submitted QC package is defensible under withheld review, given the submitted rules and uncertainty.",
}


def qc_review_feedback(detail):
    state = qc_review_state(detail)
    return {
        "review_status": QC_REVIEW_STATUS_TEXT.get(state["review_status"], QC_REVIEW_STATUS_TEXT["needs_revision"]),
        "process_stage": QC_PROCESS_STAGE_TEXT.get(state["process_stage"], QC_PROCESS_STAGE_TEXT["screening"]),
        "data_qc_status": QC_DATA_QC_TEXT.get(state["data_qc_status"], QC_DATA_QC_TEXT["weak"]),
        "event_review_status": QC_EVENT_TEXT.get(state["event_review_status"], QC_EVENT_TEXT["weak"]),
        "fault_typing_status": QC_TYPING_TEXT.get(state["fault_typing_status"], QC_TYPING_TEXT["weak"]),
        "cleaning_status": QC_CLEANING_TEXT.get(state["cleaning_status"], QC_CLEANING_TEXT["missing_or_poor"]),
        "physical_consistency_status": QC_PHYSICS_TEXT.get(state["physical_consistency_status"], QC_PHYSICS_TEXT["weak"]),
        "validation_status": QC_VALIDATION_TEXT.get(state["validation_status"], QC_VALIDATION_TEXT["not_defensible"]),
        "next_review": state["next_review"],
        "data_screening": QC_DATA_QC_TEXT.get(state["data_screening"], QC_DATA_QC_TEXT["weak"]),
        "event_grouping": QC_EVENT_TEXT.get(state["event_grouping"], QC_EVENT_TEXT["weak"]),
        "fault_typing": QC_TYPING_TEXT.get(state["fault_typing"], QC_TYPING_TEXT["weak"]),
        "cleaning_repair": QC_CLEANING_TEXT.get(state["cleaning_repair"], QC_CLEANING_TEXT["missing_or_poor"]),
        "physical_rationale": QC_PHYSICS_TEXT.get(state["physical_rationale"], QC_PHYSICS_TEXT["weak"]),
        "report_quality": QC_REPORT_TEXT.get(state["report_quality"], QC_REPORT_TEXT["missing"]),
        "public_validation_review": "Use the public validator to check public labels, true-plume false-faulting, and cleaning behavior before interpreting withheld review.",
        "withheld_qc_review": QC_VALIDATION_TEXT.get(state["validation_status"], QC_VALIDATION_TEXT["not_defensible"]),
    }


def qc_summary(detail):
    feedback = qc_review_feedback(detail)
    return (
        f"REVIEW_STATUS={feedback['review_status']}; "
        f"PROCESS_STAGE={feedback['process_stage']}; "
        f"DATA_QC_STATUS={feedback['data_qc_status']}; "
        f"EVENT_REVIEW_STATUS={feedback['event_review_status']}; "
        f"FAULT_TYPING_STATUS={feedback['fault_typing_status']}; "
        f"CLEANING_STATUS={feedback['cleaning_status']}; "
        f"VALIDATION_STATUS={feedback['validation_status']}; "
        f"NEXT_REVIEW={feedback['next_review']}"
    )


def qc_details(state):
    return [
        {"name": "data_screening", "status": "PASSED" if state["data_screening"] in {"adequate", "strong"} else "NEEDS_REVIEW", "score": None, "weight": None, "message": "record-level anomaly screening is reviewed without exposing hidden labels"},
        {"name": "event_grouping", "status": "PASSED" if state["event_grouping"] in {"adequate", "strong"} else "NEEDS_REVIEW", "score": None, "weight": None, "message": "event intervals and evidence are reviewed"},
        {"name": "fault_typing", "status": "PASSED" if state["fault_typing"] in {"adequate", "strong"} else "NEEDS_REVIEW", "score": None, "weight": None, "message": "fault type distinctions and true-plume treatment are reviewed"},
        {"name": "cleaning_repair", "status": "PASSED" if state["cleaning_repair"] in {"adequate", "strong"} else "NEEDS_REVIEW", "score": None, "weight": None, "message": "cleaned concentrations and repair actions are reviewed"},
        {"name": "physical_rationale", "status": "PASSED" if state["physical_rationale"] in {"adequate", "strong"} else "NEEDS_REVIEW", "score": None, "weight": None, "message": "Borden plume physics, neighbor consistency, and travel-time rationale are reviewed"},
        {"name": "validation_review", "status": "PASSED" if state["validation_status"] in {"acceptable", "defensible"} else "NEEDS_REVIEW", "score": None, "weight": None, "message": QC_VALIDATION_TEXT.get(state["validation_status"], QC_VALIDATION_TEXT["not_defensible"])},
    ]


def emit_structured_result(detail):
    state = qc_review_state(detail)
    feedback = qc_review_feedback(detail)
    result = {
        "valid": not bool(detail.get("errors")),
        "score": float(detail.get("total_score", 0.0) or 0.0),
        "pass_rate": max(0.0, min(1.0, float(detail.get("total_score", 0.0) or 0.0) / 100.0)),
        "summary": qc_summary(detail),
        "metrics": feedback,
        "details": qc_details(state),
    }
    print(">>>>> Start Structured Result")
    print(json.dumps(result, ensure_ascii=False))
    print(">>>>> End Structured Result")


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


def clamp01(value):
    try:
        return float(max(0.0, min(1.0, value)))
    except Exception:
        return 0.0


def band_gate(value, low, high):
    return clamp01((float(value) - float(low)) / max(float(high) - float(low), 1e-9))


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
        detail["total_score"] = 0.0
        detail["task_results"] = {"format": task_result(format_score, 3.0)}
        Path(output).write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"CASE borden_qc_fault_diagnosis OK score=0.000")
        print("TOTAL_SCORE 0.000")
        emit_structured_result(detail)
        return detail

    ans = ans.copy()
    ans["anomaly_label_pred"] = ans["anomaly_label"].astype(str).where(ans["anomaly_label"].astype(str).isin(ALLOWED), "normal")
    merged = labels.merge(ans, on="record_id", how="left")
    pred = merged["anomaly_label_pred"].fillna("normal").astype(str)
    true = merged["anomaly_label_x"].astype(str) if "anomaly_label_x" in merged.columns else merged["anomaly_label"].astype(str)
    y_true_fault = true.isin(FAULT_LABELS)
    y_pred_fault = pred.isin(FAULT_LABELS)

    record_f1 = binary_f1(y_true_fault, y_pred_fault)
    record_score_raw = scaled_score(record_f1, 0.65, 0.92, 15.0, gamma=1.5)
    type_mf1, by_label = macro_f1(true, pred, sorted(ALLOWED - {"normal"}))
    type_score_raw = scaled_score(type_mf1, 0.55, 0.88, 20.0, gamma=1.8)
    ev_quality, event_rows = event_score(labels, ans.rename(columns={"anomaly_label": "anomaly_label_pred"}) if "anomaly_label_pred" not in ans.columns else ans)
    event_score_raw = scaled_score(ev_quality, 0.60, 0.90, 15.0, gamma=1.5)

    # Stage dependency is applied as continuous release, not as a final cap.
    # Record screening has only limited standalone value: most record-level
    # credit must be confirmed by event grouping, fault typing, and physical
    # closure. Later QC artifacts then use the validated record gate.
    record_gate = release_gate(record_f1, 0.65, 0.92)
    event_gate = release_gate(ev_quality, 0.60, 0.90)
    type_gate = release_gate(type_mf1, 0.55, 0.88)
    event_artifact_gate = 1.0 if valid_events else 0.0
    type_artifact_gate = 1.0 if valid_types else 0.0
    clean_artifact_gate = 1.0 if valid_cleaned else 0.0

    plume_mask = true.eq("true_plume_arrival")
    plume_fp = float(np.mean(pred[plume_mask].isin(FAULT_LABELS))) if plume_mask.any() else 0.0
    plume_recall = float(np.mean(pred[plume_mask].eq("true_plume_arrival"))) if plume_mask.any() else 0.0
    drift_mask = true.eq("drift")
    drift_recall = float(np.mean(pred[drift_mask].eq("drift"))) if drift_mask.any() else 0.0
    shift_mask = true.eq("time_shift")
    shift_recall = float(np.mean(pred[shift_mask].eq("time_shift"))) if shift_mask.any() else 0.0
    coord_mask = true.eq("coordinate_or_depth_error")
    coord_recall = float(np.mean(pred[coord_mask].eq("coordinate_or_depth_error"))) if coord_mask.any() else 0.0
    key_type_min_recall = min(drift_recall, shift_recall, coord_recall, plume_recall)
    key_type_gate = band_gate(key_type_min_recall, 0.18, 0.62)
    true_plume_gate = min(band_gate(plume_recall, 0.20, 0.70), band_gate(1.0 - plume_fp, 0.70, 0.98))

    pred_fault_rate = float(np.mean(y_pred_fault)) if len(y_pred_fault) else 0.0
    true_fault_rate = float(np.mean(y_true_fault)) if len(y_true_fault) else 0.0
    over_ratio = pred_fault_rate / max(true_fault_rate, 1e-9)
    under_ratio = true_fault_rate / max(pred_fault_rate, 1e-9) if pred_fault_rate > 0 else float("inf")
    overlabel_gate = clamp01((2.60 - over_ratio) / 1.60) * clamp01((2.80 - under_ratio) / 1.80)
    overlabel_gate = max(0.0, min(1.0, overlabel_gate))

    event_scores = [float(r.get("event_score", 0.0)) for r in event_rows]
    event_boundary_quality = float(np.mean(event_scores)) if event_scores else 0.0
    event_boundary_gate = band_gate(event_boundary_quality, 0.50, 0.85)

    downstream_order = 1.0 - plume_fp
    complex_recall = (drift_recall + shift_recall + coord_recall) / 3.0
    physical_quality = (0.35 * downstream_order + 0.25 * drift_recall + 0.20 * shift_recall + 0.20 * coord_recall) * min(1.0, complex_recall / 0.20)
    physical_score_raw = 10.0 * clamp01(physical_quality)
    physics_quality_gate = release_gate(physical_quality, 0.20, 0.75)
    strict_closure_gate = min(
        event_gate,
        type_gate,
        physics_quality_gate,
        true_plume_gate,
        key_type_gate,
        event_boundary_gate,
        overlabel_gate,
    )
    diagnostic_closure_gate = strict_closure_gate
    record_dependency_gate = 0.06 + 0.94 * (strict_closure_gate ** 2)
    validated_record_gate = min(record_gate, record_dependency_gate)
    event_dependency_gate = validated_record_gate * event_artifact_gate
    type_dependency_gate = min(validated_record_gate, event_gate, true_plume_gate, key_type_gate) * type_artifact_gate
    diagnosis_release_gate = min(validated_record_gate, event_gate, type_gate, true_plume_gate, key_type_gate) * type_artifact_gate
    record_score = record_score_raw * record_dependency_gate
    event_score_val = event_score_raw * event_dependency_gate
    type_score = type_score_raw * type_dependency_gate
    physical_score = physical_score_raw * (strict_closure_gate ** 2) * type_artifact_gate

    clean_score = 0.0
    clean_score_raw = 0.0
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
        clean_score_raw = scaled_score(improvement, 0.05, 0.45, 25.0, gamma=1.5)
        clean_metrics = {"clean_rmse": rmse, "raw_rmse": raw_rmse, "rmse_improvement": improvement}
    clean_quality_gate = release_gate(clean_raw, 0.05, 0.45)
    cleaning_action_gate = 0.0
    if valid_cleaned and "cleaning_action" in cleaned.columns and "record_id" in cleaned.columns:
        action_map = cleaned.set_index("record_id")["cleaning_action"].fillna("").astype(str)
        action = merged["record_id"].map(action_map).fillna("").astype(str)
        normal_mask = true.eq("normal")
        plume_true_mask = true.eq("true_plume_arrival")
        fault_true_mask = true.isin(FAULT_LABELS)
        passive_actions = ["none", "keep", "unchanged", "none_real_plume", "template_blend"]
        normal_unchanged = float(np.mean(action[normal_mask].str.lower().isin(passive_actions))) if normal_mask.any() else 1.0
        plume_unrepaired = float(np.mean(action[plume_true_mask].str.lower().isin(passive_actions))) if plume_true_mask.any() else 1.0
        fault_repaired = float(np.mean(~action[fault_true_mask].str.lower().isin(["", "none", "keep", "unchanged", "none_real_plume"]))) if fault_true_mask.any() else 0.0
        cleaning_action_quality = 0.35 * normal_unchanged + 0.25 * plume_unrepaired + 0.40 * fault_repaired
        cleaning_action_gate = band_gate(cleaning_action_quality, 0.45, 0.82)
    else:
        cleaning_action_quality = 0.0
    downstream_closure_gate = strict_closure_gate ** 4
    cleaning_dependency_gate = downstream_closure_gate * clean_artifact_gate * type_artifact_gate * cleaning_action_gate
    clean_score = clean_score_raw * cleaning_dependency_gate

    report_score_raw = 0.0
    report = (sub / "fault_report.md").read_text(encoding="utf-8", errors="ignore") if (sub / "fault_report.md").exists() else ""
    long_report = len(report.strip()) >= 700
    if long_report:
        report_score_raw += 1.0
    lower = report.lower()
    for group in [
        ["borden", "groundwater", "plume"],
        ["neighbor", "spatial", "downstream", "upstream"],
        ["travel", "arrival", "time shift"],
        ["drift", "sensor", "fault"],
        ["clean", "repair", "interpolate"],
    ]:
        if long_report and any(k in lower for k in group):
            report_score_raw += 0.45
    report_score_raw = min(5.0, report_score_raw)
    report_dependency_gate = downstream_closure_gate * min(clean_quality_gate, cleaning_action_gate) * type_artifact_gate
    report_score = report_score_raw * report_dependency_gate
    robustness_raw = min(record_f1, ev_quality, type_mf1, clean_raw if clean_raw > 0 else 0.0)
    robustness_score_raw = scaled_score(robustness_raw, 0.50, 0.85, 7.0, gamma=1.5)
    stress_balance = min(record_gate, event_gate, type_gate, physics_quality_gate, true_plume_gate, key_type_gate, event_boundary_gate, overlabel_gate, cleaning_action_gate, clean_quality_gate)
    robustness_dependency_gate = (stress_balance ** 4) * event_artifact_gate * type_artifact_gate * clean_artifact_gate
    robustness_score = robustness_score_raw * robustness_dependency_gate

    raw_total_before_dependency = (
        format_score + record_score_raw + event_score_raw + type_score_raw
        + physical_score_raw + clean_score_raw + report_score_raw + robustness_score_raw
    )
    total = format_score + record_score + event_score_val + type_score + physical_score + clean_score + report_score + robustness_score
    detail.update({
        "format_score": round(format_score, 3),
        "record_anomaly_detection_score": round(record_score, 3),
        "event_detection_score": round(event_score_val, 3),
        "type_classification_score": round(type_score, 3),
        "physical_consistency_score": round(physical_score, 3),
        "cleaned_data_quality_score": round(clean_score, 3),
        "robustness_score": round(robustness_score, 3),
        "report_score": round(report_score, 3),
        "raw_total_before_dependency": round(raw_total_before_dependency, 3),
        "raw_total_before_caps": round(total, 3),
        "score_caps": [],
        "cap_reasons": [],
        "dependency_limited_reasons": [
            reason for reason, gate in [
                ("record_screening_depends_on_diagnostic_closure", record_dependency_gate),
                ("event_grouping_depends_on_record_screening", event_dependency_gate),
                ("fault_typing_depends_on_record_and_event_quality", type_dependency_gate),
                ("physical_review_depends_on_fault_typing", diagnosis_release_gate),
                ("cleaning_depends_on_diagnosis_and_physics", cleaning_dependency_gate),
                ("report_evidence_depends_on_diagnosis_and_repair", report_dependency_gate),
            ] if gate < 0.999
        ],
        "artifact_status": {
            "valid_answer": valid_answer,
            "valid_cleaned": valid_cleaned,
            "valid_events": valid_events,
            "valid_types": valid_types,
        },
        "total_score": round(max(0.0, min(100.0, total)), 3),
    })
    detail["metrics"] = {
        "score_policy": "baseline-subtracted nonlinear scoring with sequential dependency release and no final score caps",
        "release_gates": {
            "record_gate": round(record_gate, 3),
            "event_gate": round(event_gate, 3),
            "type_gate": round(type_gate, 3),
            "diagnostic_closure_gate": round(diagnostic_closure_gate, 3),
            "strict_closure_gate": round(strict_closure_gate, 3),
            "true_plume_gate": round(true_plume_gate, 3),
            "key_type_gate": round(key_type_gate, 3),
            "event_boundary_gate": round(event_boundary_gate, 3),
            "overlabel_gate": round(overlabel_gate, 3),
            "record_dependency_gate": round(record_dependency_gate, 3),
            "validated_record_gate": round(validated_record_gate, 3),
            "event_dependency_gate": round(event_dependency_gate, 3),
            "type_dependency_gate": round(type_dependency_gate, 3),
            "diagnosis_release_gate": round(diagnosis_release_gate, 3),
            "physics_quality_gate": round(physics_quality_gate, 3),
            "clean_quality_gate": round(clean_quality_gate, 3),
            "cleaning_action_gate": round(cleaning_action_gate, 3),
            "downstream_closure_gate": round(downstream_closure_gate, 3),
            "cleaning_dependency_gate": round(cleaning_dependency_gate, 3),
            "report_dependency_gate": round(report_dependency_gate, 3),
            "robustness_dependency_gate": round(robustness_dependency_gate, 3),
        },
        "clean_metrics": clean_metrics,
        "diagnostic_balance": {
            "true_plume_recall": round(plume_recall, 3),
            "true_plume_false_fault_rate": round(plume_fp, 3),
            "drift_recall": round(drift_recall, 3),
            "time_shift_recall": round(shift_recall, 3),
            "coordinate_depth_recall": round(coord_recall, 3),
            "predicted_fault_rate": round(pred_fault_rate, 3),
            "true_fault_rate": round(true_fault_rate, 3),
            "overlabel_ratio": round(over_ratio, 3),
            "underlabel_ratio": round(under_ratio if math.isfinite(under_ratio) else 999.0, 3),
            "event_boundary_quality": round(event_boundary_quality, 3),
            "cleaning_action_quality": round(cleaning_action_quality, 3),
            "stress_balance": round(stress_balance, 3),
        },
        "raw_component_scores_before_dependency": {
            "format": round(format_score, 3),
            "record_anomaly_detection": round(record_score_raw, 3),
            "event_detection": round(event_score_raw, 3),
            "type_classification": round(type_score_raw, 3),
            "physical_consistency": round(physical_score_raw, 3),
            "cleaned_data_quality": round(clean_score_raw, 3),
            "robustness_generalization": round(robustness_score_raw, 3),
            "report": round(report_score_raw, 3),
        },
        "quality_bands": {
            "detection": "low" if record_f1 < 0.65 else ("medium" if record_f1 < 0.82 else "high"),
            "classification": "missing" if not valid_types else ("low" if type_mf1 < 0.55 else ("medium" if type_mf1 < 0.75 else "high")),
            "event": "missing" if not valid_events else ("low" if ev_quality < 0.60 else ("medium" if ev_quality < 0.78 else "high")),
            "cleaning": "missing_or_poor" if (not valid_cleaned or clean_raw <= 0.05) else ("medium" if clean_raw < 0.30 else "high"),
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
    feedback = qc_review_feedback(detail)
    print(f"REVIEW_STATUS {feedback['review_status']}")
    print(f"PROCESS_STAGE {feedback['process_stage']}")
    print(f"DATA_QC_STATUS {feedback['data_qc_status']}")
    print(f"EVENT_REVIEW_STATUS {feedback['event_review_status']}")
    print(f"FAULT_TYPING_STATUS {feedback['fault_typing_status']}")
    print(f"CLEANING_STATUS {feedback['cleaning_status']}")
    print(f"VALIDATION_STATUS {feedback['validation_status']}")
    emit_structured_result(detail)
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
