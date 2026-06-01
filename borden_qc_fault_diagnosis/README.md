# Borden 3D Groundwater Monitoring Data Quality Control and Fault Diagnosis

## Role

You are an environmental data-quality engineer working with a Borden-style three-dimensional groundwater contaminant monitoring network. Your task is not source inversion, well placement, or emergency pumping optimization. Your task is to identify monitoring-data quality problems and distinguish them from real plume dynamics.

## Task

Use CPU-only statistical, time-series, spatial-consistency, and physics-informed rules to inspect noisy groundwater monitoring records. Produce:

1. `answer.csv` for the hidden evaluation records in `eval_monitoring_noisy.csv`;
2. `cleaned_monitoring_data.csv` with repaired concentrations for the same `record_id`s;
3. `fault_events.csv` with event-level fault intervals and evidence;
4. `fault_types.csv` documenting rules/physical rationale for each predicted type;
5. `fault_report.md` summarizing evidence and uncertainty.

Deep learning and GPU training are not required. The key challenge is Borden-scene physical consistency: downstream plume arrival, neighbor wells, vertical screen depth, travel-time ordering, concentration tails, and detection limits.

## Fault Labels

Allowed `anomaly_label` values:

`normal`, `spike`, `drift`, `stuck_zero`, `missing`, `unit_error`, `time_shift`, `coordinate_or_depth_error`, `true_plume_arrival`, `negative`.

`true_plume_arrival` is not a sensor fault. It is a real plume event that can look anomalous to generic detectors. Misclassifying it as a fault is penalized.

## Provided Files

- `public_problem_config.json`: Borden-style flow and transport metadata.
- `borden_grid.npz`: x/y/z grid arrays.
- `public_wells.csv`: monitoring-well geometry.
- `train_monitoring_noisy.csv`: labeled development records with noisy/faulty observations.
- `train_fault_labels.csv`: labels for training records.
- `public_validation_noisy.csv`: public validation features.
- `public_fault_labels_small.csv`: partial validation labels for local tuning.
- `public_clean_reference_subset.csv`: small clean-concentration subset for repair calibration.
- `eval_monitoring_noisy.csv`: records that must be labeled in `answer.csv`.
- `baseline_detector.py`: weak but valid baseline.
- `tools/evaluate_public.py`: public local validator.

## Output Format

`answer.csv` must contain:

```csv
record_id,well_id,time_days,anomaly_label,confidence,event_id,root_cause
```

`cleaned_monitoring_data.csv` must contain:

```csv
record_id,well_id,time_days,cleaned_concentration_mg_L,cleaning_action
```

`fault_events.csv` must contain:

```csv
event_id,anomaly_label,well_id,start_day,end_day,confidence,evidence
```

`fault_types.csv` must contain:

```csv
anomaly_label,description,detection_rule,physical_rationale
```

`confidence` should be between 0 and 1. Missing records or invalid labels are penalized.

## Recommended Workflow

1. Run `python baseline_detector.py` to generate a legal baseline.
2. For public validation, run `python baseline_detector.py --input public_validation_noisy.csv --answer public_answer.csv --cleaned public_cleaned.csv --report public_fault_report.md` and then `python tools/evaluate_public.py --answer public_answer.csv --cleaned public_cleaned.csv`.
3. For final submission, run your detector on `eval_monitoring_noisy.csv` and write task-root `answer.csv` and `cleaned_monitoring_data.csv`.
4. Improve by adding robust rolling statistics, neighbor-well checks, travel-time consistency, vertical-depth checks, event grouping, drift/time-shift detection, and conservative treatment of real plume arrivals.

## Feedback Policy

The judge keeps exact hidden labels, per-label hidden scores, cleaning RMSE, and
event-level hidden residuals private. Iterative feedback is phrased like a QC
review: it reports data screening, event grouping, fault typing, cleaning,
physical rationale, and withheld-review defensibility as qualitative statuses.
The internal review follows the practical QC workflow order: record screening
must support event grouping, event evidence must support fault typing, and
fault typing plus physical consistency must support repaired concentrations and
the final report. Downstream artifacts are therefore credited only when the
upstream diagnosis is strong enough to make them defensible.
Use the public validator for numeric local tuning, and use judge feedback to
identify which QC workflow stage needs review.

## Rules

- CPU only; no internet; no GPU required.
- Do not read `scoring/`, hidden labels, or judge files.
- Do not hard-code hidden answers or record ids.
- Do not use external MODFLOW/MT3DMS/FloPy executables.
- Submit scripts used to generate your outputs.
