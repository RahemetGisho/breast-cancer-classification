# Power BI Dashboard — Build Guide

This is a build spec, not a finished `.pbix` — Power BI Desktop isn't
available in the environment this project was built in, so the data
exports below are ready to import, and this doc tells you exactly what
to build from them. Budget roughly 1–2 hours in Power BI Desktop.

## Why a BI dashboard here, and not a live-inference tool

Power BI (like Tableau) is built for exploring data at rest, not for
calling a live REST API per user click — that's what `api/main.py` and
its `/predict` endpoint already do. This dashboard's job is different:
give a stakeholder (a hiring manager, a clinical ops lead) a way to
explore **how well the model performs and why**, using the four CSVs
below as the source data.

## Data files (in this folder)

| File                              | Grain                                    | Purpose                                                          |
| --------------------------------- | ---------------------------------------- | ---------------------------------------------------------------- |
| `test_predictions.csv`            | 1 row per held-out test case (114 rows)  | Confusion matrix, recall/precision, misclassification drill-down |
| `model_metrics.csv`               | 1 row (current model)                    | KPI cards on the overview page                                   |
| `shap_feature_importance.csv`     | 1 row per feature (30 rows)              | Global feature-importance chart                                  |
| `prediction_log.csv`              | 1 row per API request served             | Monitoring page — **see caveat below**                           |
| `prediction_log_contributors.csv` | 1 row per (request × top-5 SHAP feature) | Detail table behind the monitoring page                          |

Regenerate all five anytime with:

```bash
python -m src.reporting.export_for_powerbi
```

**Caveat to state on the dashboard itself:** `prediction_log.csv` currently
holds only the handful of requests made while testing the API locally —
not real production traffic. Label the monitoring page "based on
local test traffic" rather than implying live production volume.

## Data model (relationships to set up in Power BI)

```
test_predictions (prediction_id)  ──┐
                                     │  (independent — no shared key,
model_metrics (single row)          │   used only for KPI cards, not
                                     │   joined to anything)
shap_feature_importance (feature)  ─┘

prediction_log (prediction_id) ──1:many──> prediction_log_contributors (prediction_id)
```

`test_predictions` and the `prediction_log*` tables are **not** related to
each other (different populations: held-out test set vs. served
requests) — don't force a relationship between them, or Power BI will
silently produce a nonsensical cross-filter.

## Suggested pages

### Page 1 — Model Performance Overview

- **KPI cards** (from `model_metrics.csv`): Recall (malignant) = 95.2%,
  Precision = 97.6%, ROC-AUC = 99.6%, False Negatives = 2.
- **Confusion matrix** (matrix visual from `test_predictions.csv`,
  rows = `actual_label`, columns = `predicted_label`, values = count).
- **Recall vs. classification threshold**: a line chart — bucket
  `probability_malignant` from `test_predictions.csv` into threshold
  steps (e.g. 0.1 to 0.9) and compute recall/precision at each via a
  DAX measure (see below). This is the single most useful interactive
  element for a reviewer, since it shows the recall/precision trade-off
  your 0.5-threshold choice sits on, not just one static number.

### Page 2 — Feature Importance & Explainability

- **Horizontal bar chart** from `shap_feature_importance.csv`, sorted
  by `rank`, top 10–15 features — this is your global explainability
  view.
- Add a text callout naming the top 3 features and one clinical-sense
  sentence each (texture irregularity, radius variance, concave contour
  points — see `REPORT.md`'s SHAP section for the exact phrasing already
  written).

### Page 3 — Prediction Monitoring (labeled as local test traffic)

- **Table or card** showing request volume over `timestamp_utc`.
- **Distribution of `probability_malignant`** across served requests
  (histogram) — this is the shape a real drift-monitoring page would
  track over time once real traffic exists.
- Drill-through from a request row into
  `prediction_log_contributors.csv` to show that specific prediction's
  top-5 SHAP features (filter by `prediction_id`).

## Suggested DAX measures

```
Recall (Malignant) =
VAR TP = CALCULATE(COUNTROWS(test_predictions), test_predictions[outcome] = "true_positive")
VAR FN = CALCULATE(COUNTROWS(test_predictions), test_predictions[outcome] = "false_negative")
RETURN DIVIDE(TP, TP + FN)

Precision (Malignant) =
VAR TP = CALCULATE(COUNTROWS(test_predictions), test_predictions[outcome] = "true_positive")
VAR FP = CALCULATE(COUNTROWS(test_predictions), test_predictions[outcome] = "false_positive")
RETURN DIVIDE(TP, TP + FP)
```

Building these as measures (rather than just reading the static numbers
in `model_metrics.csv`) is what makes the confusion-matrix and
threshold-slider visuals actually interactive — filtering the page
recalculates them live.

## Design notes (avoiding a generic default look)

- Skip Power BI's default theme. Use a **restrained palette**: one
  accent color for "malignant"/risk, a neutral gray-blue for "benign",
  and avoid red/green traffic-light coding — this is a clinical context,
  not a stoplight.
- Avoid gauge/speedometer visuals for the KPI cards — they're
  the single most overused default in BI portfolios and don't convey
  more than a plain number card does here.
- Keep the confusion matrix as an actual matrix visual, not a decorative
  donut chart standing in for one.
