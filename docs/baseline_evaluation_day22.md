# Day22 Baseline Evaluation (MAE)

## Date
2026-02-22

## Goal
Evaluate the current baseline player prediction model (`baseline_rollavg_v0`) using MAE (Mean Absolute Error) by joining prediction outputs with actual player gameweek stats.

## Evaluation Script
- `backend/app/services/eval_baseline_mae.py`

## Command Run
```bash
cd backend
source .venv/bin/activate
python3 -m app.services.eval_baseline_mae
```

## Output Summary
- **Model:** `baseline_rollavg_v0`
- **GW filter:** none (all available GWs in the matched dataset)
- **Matched rows:** `3223`

## Per-GW MAE
| GW | N | MAE |
|---:|---:|---:|
| 23 | 802 | 1.1111 |
| 24 | 802 | 1.0723 |
| 25 | 802 | 0.9956 |
| 26 | 817 | 1.0453 |

## Overall MAE
- **1.0560** (N = 3223)

## Metric Definition
MAE is the average absolute difference between predicted points and actual points:

- `MAE = mean(|predicted_points - actual_points|)`

Lower MAE indicates better prediction accuracy.

## Notes
- This is a baseline evaluation result and can be used as a reference point for future model comparisons.
- The evaluation joins predictions and actual stats on:
  - `player_id`
  - `target_gw == gw`
