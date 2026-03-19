## GW30 Evaluation (n=10 finished fixtures)

**Dataset**
- Range: GW30
- Join: `fixtures (finished + scores)` × `match_predictions` on `fixture_id`
- Labels: `y_true ∈ {H, D, A}` from `(home_score, away_score)`

### Summary Metrics

| Model | Rows | Accuracy | Log Loss |
|---|---:|---:|---:|
| match_baseline_v0 | 10 | 0.4000 | 1.1762 |
| match_logreg_v0 | 10 | 0.2000 | 1.5399 |

### Confusion Matrices (rows=true, cols=pred)

**match_baseline_v0**
|  | pred_H | pred_D | pred_A |
|---|---:|---:|---:|
| true_H | 2 | 0 | 0 |
| true_D | 2 | 1 | 3 |
| true_A | 1 | 0 | 1 |

**match_logreg_v0**
|  | pred_H | pred_D | pred_A |
|---|---:|---:|---:|
| true_H | 1 | 1 | 0 |
| true_D | 4 | 0 | 2 |
| true_A | 1 | 0 | 1 |

### Interpretation (short)
- On GW30, `match_baseline_v0` outperforms `match_logreg_v0` on both **accuracy** (0.40 vs 0.20) and **log loss** (1.1762 vs 1.5399).
- The small sample size (10 matches) makes the comparison noisy, but this snapshot suggests the current logreg feature set may not generalize well.
- Most errors come from **Draw** cases (true_D=6): both models struggle to identify draws; logreg predicts no draws in this GW.
- Next step: re-train logreg on a broader finished window and/or enrich features beyond points form (e.g., goals for/against form, home/away splits, H2H summary).


## GW27–GW29 Evaluation (n=30 finished fixtures)

**Dataset**
- Range: GW27–GW29
- Join: `fixtures (finished + scores)` × `match_predictions` on `fixture_id`
- Labels: `y_true ∈ {H, D, A}` from `(home_score, away_score)`

### Summary Metrics

| Model | Rows | Accuracy | Log Loss |
|---|---:|---:|---:|
| match_baseline_v0 | 30 | 0.5333 | 1.2265 |
| match_logreg_v0 | 30 | 0.5667 | 0.8661 |

### Confusion Matrices (rows=true, cols=pred)

**match_baseline_v0**
|  | pred_H | pred_D | pred_A |
|---|---:|---:|---:|
| true_H | 5 | 3 | 3 |
| true_D | 3 | 3 | 0 |
| true_A | 2 | 3 | 8 |

**match_logreg_v0**
|  | pred_H | pred_D | pred_A |
|---|---:|---:|---:|
| true_H | 3 | 2 | 6 |
| true_D | 2 | 3 | 1 |
| true_A | 2 | 0 | 11 |

### Interpretation (short)
- The two models are close in **accuracy** (logreg +3.3pp ≈ +1 correct match out of 30).
- `match_logreg_v0` is notably better on **log loss** (0.8661 vs 1.2265), suggesting **better-calibrated probabilities** even when accuracy gains are modest.
- Error pattern: logreg improves **Away-win** detection (true_A→pred_A = 11 vs 8), but tends to over-predict Away wins on some true_H cases (true_H→pred_A = 6).
- Next step to widen the gap: enrich features beyond points form (e.g., goals for/against form, home/away splits, H2H summary).

