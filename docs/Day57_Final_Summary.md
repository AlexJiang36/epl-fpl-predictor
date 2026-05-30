# Day57 Final Summary

## Goal
Improve match prediction quality beyond the current baseline / logistic-regression stack.

## What was completed

### 1. Match classification experiments on `match_features_v2`
Used the reproducible offline dataset:

- `artifacts/offline_datasets/match_features_gw1_27_v2.csv`

Evaluation setup:

- feature version: `v2`
- split_gw: `22`
- validation gw range: `23..27`

### 2. Logistic regression candidate
Trained `match_logreg_v2` on `match_features_v2`.

Result:
- validation accuracy: **0.1961**
- validation log loss: **1.5232**

Takeaway:
- log loss improved slightly versus the older simple stack
- accuracy did not improve meaningfully
- the model overpredicted draws and did not become the best match candidate

### 3. Gradient Boosting Classifier candidate
Trained `match_gbc_v2` as a tree-based classifier.

Initial result:
- validation accuracy: **0.3137**
- validation log loss: **1.3289**

This already outperformed `match_logreg_v2`.

### 4. GBC hyperparameter sweep
Ran a small sweep across:
- `n_estimators`
- `learning_rate`
- `max_depth`
- `min_samples_leaf`
- `subsample`

Best result found:

- model: `match_gbc_v2`
- `n_estimators = 100`
- `learning_rate = 0.03`
- `max_depth = 2`
- `min_samples_leaf = 3`
- `subsample = 1.0`

Metrics:
- validation accuracy: **0.352941**
- validation log loss: **1.220291**

This became the best match model found so far.

### 5. LightGBM match classifier
Tested `match_lgbm_v2` as another tree-based classifier.

Result:
- validation accuracy: **0.2941**
- validation log loss: **1.6361**

Takeaway:
- promising model family in general
- but this first match-side run did **not** beat the best GBC configuration

### 6. Lightweight goals regression prototype
Added a small prototype to predict:

- `home_goals`
- `away_goals`

Model used:
- `match_goals_gbr_v2`

Results:
- `home_goals_mae = 0.9436`
- `away_goals_mae = 0.8653`
- `avg_goal_mae = 0.9045`
- `avg_goal_rmse = 1.1260`

Takeaway:
- the match feature layer can already support a useful first goals-prediction path
- this is still experimental, but it provides a foundation for later scoreline / exact-score work

---

## Best match model after Day57

### Selected best candidate
**`match_gbc_v2`**

Recommended parameters:
- `n_estimators = 100`
- `learning_rate = 0.03`
- `max_depth = 2`
- `min_samples_leaf = 3`
- `subsample = 1.0`

### Why it was selected
It provided the best balance of:
- accuracy
- log loss
- practical stability on the current small validation window

---

## Match model comparison

### Current ranking
1. **GBC best** — accuracy `0.3529`, log loss `1.2203`
2. **GBC initial** — accuracy `0.3137`, log loss `1.3289`
3. **LightGBM** — accuracy `0.2941`, log loss `1.6361`
4. **Logistic regression v2** — accuracy `0.1961`, log loss `1.5232`

---

## Main conclusions
- Match-side tree models are clearly stronger than the current linear/logistic version.
- The new `match_features_v2` dataset is useful and supports both classification and lightweight goals prediction.
- Rest days, ranking-derived strength differences, and goal-difference context appear to be meaningful match features.
- A lightweight goals prototype is now in place, which opens the door to future scoreline forecasting.

---

## Recommendation
Keep the following as the current match-side production candidate:

- **`match_gbc_v2`** with the best sweep configuration

Keep these as experimental / reference paths:

- `match_logreg_v2`
- `match_lgbm_v2`
- `match_goals_gbr_v2` (experimental goals layer)
