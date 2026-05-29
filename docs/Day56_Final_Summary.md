# Day56 Final Summary

## Goal
Add a small round of player intrinsic / availability proxies, then compare stronger player model candidates on the new feature set.

## Feature work completed
Built `player_features_v2_1` on top of `v2` by adding intrinsic / availability proxy features such as:

- `recent_zero_min_count`
- `recent_90_plus_count`
- `recent_start_like_rate`
- `minutes_std_roll5`
- `points_trend_last3_minus_last5`
- `minutes_trend_last3_minus_last5`
- `return_from_absence_flag`
- `matches_since_return`
- `minutes_drop_recent`

Generated reproducible dataset:

- `artifacts/offline_datasets/player_features_gw1_27_v2_1.csv`

Saved feature snapshot metadata successfully.

## Evaluation setup
- feature version: `v2_1`
- split_gw: `22`
- validation gw range: `23..27`

## Baseline results
- `pts_roll3_mean`: **1.006630**
- `pts_roll5_mean`: **1.043077**
- `pts_roll8_mean`: **1.065726**
- `pts_last1`: **1.080306**

Best baseline:
- **`pts_roll3_mean` = 1.006630**

## Model results

### Ridge
Best ridge from alpha sweep:
- `ridge_player_v2_1`
- `alpha = 10`
- **val_mae = 1.003169**

### Elastic Net
- `elasticnet_player_v2_1`
- `alpha = 0.01`
- `l1_ratio = 0.5`
- **val_mae = 0.9917**

### Gradient Boosting Regressor
Initial run:
- **0.9535**

Best sweep result:
- `gbr_player_v2_1`
- `n_estimators = 100`
- `learning_rate = 0.10`
- `max_depth = 3`
- `min_samples_leaf = 20`
- **val_mae = 0.947815**

### LightGBM
- `lgbm_player_v2_1`
- `n_estimators = 300`
- `learning_rate = 0.03`
- `num_leaves = 31`
- `min_child_samples = 30`
- `subsample = 0.9`
- `colsample_bytree = 0.9`
- **val_mae = 0.9517**

## Ranking of candidates
1. **GBR** — `0.947815`
2. **LightGBM** — `0.9517`
3. **Elastic Net** — `0.9917`
4. **Ridge** — `1.003169`
5. **Best baseline (`pts_roll3_mean`)** — `1.006630`

## Recommendation

### Best overall player model right now
**Keep `gbr_player_v2_1` as the current best player candidate.**

Recommended params:
- `n_estimators = 100`
- `learning_rate = 0.10`
- `max_depth = 3`
- `min_samples_leaf = 20`

### Best linear backup
**Keep `elasticnet_player_v2_1` as the best linear candidate.**

### LightGBM status
LightGBM is promising and very close, but for now it is still slightly behind the best GBR run.

## Interpretation
- The new `v2_1` intrinsic / availability features were useful.
- Tree-based models clearly outperformed linear models.
- Short-term availability and role-security signals remain extremely important.
- The project now has a clear stronger player-model candidate beyond simple rolling-average baselines.
