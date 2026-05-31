# Day58 Final Summary

## Goal
Choose a deliberate, limited production model set and avoid accumulating too many parallel active models.

## What was completed

### 1. Final model selection decision
A small production model set was chosen instead of keeping every recent experiment active.

#### Player-side production models
- **Primary:** `gbr_player_v2_1`
- **Secondary / backup:** `elasticnet_player_v2_1`

#### Match-side production models
- **Primary:** `match_gbc_v2`

#### Experimental model kept visible
- `match_goals_gbr_v2`

#### Archived / non-production models
- `ridge_player_v2_1`
- `lgbm_player_v2_1`
- `match_logreg_v2`
- `match_lgbm_v2`
- older rolling-average and older match baseline/logreg paths

---

## Why these models were selected

### Player-side
#### `gbr_player_v2_1`
Selected as the main player model because it was the strongest model in Day56 experiments.

Best result:
- **MAE = 0.947815**

It clearly beat:
- the strongest baseline
- ridge
- elastic net
- the first LightGBM run

#### `elasticnet_player_v2_1`
Selected as the backup player model because it was the best linear candidate.

Result:
- **MAE = 0.9917**

Why keep it active:
- more interpretable
- easier to explain
- simpler runtime behavior
- still clearly better than baseline methods

### Match-side
#### `match_gbc_v2`
Selected as the main match model because it was the strongest match classifier in Day57.

Best result:
- **accuracy = 0.352941**
- **log loss = 1.220291**

It clearly beat:
- `match_logreg_v2`
- `match_lgbm_v2`

#### `match_goals_gbr_v2`
Kept as **experimental** because it is useful and promising, but it is not the main match-result model.

Prototype results:
- `home_goals_mae = 0.9436`
- `away_goals_mae = 0.8653`
- `avg_goal_mae = 0.9045`

It remains valuable for future:
- scoreline prediction
- goals expectation layers
- futures / season-outlook work

---

## Metadata and API updates completed

### Metadata layer
The model metadata structure was extended to support:

- `status`
- `is_active`
- `is_production_default`
- `selected_reason`

### Seed metadata updated
Model metadata was seeded so selected models now carry explicit production status.

### `/models` endpoint updated
The player model listing now:
- includes metadata-defined models
- shows model status
- supports `active_only=true`

### `/match/predictions/models` endpoint updated
The match model listing now:
- includes metadata-defined match-result models
- shows model status
- supports `active_only=true`

---

## Resulting active model surfaces

### `/models?active_only=true`
Returns only:
- `gbr_player_v2_1`
- `elasticnet_player_v2_1`

### `/match/predictions/models?active_only=true`
Returns only:
- `match_gbc_v2`

This means the platform now has a real active production set instead of an uncontrolled model list.

---

## Final active set

### Player
- **Primary:** `gbr_player_v2_1`
- **Backup:** `elasticnet_player_v2_1`

### Match
- **Primary:** `match_gbc_v2`

### Experimental
- `match_goals_gbr_v2`

---

## Main conclusions
- The project now has a limited, deliberate production model set.
- Weaker or non-selected models are no longer treated as if they are equally active.
- The model listing APIs now reflect production status instead of only showing whatever exists in prediction tables.
- The project is now in a much better state for product clarity, demos, and future maintenance.

---

## Recommendation
Keep the active production model set intentionally small:

- `gbr_player_v2_1`
- `elasticnet_player_v2_1`
- `match_gbc_v2`

Keep:
- `match_goals_gbr_v2` as experimental

Archive the rest unless they are needed as baseline references.
