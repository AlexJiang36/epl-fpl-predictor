# Day58 Model Selection

## Goal
Choose a deliberate, limited production model set and stop accumulating too many parallel active models.

## Selection principles
The production set should balance:

- predictive quality
- stability
- explainability
- runtime complexity
- reproducibility
- future maintainability

The platform should not keep every model active just because it exists.
A smaller, well-justified model set is better for both product clarity and engineering discipline.

---

## Final production decision

### Player models

#### Active / production
1. **`gbr_player_v2_1`**
2. **`elasticnet_player_v2_1`**

#### Experimental / archived
- `ridge_player_v2_1`
- `lgbm_player_v2_1`
- old rolling-average style internal experiments
- sweep-only candidate variants

---

### Match models

#### Active / production
1. **`match_gbc_v2`**

#### Experimental
- `match_goals_gbr_v2`

#### Archived / non-production
- `match_logreg_v2`
- `match_lgbm_v2`
- older simple match baseline / logreg paths

---

## Why each production model was selected

# Player-side decisions

## 1. `gbr_player_v2_1` — selected as the primary player production model

### Why it was selected
This is the strongest player model found in recent experiments.

Best result:
- **MAE = 0.947815**

It clearly outperformed:
- the strongest baseline (`pts_roll3_mean`)
- ridge
- elastic net
- the first LightGBM run

### Strengths
- best predictive performance so far
- captures nonlinear interactions between recent form, availability, and team/opponent context
- uses the stronger `v2_1` player feature layer
- strong fit for the current player prediction problem

### Weaknesses
- less interpretable than linear models
- slightly more complex to explain and debug
- could be more sensitive to small training-window changes than a linear backup

### Final status
**Active**
**Primary player production model**

---

## 2. `elasticnet_player_v2_1` — selected as the linear backup player model

### Why it was selected
This was the best linear-style model in the Day56 experiments.

Result:
- **MAE = 0.9917**

It clearly beat:
- ridge
- rolling-average baselines

### Why it is worth keeping active
Even though it is not the best overall model, it provides:

- better explainability
- simpler runtime behavior
- easier debugging
- easier communication in demos and interviews

It is a strong “backup production” model because it helps the platform avoid depending only on a single tree model.

### Strengths
- best linear candidate
- more interpretable than GBR or LightGBM
- still meaningfully better than baseline methods
- good reproducibility and lower complexity

### Weaknesses
- clearly weaker than the best GBR model
- may miss nonlinear interactions that matter for player performance

### Final status
**Active**
**Secondary / backup player production model**

---

## Why the other player models were not selected

### `ridge_player_v2_1`
Not selected because:
- it beat baseline only slightly
- elastic net was clearly better
- it does not provide enough value beyond elastic net

### `lgbm_player_v2_1`
Not selected as active because:
- it was strong
- but still slightly worse than the best GBR run

It remains interesting, but not necessary in the active set right now.

### Older rolling-average models
Not selected as active because:
- they are useful reference baselines
- but weaker than the new `v2_1` model stack

They should be retained as baseline references or archived, not as active production defaults.

---

# Match-side decisions

## 3. `match_gbc_v2` — selected as the primary match production model

### Why it was selected
This was the strongest match-result classifier found in Day57.

Best result:
- **accuracy = 0.352941**
- **log loss = 1.220291**

It clearly outperformed:
- `match_logreg_v2`
- `match_lgbm_v2`

### Why this matters
This was the first match-side model that gave a clearly better result than the simple logreg path on the `match_features_v2` dataset.

### Strengths
- best match classification performance so far
- tree-based model can use nonlinear football context features better than logreg
- uses the stronger v2 match feature layer
- clear improvement in both accuracy and log loss

### Weaknesses
- still not highly accurate in absolute terms
- validation window is small
- still somewhat imperfect at handling away-win cases

### Final status
**Active**
**Primary match production model**

---

## 4. `match_goals_gbr_v2` — kept as experimental, not full production

### Why it was not promoted to main production
This model is useful, but it solves a different problem:

- predicting home goals
- predicting away goals

rather than directly acting as the main W/D/L production classifier.

It is valuable as an experimental extension and future building block, but it is not yet the main production surface.

### Why it is still important
It provides:
- a first goals-prediction layer
- a path toward scoreline / exact-score outputs
- a foundation for later futures / team-outlook work

### Current results
- `home_goals_mae = 0.9436`
- `away_goals_mae = 0.8653`
- `avg_goal_mae = 0.9045`

These are useful prototype results, but the model should still be clearly marked experimental.

### Final status
**Experimental**
**Keep visible, but not as the main production default**

---

## Why the other match models were not selected

### `match_logreg_v2`
Not selected because:
- it did not become a clear stronger candidate
- it overpredicted draws
- its overall performance was weaker than the best GBC model

It may still be kept as a simple reference model, but not as active production.

### `match_lgbm_v2`
Not selected because:
- the first match-side run did not beat GBC
- on the current small match dataset, GBC looked more stable and stronger

This can remain archived / experimental, but not active.

### Older simple match baseline / old logreg path
Not selected because:
- these were useful stepping stones
- but no longer represent the best available match models

They should remain reference / archived paths only.

---

## Final active model set

## Player production models
- **Primary:** `gbr_player_v2_1`
- **Secondary / backup:** `elasticnet_player_v2_1`

## Match production models
- **Primary:** `match_gbc_v2`

## Experimental models still worth keeping
- `match_goals_gbr_v2`

---

## Archived / non-production models

### Player
- `ridge_player_v2_1`
- `lgbm_player_v2_1`
- older rolling-average-only experimental paths

### Match
- `match_logreg_v2`
- `match_lgbm_v2`
- older simple baseline / logreg match paths

---

## Recommended metadata statuses

### Player
- `gbr_player_v2_1` → `active`
- `elasticnet_player_v2_1` → `active`
- `ridge_player_v2_1` → `archived`
- `lgbm_player_v2_1` → `experimental`

### Match
- `match_gbc_v2` → `active`
- `match_goals_gbr_v2` → `experimental`
- `match_logreg_v2` → `archived`
- `match_lgbm_v2` → `archived`

---

## Recommended defaults for product surfaces

### Player default model
**`gbr_player_v2_1`**

### Player explainable backup
**`elasticnet_player_v2_1`**

### Match default model
**`match_gbc_v2`**

### Experimental advanced layer
**`match_goals_gbr_v2`**

---

## Notes for `/models` and `/match/predictions/models`

The model listing endpoints should ideally expose:

- `model_name`
- `task_type`
- `feature_version`
- `status`
- `is_active`
- `is_production_default`
- `notes`
- key metrics summary

This will make the model layer easier to explain and easier to maintain.

---

## Final recommendation
Do **not** keep all recent models active.

The best production set right now is intentionally small:

- `gbr_player_v2_1`
- `elasticnet_player_v2_1`
- `match_gbc_v2`

with:

- `match_goals_gbr_v2` kept as experimental

This is enough to keep the platform strong, explainable, and manageable without turning it into a cluttered model zoo.
