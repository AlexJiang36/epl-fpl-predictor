# Training Workflow (Player Points Models)

## Goal
Standardize how we create, evaluate, and publish a new model.

## Inputs
Database tables:
- `player_gw_stats` (actual points per GW)
- `players` (position, cost, status)

## Output
Write predictions to:
- `predictions(player_id, target_gw, model_name, predicted_points)`

## Workflow Steps

### Step 1 — Define model
- Choose `model_name` (stable identifier)
- Define feature set and label (what is being predicted)

### Step 2 — Create dataset (no leakage)
Rules:
- Features must use only past information relative to target_gw
- Use lag/rolling features shifted by 1 GW

### Step 3 — Train + validate
- Use time-aware split by GW (train early GWs, validate later GWs)
- Metrics:
  - MAE overall
  - MAE per-GW
  - (optional) per-position MAE

### Step 4 — Publish predictions for a target GW
- Generate predictions for `target_gw`
- Write rows into `predictions` under `model_name`

### Step 5 — Register model
- Ensure `/models` includes the new model (registry is source of truth for UI)

### Step 6 — Evaluate and document
- Save results to `docs/evaluations/model_results.md`
- Record:
  - dataset range
  - split strategy
  - features used
  - MAE numbers