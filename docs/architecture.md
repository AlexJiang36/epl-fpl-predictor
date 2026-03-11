# Architecture (FPL ML Prediction Platform)

## Goal
A full-stack ML prediction platform for Fantasy Premier League:
- data ingestion
- feature pipelines
- training pipelines
- model registry
- prediction storage
- evaluation workflows
- inference APIs
- minimal UI to expose capabilities

## System Components

### 1) Database (Postgres)
Core tables:
- `players` (player metadata: position, cost, status, team_id)
- `player_gw_stats` (actuals per player per GW: minutes, goals, assists, clean sheets, total_points)
- `predictions` (model outputs per player per target_gw)
- `fixtures`, `teams`, `gameweeks` (context tables)

### 2) Backend API (FastAPI)
Primary responsibilities:
- serve predictions and squad recommendations
- expose model registry
- provide ingestion endpoints (if present)

Key endpoints:
- `GET /predictions` (filters + pagination + ordering)
- `GET /models` (model registry)
- `GET /teams`
- `GET /recommendations/squad`

### 3) Frontend (Next.js App Router)
Responsibilities:
- minimal UI for /predictions and /squad
- BFF proxy routes to backend (avoid CORS, hide backend URL)

Key pages:
- `/predictions`
- `/squad`

Key BFF routes:
- `/api/predictions` -> backend `/predictions`
- `/api/models` -> backend `/models`
- `/api/teams` -> backend `/teams`
- `/api/squad` -> backend `/recommendations/squad`

### 4) ML Pipelines (scripts / modules)
Responsibilities:
- run baseline models + ML models
- write outputs into `predictions` (model_name is the key)
- evaluation scripts to compare models

Artifacts:
- model outputs stored in DB (`predictions`)
- evaluation results stored as docs (`docs/evaluations/...`)

## Data Flow (High-level)
1. Ingest/update players, teams, fixtures, player_gw_stats
2. Run a model -> write predictions (player_id, target_gw, model_name, predicted_points)
3. `/models` lists available models from registry
4. `/predictions` queries predictions by target_gw + model_name
5. Evaluation scripts join predictions to `player_gw_stats` to compute MAE

## Guardrails
- Model/data first, avoid UI churn
- `/models` is the source of truth for model dropdown
- Predictions stored by `model_name`
- Manual fetch in UI (no auto refresh on typing)