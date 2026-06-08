# EPL / FPL Predictor

A full-stack football analytics project focused on **Fantasy Premier League decision support**, **match forecasting**, and **model evaluation**.

This project combines:
- a **FastAPI backend** for data, predictions, recommendations, and evaluation APIs
- a **Next.js frontend** for interactive product surfaces
- a **PostgreSQL + SQLAlchemy + Alembic** data layer
- an **ML workflow** for feature exports, model training, evaluation, and artifact tracking

---

## Why this project is interesting

This is not just a single prediction script. It is a small end-to-end platform that shows:

- **backend API design**
- **database schema design and migrations**
- **ML feature engineering and model comparison**
- **evaluation and reproducibility workflows**
- **frontend product thinking**
- **season-aware data modeling for long-term maintainability**

---

## Current capabilities

### 1. Player prediction workflow
The platform generates player-level gameweek forecasts and supports multiple model candidates.

Current work includes:
- baseline rolling-average player models
- richer feature exports for player form, minutes, team context, and fixture context
- multiple model experiments, including linear and tree-based approaches
- validation and comparison using **MAE**

### 2. FPL recommendation layer
The backend supports recommendation flows built on top of model outputs, including:
- squad recommendation
- transfer recommendation
- model-aware decision support for FPL-style planning

### 3. Match prediction workflow
The project also supports match-level forecasting, including:
- match result prediction
- multiple match model candidates
- evaluation using **accuracy** and **log loss**
- a lightweight goals regression prototype for richer football outputs

### 4. Evaluation product surface
The project now includes a historical evaluation layer so model quality is visible rather than hidden in notebooks or terminal logs.

This includes:
- model metadata
- active vs archived model status
- production default selection
- evaluation API endpoints
- a frontend evaluation dashboard for model comparison

### 5. Reproducibility and operations
The project tracks artifacts and operational workflows more deliberately than a typical class project.

Examples:
- feature snapshot exports
- model metadata artifacts
- refresh / validation snapshots
- season-aware export naming
- database migrations with Alembic

---

## Product surface

### Backend APIs
Examples of current API surface include:
- team and player data endpoints
- player prediction endpoints
- model listing endpoints
- squad and transfer recommendation endpoints
- match prediction endpoints
- evaluation summary endpoints

### Frontend pages
The frontend includes product-style pages such as:
- player predictions explorer
- squad recommendation page
- evaluation dashboard

The goal is not only to compute predictions, but to present them in a way that is understandable and demo-friendly.

---

## ML and evaluation workflow

### Player-side workflow
- export season-aware player feature datasets
- train / compare multiple models
- evaluate using MAE
- track feature versions and snapshot artifacts

### Match-side workflow
- export season-aware match datasets
- compare multiple match model candidates
- evaluate with accuracy and log loss
- prototype goals prediction for richer output layers

### Model management
The project supports:
- active vs archived model states
- production defaults
- model metadata for training / evaluation context
- historical comparison via API and dashboard views

---

## Season-aware design

A recent project milestone was moving from implicit single-season assumptions to explicit season-aware workflows.

This includes:
- `season` added to core fact tables
- season-aware prediction writes
- season-aware feature exports
- season-aware artifact naming such as:

```text
player_features_2025_26_gw1_27_v2.csv
match_features_2025_26_gw1_27_v2.csv
```

This makes the project much more realistic for future season rollover, historical backtesting, and previous-season priors.

---

## Architecture overview

```text
epl-fpl-predictor/
├── backend/
│   ├── app/                  # FastAPI routes, models, schemas, DB integration
│   ├── ml/                   # feature exports, training, evaluation, validation
│   ├── alembic/              # database migrations
│   └── artifacts/            # snapshots, metadata, evaluation outputs
├── frontend/
│   └── src/app/              # Next.js App Router pages and BFF routes
├── docs/                     # plans, runbooks, design notes
└── README.md
```

---

## Tech stack

### Backend
- Python 3.9
- FastAPI
- SQLAlchemy
- Alembic
- PostgreSQL 16
- Docker / Docker Compose

### Frontend
- Next.js
- React
- TypeScript
- Tailwind CSS

### ML / data
- pandas
- scikit-learn
- LightGBM experiments
- artifact-based evaluation workflow

---

## What this project demonstrates

From an engineering perspective, this project demonstrates the ability to:

- design a **full-stack application** instead of only a model notebook
- build a **database-backed prediction system**
- implement **feature engineering pipelines**
- compare models using real evaluation metrics
- build **API + frontend surfaces** around ML outputs
- improve reproducibility with **snapshots, metadata, and migrations**
- evolve a project from MVP into a more production-minded architecture

---

## Run locally

### 1. Start the database
From the project root:

```bash
docker compose up -d db
```

### 2. Start the backend

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Backend:
- `http://127.0.0.1:8000`

### 3. Configure the frontend
Create `frontend/.env.local`:

```ini
BACKEND_BASE_URL=http://127.0.0.1:8000
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

### 4. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend:
- `http://localhost:3000`

---

## Example local workflows

### Export player features
```bash
python -m ml.validation.export_player_feature_snapshot \
  --gw-start 1 \
  --gw-end 27 \
  --feature-version v2 \
  --model-name baseline_rollavg_v0
```

### Export match features
```bash
python -m ml.validation.export_match_feature_snapshot \
  --gw-start 1 \
  --gw-end 27 \
  --feature-version v2 \
  --model-name match_baseline_v0 \
  --n-form 5 \
  --n-h2h 3
```

### Run evaluation dashboard
Open:
- `http://localhost:3000/evaluation`

---

## Current project direction

Near-term work focuses on:
- completing season-aware workflows across more routes
- improving historical backtesting
- integrating previous-season priors
- strengthening model selection and evaluation surfaces
- improving dashboard polish for demos and portfolio use

---

## Notes

This project is built as a portfolio-quality engineering project rather than a one-off hackathon demo. The emphasis is on:
- clear architecture
- iterative model improvement
- reproducibility
- product-facing presentation of ML outputs
