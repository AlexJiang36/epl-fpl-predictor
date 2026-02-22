# Weekly Refresh Runbook (Terminal-Only)
## EPL-FPL Predictor
**Version:** v2 (Corrected API-route workflow)  
**Date:** 2026-02-22  
**Current target example:** GW27

## Purpose
This runbook is for the weekly refresh of FPL data and baseline predictions using backend API routes from terminal (`curl`), then validating backend, BFF, and UI.

This version is corrected for this project:
- Uses FastAPI routes (not `scripts/*`)
- Uses terminal commands (`curl`) instead of Swagger UI
- Matches current BFF + Predictions page behavior

## Preconditions
- Backend reachable at `http://127.0.0.1:8000`
- Frontend reachable at `http://localhost:3000`
- BFF routes working: `/api/predictions`, `/api/teams`, `/api/squad`

## Important Project Reminder
This project refresh flow is API-route driven.  
Do **not** use guessed Python script commands like `python -m scripts.*`.

---

## Weekly Refresh Workflow (Terminal-Only)

### Step 0 — Start services
Backend (Terminal A):
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Frontend (Terminal B):
```bash
cd frontend
npm run dev
```

### Step 1 — Refresh gameweeks metadata
```bash
curl -X POST "http://127.0.0.1:8000/gameweeks/ingest/fpl"
```

### Step 2 — Refresh fixtures
```bash
curl -X POST "http://127.0.0.1:8000/ingest/fpl/fixtures"
```

### Step 3 — Ingest finished gameweek stats (critical)
```bash
curl -X POST "http://127.0.0.1:8000/ingest/fpl/gw/finished"
```

### Step 4 — Run baseline predictions for target GW
Replace `27` with your target gameweek.
```bash
curl -X POST "http://127.0.0.1:8000/predictions/baseline/run?target_gw=27"
```

Optional variant if endpoint later requires `model_name`:
```bash
curl -X POST "http://127.0.0.1:8000/predictions/baseline/run?target_gw=27&model_name=baseline_rollavg_v0"
```

### Step 5 — Verify backend predictions (must pass)
```bash
curl -s "http://127.0.0.1:8000/predictions?target_gw=27&limit=20&offset=0&order_by=points"
```

Success criteria:
- JSON response
- `rows` non-empty
- `meta.total > 0`
- Rows show expected `target_gw`
- No `422` / `500`

### Step 6 — Verify frontend BFF route (recommended)
```bash
curl -s "http://localhost:3000/api/predictions?target_gw=27&limit=20&offset=0&order_by=points"
```

Success criteria:
- JSON response
- Non-empty rows
- No `Not Found`
- No `422`

### Step 7 — Verify Predictions page UI (recommended)
Open `http://localhost:3000/predictions`

Then:
1. Set `target_gw = 27`
2. Click **Fetch Predictions**

Expected behavior (current project):
- Manual fetch only (typing should not auto-refresh)
- Global sort is applied before client pagination
- Prev/Next paginates client-side after sorting

### Step 8 — Verify Squad page UI (optional)
Open `http://localhost:3000/squad`, set `target_gw = 27`, then generate/fetch squad.

---

## Quick Command Block (Copy/Paste)
Replace `27` with your target GW.
```bash
curl -X POST "http://127.0.0.1:8000/gameweeks/ingest/fpl"
curl -X POST "http://127.0.0.1:8000/ingest/fpl/fixtures"
curl -X POST "http://127.0.0.1:8000/ingest/fpl/gw/finished"
curl -X POST "http://127.0.0.1:8000/predictions/baseline/run?target_gw=27"
curl -s "http://127.0.0.1:8000/predictions?target_gw=27&limit=20&offset=0&order_by=points"
curl -s "http://localhost:3000/api/predictions?target_gw=27&limit=20&offset=0&order_by=points"
```


## Weekly Log Template
```md
### Weekly Refresh Log
- Date:
- Target GW:
- Commands run:
  - gameweeks ingest: success/fail
  - fixtures ingest: success/fail
  - finished gw ingest: success/fail
  - baseline run: success/fail
- Backend validation (/predictions): success/fail
- BFF validation (/api/predictions): success/fail
- UI validation (/predictions): success/fail
- UI validation (/squad): success/fail
- Notes:
```
