# EPL-FPL Predictor (MVP)

*this reflects the progress untill Day20*

A full-stack Fantasy Premier League (FPL) predictor project with:

- **FastAPI backend** (predictions / teams / squad recommendation APIs)
- **Next.js frontend** (App Router + BFF route handlers + UI pages)

---

## Current MVP Status (Day20 Checkpoint)

### Working backend endpoints
- `GET /teams`
- `GET /predictions`
- `GET /recommendations/squad`

### Working frontend pages
- `/` (home)
- `/squad` (squad recommendation UI)
- `/predictions` (filters + global sort + client pagination)

### Working frontend BFF routes
- `/api/teams`
- `/api/predictions`
- `/api/squad`

---

## Tech Stack

### Backend
- FastAPI
- Python

### Frontend
- Next.js (App Router)
- React
- TypeScript

---

## Project Structure (High Level)

```text
epl-fpl-predictor/
├── backend/                 # FastAPI app, DB/model logic, scripts
├── frontend/                # Next.js frontend (UI + BFF route handlers)
│   ├── src/
│   │   ├── app/
│   │   │   ├── api/         # BFF route handlers (/api/*)
│   │   │   ├── squad/       # Squad UI page
│   │   │   ├── predictions/ # Predictions UI page
│   │   │   └── page.tsx     # Home page
│   │   └── lib/             # Client helpers / project-specific helpers
│   └── .env.local           # Frontend env (local only)
├── docs/                    # Runbooks / constraints / screenshots
└── README.md
```

---

## Run Locally

### 1) Start Backend

Open a terminal and go to the backend folder:

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Backend should be available at:
- `http://127.0.0.1:8000`

Backend smoke tests (recommended):
- `http://127.0.0.1:8000/teams`
- `http://127.0.0.1:8000/predictions?target_gw=26&limit=20&offset=0&order_by=points`

### 2) Configure Frontend Environment

Create `frontend/.env.local` with:

```ini
BACKEND_BASE_URL=http://127.0.0.1:8000
```

If you change `.env.local`, restart the Next.js dev server.

### 3) Start Frontend

Open a second terminal and go to the frontend folder:

```bash
cd frontend
npm install
npm run dev
```

Frontend should be available at:
- `http://localhost:3000`

---

## Frontend Pages (MVP)

### `/`
Home page (entry page)

### `/squad`
Squad recommendation page:
- calls frontend BFF route `/api/squad`
- displays squad result and summary cards

### `/predictions`
Predictions explorer page with:
- filters (`target_gw`, `position`, `status`, `team_id`, etc.)
- team dropdown loaded from `/api/teams`
- manual fetch only (typing does not auto-refresh)
- global sort + client pagination (Day19 behavior)

---

## Frontend BFF Routes (Next Route Handlers)

These routes proxy frontend requests to backend endpoints:

- `/api/teams` → backend `/teams`
- `/api/predictions` → backend `/predictions`
- `/api/squad` → backend `/recommendations/squad`

This avoids frontend CORS issues and keeps backend base URL behind the frontend.

---

## Important Constraints (Read Before Changing Code)

### 1) Backend `/predictions` has a limit cap
Large limits (for example `limit=1000`) may return **422**.

✅ Do this:
- Use `limit <= 200`
- If you need “all rows”, fetch multiple pages and merge results client-side

❌ Avoid this:
- Sending `limit=1000` directly to backend

### 2) `order_by` accepted values are restricted
Backend currently accepts only:
- `points`
- `cost`
- `value`

❌ Do not send:
- `predicted_points`
- `now_cost`
- other aliases (unless normalized first)

### 3) Team filter should use `team_id`
Predictions team filter should send:
- `team_id` (number)

❌ Do not use as backend filter:
- team name string (e.g. `"Arsenal"`)
- short name string (e.g. `"ARS"`)

### 4) Manual Fetch UX is intentional
On `/predictions`, changing filters should **not** auto-refresh results.

Current intended behavior:
1. User edits filters
2. User clicks **Fetch Predictions**
3. UI fetches data and updates results

---

## Day19 Behavior (Predictions Page)

Implemented on Day19:
- fetch predictions data from backend in pages (`limit <= 200`)
- merge all rows on frontend
- apply **global sorting** on merged rows
- apply **client-side pagination** (Prev / Next)

Why this matters:
- sorting only after backend pagination gives page-local sorting, not a true global ranking

---

## Debugging Checklist (Before Editing Code)

### A) Backend direct endpoints
Check backend first:
- `http://127.0.0.1:8000/teams`
- `http://127.0.0.1:8000/predictions?...`

If these fail, fix backend before touching frontend.

### B) Frontend BFF routes
Then check frontend proxy routes:
- `http://localhost:3000/api/teams`
- `http://localhost:3000/api/predictions?...`

If backend works but BFF fails, debug Next route handlers.

### C) If you see 422 on predictions
Check:
- request URL shown in UI
- `limit <= 200`
- `order_by` is one of `points|cost|value`

---

## Common Failure Patterns (Known Issues)

1. **422 from `/api/predictions`**
   - Usually caused by invalid query params (especially too-large `limit`)

2. **`{"detail":"Not Found"}` from `/api/teams`**
   - Route handler issue, path mismatch, or dev server not restarted

3. **Using team name / short name as filter**
   - Backend expects `team_id`

4. **Accidental auto-fetch on every keystroke**
   - Makes debugging state/pagination much harder

5. **Changing proxy helper signatures without checking call sites**
   - Can break route handlers unexpectedly

---

## Development Guardrails (for Future Chats / Refactors)

When asking for coding help, include these rules:

- **MINIMAL CHANGE PATCH**
- **NO NEW FILES** (unless strictly necessary)
- **Confirm proxy/helper function signature before changing route handlers**
- **BACKEND LIMIT CAP (`<=200`)**
- **ALLOWED `order_by` = `points|cost|value`**
- **TEAM FILTER = `team_id`**
- **MANUAL FETCH**
- **GLOBAL SORT THEN CLIENT PAGINATION**

---

## Recommended Screenshots (for docs / resume / demo)

Store screenshots under `docs/screenshots/`, for example:
- `docs/screenshots/day20_squad_page.png`
- `docs/screenshots/day20_predictions_global_sort.png`

Suggested captures:
1. `/squad` page with a generated squad and summary
2. `/predictions` page showing:
   - global sort working
   - “Showing 1–20 of N”
   - filters + team dropdown

---

## Next Planned Work (Post-Day20)

- **Day21:** Weekly refresh runbook (GW update workflow)
- **Day22:** Baseline evaluation metric (e.g., MAE)
- **Day23:** Model registry endpoint + dynamic model dropdown
- **Day24:** Small cache for BFF routes
- **Day25+:** Error UX / tests / logging / demo script / resume bullets

---

## Quick Start Summary (TL;DR)

```bash
# terminal 1 (backend)
cd backend
uvicorn app.main:app --reload --port 8000
```

```bash
# terminal 2 (frontend)
cd frontend
# create frontend/.env.local with BACKEND_BASE_URL=http://127.0.0.1:8000
npm install
npm run dev
```

Then open:
- `http://localhost:3000/squad`
- `http://localhost:3000/predictions`

---

## Notes
This README reflects the **Day20 MVP freeze checkpoint** after Day19 predictions global-sort/client-pagination work.
