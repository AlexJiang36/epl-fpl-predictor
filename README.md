# EPL-FPL Predictor (MVP)

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




