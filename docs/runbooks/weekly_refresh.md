# Weekly Refresh Runbook (Local Dev)

## Prereqs
- Postgres running: `docker compose up -d`
- Backend running: `uvicorn app.main:app --reload --port 8000`
- Frontend running: `npm run dev`

## Environment
Local DB:
- user: app
- password: app
- db: epl
- host: localhost:5432

Example:
DATABASE_URL="postgresql://app:app@localhost:5432/epl"

## Step 1 — Verify DB is up
```bash
docker compose ps