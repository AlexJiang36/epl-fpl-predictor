from fastapi import FastAPI
from app.api.routes.health import router as health_router
from app.api.routes.db_ping import router as db_ping_router
from app.api.routes.teams import router as teams_router
from app.api.routes.players import router as players_router
from app.api.routes.ingest import router as ingest_router
from app.api.routes.fixtures import router as fixtures_router


app = FastAPI(title="EPL/FPL Predictor")

app.include_router(health_router)
app.include_router(db_ping_router)
app.include_router(teams_router)
app.include_router(players_router)
app.include_router(ingest_router)
app.include_router(fixtures_router)
